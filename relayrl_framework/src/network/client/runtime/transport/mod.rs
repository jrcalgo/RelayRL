use crate::network::TransportType;
use crate::network::client::runtime::router::RoutedMessage;
use crate::types::trajectory::RelayRLTrajectory;
use crate::utilities::configuration::ClientConfigLoader;
use async_trait::async_trait;
use serde::Serialize;
use serde_pickle as pickle;
use std::io::Cursor;
use std::sync::Arc;
use tch::CModule;
use tokio::sync::mpsc::Sender;

pub mod tonic;
pub mod zmq;

#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::tonic::rl_service::Trajectory;

pub fn serialize_trajectory<T: Serialize>(trajectory: &T) -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    pickle::to_writer(&mut buf, trajectory, Default::default())
        .expect("Failed to serialize trajectory");
    buf.into_inner()
}

pub enum TransportClient {
    #[cfg(feature = "zmq_network")]
    Sync(Box<dyn SyncClientTransport + Send + Sync>),
    #[cfg(feature = "grpc_network")]
    Async(Box<dyn AsyncClientTransport + Send + Sync>),
}

#[cfg(feature = "grpc_network")]
#[async_trait]
pub trait AsyncClientTransport: Send + Sync {
    async fn initial_model_handshake(
        &self,
        model_server_address: &str,
    ) -> Result<Option<CModule>, String>;
    async fn send_traj_to_server(&self, trajectory: Trajectory, training_server_address: &str);
    async fn listen_for_model(&self, model_server_address: &str);
    fn convert_relayrl_to_proto_trajectory(&self, traj: &RelayRLTrajectory) -> Trajectory;
}

#[cfg(feature = "zmq_network")]
pub trait SyncClientTransport: Send + Sync {
    fn initial_model_handshake(
        &self,
        model_server_address: &str,
    ) -> Result<Option<CModule>, String>;
    fn send_traj_to_server(
        &self,
        trajectory: RelayRLTrajectory,
        training_server_address: &str,
    ) -> Result<(), String>;
    fn listen_for_model(&self, model_server_address: &str, tx_to_router: Sender<RoutedMessage>);
}

pub fn client_transport_factory(
    transport_type: TransportType,
    config: &ClientConfigLoader,
) -> TransportClient {
    match transport_type {
        #[cfg(feature = "grpc_network")]
        TransportType::GRPC => TransportClient::Async(Box::new(tonic::TonicClient::new(config))),
        #[cfg(feature = "zmq_network")]
        TransportType::ZMQ => TransportClient::Sync(Box::new(zmq::ZmqClient::new(config))),
    }
}
