
use crate::network::TransportType;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::ClientConfigLoader;

use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::EncodedTrajectory;
use relayrl_types::types::model::ModelModule;

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use serde::Serialize;
use serde_pickle as pickle;
use std::io::Cursor;
use tokio::sync::mpsc::Sender;

pub mod tonic;
pub mod zmq;

pub enum TransportClient<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "zmq_network")]
    Sync(Box<dyn SyncClientTransport<B> + Send + Sync>),
    #[cfg(feature = "grpc_network")]
    Async(Box<dyn AsyncClientTransport<B> + Send + Sync>),
}

#[cfg(feature = "grpc_network")]
#[async_trait]
pub trait AsyncClientTransport<B: Backend + BackendMatcher<Backend = B>>: Send + Sync {
    async fn initial_model_handshake(
        &self,
        model_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, String>;
    async fn send_traj_to_server(
        &self,
        encoded_trajectory: EncodedTrajectory,
        training_server_address: &str,
    );
    async fn listen_for_model(&self, model_server_address: &str);
    async fn send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), String>;
    async fn send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), String>;
}

#[cfg(feature = "zmq_network")]
pub trait SyncClientTransport<B: Backend + BackendMatcher<Backend = B>>: Send + Sync {
    fn initial_model_handshake(
        &self,
        model_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, String>;
    fn send_traj_to_server(
        &self,
        encoded_trajectory: EncodedTrajectory,
        training_server_address: &str,
    ) -> Result<(), String>;
    fn listen_for_model(&self, model_server_address: &str, tx_to_router: Sender<RoutedMessage>);
    fn send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), String>;
    fn send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), String>;
}

pub fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
    transport_type: TransportType,
    config: &ClientConfigLoader,
) -> TransportClient<B> {
    match transport_type {
        #[cfg(feature = "grpc_network")]
        TransportType::GRPC => {
            TransportClient::<B>::Async(Box::new(tonic::TonicClient::new(config)))
        }
        #[cfg(feature = "zmq_network")]
        TransportType::ZMQ => TransportClient::<B>::Sync(Box::new(zmq::ZmqClient::new(config))),
    }
}
