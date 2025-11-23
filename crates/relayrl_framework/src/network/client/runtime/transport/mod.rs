use crate::network::TransportType;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::ClientConfigLoader;

use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::EncodedTrajectory;
use relayrl_types::types::model::ModelModule;

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

pub mod tonic;
pub mod zmq;

type TransportUuid = Uuid;

static TRANSPORT_IDX: Vec<TransportUuid> = Vec::new();

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("No transport configured: {0}")]
    NoTransportConfiguredError(String),
    #[error("Model handshake failed: {0}")]
    ModelHandshakeError(String),
    #[error("Send trajectory failed: {0}")]
    SendTrajError(String),
    #[error("Listen for model failed: {0}")]
    ListenForModelError(String),
    #[error("Send scaling warning failed: {0}")]
    SendScalingWarningError(String),
    #[error("Send scaling complete failed: {0}")]
    SendScalingCompleteError(String),
}

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
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    async fn send_traj_to_server(
        &self,
        encoded_trajectory: EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn listen_for_model(
        &self,
        agent_listener_address: &str,
        global_dispatcher_tx: Sender<RoutedMessage>,
    ) -> Result<(), TransportError>;
    async fn send_scaling_warning(&self, operation: ScalingOperation)
    -> Result<(), TransportError>;
    async fn send_scaling_complete(
        &self,
        operation: ScalingOperation,
    ) -> Result<(), TransportError>;
    async fn shutdown(&self) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq_network")]
pub trait SyncClientTransport<B: Backend + BackendMatcher<Backend = B>>: Send + Sync {
    fn initial_model_handshake(
        &self,
        model_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn send_traj_to_server(
        &self,
        encoded_trajectory: EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError>;
    fn listen_for_model(
        &self,
        agent_listener_address: &str,
        global_dispatcher_tx: Sender<RoutedMessage>,
    ) -> Result<(), TransportError>;
    fn send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), TransportError>;
    fn send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), TransportError>;
    fn shutdown(&self) -> Result<(), TransportError>;
}

pub fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
    transport_type: TransportType,
) -> TransportClient<B> {
    match transport_type {
        #[cfg(feature = "grpc_network")]
        TransportType::GRPC => TransportClient::<B>::Async(Box::new(tonic::TonicClient::new())),
        #[cfg(feature = "zmq_network")]
        TransportType::ZMQ => TransportClient::<B>::Sync(Box::new(zmq::ZmqClient::new())),
    }
}
