use crate::network::TransportType;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::zmq::ZmqClientError;
use crate::network::client::runtime::data::transport_sink::zmq::interface::ZmqInterface;
use crate::network::client::runtime::router::{InferenceRequest, RoutedMessage};
use crate::utilities::configuration::{Algorithm, ClientConfigLoader};

use active_uuid_registry::UuidPoolError;
use relayrl_types::HyperparameterArgs;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::EncodedTrajectory;
use relayrl_types::types::model::ModelModule;

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

#[cfg(feature = "nats_transport")]
pub(crate) mod nats;
#[cfg(feature = "zmq_transport")]
pub(crate) mod zmq;

pub(crate) mod transport_dispatcher;

type TransportUuid = Uuid;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("Transport initilization failed: {0}")]
    TransportInitializationError(String),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
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
    SendScalingCompleteError(Strinssg),
    #[error("Send client IDs to server failed: {0}")]
    SendClientIdsToServerError(String),
    #[error("Send shutdown signal to server failed: {0}")]
    SendShutdownSignalError(String),
    #[error("Send algorithm init request failed: {0}")]
    SendAlgorithmInitRequestError(String),
    #[cfg(feature = "zmq_transport")]
    #[error("ZMQ transport error: {0}")]
    ZmqClientError(String),
    #[cfg(feature = "nats_transport")]
    #[error("NATS transport error: {0}")]
    NatsClientError(String),
}

pub(crate) enum ClientTransportInterface<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "sync_transport")]
    Sync(Box<dyn SyncClientTransportInterface<B>>),
    #[cfg(feature = "async_transport")]
    Async(Box<dyn AsyncClientTransportInterface<B>>),
}

#[cfg(feature = "async_transport")]
pub(crate) trait AsyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    AsyncClientInferenceTransportOps<B> + AsyncClientTrainingTransportOps<B>
{
    async fn new() -> Result<Self, TransportError>
    where
        Self: Sized;
    async fn shutdown(&self, server_addresses: ServerAddresses) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    SyncClientInferenceTransportOps<B> + SyncClientTrainingTransportOps<B>
{
    fn new(shared_client_capabilities: Arc<ClientCapabilities>) -> Result<Self, TransportError>
    where
        Self: Sized;
    fn shutdown(&self, server_addresses: ServerAddresses) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
#[async_trait]
pub(crate) trait AsyncClientInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncClientScalingTransportOps<B>
{
    async fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        server_addresses: ServerAddresses,
    ) -> Result<RelayRLAction, TransportError>;
    async fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncClientInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncClientScalingTransportOps<B>
{
    fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        server_addresses: ServerAddresses,
    ) -> Result<RelayRLAction, TransportError>;
    fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
#[async_trait]
pub(crate) trait AsyncClientTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncClientScalingTransportOps<B>
{
    async fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    async fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        server_addresses: ServerAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    async fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    async fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncClientTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncClientScalingTransportOps<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        server_addresses: ServerAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
pub(crate) trait AsyncClientScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    async fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    async fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncClientScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
    fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        server_addresses: ServerAddresses,
    ) -> Result<(), TransportError>;
}

pub(crate) fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
    transport_type: TransportType,
    shared_client_capabilities: Arc<ClientCapabilities>,
) -> Result<ClientTransportInterface<B>, TransportError> {
    match transport_type {
        #[cfg(feature = "zmq_transport")]
        TransportType::ZMQ => Ok(ClientTransportInterface::<B>::Sync(Box::new(
            ZmqInterface::<B>::new(shared_client_capabilities)
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
        #[cfg(feature = "nats_transport")]
        TransportType::NATS => Ok(ClientTransportInterface::<B>::Async(Box::new(
            NatsInterface::<B>::new(shared_client_capabilities)
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
    }
}
