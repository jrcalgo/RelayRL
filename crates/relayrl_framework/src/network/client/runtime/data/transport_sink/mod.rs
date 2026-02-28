use crate::network::TransportType;
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::zmq::ZmqClientError;
use crate::network::client::runtime::data::transport_sink::zmq::interface::ZmqInterface;
use crate::network::client::runtime::router::{InferenceRequest, RoutedMessage};
use crate::network::client::agent::ModelMode;
use crate::prelude::network::ClientModes;
use crate::utilities::configuration::{Algorithm, ClientConfigLoader};

use active_uuid_registry::UuidPoolError;
use relayrl_types::HyperparameterArgs;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::BackendMatcher;
use relayrl_types::data::trajectory::EncodedTrajectory;
use relayrl_types::model::ModelModule;

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

#[cfg(feature = "nats-transport")]
pub(crate) mod nats;
#[cfg(feature = "zmq-transport")]
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
    SendScalingCompleteError(String),
    #[error("Send client IDs to server failed: {0}")]
    SendClientIdsToServerError(String),
    #[error("Send shutdown signal to server failed: {0}")]
    SendShutdownSignalError(String),
    #[error("Send algorithm init request failed: {0}")]
    SendAlgorithmInitRequestError(String),
    #[cfg(feature = "zmq-transport")]
    #[error(transparent)]
    ZmqClientError(#[from] ZmqClientError),
    #[cfg(feature = "nats-transport")]
    #[error("NATS transport error: {0}")]
    NatsClientError(String),
    #[error("Max transport retries exceeded: {cause}, attempts: {attempts}")]
    MaxRetriesExceeded { cause: String, attempts: u32 },
    #[error("Circuit open, server appears unavailable")]
    CircuitOpen,
    #[error("Invalid state: {0}")]
    InvalidState(String),
    #[error("Task join error: {0}")]
    JoinError(String),
    #[error("Multiple errors: \"{0}\" and \"{1}\"")]
    MultipleErrors(String, String),
}

pub(crate) enum ClientTransportInterface<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "zmq-transport")]
    Sync(Box<dyn SyncClientTransportInterface<B>>),
    #[cfg(feature = "nats-transport")]
    Async(Box<dyn AsyncClientTransportInterface<B>>),
}

#[cfg(feature = "nats-transport")]
pub(crate) trait AsyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    AsyncClientInferenceTransportOps<B> + AsyncClientTrainingTransportOps<B>
{
    async fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
    ) -> Result<Self, TransportError>
    where
        Self: Sized;
    async fn shutdown(
        &self,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    SyncClientInferenceTransportOps<B> + SyncClientTrainingTransportOps<B>
{
    fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
    ) -> Result<Self, TransportError>
    where
        Self: Sized;
    fn shutdown(&self, server_addresses: SharedTransportAddresses) -> Result<(), TransportError>;
}

#[cfg(feature = "nats-transport")]
#[async_trait]
pub(crate) trait AsyncClientInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncClientScalingTransportOps<B>
{
    async fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        server_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError>;
    async fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncClientScalingTransportOps<B>
{
    fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        server_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError>;
    fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "nats-transport")]
#[async_trait]
pub(crate) trait AsyncClientTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncClientScalingTransportOps<B>
{
    async fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        server_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    async fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncClientScalingTransportOps<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        server_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "nats-transport")]
pub(crate) trait AsyncClientScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    async fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

pub(crate) fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
    transport_type: TransportType,
    client_namespace: Arc<str>,
    shared_client_modes: Arc<ClientModes>,
) -> Result<ClientTransportInterface<B>, TransportError> {
    match transport_type {
        #[cfg(feature = "zmq-transport")]
        TransportType::ZMQ => Ok(ClientTransportInterface::<B>::Sync(Box::new(
            ZmqInterface::<B>::new(client_namespace, shared_client_modes)
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
        #[cfg(feature = "nats-transport")]
        TransportType::NATS => Ok(ClientTransportInterface::<B>::Async(Box::new(
            NatsInterface::<B>::new(client_namespace, shared_client_modes)
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
    }
}
