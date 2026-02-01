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
    SendScalingCompleteError(String),
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

pub(crate) enum TransportClient<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "sync_transport")]
    Sync(Box<dyn SyncClientTransportInterface<B>>),
    #[cfg(feature = "async_transport")]
    Async(Box<dyn AsyncClientTransportInterface<B>>),
}

#[cfg(feature = "async_transport")]
pub(crate) trait AsyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    AsyncInferenceTransportOps<B> + AsyncTrainingTransportOps<B>
{
    async fn new() -> Result<Self, TransportError>
    where
        Self: Sized;
    async fn shutdown(&self, shared_server_addresses: Arc<RwLock<ServerAddresses>>) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    SyncInferenceTransportOps<B> + SyncTrainingTransportOps<B>
{
    fn new() -> Result<Self, TransportError>
    where
        Self: Sized;
    fn shutdown(&self, shared_server_addresses: Arc<RwLock<ServerAddresses>>) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
#[async_trait]
pub(crate) trait AsyncInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncScalingTransportOps<B>
{
    async fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError>;
    async fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncScalingTransportOps<B>
{
    fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError>;
    fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
#[async_trait]
pub(crate) trait AsyncTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncScalingTransportOps<B>
{
    async fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError>;
    async fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        model_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    async fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        agent_listener_address: &str,
        global_dispatcher_tx: Sender<RoutedMessage>,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncScalingTransportOps<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError>;
    fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        model_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError>;
    fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        agent_listener_address: &str,
        global_dispatcher_tx: Sender<RoutedMessage>,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
pub(crate) trait AsyncScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    async fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    async fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
}

pub(crate) fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
    transport_type: TransportType,
    shared_client_capabilities: Arc<ClientCapabilities>,
) -> Result<TransportClient<B>, TransportError> {
    match transport_type {
        #[cfg(feature = "zmq_transport")]
        TransportType::ZMQ => Ok(TransportClient::<B>::Sync(Box::new(
            ZmqInterface::<B>::new(shared_client_capabilities)
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
        #[cfg(feature = "nats_transport")]
        TransportType::NATS => Ok(TransportClient::<B>::Async(Box::new(
            NatsInterface::<B>::new(shared_client_capabilities)
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
    }
}
