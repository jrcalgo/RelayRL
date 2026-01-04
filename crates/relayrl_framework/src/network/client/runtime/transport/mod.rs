use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::router::{InferenceRequest, RoutedMessage};
use crate::network::{HyperparameterArgs, TransportType};
use crate::utilities::configuration::{Algorithm, ClientConfigLoader};

use relayrl_types::Hyperparams;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::EncodedTrajectory;
use relayrl_types::types::model::ModelModule;
use active_uuid_registry::UuidPoolError;

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

#[cfg(feature = "zmq_transport")]
pub mod zmq;

pub mod transport_dispatcher;
pub use transport_dispatcher::{
    BackpressureController, CircuitBreaker, CircuitState, DispatcherConfig, DispatcherError,
    InferenceDispatcher, RetryPolicy, ScalingDispatcher, TrainingDispatcher,
};

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
}

pub(crate) enum TransportClient<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "sync_transport")]
    Sync(Box<dyn SyncClientTransport<B>>),
    #[cfg(feature = "async_transport")]
    Async(Box<dyn AsyncClientTransport<B>>),
}

#[cfg(feature = "async_transport")]
pub(crate) trait AsyncClientTransport<B: Backend + BackendMatcher<Backend = B>>:
    AsyncInferenceServerTransport<B> + AsyncTrainingServerTransport<B>
{
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncClientTransport<B: Backend + BackendMatcher<Backend = B>>:
    SyncInferenceServerTransport<B> + SyncTrainingServerTransport<B>
{
}

#[cfg(feature = "async_transport")]
#[async_trait]
pub(crate) trait AsyncInferenceServerTransport<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    async fn send_action_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError>;
    async fn send_flag_last_action(
        &self,
        actor_id: &Uuid,
        reward: f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncInferenceServerTransport<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    fn send_action_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError>;
    fn send_flag_last_action(
        &self,
        actor_id: &Uuid,
        reward: f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "async_transport")]
#[async_trait]
pub(crate) trait AsyncTrainingServerTransport<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    async fn send_client_ids_to_server(
        &self,
        scaling_id: &Uuid,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
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
    async fn send_traj_to_server(
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
    async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn send_shutdown_signal_to_server(
        &self,
        scaling_id: &Uuid,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn shutdown(&self) -> Result<(), TransportError>;
}

#[cfg(feature = "sync_transport")]
pub(crate) trait SyncTrainingServerTransport<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    fn send_client_ids_to_server(
        &self,
        scaling_id: &Uuid,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
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
    fn send_traj_to_server(
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
    fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn send_shutdown_signal_to_server(
        &self,
        scaling_id: &Uuid,
        scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn shutdown(&self) -> Result<(), TransportError>;
}

pub(crate) fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
    transport_type: TransportType,
) -> Result<TransportClient<B>, TransportError> {
    match transport_type {
        #[cfg(feature = "sync_transport")]
        TransportType::ZMQ => Ok(TransportClient::<B>::Sync(Box::new(
            zmq::ZmqClient::new()
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
    }
}
