//! Transport dispatch abstractions for experimental client networking paths.
//!
//! These components sit behind `zmq-transport` and `nats-transport`. They remain experimental in
//! `0.5.0-beta`; the local/default client runtime is the supported beta path.

use crate::network::TransportType;
use crate::network::client::agent::ModelMode;
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
#[cfg(feature = "nats-transport")]
use crate::network::client::runtime::data::sinks::transport_sink::nats::interface::NatsInterface;
#[cfg(feature = "zmq-transport")]
use crate::network::client::runtime::data::sinks::transport_sink::zmq::ZmqClientError;
#[cfg(feature = "zmq-transport")]
use crate::network::client::runtime::data::sinks::transport_sink::zmq::interface::ZmqInterface;
use crate::network::client::runtime::router::RoutedMessage;
use crate::prelude::network::ClientModes;
use crate::utilities::configuration::Algorithm;

use relayrl_types::HyperparameterArgs;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::BackendMatcher;
use relayrl_types::data::trajectory::EncodedTrajectory;
use relayrl_types::model::ModelModule;

use active_uuid_registry::{ContextString, NamespaceString, UuidPoolError, registry_uuid::Uuid};

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc::Sender;

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

fn combine_scaling_results(
    result1: Option<Result<(), TransportError>>,
    result2: Option<Result<(), TransportError>>,
) -> Result<(), TransportError> {
    match (result1, result2) {
        (Some(Err(e)), Some(Err(e2))) => Err(TransportError::MultipleErrors(
            e.to_string(),
            e2.to_string(),
        )),
        (Some(Err(e)), None) => Err(e),
        (None, Some(Err(e))) => Err(e),
        (None, None) => Err(TransportError::InvalidState(
            "Received a scaling operation before either transport server was initialized"
                .to_string(),
        )),
        _ => Ok(()),
    }
}

pub(crate) enum ClientTransportInterface<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "zmq-transport")]
    Sync(Box<dyn SyncClientTransportInterface<B>>),
    #[cfg(feature = "nats-transport")]
    Async(Box<dyn AsyncClientTransportInterface<B>>),
}

#[cfg(feature = "nats-transport")]
#[async_trait]
pub(crate) trait AsyncClientTransportInterface<B: Backend + BackendMatcher<Backend = B>>:
    AsyncClientInferenceTransportOps<B> + AsyncClientTrainingTransportOps<B>
{
    async fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
    ) -> Result<Self, TransportError>
    where
        Self: Sized;
    async fn shutdown(&self) -> Result<(), TransportError>;
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
    fn shutdown(&self) -> Result<(), TransportError>;
}

#[cfg(feature = "nats-transport")]
#[async_trait]
pub(crate) trait AsyncClientInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncClientScalingTransportOps<B>
{
    async fn send_inference_model_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        model_mode: ModelMode,
        model_module: Option<ModelModule<B>>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_inference_request(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        obs_bytes: Vec<u8>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError>;
    async fn send_flag_last_inference(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        reward: f32,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientInferenceTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncClientScalingTransportOps<B>
{
    fn send_inference_model_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        model_mode: ModelMode,
        model_module: Option<ModelModule<B>>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_inference_request(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        obs_bytes: Vec<u8>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError>;
    fn send_flag_last_inference(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        reward: f32,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "nats-transport")]
#[async_trait]
pub(crate) trait AsyncClientTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + AsyncClientScalingTransportOps<B>
{
    async fn send_algorithm_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn initial_model_handshake(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    async fn send_trajectory(
        &self,
        buffer_entry: (NamespaceString, ContextString, Uuid),
        encoded_trajectory: EncodedTrajectory,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn listen_for_model(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
        model_update_tx: Sender<RoutedMessage>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn stop_model_listener(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientTrainingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync + SyncClientScalingTransportOps<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn initial_model_handshake(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn send_trajectory(
        &self,
        buffer_entry: (NamespaceString, ContextString, Uuid),
        encoded_trajectory: EncodedTrajectory,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn listen_for_model(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
        model_update_tx: Sender<RoutedMessage>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn stop_model_listener(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "nats-transport")]
#[async_trait]
pub(crate) trait AsyncClientScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    async fn send_client_ids(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        client_ids: Vec<(NamespaceString, ContextString, Uuid)>,
        replace_context: bool,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_scaling_warning(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_scaling_complete(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    async fn send_shutdown_signal(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

#[cfg(feature = "zmq-transport")]
pub(crate) trait SyncClientScalingTransportOps<B: Backend + BackendMatcher<Backend = B>>:
    Send + Sync
{
    fn send_client_ids(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        client_ids: Vec<(NamespaceString, ContextString, Uuid)>,
        replace_context: bool,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_scaling_warning(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_scaling_complete(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
    fn send_shutdown_signal(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError>;
}

pub(crate) async fn client_transport_factory<B: Backend + BackendMatcher<Backend = B>>(
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
                .await
                .map_err(|e| TransportError::TransportInitializationError(e.to_string()))?,
        ))),
    }
}
