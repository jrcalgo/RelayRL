pub(crate) mod interface;
pub(super) mod ops;
pub(super) mod policies;

use crate::network::client::agent::ModelMode;
use crate::network::client::runtime::data::transport_sink::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::TransportError;
use crate::network::client::runtime::data::transport_sink::zmq::ops::ZmqPoolError;
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::Algorithm;

use relayrl_types::HyperparameterArgs;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use active_uuid_registry::{ContextString, NamespaceString, registry_uuid::Uuid};

use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc::Sender;

#[derive(Debug, Error, Clone)]
pub enum ZmqClientError {
    #[error("ZMQ transport error: {0}")]
    ZmqTransportError(String),
    #[error("Max retries exceeded after {attempts} attempts: {cause}")]
    MaxRetriesExceeded { cause: String, attempts: u32 },
    #[error("Circuit breaker open - server appears unavailable")]
    CircuitOpen,
    #[error("Backpressure limit exceeded - too many concurrent requests")]
    BackpressureExceeded,
    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),
    #[error("Dispatcher error: {0}")]
    InvalidState(String),
    #[error("Task join error: {0}")]
    JoinError(String),
    #[error(transparent)]
    ZmqPoolError(#[from] ZmqPoolError),
}
pub(super) trait ZmqInferenceExecution {
    fn execute_send_inference_request(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError>;
    fn execute_send_flag_last_inference(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        reward: &f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_inference_model_init_request<B: Backend + BackendMatcher<Backend = B>>(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        model_mode: &ModelMode,
        model_module: &Option<ModelModule<B>>,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
}

pub(super) trait ZmqTrainingExecution<B: Backend + BackendMatcher<Backend = B>> {
    fn execute_listen_for_model(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
        model_update_tx: &Sender<RoutedMessage>,
        model_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_algorithm_init_request(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        actor_entries: &[(NamespaceString, ContextString, Uuid)],
        model_mode: &ModelMode,
        algorithm: &Algorithm,
        hyperparams: &HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_initial_model_handshake(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn execute_send_trajectory(
        &self,
        buffer_entry: &(NamespaceString, ContextString, Uuid),
        encoded_trajectory: &EncodedTrajectory,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
    fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError>;
}

#[cfg(test)]
mod tests {}
