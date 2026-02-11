pub(crate) mod interface;
pub(crate) mod ops;
pub(crate) mod policies;

use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::data::transport_sink::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::TransportError;
use crate::utilities::configuration::Algorithm;

use relayrl_types::HyperparameterArgs;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

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
}

pub(crate) trait ZmqInferenceExecution {
    fn execute_send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<RelayRLAction, TransportError>;
    fn execute_send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &Vec<(String, Uuid)>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
}

pub(crate) trait ZmqTrainingExecution<B: Backend + BackendMatcher<Backend = B>> {
    fn execute_send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_initial_model_handshake(
        &self,
        actor_id: &Uuid,
        model_server_address: &str,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    fn execute_send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &Vec<(String, Uuid)>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError>;
}
