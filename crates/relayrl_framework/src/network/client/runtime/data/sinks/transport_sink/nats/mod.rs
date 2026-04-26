pub(crate) mod interface;
pub(super) mod ops;
pub(super) mod policies;

use crate::network::client::agent::ModelMode;
use crate::network::client::runtime::data::sinks::transport_sink::ScalingOperation;
use crate::network::client::runtime::data::sinks::transport_sink::TransportError;
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
use thiserror::Error;
use tokio::sync::mpsc::Sender;

pub(super) mod inference_subjects {
    pub(super) const INFERENCE_REQUEST_SUBJECT: &str = "inference-server.model.inference.request";
    pub(super) const FLAG_LAST_INFERENCE_SUBJECT: &str =
        "inference-server.model.inference.flag-last";

    pub(super) const INFERENCE_MODEL_INIT_REQUEST_SUBJECT: &str = "inference-server.model.init";

    pub(super) const INFERENCE_SCALING_CLIENT_IDS_SUBJECT: &str =
        "inference-server.scaling.client-ids";
    pub(super) const INFERENCE_SCALING_WARNING_SUBJECT: &str = "inference-server.scaling.warning";
    pub(super) const INFERENCE_SCALING_COMPLETE_SUBJECT: &str = "inference-server.scaling.complete";
    pub(super) const INFERENCE_SCALING_SHUTDOWN_SUBJECT: &str = "inference-server.scaling.shutdown";
}

pub(super) mod training_subjects {
    pub(super) const TRAINING_MODEL_LISTENING_SUBJECT: &str = "training-server.model.listening";
    pub(super) const TRAINING_MODEL_HANDSHAKE_SUBJECT: &str = "training-server.model.handshake";

    pub(super) const TRAINING_ALGORITHM_INIT_REQUEST_SUBJECT: &str =
        "training-server.algorithm.init";

    pub(super) const TRAINING_SEND_TRAJECTORY_SUBJECT: &str = "training-server.trajectory";

    pub(super) const TRAINING_SCALING_CLIENT_IDS_SUBJECT: &str =
        "training-server.scaling.client-ids";
    pub(super) const TRAINING_SCALING_WARNING_SUBJECT: &str = "training-server.scaling.warning";
    pub(super) const TRAINING_SCALING_COMPLETE_SUBJECT: &str = "training-server.scaling.complete";
    pub(super) const TRAINING_SCALING_SHUTDOWN_SUBJECT: &str = "training-server.scaling.shutdown";
}

#[derive(Debug, Error, Clone)]
pub enum NatsClientError {
    #[error("NATS transport error: {0}")]
    NatsTransportError(String),
    #[error("Task join error: {0}")]
    JoinError(String),
}

pub(super) trait NatsInferenceExecution {
    async fn execute_send_inference_request(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError>;
    async fn execute_send_flag_last_inference(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        reward: &f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_inference_model_init_request<B: Backend + BackendMatcher<Backend = B>>(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        model_mode: &ModelMode,
        model_module: &Option<ModelModule<B>>,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        inference_server_address: &str,
    ) -> Result<(), TransportError>;
}

pub(super) trait NatsTrainingExecution<B: Backend + BackendMatcher<Backend = B>> {
    async fn execute_listen_for_model(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
        model_update_tx: &Sender<RoutedMessage>,
        training_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_algorithm_init_request(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        actor_entries: &[(NamespaceString, ContextString, Uuid)],
        model_mode: &ModelMode,
        algorithm: &Algorithm,
        hyperparams: &HashMap<Algorithm, HyperparameterArgs>,
        training_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_initial_model_handshake(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        training_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError>;
    async fn execute_send_trajectory(
        &self,
        buffer_entry: &(NamespaceString, ContextString, Uuid),
        encoded_trajectory: &EncodedTrajectory,
        training_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        training_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_server_address: &str,
    ) -> Result<(), TransportError>;
    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        training_server_address: &str,
    ) -> Result<(), TransportError>;
}

#[cfg(test)]
mod tests {}
