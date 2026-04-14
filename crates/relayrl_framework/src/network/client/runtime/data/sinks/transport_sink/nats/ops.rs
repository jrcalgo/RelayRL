//! NATS transport operations for the experimental client transport path.
//!
//! The local/default client runtime is the supported `0.5.0-beta` path. NATS-backed workflows in
//! this module remain experimental.

use crate::network::client::agent::ModelMode;
use crate::network::client::runtime::data::transport_sink::{ScalingOperation, TransportError};
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::utilities::configuration::Algorithm;

use super::inference_subjects::{
    FLAG_LAST_INFERENCE_SUBJECT, INFERENCE_MODEL_INIT_REQUEST_SUBJECT, INFERENCE_REQUEST_SUBJECT,
    INFERENCE_SCALING_CLIENT_IDS_SUBJECT, INFERENCE_SCALING_COMPLETE_SUBJECT,
    INFERENCE_SCALING_SHUTDOWN_SUBJECT, INFERENCE_SCALING_WARNING_SUBJECT,
};
use super::training_subjects::{
    TRAINING_ALGORITHM_INIT_REQUEST_SUBJECT, TRAINING_MODEL_HANDSHAKE_SUBJECT,
    TRAINING_MODEL_LISTENING_SUBJECT, TRAINING_SCALING_CLIENT_IDS_SUBJECT,
    TRAINING_SCALING_COMPLETE_SUBJECT, TRAINING_SCALING_SHUTDOWN_SUBJECT,
    TRAINING_SCALING_WARNING_SUBJECT, TRAINING_SEND_TRAJECTORY_SUBJECT,
};
use super::{NatsInferenceExecution, NatsTrainingExecution};

use relayrl_types::HyperparameterArgs;
use relayrl_types::model::utils::validate_module;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use active_uuid_registry::{ContextString, NamespaceString, registry_uuid::Uuid};

use burn_tensor::backend::Backend;
use bytes::Bytes;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use tokio_stream::{Stream, StreamExt};

// Every payload struct is serialized to `bytes::Bytes` via bincode v2 and sent
// as the body of a NATS/JetStream message.
mod payloads {
    use crate::utilities::configuration::Algorithm;
    use relayrl_types::HyperparameterArgs;
    use relayrl_types::prelude::trajectory::EncodedTrajectory;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Serialize, Deserialize)]
    pub(super) struct InferenceRequestPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) actor_namespace: String,
        pub(super) actor_context: String,
        pub(super) actor_id: String,
        pub(super) obs_bytes: Vec<u8>,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct FlagLastInferencePayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) actor_namespace: String,
        pub(super) actor_context: String,
        pub(super) actor_id: String,
        pub(super) reward: f32,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct InferenceModelInitRequestPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) scaling_namespace: String,
        pub(super) scaling_context: String,
        pub(super) scaling_id: String,
        pub(super) model_mode_string: String,
        pub(super) model_files_bundle_bytes: Vec<u8>,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ClientIdEntry {
        pub(super) namespace: String,
        pub(super) context: String,
        pub(super) id: String,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ClientIdsPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) scaling_namespace: String,
        pub(super) scaling_context: String,
        pub(super) scaling_id: String,
        pub(super) client_id_entries: Vec<ClientIdEntry>,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ScalingOperationPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) scaling_namespace: String,
        pub(super) scaling_context: String,
        pub(super) scaling_id: String,
        pub(super) operation_string: String,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ShutdownPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) scaling_namespace: String,
        pub(super) scaling_context: String,
        pub(super) scaling_id: String,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ModelHandshakeRequestPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) actor_namespace: String,
        pub(super) actor_context: String,
        pub(super) actor_id: String,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ModelHandshakeResponsePayload {
        pub(super) model_files_bundle_bytes: Vec<u8>,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct AlgorithmInitRequestPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) scaling_namespace: String,
        pub(super) scaling_context: String,
        pub(super) scaling_id: String,
        pub(super) actor_entries_string: String,
        pub(super) model_mode_string: String,
        pub(super) algorithm: Algorithm,
        pub(super) hyperparams: HashMap<Algorithm, HyperparameterArgs>,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct TrajectoryPublishPayload {
        pub(super) transport_namespace: String,
        pub(super) transport_context: String,
        pub(super) buffer_namespace: String,
        pub(super) buffer_context: String,
        pub(super) buffer_id: String,
        pub(super) encoded_trajectory: EncodedTrajectory,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ModelUpdateBroadcastMessage {
        pub(super) model_bytes: Vec<u8>,
        pub(super) actor_id_bytes: Vec<u8>,
        pub(super) model_version: i64,
    }

    #[derive(Serialize, Deserialize)]
    pub(super) struct ModelFilesBundle {
        pub(super) metadata_json_bytes: Vec<u8>,
        pub(super) model_file_name: String,
        pub(super) model_file_bytes: Vec<u8>,
    }
}

use payloads::{
    AlgorithmInitRequestPayload, ClientIdEntry, ClientIdsPayload, FlagLastInferencePayload,
    InferenceModelInitRequestPayload, InferenceRequestPayload, ModelFilesBundle,
    ModelHandshakeRequestPayload, ModelHandshakeResponsePayload, ModelUpdateBroadcastMessage,
    ScalingOperationPayload, ShutdownPayload, TrajectoryPublishPayload,
};

fn serialize_payload_to_nats_bytes<T: Serialize>(
    payload: &T,
    operation_name: &str,
) -> Result<Bytes, TransportError> {
    bincode::serde::encode_to_vec(payload, bincode::config::standard())
        .map(Bytes::from)
        .map_err(|serialization_error| {
            TransportError::InvalidState(format!(
                "Failed to serialize {} payload: {}",
                operation_name, serialization_error
            ))
        })
}

fn deserialize_nats_response_bytes<T: for<'de> Deserialize<'de>>(
    response_bytes: &[u8],
    operation_name: &str,
) -> Result<T, TransportError> {
    bincode::serde::decode_from_slice(response_bytes, bincode::config::standard())
        .map(|(deserialized_value, _bytes_consumed)| deserialized_value)
        .map_err(|deserialization_error| {
            TransportError::InvalidState(format!(
                "Failed to deserialize {} response: {}",
                operation_name, deserialization_error
            ))
        })
}

fn serialize_model_module_to_bundle<B: Backend + BackendMatcher<Backend = B>>(
    model_module: &ModelModule<B>,
) -> Result<ModelFilesBundle, TransportError> {
    let model_serialization_temp_dir =
        tempfile::TempDir::new().map_err(|temp_dir_creation_error| {
            TransportError::InvalidState(format!(
                "Failed to create temporary directory for model serialization: {}",
                temp_dir_creation_error
            ))
        })?;

    model_module
        .save(model_serialization_temp_dir.path())
        .map_err(|model_save_error| {
            TransportError::InvalidState(format!(
                "Failed to save model module to temporary directory: {:?}",
                model_save_error
            ))
        })?;

    let metadata_json_file_path = model_serialization_temp_dir.path().join("metadata.json");
    let metadata_json_bytes =
        std::fs::read(&metadata_json_file_path).map_err(|metadata_read_error| {
            TransportError::InvalidState(format!(
                "Failed to read metadata.json from temporary directory: {}",
                metadata_read_error
            ))
        })?;

    let raw_metadata_json_string = String::from_utf8_lossy(&metadata_json_bytes).into_owned();
    let metadata_value: serde_json::Value = serde_json::from_str(&raw_metadata_json_string)
        .map_err(|json_parse_error| {
            TransportError::InvalidState(format!(
                "Failed to parse metadata.json for model file name extraction: {}",
                json_parse_error
            ))
        })?;

    let model_file_name = metadata_value
        .get("model_file")
        .and_then(|model_file_value| model_file_value.as_str())
        .ok_or_else(|| {
            TransportError::InvalidState(
                "metadata.json does not contain a valid 'model_file' field".to_string(),
            )
        })?
        .to_string();

    let model_file_path = model_serialization_temp_dir.path().join(&model_file_name);
    let model_file_bytes = std::fs::read(&model_file_path).map_err(|model_file_read_error| {
        TransportError::InvalidState(format!(
            "Failed to read model file '{}' from temporary directory: {}",
            model_file_name, model_file_read_error
        ))
    })?;

    Ok(ModelFilesBundle {
        metadata_json_bytes,
        model_file_name,
        model_file_bytes,
    })
}

fn deserialize_model_module_from_bundle<B: Backend + BackendMatcher<Backend = B>>(
    model_files_bundle: ModelFilesBundle,
) -> Result<ModelModule<B>, TransportError> {
    let model_deserialization_temp_dir =
        tempfile::TempDir::new().map_err(|temp_dir_creation_error| {
            TransportError::ModelHandshakeError(format!(
                "Failed to create temporary directory for model deserialization: {}",
                temp_dir_creation_error
            ))
        })?;

    let metadata_json_file_path = model_deserialization_temp_dir.path().join("metadata.json");
    std::fs::write(
        &metadata_json_file_path,
        &model_files_bundle.metadata_json_bytes,
    )
    .map_err(|metadata_write_error| {
        TransportError::ModelHandshakeError(format!(
            "Failed to write metadata.json to temporary directory: {}",
            metadata_write_error
        ))
    })?;

    let model_file_path = model_deserialization_temp_dir
        .path()
        .join(&model_files_bundle.model_file_name);
    std::fs::write(&model_file_path, &model_files_bundle.model_file_bytes).map_err(
        |model_file_write_error| {
            TransportError::ModelHandshakeError(format!(
                "Failed to write model file '{}' to temporary directory: {}",
                model_files_bundle.model_file_name, model_file_write_error
            ))
        },
    )?;

    let loaded_model_module = ModelModule::<B>::load_from_path(
        model_deserialization_temp_dir.path(),
    )
    .map_err(|model_load_error| {
        TransportError::ModelHandshakeError(format!(
            "Failed to load model module from temporary directory: {:?}",
            model_load_error
        ))
    })?;

    validate_module::<B>(&loaded_model_module).map_err(|model_validation_error| {
        TransportError::ModelHandshakeError(format!(
            "Failed to validate deserialized model module: {:?}",
            model_validation_error
        ))
    })?;

    Ok(loaded_model_module)
}

// Computes a u64 hash of the server address string using `DefaultHasher`.
// This gives O(1) equality comparison between the cached fingerprint and the
// incoming address without storing or comparing the full string.
fn address_fingerprint(server_address: &str) -> u64 {
    let mut address_hasher = DefaultHasher::new();
    server_address.hash(&mut address_hasher);
    address_hasher.finish()
}

fn build_routed_model_update_message(
    model_update_payload_bytes: &[u8],
) -> Result<RoutedMessage, TransportError> {
    let deserialized_model_update_broadcast: ModelUpdateBroadcastMessage =
        deserialize_nats_response_bytes(model_update_payload_bytes, "model update broadcast")?;

    let model_bytes_vector = deserialized_model_update_broadcast.model_bytes;
    let actor_id_bytes_vector = deserialized_model_update_broadcast.actor_id_bytes;
    let model_version = deserialized_model_update_broadcast.model_version;

    if model_bytes_vector.is_empty() {
        return Err(TransportError::ListenForModelError(
            "Received model update broadcast with empty model bytes".to_string(),
        ));
    }

    if actor_id_bytes_vector.len() != 16 {
        return Err(TransportError::ListenForModelError(format!(
            "Received model update broadcast with invalid actor ID byte length: \
             expected 16, got {}",
            actor_id_bytes_vector.len()
        )));
    }

    let actor_id_byte_array: [u8; 16] =
        actor_id_bytes_vector
            .as_slice()
            .try_into()
            .map_err(|conversion_error| {
                TransportError::ListenForModelError(format!(
                    "Failed to convert actor ID bytes to fixed-size array: {}",
                    conversion_error
                ))
            })?;

    Ok(RoutedMessage {
        actor_id: Uuid::from_bytes(actor_id_byte_array),
        protocol: RoutingProtocol::ModelUpdate,
        payload: RoutedPayload::ModelUpdate {
            model_bytes: model_bytes_vector,
            version: model_version,
        },
    })
}

async fn forward_model_update_payloads<S, F, Fut>(
    mut payload_stream: S,
    global_dispatcher_tx: &Sender<RoutedMessage>,
    mut is_shutting_down: F,
) -> Result<(), TransportError>
where
    S: Stream<Item = Bytes> + Unpin,
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    while let Some(model_update_payload) = payload_stream.next().await {
        let routed_model_update_message =
            build_routed_model_update_message(model_update_payload.as_ref())?;

        if let Err(send_error) = global_dispatcher_tx.send(routed_model_update_message).await {
            if is_shutting_down().await {
                return Ok(());
            }

            return Err(TransportError::ListenForModelError(format!(
                "Failed to dispatch model update message to global dispatcher: {}",
                send_error
            )));
        }
    }

    if is_shutting_down().await {
        Ok(())
    } else {
        Err(TransportError::ListenForModelError(
            "Model update subscriber closed unexpectedly".to_string(),
        ))
    }
}

struct NatsModelListenerHandle {
    receiver_id: Uuid,
    listener_shutdown: Arc<AtomicBool>,
    model_listener_shutdown_flags: Arc<DashMap<Uuid, Arc<AtomicBool>>>,
}

impl NatsModelListenerHandle {
    fn is_stopping(&self) -> bool {
        self.listener_shutdown.load(Ordering::SeqCst)
    }
}

impl Drop for NatsModelListenerHandle {
    fn drop(&mut self) {
        let should_remove = self
            .model_listener_shutdown_flags
            .get(&self.receiver_id)
            .map(|entry| Arc::ptr_eq(entry.value(), &self.listener_shutdown))
            .unwrap_or(false);

        if should_remove {
            self.model_listener_shutdown_flags.remove(&self.receiver_id);
        }
    }
}

pub(super) struct NatsConnectionManager {
    pub(super) client_namespace: Arc<str>,
    inference_client: Option<async_nats::Client>,
    inference_address_fingerprint: Option<u64>,
    training_client: Option<async_nats::Client>,
    training_jetstream_context: Option<async_nats::jetstream::Context>,
    training_address_fingerprint: Option<u64>,
    model_listener_shutdown_flags: Arc<DashMap<Uuid, Arc<AtomicBool>>>,
    shutting_down: bool,
}

struct NatsShutdownClients {
    client_namespace: Arc<str>,
    inference_client: Option<async_nats::Client>,
    training_client: Option<async_nats::Client>,
}

impl NatsConnectionManager {
    pub(super) fn new(client_namespace: Arc<str>) -> Self {
        Self {
            client_namespace,
            inference_client: None,
            inference_address_fingerprint: None,
            training_client: None,
            training_jetstream_context: None,
            training_address_fingerprint: None,
            model_listener_shutdown_flags: Arc::new(DashMap::new()),
            shutting_down: false,
        }
    }

    pub(super) fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    fn register_model_listener(&self, receiver_id: &Uuid) -> NatsModelListenerHandle {
        let listener_shutdown = self
            .model_listener_shutdown_flags
            .entry(*receiver_id)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .clone();
        listener_shutdown.store(false, Ordering::SeqCst);

        NatsModelListenerHandle {
            receiver_id: *receiver_id,
            listener_shutdown,
            model_listener_shutdown_flags: self.model_listener_shutdown_flags.clone(),
        }
    }

    fn stop_model_listener(&self, receiver_id: &Uuid) {
        if let Some(listener_shutdown) = self.model_listener_shutdown_flags.get(receiver_id) {
            listener_shutdown.store(true, Ordering::SeqCst);
        }
    }

    fn begin_shutdown(&mut self) -> Option<NatsShutdownClients> {
        if self.shutting_down {
            return None;
        }

        self.shutting_down = true;
        for listener_shutdown in self.model_listener_shutdown_flags.iter() {
            listener_shutdown.value().store(true, Ordering::SeqCst);
        }
        self.inference_address_fingerprint = None;
        self.training_jetstream_context = None;
        self.training_address_fingerprint = None;

        Some(NatsShutdownClients {
            client_namespace: self.client_namespace.clone(),
            inference_client: self.inference_client.take(),
            training_client: self.training_client.take(),
        })
    }

    /// Returns a clone of the cached inference `Client`, reconnecting first if
    /// `nats_inference_server_address` differs from the cached fingerprint.
    pub(super) async fn get_inference_client(
        &mut self,
        nats_inference_server_address: &str,
    ) -> Result<async_nats::Client, TransportError> {
        if self.shutting_down {
            return Err(TransportError::InvalidState(
                "NATS transport is shutting down".to_string(),
            ));
        }

        let incoming_address_fingerprint = address_fingerprint(nats_inference_server_address);

        let address_has_changed = match self.inference_address_fingerprint {
            Some(cached_address_fingerprint) => {
                cached_address_fingerprint != incoming_address_fingerprint
            }
            None => true,
        };

        if address_has_changed {
            if let Some(existing_inference_client) = self.inference_client.take() {
                existing_inference_client
                    .drain()
                    .await
                    .map_err(|drain_error| {
                        TransportError::InvalidState(format!(
                            "Failed to drain existing inference NATS client \
                             before reconnecting to '{}': {}",
                            nats_inference_server_address, drain_error
                        ))
                    })?;
            }

            let new_inference_client = async_nats::connect(nats_inference_server_address)
                .await
                .map_err(|nats_connection_error| {
                    TransportError::NatsClientError(format!(
                        "Failed to connect inference NATS client to '{}': {}",
                        nats_inference_server_address, nats_connection_error
                    ))
                })?;

            self.inference_client = Some(new_inference_client);
            self.inference_address_fingerprint = Some(incoming_address_fingerprint);
        }

        self.inference_client.clone().ok_or_else(|| {
            TransportError::InvalidState(
                "Inference NATS client is unavailable after connection attempt".to_string(),
            )
        })
    }

    /// Returns clones of the cached training `Client` and its derived JetStream
    /// `Context`, reconnecting both if `nats_training_server_address` differs
    /// from the cached fingerprint.
    pub(super) async fn get_training_client(
        &mut self,
        nats_training_server_address: &str,
    ) -> Result<(async_nats::Client, async_nats::jetstream::Context), TransportError> {
        if self.shutting_down {
            return Err(TransportError::InvalidState(
                "NATS transport is shutting down".to_string(),
            ));
        }

        let incoming_address_fingerprint = address_fingerprint(nats_training_server_address);

        let address_has_changed = match self.training_address_fingerprint {
            Some(cached_address_fingerprint) => {
                cached_address_fingerprint != incoming_address_fingerprint
            }
            None => true,
        };

        if address_has_changed {
            if let Some(existing_training_client) = self.training_client.take() {
                existing_training_client
                    .drain()
                    .await
                    .map_err(|drain_error| {
                        TransportError::InvalidState(format!(
                            "Failed to drain existing training NATS client \
                             before reconnecting to '{}': {}",
                            nats_training_server_address, drain_error
                        ))
                    })?;
            }
            self.training_jetstream_context = None;

            let new_training_client = async_nats::connect(nats_training_server_address)
                .await
                .map_err(|nats_connection_error| {
                    TransportError::NatsClientError(format!(
                        "Failed to connect training NATS client to '{}': {}",
                        nats_training_server_address, nats_connection_error
                    ))
                })?;

            let new_training_jetstream_context =
                async_nats::jetstream::new(new_training_client.clone());

            self.training_client = Some(new_training_client);
            self.training_jetstream_context = Some(new_training_jetstream_context);
            self.training_address_fingerprint = Some(incoming_address_fingerprint);
        }

        let training_client = self.training_client.clone().ok_or_else(|| {
            TransportError::InvalidState(
                "Training NATS client is unavailable after connection attempt".to_string(),
            )
        })?;

        let training_jetstream_context =
            self.training_jetstream_context.clone().ok_or_else(|| {
                TransportError::InvalidState(
                    "Training NATS JetStream context is unavailable after connection attempt"
                        .to_string(),
                )
            })?;

        Ok((training_client, training_jetstream_context))
    }
}

pub(super) struct NatsInferenceOps {
    pub(super) transport_entry: (String, String),
    pub(super) nats_connection_manager: Arc<RwLock<NatsConnectionManager>>,
}

impl NatsInferenceOps {
    pub(super) fn new(
        transport_entry: (String, String),
        nats_connection_manager: Arc<RwLock<NatsConnectionManager>>,
    ) -> Self {
        Self {
            transport_entry,
            nats_connection_manager,
        }
    }
}

impl NatsInferenceExecution for NatsInferenceOps {
    async fn execute_send_inference_request(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        obs_bytes: &[u8],
        nats_inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        let (actor_namespace, actor_context, actor_id) = actor_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::NatsClientError(
                "Inference server address is empty".to_string(),
            ));
        }

        let inference_request_payload = InferenceRequestPayload {
            transport_namespace,
            transport_context,
            actor_namespace: actor_namespace.to_string(),
            actor_context: actor_context.to_string(),
            actor_id: actor_id.to_string(),
            obs_bytes: obs_bytes.to_vec(),
        };

        let serialized_inference_request_payload =
            serialize_payload_to_nats_bytes(&inference_request_payload, "inference request")?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        let inference_response_message = nats_inference_client
            .request(
                INFERENCE_REQUEST_SUBJECT,
                serialized_inference_request_payload,
            )
            .await
            .map_err(|nats_request_error| {
                TransportError::NatsClientError(format!(
                    "NATS request on '{}' failed for inference: {}",
                    INFERENCE_REQUEST_SUBJECT, nats_request_error
                ))
            })?;

        let (relayrl_action, _bytes_consumed): (RelayRLAction, usize) =
            bincode::serde::decode_from_slice(
                &inference_response_message.payload,
                bincode::config::standard(),
            )
            .map_err(|deserialization_error| {
                TransportError::NatsClientError(format!(
                    "Failed to deserialize inference response as RelayRLAction: {}",
                    deserialization_error
                ))
            })?;

        Ok(relayrl_action)
    }

    async fn execute_send_flag_last_inference(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        reward: &f32,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        let (actor_namespace, actor_context, actor_id) = actor_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::NatsClientError(
                "Inference server address is empty".to_string(),
            ));
        }

        let flag_last_inference_payload = FlagLastInferencePayload {
            transport_namespace,
            transport_context,
            actor_namespace: actor_namespace.to_string(),
            actor_context: actor_context.to_string(),
            actor_id: actor_id.to_string(),
            reward: *reward,
        };

        let serialized_flag_last_payload =
            serialize_payload_to_nats_bytes(&flag_last_inference_payload, "flag-last-inference")?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        nats_inference_client
            .publish(FLAG_LAST_INFERENCE_SUBJECT, serialized_flag_last_payload)
            .await
            .map_err(|nats_publish_error| {
                TransportError::NatsClientError(format!(
                    "NATS publish on '{}' failed for flag-last-inference: {}",
                    FLAG_LAST_INFERENCE_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_inference_model_init_request<B: Backend + BackendMatcher<Backend = B>>(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        model_mode: &ModelMode,
        model_module: &Option<ModelModule<B>>,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::NatsClientError(
                "Inference server address is empty".to_string(),
            ));
        }

        let model_mode_string = match model_mode {
            ModelMode::Independent => "Independent",
            ModelMode::Shared => "Shared",
        }
        .to_string();

        let model_files_bundle_bytes: Vec<u8> = match model_module {
            Some(model_module_reference) => {
                let model_files_bundle = serialize_model_module_to_bundle(model_module_reference)?;
                serialize_payload_to_nats_bytes(&model_files_bundle, "model files bundle")?.to_vec()
            }
            None => vec![],
        };

        let inference_model_init_request_payload = InferenceModelInitRequestPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            model_mode_string,
            model_files_bundle_bytes,
        };

        let serialized_model_init_request_payload = serialize_payload_to_nats_bytes(
            &inference_model_init_request_payload,
            "inference model init request",
        )?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        nats_inference_client
            .publish(
                INFERENCE_MODEL_INIT_REQUEST_SUBJECT,
                serialized_model_init_request_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::NatsClientError(format!(
                    "NATS publish on '{}' failed for inference model init request: {}",
                    INFERENCE_MODEL_INIT_REQUEST_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::SendClientIdsToServerError(
                "Inference server address is empty".to_string(),
            ));
        }

        let client_id_entries: Vec<ClientIdEntry> = client_ids
            .iter()
            .map(
                |(client_namespace, client_context, client_id)| ClientIdEntry {
                    namespace: client_namespace.to_string(),
                    context: client_context.to_string(),
                    id: client_id.to_string(),
                },
            )
            .collect();

        let inference_client_ids_payload = ClientIdsPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            client_id_entries,
        };

        let serialized_inference_client_ids_payload =
            serialize_payload_to_nats_bytes(&inference_client_ids_payload, "inference client IDs")?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        let client_ids_acknowledgement_message = nats_inference_client
            .request(
                INFERENCE_SCALING_CLIENT_IDS_SUBJECT,
                serialized_inference_client_ids_payload,
            )
            .await
            .map_err(|nats_request_error| {
                TransportError::SendClientIdsToServerError(format!(
                    "NATS request on '{}' failed for inference client IDs: {}",
                    INFERENCE_SCALING_CLIENT_IDS_SUBJECT, nats_request_error
                ))
            })?;

        let acknowledgement_string =
            String::from_utf8_lossy(&client_ids_acknowledgement_message.payload);
        match acknowledgement_string.trim().parse::<i64>() {
            Ok(0) => Ok(()),
            Ok(_non_zero_response_code) => Err(TransportError::SendClientIdsToServerError(
                "Inference server responded with failure to client IDs request".to_string(),
            )),
            Err(parse_error) => Err(TransportError::SendClientIdsToServerError(format!(
                "Failed to parse inference client IDs acknowledgement response: {}",
                parse_error
            ))),
        }
    }

    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::SendScalingWarningError(
                "Inference server address is empty".to_string(),
            ));
        }

        let operation_string = match operation {
            ScalingOperation::ScaleOut => "scale_out",
            ScalingOperation::ScaleIn => "scale_in",
        }
        .to_string();

        let inference_scaling_warning_payload = ScalingOperationPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            operation_string,
        };

        let serialized_scaling_warning_payload = serialize_payload_to_nats_bytes(
            &inference_scaling_warning_payload,
            "inference scaling warning",
        )?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        nats_inference_client
            .publish(
                INFERENCE_SCALING_WARNING_SUBJECT,
                serialized_scaling_warning_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendScalingWarningError(format!(
                    "NATS publish on '{}' failed for inference scaling warning: {}",
                    INFERENCE_SCALING_WARNING_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::SendScalingCompleteError(
                "Inference server address is empty".to_string(),
            ));
        }

        let operation_string = match operation {
            ScalingOperation::ScaleOut => "scale_out",
            ScalingOperation::ScaleIn => "scale_in",
        }
        .to_string();

        let inference_scaling_complete_payload = ScalingOperationPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            operation_string,
        };

        let serialized_scaling_complete_payload = serialize_payload_to_nats_bytes(
            &inference_scaling_complete_payload,
            "inference scaling complete",
        )?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        nats_inference_client
            .publish(
                INFERENCE_SCALING_COMPLETE_SUBJECT,
                serialized_scaling_complete_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendScalingCompleteError(format!(
                    "NATS publish on '{}' failed for inference scaling complete: {}",
                    INFERENCE_SCALING_COMPLETE_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_inference_server_address.is_empty() {
            return Err(TransportError::SendShutdownSignalError(
                "Inference server address is empty".to_string(),
            ));
        }

        let inference_shutdown_payload = ShutdownPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
        };

        let serialized_inference_shutdown_payload =
            serialize_payload_to_nats_bytes(&inference_shutdown_payload, "inference shutdown")?;

        let nats_inference_client = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_inference_client(nats_inference_server_address)
                .await?
        };

        nats_inference_client
            .publish(
                INFERENCE_SCALING_SHUTDOWN_SUBJECT,
                serialized_inference_shutdown_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendShutdownSignalError(format!(
                    "NATS publish on '{}' failed for inference shutdown signal: {}",
                    INFERENCE_SCALING_SHUTDOWN_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }
}

pub(super) struct NatsTrainingOps {
    pub(super) transport_entry: (String, String),
    pub(super) nats_connection_manager: Arc<RwLock<NatsConnectionManager>>,
}

impl NatsTrainingOps {
    pub(super) fn new(
        transport_entry: (String, String),
        nats_connection_manager: Arc<RwLock<NatsConnectionManager>>,
    ) -> Self {
        Self {
            transport_entry,
            nats_connection_manager,
        }
    }

    pub(super) async fn is_shutting_down(&self) -> bool {
        self.nats_connection_manager.read().await.is_shutting_down()
    }

    pub(super) async fn stop_model_listener(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
    ) -> Result<(), TransportError> {
        let (_, _, receiver_id) = receiver_entry;
        self.nats_connection_manager
            .read()
            .await
            .stop_model_listener(receiver_id);
        Ok(())
    }

    pub(super) async fn shutdown(&self) -> Result<(), TransportError> {
        let shutdown_clients = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager.begin_shutdown()
        };

        let Some(shutdown_clients) = shutdown_clients else {
            return Ok(());
        };

        let mut drain_errors: Vec<String> = Vec::new();

        if let Some(existing_inference_client) = shutdown_clients.inference_client
            && let Err(drain_error) = existing_inference_client.drain().await
        {
            drain_errors.push(format!(
                "Failed to drain inference NATS client for '{}': {}",
                shutdown_clients.client_namespace, drain_error
            ));
        }

        if let Some(existing_training_client) = shutdown_clients.training_client
            && let Err(drain_error) = existing_training_client.drain().await
        {
            drain_errors.push(format!(
                "Failed to drain training NATS client for '{}': {}",
                shutdown_clients.client_namespace, drain_error
            ));
        }

        match drain_errors.len() {
            0 => Ok(()),
            1 => Err(TransportError::InvalidState(drain_errors.remove(0))),
            _ => {
                let second = drain_errors.pop().unwrap_or_default();
                let first = drain_errors.pop().unwrap_or_default();
                Err(TransportError::MultipleErrors(first, second))
            }
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> NatsTrainingExecution<B> for NatsTrainingOps {
    async fn execute_listen_for_model(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
        model_update_tx: &Sender<RoutedMessage>,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        if nats_training_server_address.is_empty() {
            return Err(TransportError::ListenForModelError(
                "Training server address is empty".to_string(),
            ));
        }

        if model_update_tx.is_closed() {
            return Err(TransportError::ListenForModelError(
                "Model update transmitter channel is closed".to_string(),
            ));
        }

        if self.is_shutting_down().await {
            return Ok(());
        }

        let (_, _, receiver_id) = receiver_entry;

        let (nats_training_client, listener_stop_flag, _listener_handle) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            let listener_handle = nats_connection_manager.register_model_listener(receiver_id);
            let listener_stop_flag = listener_handle.listener_shutdown.clone();
            let (training_client, _training_jetstream_context) = nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?;
            (training_client, listener_stop_flag, listener_handle)
        };

        let model_update_subscriber = nats_training_client
            .subscribe(TRAINING_MODEL_LISTENING_SUBJECT)
            .await
            .map_err(|subscribe_error| {
                TransportError::ListenForModelError(format!(
                    "Failed to subscribe to '{}' for model updates: {}",
                    TRAINING_MODEL_LISTENING_SUBJECT, subscribe_error
                ))
            })?;

        let payload_stream =
            model_update_subscriber.map(|received_nats_message| received_nats_message.payload);

        forward_model_update_payloads(payload_stream, model_update_tx, || async {
            listener_stop_flag.load(Ordering::SeqCst) || self.is_shutting_down().await
        })
        .await
    }

    async fn execute_send_algorithm_init_request(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        actor_entries: &[(NamespaceString, ContextString, Uuid)],
        model_mode: &ModelMode,
        algorithm: &Algorithm,
        hyperparams: &HashMap<Algorithm, HyperparameterArgs>,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::SendAlgorithmInitRequestError(
                "Training server address is empty".to_string(),
            ));
        }

        let actor_entries_string = actor_entries
            .iter()
            .map(|(actor_namespace, actor_context, actor_id)| {
                format!("{}:{}:{}", actor_namespace, actor_context, actor_id)
            })
            .collect::<Vec<String>>()
            .join(",");

        let model_mode_string = match model_mode {
            ModelMode::Independent => "Independent",
            ModelMode::Shared => "Shared",
        }
        .to_string();

        let algorithm_init_request_payload = AlgorithmInitRequestPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            actor_entries_string,
            model_mode_string,
            algorithm: algorithm.clone(),
            hyperparams: hyperparams.clone(),
        };

        let serialized_algorithm_init_request_payload = serialize_payload_to_nats_bytes(
            &algorithm_init_request_payload,
            "algorithm init request",
        )?;

        let (nats_training_client, _training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        nats_training_client
            .publish(
                TRAINING_ALGORITHM_INIT_REQUEST_SUBJECT,
                serialized_algorithm_init_request_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendAlgorithmInitRequestError(format!(
                    "NATS publish on '{}' failed for algorithm init request: {}",
                    TRAINING_ALGORITHM_INIT_REQUEST_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_initial_model_handshake(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        nats_training_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        let (actor_namespace, actor_context, actor_id) = actor_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::ModelHandshakeError(
                "Training server address is empty".to_string(),
            ));
        }

        let model_handshake_request_payload = ModelHandshakeRequestPayload {
            transport_namespace,
            transport_context,
            actor_namespace: actor_namespace.to_string(),
            actor_context: actor_context.to_string(),
            actor_id: actor_id.to_string(),
        };

        let serialized_model_handshake_request_payload = serialize_payload_to_nats_bytes(
            &model_handshake_request_payload,
            "model handshake request",
        )?;

        let (nats_training_client, _training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        let model_handshake_response_message = nats_training_client
            .request(
                TRAINING_MODEL_HANDSHAKE_SUBJECT,
                serialized_model_handshake_request_payload,
            )
            .await
            .map_err(|nats_request_error| {
                TransportError::ModelHandshakeError(format!(
                    "NATS request on '{}' failed for model handshake: {}",
                    TRAINING_MODEL_HANDSHAKE_SUBJECT, nats_request_error
                ))
            })?;

        let deserialized_handshake_response: ModelHandshakeResponsePayload =
            deserialize_nats_response_bytes(
                &model_handshake_response_message.payload,
                "model handshake response",
            )?;

        if deserialized_handshake_response
            .model_files_bundle_bytes
            .is_empty()
        {
            return Ok(None);
        }

        let model_files_bundle: ModelFilesBundle = deserialize_nats_response_bytes(
            &deserialized_handshake_response.model_files_bundle_bytes,
            "model files bundle",
        )?;

        let loaded_model_module = deserialize_model_module_from_bundle::<B>(model_files_bundle)?;

        Ok(Some(loaded_model_module))
    }

    async fn execute_send_trajectory(
        &self,
        buffer_entry: &(NamespaceString, ContextString, Uuid),
        encoded_trajectory: &EncodedTrajectory,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        let (buffer_namespace, buffer_context, buffer_id) = buffer_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::SendTrajError(
                "Training server address is empty".to_string(),
            ));
        }

        let trajectory_publish_payload = TrajectoryPublishPayload {
            transport_namespace,
            transport_context,
            buffer_namespace: buffer_namespace.to_string(),
            buffer_context: buffer_context.to_string(),
            buffer_id: buffer_id.to_string(),
            encoded_trajectory: encoded_trajectory.clone(),
        };

        let serialized_trajectory_publish_payload =
            serialize_payload_to_nats_bytes(&trajectory_publish_payload, "trajectory publish")?;

        let (_nats_training_client, nats_training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        nats_training_jetstream_context
            .publish(
                TRAINING_SEND_TRAJECTORY_SUBJECT,
                serialized_trajectory_publish_payload,
            )
            .await
            .map_err(|jetstream_publish_error| {
                TransportError::SendTrajError(format!(
                    "JetStream publish on '{}' failed for trajectory: {}",
                    TRAINING_SEND_TRAJECTORY_SUBJECT, jetstream_publish_error
                ))
            })?
            .await
            .map_err(|trajectory_publish_ack_error| {
                TransportError::SendTrajError(format!(
                    "JetStream publish acknowledgement on '{}' failed for trajectory: {}",
                    TRAINING_SEND_TRAJECTORY_SUBJECT, trajectory_publish_ack_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::SendClientIdsToServerError(
                "Training server address is empty".to_string(),
            ));
        }

        let client_id_entries: Vec<ClientIdEntry> = client_ids
            .iter()
            .map(
                |(client_namespace, client_context, client_id)| ClientIdEntry {
                    namespace: client_namespace.to_string(),
                    context: client_context.to_string(),
                    id: client_id.to_string(),
                },
            )
            .collect();

        let training_client_ids_payload = ClientIdsPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            client_id_entries,
        };

        let serialized_training_client_ids_payload =
            serialize_payload_to_nats_bytes(&training_client_ids_payload, "training client IDs")?;

        let (nats_training_client, _training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        let client_ids_acknowledgement_message = nats_training_client
            .request(
                TRAINING_SCALING_CLIENT_IDS_SUBJECT,
                serialized_training_client_ids_payload,
            )
            .await
            .map_err(|nats_request_error| {
                TransportError::SendClientIdsToServerError(format!(
                    "NATS request on '{}' failed for training client IDs: {}",
                    TRAINING_SCALING_CLIENT_IDS_SUBJECT, nats_request_error
                ))
            })?;

        let acknowledgement_string =
            String::from_utf8_lossy(&client_ids_acknowledgement_message.payload);
        match acknowledgement_string.trim().parse::<i64>() {
            Ok(0) => Ok(()),
            Ok(_non_zero_response_code) => Err(TransportError::SendClientIdsToServerError(
                "Training server responded with failure to client IDs request".to_string(),
            )),
            Err(parse_error) => Err(TransportError::SendClientIdsToServerError(format!(
                "Failed to parse training client IDs acknowledgement response: {}",
                parse_error
            ))),
        }
    }

    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::SendScalingWarningError(
                "Training server address is empty".to_string(),
            ));
        }

        let operation_string = match operation {
            ScalingOperation::ScaleOut => "scale_out",
            ScalingOperation::ScaleIn => "scale_in",
        }
        .to_string();

        let training_scaling_warning_payload = ScalingOperationPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            operation_string,
        };

        let serialized_training_scaling_warning_payload = serialize_payload_to_nats_bytes(
            &training_scaling_warning_payload,
            "training scaling warning",
        )?;

        let (nats_training_client, _training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        nats_training_client
            .publish(
                TRAINING_SCALING_WARNING_SUBJECT,
                serialized_training_scaling_warning_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendScalingWarningError(format!(
                    "NATS publish on '{}' failed for training scaling warning: {}",
                    TRAINING_SCALING_WARNING_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::SendScalingCompleteError(
                "Training server address is empty".to_string(),
            ));
        }

        let operation_string = match operation {
            ScalingOperation::ScaleOut => "scale_out",
            ScalingOperation::ScaleIn => "scale_in",
        }
        .to_string();

        let training_scaling_complete_payload = ScalingOperationPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
            operation_string,
        };

        let serialized_training_scaling_complete_payload = serialize_payload_to_nats_bytes(
            &training_scaling_complete_payload,
            "training scaling complete",
        )?;

        let (nats_training_client, _training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        nats_training_client
            .publish(
                TRAINING_SCALING_COMPLETE_SUBJECT,
                serialized_training_scaling_complete_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendScalingCompleteError(format!(
                    "NATS publish on '{}' failed for training scaling complete: {}",
                    TRAINING_SCALING_COMPLETE_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }

    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        let (scaling_namespace, scaling_context, scaling_id) = scaling_entry;
        let (transport_namespace, transport_context) = self.transport_entry.clone();

        if nats_training_server_address.is_empty() {
            return Err(TransportError::SendShutdownSignalError(
                "Training server address is empty".to_string(),
            ));
        }

        let training_shutdown_payload = ShutdownPayload {
            transport_namespace,
            transport_context,
            scaling_namespace: scaling_namespace.to_string(),
            scaling_context: scaling_context.to_string(),
            scaling_id: scaling_id.to_string(),
        };

        let serialized_training_shutdown_payload =
            serialize_payload_to_nats_bytes(&training_shutdown_payload, "training shutdown")?;

        let (nats_training_client, _training_jetstream_context) = {
            let mut nats_connection_manager = self.nats_connection_manager.write().await;
            nats_connection_manager
                .get_training_client(nats_training_server_address)
                .await?
        };

        nats_training_client
            .publish(
                TRAINING_SCALING_SHUTDOWN_SUBJECT,
                serialized_training_shutdown_payload,
            )
            .await
            .map_err(|nats_publish_error| {
                TransportError::SendShutdownSignalError(format!(
                    "NATS publish on '{}' failed for training shutdown signal: {}",
                    TRAINING_SCALING_SHUTDOWN_SUBJECT, nats_publish_error
                ))
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    use tokio::sync::mpsc;

    fn make_model_update_payload(
        actor_id_bytes: [u8; 16],
        version: i64,
        model_bytes: &[u8],
    ) -> Bytes {
        serialize_payload_to_nats_bytes(
            &ModelUpdateBroadcastMessage {
                model_bytes: model_bytes.to_vec(),
                actor_id_bytes: actor_id_bytes.to_vec(),
                model_version: version,
            },
            "model update broadcast",
        )
        .unwrap()
    }

    #[tokio::test]
    async fn forward_model_update_payloads_dispatches_multiple_updates() {
        let (tx, mut rx) = mpsc::channel::<RoutedMessage>(4);
        let payload_stream = tokio_stream::iter(vec![
            make_model_update_payload([1; 16], 3, &[10, 20]),
            make_model_update_payload([2; 16], 4, &[30, 40]),
        ]);

        forward_model_update_payloads(payload_stream, &tx, || std::future::ready(true))
            .await
            .unwrap();

        let first_message = rx.recv().await.unwrap();
        assert_eq!(first_message.actor_id, Uuid::from_bytes([1; 16]));
        assert!(matches!(
            first_message.protocol,
            RoutingProtocol::ModelUpdate
        ));
        match first_message.payload {
            RoutedPayload::ModelUpdate {
                model_bytes,
                version,
            } => {
                assert_eq!(model_bytes, vec![10, 20]);
                assert_eq!(version, 3);
            }
            _ => panic!("expected model update payload"),
        }

        let second_message = rx.recv().await.unwrap();
        assert_eq!(second_message.actor_id, Uuid::from_bytes([2; 16]));
        match second_message.payload {
            RoutedPayload::ModelUpdate {
                model_bytes,
                version,
            } => {
                assert_eq!(model_bytes, vec![30, 40]);
                assert_eq!(version, 4);
            }
            _ => panic!("expected model update payload"),
        }
    }

    #[tokio::test]
    async fn forward_model_update_payloads_returns_ok_when_shutdown_closes_stream() {
        let (tx, _rx) = mpsc::channel::<RoutedMessage>(1);
        let payload_stream = tokio_stream::iter(Vec::<Bytes>::new());

        let result =
            forward_model_update_payloads(payload_stream, &tx, || std::future::ready(true)).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn forward_model_update_payloads_errors_when_stream_closes_unexpectedly() {
        let (tx, _rx) = mpsc::channel::<RoutedMessage>(1);
        let payload_stream = tokio_stream::iter(Vec::<Bytes>::new());

        let result =
            forward_model_update_payloads(payload_stream, &tx, || std::future::ready(false)).await;

        assert!(matches!(
            result,
            Err(TransportError::ListenForModelError(message))
                if message.contains("closed unexpectedly")
        ));
    }

    #[tokio::test]
    async fn connection_manager_shutdown_marks_state_and_blocks_reconnects() {
        let mut connection_manager = NatsConnectionManager::new(Arc::from("test-client"));
        assert!(!connection_manager.is_shutting_down());

        let shutdown_clients = connection_manager.begin_shutdown();

        assert!(shutdown_clients.is_some());
        assert!(connection_manager.is_shutting_down());
        assert!(matches!(
            connection_manager
                .get_training_client("nats://127.0.0.1:4222")
                .await,
            Err(TransportError::InvalidState(message))
                if message.contains("shutting down")
        ));
    }

    #[test]
    fn register_and_stop_model_listener_updates_flag() {
        let connection_manager = NatsConnectionManager::new(Arc::from("test-client"));
        let receiver_id = Uuid::new_v4();
        let listener_handle = connection_manager.register_model_listener(&receiver_id);

        assert!(!listener_handle.is_stopping());

        connection_manager.stop_model_listener(&receiver_id);

        assert!(listener_handle.is_stopping());
    }

    #[test]
    fn listener_handle_drop_unregisters_listener_flag() {
        let connection_manager = NatsConnectionManager::new(Arc::from("test-client"));
        let receiver_id = Uuid::new_v4();
        let listener_handle = connection_manager.register_model_listener(&receiver_id);

        assert!(
            connection_manager
                .model_listener_shutdown_flags
                .get(&receiver_id)
                .is_some()
        );

        drop(listener_handle);

        assert!(
            connection_manager
                .model_listener_shutdown_flags
                .get(&receiver_id)
                .is_none()
        );
    }

    #[test]
    fn begin_shutdown_marks_registered_listeners_stopping() {
        let mut connection_manager = NatsConnectionManager::new(Arc::from("test-client"));
        let receiver_id = Uuid::new_v4();
        let listener_handle = connection_manager.register_model_listener(&receiver_id);

        let shutdown_clients = connection_manager.begin_shutdown();

        assert!(shutdown_clients.is_some());
        assert!(connection_manager.is_shutting_down());
        assert!(listener_handle.is_stopping());
    }
}
