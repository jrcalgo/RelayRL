use crate::network::HyperparameterArgs;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    SharedInferenceAddresses, SharedTrainingAddresses, SharedTransportAddresses,
};
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::zmq::{
    ZmqClientError, ZmqInferenceExecution, ZmqTrainingExecution,
};
use crate::network::client::runtime::data::transport_sink::{
    SyncClientInferenceTransportOps, SyncClientTrainingTransportOps, TransportError, TransportUuid,
};
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
use crate::network::client::agent::ModelMode;
use crate::utilities::configuration::{Algorithm, ClientConfigLoader};

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::{add_id, remove_id, reserve_id_with};
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::BackendMatcher;
use relayrl_types::data::trajectory::{EncodedTrajectory, RelayRLTrajectory};
use relayrl_types::model::utils::validate_module;
use relayrl_types::model::{HotReloadableModel, ModelModule};

use burn_tensor::backend::Backend;
use std::io::Write;
use zmq::SocketType::PAIR;

use dashmap::DashMap;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use tempfile::NamedTempFile;
use tokio::sync::mpsc::Sender;
use tokio::task;
use uuid::Uuid;
use zmq::{Context, Socket, SocketType};

use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum ZmqPoolError {
    #[error(transparent)]
    SocketError(#[from] zmq::Error),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error("Failed to read ZMQ pool: {0}")]
    ReadError(String),
    #[error("Failed to initialize ZMQ pool: {0}")]
    InitializationError(String),
}

type SocketUuid = Uuid;

pub(super) struct ZmqSocketPool {
    pub(super) inference_dealer_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(super) model_dealer_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(super) model_sub_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(super) traj_push_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(super) scaling_dealer_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
}

/// Raw ZMQ transport operations.
///
/// This struct handles only ZMQ-specific concerns:
/// - Socket creation and caching
/// - Address caching
/// - Message framing and protocol
///
/// Application-level state (model version, algorithm initialization) is managed
/// by the dispatcher layer (see `transport_dispatcher.rs`).
pub(super) struct ZmqPool {
    pub(super) context: Context,
    cached_addresses: Option<DashMap<Uuid, Arc<RwLock<SharedTransportAddresses>>>>,
    cached_sockets: Arc<ZmqSocketPool>,
}

#[derive(Debug, Clone, Copy)]
enum CacheAddressType {
    InferenceServer,
    AgentListener,
    ModelServer,
    TrajectoryServer,
    InferenceScalingServer,
    TrainingScalingServer,
}

#[derive(Debug, Clone, Copy)]
enum SocketPoolType {
    InferenceDealer,
    ModelDealer,
    ModelSub,
    TrajPush,
    ScalingDealer,
}

impl ZmqPool {
    pub fn new() -> Result<Self, ZmqPoolError> {
        Ok(Self {
            context: Context::new(),
            cached_addresses: None,
            cached_sockets: Arc::new(ZmqSocketPool {
                inference_dealer_socket: None,
                model_dealer_socket: None,
                model_sub_socket: None,
                traj_push_socket: None,
                scaling_dealer_socket: None,
            }),
        })
    }

    fn create_dealer_socket(
        &self,
        context: &Context,
        address: &str,
    ) -> Result<Socket, ZmqPoolError> {
        let socket = context.socket(zmq::DEALER)?;

        // Set socket identity
        let identity: SocketUuid =
            reserve_id_with("client", "zmq_dealer_socket", 117, 100).map_err(ZmqPoolError::from)?;
        socket.set_identity(identity.as_bytes())?;

        // Set socket options for performance
        socket.set_rcvtimeo(30000)?;
        socket.set_sndhwm(1000)?;
        socket.set_rcvhwm(1000)?;
        socket.set_maxmsgsize(-1)?;

        // Connect to the server
        socket.connect(address)?;

        Ok(socket)
    }

    fn create_push_socket(&self, context: &Context, address: &str) -> Result<Socket, ZmqPoolError> {
        let socket = context.socket(zmq::PUSH)?;

        let identity: SocketUuid =
            reserve_id_with("client", "zmq_push_socket", 67, 100).map_err(ZmqPoolError::from)?;
        socket.set_identity(identity.as_bytes())?;

        // Set send timeout to non-blocking
        socket.set_sndtimeo(5000)?; // 5 second timeout

        // Connect to trajectory server
        socket.connect(address)?;

        Ok(socket)
    }

    fn create_sub_socket(&self, context: &Context, address: &str) -> Result<Socket, ZmqPoolError> {
        let socket = context.socket(zmq::SUB)?;

        let identity: SocketUuid =
            reserve_id_with("client", "zmq_sub_socket", 69, 100).map_err(ZmqPoolError::from)?;
        socket.set_identity(identity.as_bytes())?;

        socket.set_subscribe(b"")?;

        socket.connect(address)?;

        Ok(socket)
    }

    #[inline(always)]
    fn update_cache(
        &self,
        identity: &Uuid,
        new_address: &str,
        address_type: CacheAddressType,
        socket_type: SocketPoolType,
    ) -> Result<bool, ZmqPoolError> {
        let cached_addresses = self
            .cached_addresses
            .as_ref()
            .ok_or_else(|| ZmqPoolError::ReadError("Cached addresses not available".to_string()))?;

        // Check if we need to update the cache
        let needs_update: bool = match cached_addresses.get(identity) {
            Some(addresses) => {
                let addr_guard = addresses.read().map_err(|e| {
                    ZmqPoolError::ReadError(format!("Failed to read cached addresses: {}", e))
                })?;
                match address_type {
                    CacheAddressType::InferenceServer => {
                        addr_guard
                            .inference_addresses
                            .inference_server_address
                            .as_ref()
                            != new_address
                    }
                    CacheAddressType::AgentListener => {
                        addr_guard
                            .training_addresses
                            .agent_listener_address
                            .as_ref()
                            != new_address
                    }
                    CacheAddressType::ModelServer => {
                        addr_guard.training_addresses.model_server_address.as_ref() != new_address
                    }
                    CacheAddressType::TrajectoryServer => {
                        addr_guard
                            .training_addresses
                            .trajectory_server_address
                            .as_ref()
                            != new_address
                    }
                    CacheAddressType::TrainingScalingServer => {
                        addr_guard
                            .training_addresses
                            .training_scaling_server_address
                            .as_ref()
                            != new_address
                    }
                    CacheAddressType::InferenceScalingServer => {
                        addr_guard
                            .inference_addresses
                            .inference_scaling_server_address
                            .as_ref()
                            != new_address
                    }
                }
            }
            None => true,
        };

        if !needs_update {
            return Ok(false);
        }

        // Build updated SharedTransportAddresses
        let address_entry = cached_addresses
            .get(identity)
            .ok_or_else(|| ZmqPoolError::ReadError("Cached addresses not available".to_string()))?;
        let current_addresses = address_entry.read().map_err(|e| {
            ZmqPoolError::ReadError(format!("Failed to read cached addresses: {}", e))
        })?;

        let updated_addresses = match address_type {
            CacheAddressType::InferenceServer => SharedTransportAddresses {
                inference_addresses: SharedInferenceAddresses {
                    inference_server_address: Arc::from(new_address),
                    inference_scaling_server_address: current_addresses
                        .inference_addresses
                        .inference_scaling_server_address
                        .clone()
                        .clone(),
                },
                training_addresses: SharedTrainingAddresses {
                    agent_listener_address: current_addresses
                        .training_addresses
                        .agent_listener_address
                        .clone(),
                    model_server_address: current_addresses
                        .training_addresses
                        .model_server_address
                        .clone(),
                    trajectory_server_address: current_addresses
                        .training_addresses
                        .trajectory_server_address
                        .clone(),
                    training_scaling_server_address: current_addresses
                        .training_addresses
                        .training_scaling_server_address
                        .clone(),
                },
            },
            CacheAddressType::AgentListener => SharedTransportAddresses {
                inference_addresses: SharedInferenceAddresses {
                    inference_server_address: current_addresses
                        .inference_addresses
                        .inference_server_address
                        .clone(),
                    inference_scaling_server_address: current_addresses
                        .inference_addresses
                        .inference_scaling_server_address
                        .clone()
                        .clone(),
                },
                training_addresses: SharedTrainingAddresses {
                    agent_listener_address: Arc::from(new_address),
                    model_server_address: current_addresses
                        .training_addresses
                        .model_server_address
                        .clone(),
                    trajectory_server_address: current_addresses
                        .training_addresses
                        .trajectory_server_address
                        .clone(),
                    training_scaling_server_address: current_addresses
                        .training_addresses
                        .training_scaling_server_address
                        .clone(),
                },
            },
            CacheAddressType::ModelServer => SharedTransportAddresses {
                inference_addresses: SharedInferenceAddresses {
                    inference_server_address: current_addresses
                        .inference_addresses
                        .inference_server_address
                        .clone(),
                    inference_scaling_server_address: current_addresses
                        .inference_addresses
                        .inference_scaling_server_address
                        .clone()
                        .clone(),
                },
                training_addresses: SharedTrainingAddresses {
                    agent_listener_address: current_addresses
                        .training_addresses
                        .agent_listener_address
                        .clone(),
                    model_server_address: Arc::from(new_address),
                    trajectory_server_address: current_addresses
                        .training_addresses
                        .trajectory_server_address
                        .clone(),
                    training_scaling_server_address: current_addresses
                        .training_addresses
                        .training_scaling_server_address
                        .clone(),
                },
            },
            CacheAddressType::TrajectoryServer => SharedTransportAddresses {
                inference_addresses: SharedInferenceAddresses {
                    inference_server_address: current_addresses
                        .inference_addresses
                        .inference_server_address
                        .clone(),
                    inference_scaling_server_address: current_addresses
                        .inference_addresses
                        .inference_scaling_server_address
                        .clone()
                        .clone(),
                },
                training_addresses: SharedTrainingAddresses {
                    agent_listener_address: current_addresses
                        .training_addresses
                        .agent_listener_address
                        .clone(),
                    model_server_address: current_addresses
                        .training_addresses
                        .model_server_address
                        .clone(),
                    trajectory_server_address: Arc::from(new_address),
                    training_scaling_server_address: current_addresses
                        .training_addresses
                        .training_scaling_server_address
                        .clone(),
                },
            },
            CacheAddressType::InferenceScalingServer => SharedTransportAddresses {
                inference_addresses: SharedInferenceAddresses {
                    inference_server_address: current_addresses
                        .inference_addresses
                        .inference_server_address
                        .clone(),
                    inference_scaling_server_address: Arc::from(new_address),
                },
                training_addresses: SharedTrainingAddresses {
                    agent_listener_address: current_addresses
                        .training_addresses
                        .agent_listener_address
                        .clone(),
                    model_server_address: current_addresses
                        .training_addresses
                        .model_server_address
                        .clone(),
                    trajectory_server_address: current_addresses
                        .training_addresses
                        .trajectory_server_address
                        .clone(),
                    training_scaling_server_address: current_addresses
                        .training_addresses
                        .training_scaling_server_address
                        .clone(),
                },
            },
            CacheAddressType::TrainingScalingServer => SharedTransportAddresses {
                inference_addresses: SharedInferenceAddresses {
                    inference_server_address: current_addresses
                        .inference_addresses
                        .inference_server_address
                        .clone(),
                    inference_scaling_server_address: current_addresses
                        .inference_addresses
                        .inference_scaling_server_address
                        .clone()
                        .clone(),
                },
                training_addresses: SharedTrainingAddresses {
                    agent_listener_address: current_addresses
                        .training_addresses
                        .agent_listener_address
                        .clone(),
                    model_server_address: current_addresses
                        .training_addresses
                        .model_server_address
                        .clone(),
                    trajectory_server_address: current_addresses
                        .training_addresses
                        .trajectory_server_address
                        .clone(),
                    training_scaling_server_address: Arc::from(new_address),
                },
            },
        };

        cached_addresses.insert(*identity, Arc::new(RwLock::new(updated_addresses)));

        // Create and cache the appropriate socket
        let socket_result = match socket_type {
            SocketPoolType::ModelDealer | SocketPoolType::ScalingDealer => {
                self.create_dealer_socket(&self.context, new_address)
            }
            SocketPoolType::InferenceDealer => {
                self.create_dealer_socket(&self.context, new_address)
            }
            SocketPoolType::ModelSub => self.create_sub_socket(&self.context, new_address),
            SocketPoolType::TrajPush => self.create_push_socket(&self.context, new_address),
        };

        let socket = socket_result.map_err(|e| {
            ZmqPoolError::InitializationError(format!(
                "Failed to create {:?} socket: {}",
                socket_type, e
            ))
        })?;

        let socket_pool = match socket_type {
            SocketPoolType::InferenceDealer => &self.cached_sockets.inference_dealer_socket,
            SocketPoolType::ModelDealer => &self.cached_sockets.model_dealer_socket,
            SocketPoolType::ModelSub => &self.cached_sockets.model_sub_socket,
            SocketPoolType::TrajPush => &self.cached_sockets.traj_push_socket,
            SocketPoolType::ScalingDealer => &self.cached_sockets.scaling_dealer_socket,
        };

        socket_pool
            .as_ref()
            .ok_or_else(|| ZmqPoolError::ReadError("Socket pool not initialized".to_string()))?
            .insert(*identity, Arc::new(Mutex::new(socket)));

        Ok(true)
    }
}

#[repr(i64)]
enum ServerResponse {
    Success = 0,
    Failure = 1,
}

impl ServerResponse {
    fn from_i64(value: i64) -> Self {
        match value {
            0 => ServerResponse::Success,
            1 => ServerResponse::Failure,
            _ => ServerResponse::Failure,
        }
    }
}

pub(super) struct ZmqInferenceOps {
    transport_id: TransportUuid,
    zmq_pool: Arc<RwLock<ZmqPool>>,
}

impl ZmqInferenceOps {
    pub(super) fn new(transport_id: TransportUuid, zmq_pool: Arc<RwLock<ZmqPool>>) -> Self {
        Self {
            transport_id,
            zmq_pool,
        }
    }

    pub(super) fn shutdown(&self) -> Result<(), TransportError> {
        unimplemented!();
    }
}

impl ZmqInferenceExecution for ZmqInferenceOps {
    fn execute_send_inference_request(
        &self,
        actor_id: &Uuid,
        action_request: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        unimplemented!();
    }

    fn execute_send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError> {
        unimplemented!();
    }

    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        unimplemented!();
    }

    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        unimplemented!();
    }

    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        unimplemented!();
    }

    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        unimplemented!();
    }
}

pub(super) struct ZmqTrainingOps {
    transport_id: TransportUuid,
    zmq_pool: Arc<RwLock<ZmqPool>>,
}

impl ZmqTrainingOps {
    pub(super) fn new(transport_id: TransportUuid, zmq_pool: Arc<RwLock<ZmqPool>>) -> Self {
        Self {
            transport_id,
            zmq_pool,
        }
    }

    pub(super) fn shutdown(&self) -> Result<(), TransportError> {
        if let Some(sockets) = &self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to read ZMQ pool during cache removal: {}",
                    e
                ))
            })?
            .cached_sockets
            .model_dealer_socket
        {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_id("client", "zmq_dealer_socket", socket_id.clone())
                    .map_err(TransportError::from)?;

                sockets.remove(&socket_id);
            }
        }

        if let Some(sockets) = &self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to read ZMQ pool during cache removal: {}",
                    e
                ))
            })?
            .cached_sockets
            .model_sub_socket
        {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_id("client", "zmq_sub_socket", socket_id.clone())
                    .map_err(TransportError::from)?;

                sockets.remove(&socket_id);
            }
        }

        if let Some(sockets) = &self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to read ZMQ pool during cache removal: {}",
                    e
                ))
            })?
            .cached_sockets
            .traj_push_socket
        {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_id("client", "zmq_push_socket", socket_id.clone())
                    .map_err(TransportError::from)?;

                sockets.remove(&socket_id);
            }
        }

        if let Some(sockets) = &self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to read ZMQ pool during cache removal: {}",
                    e
                ))
            })?
            .cached_sockets
            .scaling_dealer_socket
        {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_id("client", "zmq_dealer_socket", socket_id.clone())
                    .map_err(TransportError::from)?;

                sockets.remove(&socket_id);
            }
        }

        remove_id("client", "zmq_transport_client", self.transport_id.clone())
            .map_err(TransportError::from)?;
        Ok(())
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ZmqTrainingExecution<B> for ZmqTrainingOps {
    fn execute_listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        model_server_address: &str,
    ) -> Result<(), TransportError> {
        if receiver_id.is_nil() {
            return Err(TransportError::ListenForModelError(
                "Receiver ID is nil".to_string(),
            ));
        }

        if model_server_address.is_empty() {
            return Err(TransportError::ListenForModelError(
                "Model server address is empty".to_string(),
            ));
        }

        if global_dispatcher_tx.is_closed() {
            return Err(TransportError::ListenForModelError(
                "Global dispatcher is closed".to_string(),
            ));
        }

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::ListenForModelError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                receiver_id,
                model_server_address,
                CacheAddressType::ModelServer,
                SocketPoolType::ModelSub,
            )
            .map_err(ZmqClientError::from)?;

        let socket = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::ListenForModelError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?
            .cached_sockets
            .model_sub_socket
            .as_ref()
            .unwrap()
            .get(receiver_id)
            .unwrap()
            .clone();

        let model_server_address = model_server_address.to_string();
        let global_dispatcher_tx = global_dispatcher_tx.clone();

        task::spawn_blocking(move || {
            println!(
                "[ZmqClient] Listening for model updates at {}",
                model_server_address
            );

            loop {
                match socket
                    .try_lock()
                    .map_err(|e| {
                        TransportError::ListenForModelError(format!(
                            "Failed to lock sub socket: {}",
                            e
                        ))
                    })?
                    .recv_bytes(0)
                {
                    Ok(model_bytes) => {
                        let msg = RoutedMessage {
                            actor_id: Uuid::nil(), // broadcast placeholder
                            protocol: RoutingProtocol::ModelUpdate,
                            payload: RoutedPayload::ModelUpdate {
                                model_bytes,
                                version: 0,
                            },
                        };
                        let _ = global_dispatcher_tx.blocking_send(msg);
                    }
                    Err(e) => {
                        eprintln!("[ZmqClient] SUB socket recv error: {}", e);
                        return Err::<(), TransportError>(TransportError::ListenForModelError(
                            format!("SUB socket recv error: {}", e),
                        ));
                    }
                }
            }
        });

        Ok(())
    }

    fn execute_send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError> {
        // TODO: Reqeust that the server initializes a shared algorithm OR individual algorithms per actor (must be the same algorithm for now)
        if scaling_id.is_nil() {
            return Err(TransportError::SendAlgorithmInitRequestError(
                "Scaling ID is nil".to_string(),
            ));
        }

        if agent_listener_address.is_empty() {
            return Err(TransportError::SendAlgorithmInitRequestError(
                "Agent listener address is empty".to_string(),
            ));
        }

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendAlgorithmInitRequestError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                scaling_id,
                agent_listener_address,
                CacheAddressType::TrainingScalingServer,
                SocketPoolType::ScalingDealer,
            )
            .map_err(ZmqClientError::from)?;

        let transport_id_string = self.transport_id.to_string();
        let scaling_id_string = scaling_id.to_string();
        let algorithm_name_string = algorithm.as_str().to_string();
        let hyperparams_string = serde_json::to_string(&hyperparams).unwrap_or_default();

        let empty_frame: Vec<u8> = vec![];
        let transport_id_frame: Vec<u8> = transport_id_string.as_bytes().to_vec();
        let scaling_id_frame: Vec<u8> = scaling_id_string.as_bytes().to_vec();
        let algorithm_init_payload: Vec<u8> = b"ALGORITHM_INIT".to_vec();
        let algorithm_name_frame: Vec<u8> = algorithm_name_string.as_bytes().to_vec();
        let _hyperparams_payload: Vec<u8> = hyperparams_string.as_bytes().to_vec();

        let socket = {
            let pool = self.zmq_pool.read().map_err(|e| {
                TransportError::SendAlgorithmInitRequestError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?;
            let socket_entry = pool
                .cached_sockets
                .scaling_dealer_socket
                .as_ref()
                .unwrap()
                .get(scaling_id)
                .unwrap();
            socket_entry.value().clone()
        };

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendAlgorithmInitRequestError(format!(
                    "Failed to lock scaling dealer socket: {}",
                    e
                ))
            })?
            .send_multipart(
                [
                    empty_frame,
                    transport_id_frame,
                    scaling_id_frame,
                    algorithm_init_payload,
                    algorithm_name_frame,
                ],
                0,
            ) {
            Ok(_) => {
                println!("[ZmqClient] Sent algorithm init request");
            }
            Err(e) => {
                return Err(TransportError::SendAlgorithmInitRequestError(format!(
                    "Failed to send algorithm init request: {}",
                    e
                )));
            }
        }

        Ok(())
    }

    fn execute_initial_model_handshake(
        &self,
        actor_id: &Uuid,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        if actor_id.is_nil() {
            return Err(TransportError::ModelHandshakeError(
                "Actor ID is nil".to_string(),
            ));
        }

        if agent_listener_address.is_empty() {
            return Err(TransportError::ModelHandshakeError(
                "Agent listener address is empty".to_string(),
            ));
        }

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::ModelHandshakeError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                actor_id,
                agent_listener_address,
                CacheAddressType::AgentListener,
                SocketPoolType::ModelDealer,
            )
            .map_err(ZmqClientError::from)?;

        println!("[ZmqClient] Starting initial model handshake...");

        let transport_id_string = self.transport_id.to_string();
        let actor_id_string = actor_id.to_string();

        let empty_frame: Vec<u8> = vec![];
        let transport_id_frame: &[u8] = transport_id_string.as_bytes();
        let actor_id_frame: &[u8] = actor_id_string.as_bytes();
        let get_model_payload: &[u8] = b"GET_MODEL";

        let socket = {
            let pool = self.zmq_pool.read().map_err(|e| {
                TransportError::ModelHandshakeError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?;
            let socket_entry = pool
                .cached_sockets
                .model_dealer_socket
                .as_ref()
                .unwrap()
                .get(actor_id)
                .unwrap();
            socket_entry.value().clone()
        };

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::ModelHandshakeError(format!("Failed to lock dealer socket: {}", e))
            })?
            .send_multipart(
                [
                    &empty_frame,
                    transport_id_frame,
                    actor_id_frame,
                    get_model_payload,
                ],
                0,
            ) {
            Ok(_) => println!("[ZmqClient] Sent GET_MODEL request"),
            Err(e) => {
                eprintln!("[ZmqClient] Failed to send GET_MODEL: {}", e);
                return Err(TransportError::ModelHandshakeError(format!(
                    "Failed to send GET_MODEL: {}",
                    e
                )));
            }
        }

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::ModelHandshakeError(format!("Failed to lock dealer socket: {}", e))
            })?
            .recv_multipart(0)
        {
            Ok(message_parts) => {
                if message_parts.len() < 2 {
                    eprintln!("[ZmqClient] Malformed handshake response");
                    return Err(TransportError::ModelHandshakeError(
                        "Malformed handshake response".to_string(),
                    ));
                }

                let model_bytes: &Vec<u8> = &message_parts[1];
                println!(
                    "[ZmqClient] Received initial model ({} bytes)",
                    model_bytes.len()
                );

                // Save model to temporary file and load it
                match NamedTempFile::new() {
                    Ok(mut temp_file) => {
                        if let Err(e) = temp_file.write_all(model_bytes) {
                            eprintln!("[ZmqClient] Failed to write model to temp file: {}", e);
                            return Err(TransportError::ModelHandshakeError(format!(
                                "Failed to write model to temp file: {}",
                                e
                            )));
                        }

                        match ModelModule::<B>::load_from_path(temp_file.path()) {
                            Ok(model) => {
                                if let Err(e) = validate_module::<B>(&model) {
                                    eprintln!("[ZmqClient] Failed to validate model: {:?}", e);
                                    return Err(TransportError::ModelHandshakeError(format!(
                                        "Failed to validate model: {:?}",
                                        e
                                    )));
                                }
                                println!("[ZmqClient] Model loaded and validated successfully");
                                Ok(Some(model))
                            }
                            Err(e) => {
                                eprintln!("[ZmqClient] Failed to load model: {:?}", e);
                                Err(TransportError::ModelHandshakeError(format!(
                                    "Failed to load model: {:?}",
                                    e
                                )))
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[ZmqClient] Failed to create temp file: {}", e);
                        Err(TransportError::ModelHandshakeError(format!(
                            "Failed to create temp file: {}",
                            e
                        )))
                    }
                }
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to receive model: {}", e);
                Err(TransportError::ModelHandshakeError(format!(
                    "Failed to receive model: {}",
                    e
                )))
            }
        }
    }

    fn execute_send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        if sender_id.is_nil() {
            return Err(TransportError::SendTrajError(
                "Sender ID is nil".to_string(),
            ));
        }

        if trajectory_server_address.is_empty() {
            return Err(TransportError::SendTrajError(
                "Trajectory server address is empty".to_string(),
            ));
        }

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                sender_id,
                trajectory_server_address,
                CacheAddressType::TrajectoryServer,
                SocketPoolType::TrajPush,
            )
            .map_err(ZmqClientError::from)?;

        // Serialize the trajectory
        let serialized_traj: Vec<u8> = serde_json::to_vec(&encoded_trajectory).map_err(|e| {
            TransportError::SendTrajError(format!("Failed to serialize trajectory: {}", e))
        })?;

        println!(
            "[ZmqClient] Sending trajectory ({} bytes, {} actions)",
            serialized_traj.len(),
            encoded_trajectory.num_actions
        );

        let socket = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?
            .cached_sockets
            .traj_push_socket
            .as_ref()
            .unwrap()
            .get(sender_id)
            .unwrap()
            .value()
            .clone();

        // Send the trajectory
        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendTrajError(format!("Failed to lock push socket: {}", e))
            })?
            .send(serialized_traj, 0)
        {
            Ok(_) => {
                println!("[ZmqClient] Trajectory sent successfully");
                Ok(())
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to send trajectory: {}", e);
                Err(TransportError::SendTrajError(format!(
                    "Failed to send trajectory: {}",
                    e
                )))
            }
        }
    }

    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        if scaling_id.is_nil() {
            return Err(TransportError::SendClientIdsToServerError(
                "Coordinator ID is nil".to_string(),
            ));
        }

        if training_scaling_server_address.is_empty() {
            return Err(TransportError::SendClientIdsToServerError(
                "Agent listener address is empty".to_string(),
            ));
        }

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                scaling_id,
                training_scaling_server_address,
                CacheAddressType::TrainingScalingServer,
                SocketPoolType::ScalingDealer,
            )
            .map_err(ZmqClientError::from)?;

        // TODO: Send client IDs to server for caching, validation, and routing
        let transport_id_string = self.transport_id.to_string();
        let scaling_id_string = scaling_id.to_string();

        let empty_frame: Vec<u8> = vec![];
        let transport_id_frame: &[u8] = transport_id_string.as_bytes();
        let scaling_id_frame: &[u8] = scaling_id_string.as_bytes();
        let pairs_payload = client_ids
            .iter()
            .map(|(name, id)| name.to_string() + " " + id.to_string().as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let socket = {
            let pool = self.zmq_pool.read().map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?;
            let socket_entry = pool
                .cached_sockets
                .scaling_dealer_socket
                .as_ref()
                .unwrap()
                .get(scaling_id)
                .unwrap();
            socket_entry.value().clone()
        };

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to lock scaling dealer socket: {}",
                    e
                ))
            })?
            .send_multipart(
                [
                    &empty_frame,
                    transport_id_frame,
                    scaling_id_frame,
                    pairs_payload.as_bytes(),
                ],
                0,
            ) {
            Ok(_) => println!("[ZmqClient] Sent client IDs to server"),
            Err(e) => {
                return Err(TransportError::SendClientIdsToServerError(format!(
                    "Failed to send client IDs to server: {}",
                    e
                )));
            }
        }

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to lock scaling dealer socket: {}",
                    e
                ))
            })?
            .recv_multipart(0)
        {
            Ok(message_parts) => {
                if message_parts.len() < 2 {
                    return Err(TransportError::SendClientIdsToServerError(
                        "Malformed response".to_string(),
                    ));
                }

                let message_bytes: Vec<u8> = message_parts[1].to_vec();

                match String::from_utf8_lossy(&message_bytes).parse::<i64>() {
                    Ok(value) => match ServerResponse::from_i64(value) {
                        ServerResponse::Success => {
                            println!("[ZmqClient] Server updated cache with client IDs");
                            return Ok(());
                        }
                        ServerResponse::Failure => {
                            return Err(TransportError::SendClientIdsToServerError(
                                "Server failed to acknowledge client IDs".to_string(),
                            ));
                        }
                    },
                    Err(e) => {
                        return Err(TransportError::SendClientIdsToServerError(format!(
                            "Failed to parse server response: {}",
                            e
                        )));
                    }
                }
            }
            Err(e) => {
                return Err(TransportError::SendClientIdsToServerError(format!(
                    "Failed to receive client IDs from server: {}",
                    e
                )));
            }
        }
    }

    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        if scaling_id.is_nil() {
            return Err(TransportError::SendScalingWarningError(
                "Scaling ID is nil".to_string(),
            ));
        }

        if training_scaling_server_address.is_empty() {
            return Err(TransportError::SendScalingWarningError(
                "Scaling server address is empty".to_string(),
            ));
        }

        let operation_type = match operation {
            ScalingOperation::ScaleOut => "scale_out",
            ScalingOperation::ScaleIn => "scale_in",
        };

        println!(
            "[ZmqClient] Scaling warning notification send for {}",
            operation_type
        );

        // TODO: In a full implementation, this would send a ZMQ message to the training server
        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendScalingWarningError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                scaling_id,
                training_scaling_server_address,
                CacheAddressType::TrainingScalingServer,
                SocketPoolType::ScalingDealer,
            )
            .map_err(ZmqClientError::from)?;

        println!(
            "[ZmqClient] Sending scaling warning to {}",
            training_scaling_server_address
        );

        let transport_id_string = self.transport_id.to_string();
        let scaling_id_string = scaling_id.to_string();

        let empty_frame: Vec<u8> = vec![];
        let transport_id_frame: &[u8] = transport_id_string.as_bytes();
        let scaling_id_frame: &[u8] = scaling_id_string.as_bytes();
        let scaling_warning_payload: &[u8] = b"ROUTER_SCALE_WARNING";

        let socket = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendScalingWarningError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?
            .cached_sockets
            .scaling_dealer_socket
            .as_ref()
            .unwrap()
            .get(scaling_id)
            .unwrap()
            .value()
            .clone();

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendScalingWarningError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .send_multipart(
                [
                    &empty_frame,
                    transport_id_frame,
                    scaling_id_frame,
                    scaling_warning_payload,
                ],
                0,
            ) {
            Ok(_) => {
                println!("[ZmqClient] Scaling warning sent successfully");
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to send scaling warning: {}", e);
                return Err(TransportError::SendScalingWarningError(format!(
                    "Failed to send scaling warning: {}",
                    e
                )));
            }
        }

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendScalingWarningError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .recv_multipart(0)
        {
            Ok(message_parts) => {
                if message_parts.len() < 2 {
                    eprintln!("[ZmqClient] Malformed scaling warning response");
                    return Err(TransportError::SendScalingWarningError(
                        "Malformed scaling warning response".to_string(),
                    ));
                }

                let response_bytes: &Vec<u8> = &message_parts[1];
                println!(
                    "[ZmqClient] Scaling warning response: {}",
                    String::from_utf8_lossy(response_bytes)
                );

                match String::from_utf8_lossy(response_bytes).parse::<i64>() {
                    Ok(value) => match ServerResponse::from_i64(value) {
                        ServerResponse::Success => {
                            println!("[ZmqClient] Server acknowledged scaling warning");
                        }
                        ServerResponse::Failure => {
                            println!("[ZmqClient] Server failed to acknowledge scaling warning");
                            return Err(TransportError::SendScalingWarningError(
                                "Server failed to acknowledge scaling warning".to_string(),
                            ));
                        }
                    },
                    Err(e) => {
                        eprintln!(
                            "[ZmqClient] Failed to parse scaling warning response: {}",
                            e
                        );
                        return Err(TransportError::SendScalingWarningError(format!(
                            "Failed to parse scaling warning response: {}",
                            e
                        )));
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[ZmqClient] Failed to receive scaling warning response: {}",
                    e
                );
                return Err(TransportError::SendScalingWarningError(format!(
                    "Failed to receive scaling warning response: {}",
                    e
                )));
            }
        }

        Ok(())
    }

    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        if scaling_id.is_nil() {
            return Err(TransportError::SendScalingCompleteError(
                "Scaling ID is nil".to_string(),
            ));
        }

        if training_scaling_server_address.is_empty() {
            return Err(TransportError::SendScalingCompleteError(
                "Scaling server address is empty".to_string(),
            ));
        }

        let operation_type = match operation {
            ScalingOperation::ScaleOut => "scale_out",
            ScalingOperation::ScaleIn => "scale_in",
        };

        println!(
            "[ZmqClient] Scaling complete notification send for {}",
            operation_type
        );

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendScalingCompleteError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                scaling_id,
                training_scaling_server_address,
                CacheAddressType::TrainingScalingServer,
                SocketPoolType::ScalingDealer,
            )
            .map_err(ZmqClientError::from)?;

        println!(
            "[ZmqClient] Sending scaling complete to {}",
            training_scaling_server_address
        );

        let transport_id_string = self.transport_id.to_string();
        let scaling_id_string = scaling_id.to_string();

        let empty_frame: Vec<u8> = vec![];
        let transport_id_frame: &[u8] = transport_id_string.as_bytes();
        let scaling_id_frame: &[u8] = scaling_id_string.as_bytes();
        let scaling_complete_payload: &[u8] = b"ROUTER_SCALE_COMPLETE";

        let socket = {
            let pool = self.zmq_pool.read().map_err(|e| {
                TransportError::SendScalingCompleteError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?;
            let socket_entry = pool
                .cached_sockets
                .scaling_dealer_socket
                .as_ref()
                .unwrap()
                .get(scaling_id)
                .unwrap();
            socket_entry.value().clone()
        };

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendScalingCompleteError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .send_multipart(
                [
                    &empty_frame,
                    transport_id_frame,
                    scaling_id_frame,
                    scaling_complete_payload,
                ],
                0,
            ) {
            Ok(_) => {
                println!("[ZmqClient] Scaling complete sent successfully");
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to send scaling complete: {}", e);
                return Err(TransportError::SendScalingCompleteError(format!(
                    "Failed to send scaling complete: {}",
                    e
                )));
            }
        }

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendScalingCompleteError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .recv_multipart(0)
        {
            Ok(message_parts) => {
                if message_parts.len() < 2 {
                    eprintln!("[ZmqClient] Malformed scaling complete response");
                    return Err(TransportError::SendScalingCompleteError(
                        "Malformed scaling complete response".to_string(),
                    ));
                }

                let response_bytes: &Vec<u8> = &message_parts[1];
                println!(
                    "[ZmqClient] Scaling complete response: {}",
                    String::from_utf8_lossy(response_bytes)
                );

                match String::from_utf8_lossy(response_bytes).parse::<i64>() {
                    Ok(value) => match ServerResponse::from_i64(value) {
                        ServerResponse::Success => {
                            println!("[ZmqClient] Server acknowledged scaling complete");
                        }
                        ServerResponse::Failure => {
                            println!("[ZmqClient] Server failed to acknowledge scaling complete");
                            return Err(TransportError::SendScalingCompleteError(
                                "Server failed to acknowledge scaling complete".to_string(),
                            ));
                        }
                    },
                    Err(e) => {
                        eprintln!(
                            "[ZmqClient] Failed to parse scaling complete response: {}",
                            e
                        );
                        return Err(TransportError::SendScalingCompleteError(format!(
                            "Failed to parse scaling complete response: {}",
                            e
                        )));
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[ZmqClient] Failed to receive scaling complete response: {}",
                    e
                );
                return Err(TransportError::SendScalingCompleteError(format!(
                    "Failed to receive scaling complete response: {}",
                    e
                )));
            }
        }

        Ok(())
    }

    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        if scaling_id.is_nil() {
            return Err(TransportError::SendShutdownSignalError(
                "Scaling ID is nil".to_string(),
            ));
        }

        if training_scaling_server_address.is_empty() {
            return Err(TransportError::SendShutdownSignalError(
                "Scaling server address is empty".to_string(),
            ));
        }

        println!(
            "[ZmqClient] Sending shutdown signal to {}",
            training_scaling_server_address
        );

        let _ = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendShutdownSignalError(format!(
                    "Failed to read ZMQ pool during cache update: {}",
                    e
                ))
            })?
            .update_cache(
                scaling_id,
                training_scaling_server_address,
                CacheAddressType::TrainingScalingServer,
                SocketPoolType::ScalingDealer,
            )
            .map_err(ZmqClientError::from)?;

        let transport_id_string = self.transport_id.to_string();
        let scaling_id_string = scaling_id.to_string();

        let empty_frame: Vec<u8> = vec![];
        let transport_id_frame: &[u8] = transport_id_string.as_bytes();
        let scaling_id_frame: &[u8] = scaling_id_string.as_bytes();
        let shutdown_payload: &[u8] = b"CLIENT_SHUTDOWN";

        let socket = self
            .zmq_pool
            .read()
            .map_err(|e| {
                TransportError::SendShutdownSignalError(format!(
                    "Failed to read ZMQ pool during socket retrieval: {}",
                    e
                ))
            })?
            .cached_sockets
            .scaling_dealer_socket
            .as_ref()
            .unwrap()
            .get(scaling_id)
            .unwrap()
            .value()
            .clone();

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendShutdownSignalError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .send_multipart(
                [
                    &empty_frame,
                    transport_id_frame,
                    scaling_id_frame,
                    shutdown_payload,
                ],
                0,
            ) {
            Ok(_) => println!("[ZmqClient] Sent shutdown signal to server"),
            Err(e) => {
                return Err(TransportError::SendShutdownSignalError(format!(
                    "Failed to send shutdown signal to server: {}",
                    e
                )));
            }
        }

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendShutdownSignalError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .recv_multipart(0)
        {
            Ok(message_parts) => {
                if message_parts.len() < 2 {
                    return Err(TransportError::SendShutdownSignalError(
                        "Malformed response".to_string(),
                    ));
                }

                let response_bytes: &Vec<u8> = &message_parts[1];
                println!(
                    "[ZmqClient] Shutdown signal response: {}",
                    String::from_utf8_lossy(response_bytes)
                );

                match String::from_utf8_lossy(response_bytes).parse::<i64>() {
                    Ok(value) => match ServerResponse::from_i64(value) {
                        ServerResponse::Success => {
                            println!("[ZmqClient] Server acknowledged shutdown signal");
                            return Ok(());
                        }
                        ServerResponse::Failure => {
                            println!("[ZmqClient] Server failed to acknowledge shutdown signal");
                            return Err(TransportError::SendShutdownSignalError(
                                "Server failed to acknowledge shutdown signal".to_string(),
                            ));
                        }
                    },
                    Err(e) => {
                        return Err(TransportError::SendShutdownSignalError(format!(
                            "Failed to parse shutdown signal response: {}",
                            e
                        )));
                    }
                }
            }
            Err(e) => {
                return Err(TransportError::SendShutdownSignalError(format!(
                    "Failed to receive shutdown signal from server: {}",
                    e
                )));
            }
        }
    }
}
