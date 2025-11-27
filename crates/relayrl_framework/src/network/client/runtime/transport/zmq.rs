use crate::network::UuidPoolError;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::transport::{
    SyncClientTransport, TransportError, TransportUuid,
};
use crate::network::remove_uuid_from_pool;
use crate::network::{GLOBAL_UUID_POOL, add_uuid_to_pool, random_uuid};
use crate::utilities::configuration::ClientConfigLoader;
use crate::utilities::misc_utils::ServerAddresses;

use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::{EncodedTrajectory, RelayRLTrajectory};
use relayrl_types::types::model::utils::validate_module;
use relayrl_types::types::model::{HotReloadableModel, ModelModule};

use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::io::Write;
use zmq::SocketType::PAIR;

use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicI64};
use tempfile::NamedTempFile;
use tokio::task;
use uuid::Uuid;
use zmq::{Context, Socket, SocketType};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ZmqClientError {
    #[error(transparent)]
    ZmqClientError(#[from] zmq::Error),
    #[error(transparent)]
    FailedToGenerateUniqueUuidError(#[from] UuidPoolError),
}

type SocketUuid = Uuid;

pub(crate) struct ZmqSocketPool {
    pub(crate) model_dealer_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(crate) model_sub_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(crate) traj_push_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
    pub(crate) scaling_dealer_socket: Option<DashMap<SocketUuid, Arc<Mutex<Socket>>>>,
}

pub(crate) struct ZmqClient {
    transport_id: TransportUuid,
    current_version: Arc<AtomicI64>,
    algorithm_initialized: Arc<AtomicBool>,
    pub(crate) context: Context,
    cached_addresses: Option<DashMap<Uuid, Arc<RwLock<ServerAddresses>>>>,
    cached_sockets: Arc<ZmqSocketPool>,
}

impl ZmqClient {
    pub fn new() -> Result<Self, ZmqClientError> {
        let pid: u32 = std::process::id();
        let pid_bytes: [u8; _] = pid.to_be_bytes();

        let mut pid_buf: [u8; 16] = [0u8; 16];
        pid_buf[..4].copy_from_slice(&pid_bytes);

        let transport_id = random_uuid(
            "zmq_transport_client",
            pid_buf.into_iter().sum::<u8>() as u32,
            100,
            0,
        )
        .map_err(ZmqClientError::from)?;

        Ok(Self {
            transport_id,
            current_version: Arc::new(AtomicI64::new(0)),
            algorithm_initialized: Arc::new(AtomicBool::new(false)),
            context: Context::new(),
            cached_addresses: None,
            cached_sockets: Arc::new(ZmqSocketPool {
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
    ) -> Result<Socket, ZmqClientError> {
        let socket = context.socket(zmq::DEALER)?;

        // Set socket identity
        let identity: SocketUuid = random_uuid("zmq_dealer_socket", rand::random::<u32>(), 100, 0)
            .map_err(ZmqClientError::from)?;
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

    fn create_push_socket(
        &self,
        context: &Context,
        address: &str,
    ) -> Result<Socket, ZmqClientError> {
        let socket = context.socket(zmq::PUSH)?;

        let identity: SocketUuid = random_uuid("zmq_push_socket", rand::random::<u32>(), 100, 0)
            .map_err(ZmqClientError::from)?;
        socket.set_identity(identity.as_bytes())?;

        // Set send timeout to non-blocking
        socket.set_sndtimeo(5000)?; // 5 second timeout

        // Connect to trajectory server
        socket.connect(address)?;

        Ok(socket)
    }

    fn create_sub_socket(
        &self,
        context: &Context,
        address: &str,
    ) -> Result<Socket, ZmqClientError> {
        let socket = context.socket(zmq::SUB)?;

        let identity: SocketUuid = random_uuid("zmq_sub_socket", rand::random::<u32>(), 100, 0)
            .map_err(ZmqClientError::from)?;
        socket.set_identity(identity.as_bytes())?;

        socket.set_subscribe(b"")?;

        socket.connect(address)?;

        Ok(socket)
    }
}

#[repr(i64)]
enum ScalingResponse {
    Success(i64) = 0,
    Failure(i64) = 1,
}
impl ScalingResponse {
    fn from_i64(value: i64) -> Self {
        match value {
            0 => ScalingResponse::Success(value),
            1 => ScalingResponse::Failure(value),
            _ => ScalingResponse::Failure(value),
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientTransport<B> for ZmqClient {
    fn send_algorithm_init_request(
        &self,
        identity: &Uuid,
        agent_listener_address: &str,
    ) -> Result<(), TransportError> {
        Ok(())
    }

    fn initial_model_handshake(
        &self,
        identity: &Uuid,
        _model_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        if agent_listener_address.is_empty() {
            return Err(TransportError::ModelHandshakeError(
                "Agent listener address is empty".to_string()
            ));
        }

        {
            let cached_addresses = self.cached_addresses.as_ref().ok_or_else(|| {
                TransportError::ModelHandshakeError("Cached addresses not available".to_string())
            })?;
            // Check if we need to update the cache
            let needs_update: bool = {
                match cached_addresses.get(identity) {
                    Some(addresses) => {
                        addresses
                            .read()
                            .map_err(|e| {
                                TransportError::ModelHandshakeError(format!(
                                    "Failed to read cached addresses: {}",
                                    e
                                ))
                            })?
                            .agent_listener_address
                            != agent_listener_address
                    }
                    None => true,
                }
            };

            // Update cache if needed
            if needs_update {
                cached_addresses.insert(
                    *identity,
                    Arc::new(RwLock::new(ServerAddresses {
                        agent_listener_address: agent_listener_address.to_string(),
                        model_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .model_server_address
                            .clone(),
                        trajectory_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .trajectory_server_address
                            .clone(),
                        scaling_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .scaling_server_address
                            .clone(),
                    })),
                );

                let dealer_socket: Socket = self
                    .create_dealer_socket(&self.context, agent_listener_address)
                    .map_err(|e| {
                        TransportError::ModelHandshakeError(format!(
                            "Failed to create dealer socket: {}",
                            e
                        ))
                    })?;

                self.cached_sockets
                    .model_dealer_socket
                    .as_ref()
                    .unwrap()
                    .insert(*identity, Arc::new(Mutex::new(dealer_socket)));
            }
        }

        println!("[ZmqClient] Starting initial model handshake...");

        // Send GET_MODEL request
        let empty_frame: Vec<u8> = vec![];
        let get_model_frame: &[u8] = b"GET_MODEL";

        let socket = self
            .cached_sockets
            .model_dealer_socket
            .as_ref()
            .unwrap()
            .get(identity)
            .unwrap();

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::ModelHandshakeError(format!("Failed to lock dealer socket: {}", e))
            })?
            .send_multipart([&empty_frame, get_model_frame], 0)
        {
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
                    return Err(TransportError::ModelHandshakeError(format!(
                        "Malformed handshake response"
                    )));
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
                                return Err(TransportError::ModelHandshakeError(format!(
                                    "Failed to load model: {:?}",
                                    e
                                )));
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[ZmqClient] Failed to create temp file: {}", e);
                        return Err(TransportError::ModelHandshakeError(format!(
                            "Failed to create temp file: {}",
                            e
                        )));
                    }
                }
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to receive model: {}", e);
                return Err(TransportError::ModelHandshakeError(format!(
                    "Failed to receive model: {}",
                    e
                )));
            }
        }
    }

    fn send_traj_to_server(
        &self,
        identity: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        _model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        if trajectory_server_address.is_empty() {
            return Err(TransportError::SendTrajError(format!(
                "Trajectory server address is empty"
            )));
        }

        {
            let cached_addresses = self.cached_addresses.as_ref().ok_or_else(|| {
                TransportError::SendTrajError("Cached addresses not available".to_string())
            })?;
            let needs_update: bool = {
                cached_addresses
                    .get(identity)
                    .unwrap()
                    .read()
                    .unwrap()
                    .trajectory_server_address
                    != trajectory_server_address
            };

            if needs_update {
                cached_addresses.insert(
                    *identity,
                    Arc::new(RwLock::new(ServerAddresses {
                        agent_listener_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .agent_listener_address
                            .clone(),
                        model_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .model_server_address
                            .clone(),
                        trajectory_server_address: trajectory_server_address.to_string(),
                        scaling_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .scaling_server_address
                            .clone(),
                    })),
                );

                let push_socket: Socket = self
                    .create_push_socket(&self.context, trajectory_server_address)
                    .map_err(|e| {
                        TransportError::SendTrajError(format!(
                            "Failed to create push socket: {}",
                            e
                        ))
                    })?;
                self.cached_sockets
                    .traj_push_socket
                    .as_ref()
                    .unwrap()
                    .insert(*identity, Arc::new(Mutex::new(push_socket)));
            }
        }

        // Serialize the trajectory
        let serialized_traj: Vec<u8> = serde_json::to_vec(&encoded_trajectory).map_err(|e| {
            TransportError::SendTrajError(format!(
                "Failed to serialize trajectory: {}",
                e
            ))
        })?;

        println!(
            "[ZmqClient] Sending trajectory ({} bytes, {} actions)",
            serialized_traj.len(),
            encoded_trajectory.num_actions
        );

        let socket = self
            .cached_sockets
            .traj_push_socket
            .as_ref()
            .unwrap()
            .get(identity)
            .unwrap();

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

    fn listen_for_model(
        &self,
        identity: &Uuid,
        model_server_address: &str,
        global_dispatcher_tx: tokio::sync::mpsc::Sender<RoutedMessage>,
    ) -> Result<(), TransportError> {
        if model_server_address.is_empty() {
            return Err(TransportError::ListenForModelError(format!(
                "Model server address is empty"
            )));
        }

        {
            let cached_addresses = self.cached_addresses.as_ref().ok_or_else(|| {
                TransportError::ListenForModelError("Cached addresses not available".to_string())
            })?;
            let needs_update: bool = {
                cached_addresses
                    .get(identity)
                    .unwrap()
                    .read()
                    .unwrap()
                    .model_server_address
                    != model_server_address
            };
            if needs_update {
                cached_addresses.insert(
                    *identity,
                    Arc::new(RwLock::new(ServerAddresses {
                        agent_listener_address: cached_addresses
                            .get(&identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .agent_listener_address
                            .clone(),
                        model_server_address: model_server_address.to_string(),
                        trajectory_server_address: cached_addresses
                            .get(&identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .trajectory_server_address
                            .clone(),
                        scaling_server_address: cached_addresses
                            .get(&identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .scaling_server_address
                            .clone(),
                    })),
                );

                let sub_socket: Socket = self
                    .create_sub_socket(&self.context, model_server_address)
                    .map_err(|e| {
                        TransportError::ListenForModelError(format!(
                            "Failed to create sub socket: {}",
                            e
                        ))
                    })?;
                self.cached_sockets
                    .model_sub_socket
                    .as_ref()
                    .unwrap()
                    .insert(*identity, Arc::new(Mutex::new(sub_socket)));
            }
        }

        let socket = self
            .cached_sockets
            .model_sub_socket
            .as_ref()
            .unwrap()
            .get(identity)
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

    fn send_scaling_warning(
        &self,
        identity: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        let operation_type = match operation {
            ScalingOperation::ScaleUp => "scale_up",
            ScalingOperation::ScaleDown => "scale_down",
        };

        println!(
            "[ZmqClient] Scaling warning: {} operation initiated",
            operation_type
        );

        // TODO: In a full implementation, this would send a ZMQ message to the training server
        // For now, we log the warning and return success
        // The training server can use this to prepare for scaling operations
        {
            let cached_addresses = self.cached_addresses.as_ref().ok_or_else(|| {
                TransportError::SendScalingWarningError(
                    "Cached addresses not available".to_string(),
                )
            })?;
            let needs_update: bool = {
                cached_addresses
                    .get(identity)
                    .unwrap()
                    .read()
                    .unwrap()
                    .scaling_server_address
                    != scaling_server_address
            };
            if needs_update {
                cached_addresses.insert(
                    *identity,
                    Arc::new(RwLock::new(ServerAddresses {
                        agent_listener_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .agent_listener_address
                            .clone(),
                        model_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .model_server_address
                            .clone(),
                        trajectory_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .trajectory_server_address
                            .clone(),
                        scaling_server_address: scaling_server_address.to_string(),
                    })),
                );

                let dealer_socket: Socket = self
                    .create_dealer_socket(&self.context, scaling_server_address)
                    .map_err(|e| {
                        TransportError::SendScalingWarningError(format!(
                            "Failed to create dealer socket: {}",
                            e
                        ))
                    })?;
                self.cached_sockets
                    .scaling_dealer_socket
                    .as_ref()
                    .unwrap()
                    .insert(*identity, Arc::new(Mutex::new(dealer_socket)));
            }
        }

        println!(
            "[ZmqClient] Sending scaling warning to {}",
            scaling_server_address
        );

        let empty_frame: Vec<u8> = vec![];
        let get_model_frame: &[u8] = b"ROUTER_SCALE_WARNING";

        let socket = self
            .cached_sockets
            .scaling_dealer_socket
            .as_ref()
            .unwrap()
            .get(identity)
            .unwrap();

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendScalingWarningError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .send_multipart([&empty_frame, get_model_frame], 0)
        {
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
                    return Err(TransportError::SendScalingWarningError(format!(
                        "Malformed scaling warning response"
                    )));
                }

                let response_bytes: &Vec<u8> = &message_parts[1];
                println!(
                    "[ZmqClient] Scaling warning response: {}",
                    String::from_utf8_lossy(response_bytes)
                );

                match String::from_utf8_lossy(response_bytes).parse::<i64>() {
                    Ok(value) => match ScalingResponse::from_i64(value) {
                        ScalingResponse::Success(version) => {
                            println!("[ZmqClient] Server acknowledged scaling warning");
                        }
                        ScalingResponse::Failure(version) => {
                            println!("[ZmqClient] Server failed to acknowledge scaling warning");
                            return Err(TransportError::SendScalingWarningError(format!(
                                "Server failed to acknowledge scaling warning: {}",
                                version
                            )));
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

    fn send_scaling_complete(
        &self,
        identity: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        let operation_type = match operation {
            ScalingOperation::ScaleUp => "scale_up",
            ScalingOperation::ScaleDown => "scale_down",
        };

        println!(
            "[ZmqClient] Scaling complete: {} operation finished",
            operation_type
        );

        // TODO: In a full implementation, this would send a ZMQ message to the training server
        // to signal that scaling has completed and normal operations can resume
        // The server can acknowledge the completion and adjust its internal state
        {
            let cached_addresses = self.cached_addresses.as_ref().ok_or_else(|| {
                TransportError::SendScalingWarningError(
                    "Cached addresses not available".to_string(),
                )
            })?;
            let needs_update: bool = {
                cached_addresses
                    .get(identity)
                    .unwrap()
                    .read()
                    .unwrap()
                    .scaling_server_address
                    != scaling_server_address
            };
            if needs_update {
                cached_addresses.insert(
                    *identity,
                    Arc::new(RwLock::new(ServerAddresses {
                        agent_listener_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .agent_listener_address
                            .clone(),
                        model_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .model_server_address
                            .clone(),
                        trajectory_server_address: cached_addresses
                            .get(identity)
                            .unwrap()
                            .read()
                            .unwrap()
                            .trajectory_server_address
                            .clone(),
                        scaling_server_address: scaling_server_address.to_string(),
                    })),
                );

                let dealer_socket: Socket = self
                    .create_dealer_socket(&self.context, scaling_server_address)
                    .map_err(|e| {
                        TransportError::SendScalingWarningError(format!(
                            "Failed to create dealer socket: {}",
                            e
                        ))
                    })?;
                self.cached_sockets
                    .scaling_dealer_socket
                    .as_ref()
                    .unwrap()
                    .insert(*identity, Arc::new(Mutex::new(dealer_socket)));
            }
        }

        println!(
            "[ZmqClient] Sending scaling complete to {}",
            scaling_server_address
        );

        let empty_frame: Vec<u8> = vec![];
        let get_model_frame: &[u8] = b"ROUTER_SCALE_COMPLETE";

        let socket = self
            .cached_sockets
            .scaling_dealer_socket
            .as_ref()
            .unwrap()
            .get(identity)
            .unwrap();

        match socket
            .try_lock()
            .map_err(|e| {
                TransportError::SendScalingCompleteError(format!(
                    "Failed to lock dealer socket: {}",
                    e
                ))
            })?
            .send_multipart([&empty_frame, get_model_frame], 0)
        {
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
                    return Err(TransportError::SendScalingCompleteError(format!(
                        "Malformed scaling complete response"
                    )));
                }

                let response_bytes: &Vec<u8> = &message_parts[1];
                println!(
                    "[ZmqClient] Scaling complete response: {}",
                    String::from_utf8_lossy(response_bytes)
                );

                match String::from_utf8_lossy(response_bytes).parse::<i64>() {
                    Ok(value) => match ScalingResponse::from_i64(value) {
                        ScalingResponse::Success(_) => {
                            println!("[ZmqClient] Server acknowledged scaling complete");
                        }
                        ScalingResponse::Failure(version) => {
                            println!("[ZmqClient] Server failed to acknowledge scaling complete");
                            return Err(TransportError::SendScalingCompleteError(format!(
                                "Server failed to acknowledge scaling complete: {}",
                                version
                            )));
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

    fn shutdown(&self) -> Result<(), TransportError> {
        if let Some(sockets) = &self.cached_sockets.model_dealer_socket {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_uuid_from_pool("zmq_dealer_socket", &socket_id)
                    .map_err(|e| TransportError::UuidPoolError(e))?;

                sockets.remove(&socket_id);
            }
        }

        if let Some(sockets) = &self.cached_sockets.model_sub_socket {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_uuid_from_pool("zmq_sub_socket", &socket_id)
                    .map_err(|e| TransportError::UuidPoolError(e))?;

                sockets.remove(&socket_id);
            }
        }

        if let Some(sockets) = &self.cached_sockets.traj_push_socket {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_uuid_from_pool("zmq_push_socket", &socket_id)
                    .map_err(|e| TransportError::UuidPoolError(e))?;

                sockets.remove(&socket_id);
            }
        }

        if let Some(sockets) = &self.cached_sockets.scaling_dealer_socket {
            for entry in sockets.iter() {
                let socket_id = *entry.key();
                remove_uuid_from_pool("zmq_dealer_socket", &socket_id)
                    .map_err(|e| TransportError::UuidPoolError(e))?;

                sockets.remove(&socket_id);
            }
        }

        remove_uuid_from_pool("zmq_transport_client", &self.transport_id)
            .map_err(|e| TransportError::UuidPoolError(e))?;
        Ok(())
    }
}
