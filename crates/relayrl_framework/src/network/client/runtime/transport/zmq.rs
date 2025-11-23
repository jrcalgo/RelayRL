use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::transport::{SyncClientTransport, TransportError, TransportUuid};
use crate::network::random_uuid;
use crate::utilities::configuration::ClientConfigLoader;

use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::{EncodedTrajectory, RelayRLTrajectory};
use relayrl_types::types::model::utils::validate_module;
use relayrl_types::types::model::{HotReloadableModel, ModelModule};

use burn_tensor::backend::Backend;
use std::io::Write;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64};
use tempfile::NamedTempFile;
use tokio::sync::RwLock;
use tokio::task;
use uuid::Uuid;
use zmq::{Context, Socket};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ZmqClientError {
    #[error(transparent)]
    ZmqClientError(#[from] zmq::Error),
}

pub struct ZmqClient {
    transport_id: TransportUuid,
    current_version: Arc<AtomicI64>,
    algorithm_initialized: Arc<AtomicBool>,
}

impl ZmqClient {
    pub fn new() -> Self {
        let pid: u32 = std::process::id();
        let pid_bytes: [u8; _] = pid.to_be_bytes();

        let mut pid_buf: [u8; 16] = [0u8; 16];
        pid_buf[..4].copy_from_slice(&pid_bytes);

        Self {
            transport_id: random_uuid(pid_buf.into_iter().sum::<u8>() as u32),
            current_version: Arc::new(AtomicI64::new(0)),
            algorithm_initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    fn create_dealer_socket(
        &self,
        context: &Context,
        address: &str,
    ) -> Result<Socket, ZmqClientError> {
        let socket = context.socket(zmq::DEALER)?;

        // Set socket identity
        socket.set_identity(self.transport_id.to_string().as_bytes())?;

        // Set socket options for performance
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

        // Set send timeout to non-blocking
        socket.set_sndtimeo(5000)?; // 5 second timeout

        // Connect to trajectory server
        socket.connect(address)?;

        Ok(socket)
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientTransport<B> for ZmqClient {
    fn initial_model_handshake(
        &self,
        training_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        let context = Context::new();

        // Use agent_listener_address for handshake
        let socket = match self.create_dealer_socket(&context, agent_listener_address) {
            Ok(socket) => socket,
            Err(e) => {
                eprintln!("[ZmqClient] Failed to create handshake socket: {}", e);
                return Err(TransportError::ModelHandshakeError(format!(
                    "Failed to create handshake socket: {}",
                    e
                )));
            }
        };

        println!("[ZmqClient] Starting initial model handshake...");

        // Send GET_MODEL request
        let empty_frame: Vec<u8> = vec![];
        let get_model_frame: &[u8] = b"GET_MODEL";

        match socket.send_multipart([&empty_frame, get_model_frame], 0) {
            Ok(_) => println!("[ZmqClient] Sent GET_MODEL request"),
            Err(e) => {
                eprintln!("[ZmqClient] Failed to send GET_MODEL: {}", e);
                return Err(TransportError::ModelHandshakeError(format!(
                    "Failed to send GET_MODEL: {}",
                    e
                )));
            }
        }

        // Wait for model response with timeout
        socket.set_rcvtimeo(30000).map_err(|e| {
            TransportError::ModelHandshakeError(format!("Failed to set receive timeout: {}", e))
        })?;

        match socket.recv_multipart(0) {
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
        encoded_trajectory: EncodedTrajectory,
        training_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        let context = Context::new();

        // Use trajectory_server_address for sending trajectories
        let socket = self
            .create_push_socket(&context, trajectory_server_address)
            .map_err(|e| {
                TransportError::SendTrajError(format!(
                    "Failed to create trajectory socket: {}",
                    e.to_string()
                ))
            })?;

        // Serialize the trajectory
        let serialized_traj: Vec<u8> = serde_json::to_vec(&encoded_trajectory).map_err(|e| {
            TransportError::SendTrajError(format!(
                "Failed to serialize trajectory: {}",
                e.to_string()
            ))
        })?;

        println!(
            "[ZmqClient] Sending trajectory ({} bytes, {} actions)",
            serialized_traj.len(),
            encoded_trajectory.num_actions
        );

        // Send the trajectory
        match socket.send(serialized_traj, 0) {
            Ok(_) => {
                println!("[ZmqClient] Trajectory sent successfully");
                Ok(())
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to send trajectory: {}", e);
                return Err(TransportError::SendTrajError(format!(
                    "Failed to send trajectory: {}",
                    e
                )));
            }
        }
    }

    fn listen_for_model(
        &self,
        training_server_address: &str,
        global_dispatcher_tx: tokio::sync::mpsc::Sender<RoutedMessage>,
    ) -> Result<(), TransportError> {
        let sub_address = training_server_address.to_string();

        task::spawn_blocking(move || {
            let context = Context::new();
            let socket = context.socket(zmq::SUB).map_err(|e| {
                TransportError::ListenForModelError(format!(
                    "Failed to create SUB socket: {}",
                    e.to_string()
                ))
            })?;
            socket.set_subscribe(b"").map_err(|e| {
                TransportError::ListenForModelError(format!(
                    "SUB subscribe failed: {}",
                    e.to_string()
                ))
            })?;
            if let Err(e) = socket.connect(&sub_address) {
                eprintln!("[ZmqClient] Failed to connect SUB socket: {}", e);
                return Err::<(), TransportError>(TransportError::ListenForModelError(format!(
                    "Failed to connect SUB socket: {}",
                    e
                )));
            }
            println!("[ZmqClient] Listening for model updates at {}", sub_address);
            loop {
                match socket.recv_bytes(0) {
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
                        return Err(TransportError::ListenForModelError(format!(
                            "SUB socket recv error: {}",
                            e
                        )));
                    }
                }
            }
        });
        Ok(())
    }

    fn send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), TransportError> {
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

        Ok(())
    }

    fn send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), TransportError> {
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

        Ok(())
    }

    fn shutdown(&self) -> Result<(), TransportError> {
        // TODO: implement shutdown logic
        Ok(())
    }
}
