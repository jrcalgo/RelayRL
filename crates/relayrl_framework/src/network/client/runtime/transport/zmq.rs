use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::transport::SyncClientTransport;
use crate::utilities::configuration::ClientConfigLoader;

use relayrl_types::types::model::utils::validate_module;
use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::{EncodedTrajectory, RelayRLTrajectory};
use relayrl_types::types::model::{ModelModule, HotReloadableModel};

use burn_tensor::backend::Backend;
use std::io::Write;

use tempfile::NamedTempFile;
use tokio::task;
use uuid::Uuid;
use zmq::{Context, Socket};

pub struct ZmqClient {
    agent_listener_address: String,
    trajectory_server_address: String,
    training_server_address: String,
    max_traj_length: u128,
    client_identity: String,
}

impl ZmqClient {
    pub fn new(config: &ClientConfigLoader) -> Self {
        let agent_listener = config.transport_config.get_agent_listener_address();
        let trajectory_server = config.transport_config.get_trajectory_server_address();
        let training_server = config.transport_config.get_training_server_address();
        let max_traj_length = config.transport_config.max_traj_length;

        let agent_listener_address = format!(
            "{}{}:{}",
            agent_listener.prefix, agent_listener.host, agent_listener.port
        );
        let trajectory_server_address = format!(
            "{}{}:{}",
            trajectory_server.prefix, trajectory_server.host, trajectory_server.port
        );
        let training_server_address = format!(
            "{}{}:{}",
            training_server.prefix, training_server.host, training_server.port
        );

        let pid: u32 = std::process::id();
        let pid_bytes = pid.to_be_bytes();

        let mut pid_buf = [0u8; 16];
        pid_buf[..4].copy_from_slice(&pid_bytes);

        // Generate a unique client identity
        let client_identity = format!("RelayRLClient-{}", Uuid::new_v8(pid_buf));

        Self {
            agent_listener_address,
            trajectory_server_address,
            training_server_address,
            max_traj_length,
            client_identity,
        }
    }

    fn create_dealer_socket(&self, context: &Context, address: &str) -> Result<Socket, zmq::Error> {
        let socket = context.socket(zmq::DEALER)?;

        // Set socket identity
        socket.set_identity(self.client_identity.as_bytes())?;

        // Set socket options for performance
        socket.set_sndhwm(1000)?;
        socket.set_rcvhwm(1000)?;
        socket.set_maxmsgsize(-1)?;

        // Connect to the server
        socket.connect(address)?;

        Ok(socket)
    }

    fn create_push_socket(&self, context: &Context, address: &str) -> Result<Socket, zmq::Error> {
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
        _model_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, String> {
        let context = Context::new();

        // Use agent_listener_address for handshake
        let socket = match self.create_dealer_socket(&context, &self.agent_listener_address) {
            Ok(socket) => socket,
            Err(e) => {
                eprintln!("[ZmqClient] Failed to create handshake socket: {}", e);
                return Ok(None);
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
                return Ok(None);
            }
        }

        // Wait for model response with timeout
        socket
            .set_rcvtimeo(30000)
            .expect("Failed to set receive timeout"); // 30 second timeout

        match socket.recv_multipart(0) {
            Ok(message_parts) => {
                if message_parts.len() < 2 {
                    eprintln!("[ZmqClient] Malformed handshake response");
                    return Ok(None);
                }

                let model_bytes = &message_parts[1];
                println!(
                    "[ZmqClient] Received initial model ({} bytes)",
                    model_bytes.len()
                );

                // Save model to temporary file and load it
                match NamedTempFile::new() {
                    Ok(mut temp_file) => {
                        if let Err(e) = temp_file.write_all(model_bytes) {
                            eprintln!("[ZmqClient] Failed to write model to temp file: {}", e);
                            return Ok(None);
                        }

                        match ModelModule::<B>::load_from_path(temp_file.path()) {
                            Ok(model) => {
                                if let Err(e) = validate_module::<B>(&model) {
                                    eprintln!("[ZmqClient] Failed to validate model: {:?}", e);
                                    return Ok(None);
                                }
                                println!("[ZmqClient] Model loaded and validated successfully");
                                Ok(Some(model))
                            }
                            Err(e) => {
                                eprintln!("[ZmqClient] Failed to load model: {:?}", e);
                                Ok(None)
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[ZmqClient] Failed to create temp file: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                eprintln!("[ZmqClient] Failed to receive model: {}", e);
                Ok(None)
            }
        }
    }

    fn send_traj_to_server(
        &self,
        encoded_trajectory: EncodedTrajectory,
        _training_server_address: &str,
    ) -> Result<(), String> {
        let context = Context::new();

        // Use trajectory_server_address for sending trajectories
        let socket = self
            .create_push_socket(&context, &self.trajectory_server_address)
            .map_err(|e| format!("Failed to create trajectory socket: {}", e))?;

        // Serialize the trajectory
        let serialized_traj: Vec<u8> = serde_json::to_vec(&encoded_trajectory).map_err(|e| format!("Failed to serialize trajectory: {}", e))?;

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
                let error_msg = format!("Failed to send trajectory: {}", e);
                eprintln!("[ZmqClient] {}", error_msg);
                Err(error_msg)
            }
        }
    }

    fn listen_for_model(
        &self,
        _model_server_address: &str,
        tx_to_router: tokio::sync::mpsc::Sender<RoutedMessage>,
    ) {
        let sub_address = self.training_server_address.clone();

        task::spawn_blocking(move || {
            let context = Context::new();
            let socket = context
                .socket(zmq::SUB)
                .expect("Failed to create SUB socket");
            socket.set_subscribe(b"").expect("SUB subscribe failed");
            if let Err(e) = socket.connect(&sub_address) {
                eprintln!("[ZmqClient] Failed to connect SUB socket: {}", e);
                return;
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
                        let _ = tx_to_router.blocking_send(msg);
                    }
                    Err(e) => {
                        eprintln!("[ZmqClient] SUB socket recv error: {}", e);
                        break;
                    }
                }
            }
        });
    }

    fn send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), String> {
        // TODO: implement
        Ok(())
    }

    fn send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), String> {
        // TODO: implement
        Ok(())
    }
}
