#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::AsyncClientTransport;
#[cfg(feature = "grpc_network")]
use crate::network::validate_model;
#[cfg(feature = "grpc_network")]
use crate::types::trajectory::RelayRLTrajectory;
#[cfg(feature = "grpc_network")]
use crate::utilities::configuration::ClientConfigLoader;
#[cfg(feature = "grpc_network")]
use crate::utilities::orchestration::tonic_utils::deserialize_model;
#[cfg(feature = "grpc_network")]
use async_trait::async_trait;
#[cfg(feature = "grpc_network")]
use std::collections::HashMap;
#[cfg(feature = "grpc_network")]
use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};
#[cfg(feature = "grpc_network")]
use tch::CModule;
#[cfg(feature = "grpc_network")]
use tokio::time::{Duration, timeout};
#[cfg(feature = "grpc_network")]
use tonic::Request;
#[cfg(feature = "grpc_network")]
use tonic::transport::Channel;

// Generated proto definitions
#[cfg(feature = "grpc_network")]
pub mod rl_service {
    tonic::include_proto!("relayrl");
}

#[cfg(feature = "grpc_network")]
use rl_service::{
    Action, GetModelRequest, InitRequest, ModelResponse, ParameterValue, SendTrajectoriesRequest,
    SendTrajectoriesResponse, Trajectory, rl_service_client::RlServiceClient,
};

#[cfg(feature = "grpc_network")]
pub struct TonicClient {
    client: Option<RlServiceClient<Channel>>,
    client_id: String,
    current_version: Arc<AtomicI64>,
    config: Arc<ClientConfigLoader>,
    algorithm_initialized: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "grpc_network")]
impl TonicClient {
    pub fn new(config: &ClientConfigLoader) -> Self {
        let pid: u32 = std::process::id();
        let pid_bytes = pid.to_be_bytes();

        let mut pid_buf = [0u8; 16];
        pid_buf[..4].copy_from_slice(&pid_bytes);

        Self {
            client: None,
            client_id: uuid::Uuid::new_v8(pid_buf).to_string(),
            current_version: Arc::new(AtomicI64::new(0)),
            config: Arc::new(config.clone()),
            algorithm_initialized: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    async fn ensure_client(
        &mut self,
        server_address: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.client.is_none() {
            let channel = Channel::from_shared(format!("http://{}", server_address))?
                .connect()
                .await?;
            self.client = Some(RlServiceClient::new(channel));
        }
        Ok(())
    }

    async fn get_model_with_timeout(
        &mut self,
        server_address: &str,
        expected_version: i64,
    ) -> Result<ModelResponse, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_client(server_address).await?;

        let current_version = self.current_version.load(Ordering::SeqCst);
        let request = Request::new(GetModelRequest {
            client_id: self.client_id.clone(),
            client_version: current_version as i32,
            expected_version: expected_version as i32,
        });

        let response = timeout(
            Duration::from_secs(30),
            self.client.as_mut().unwrap().get_model(request),
        )
        .await??;

        Ok(response.into_inner())
    }

    async fn send_trajectories_with_timeout(
        &mut self,
        server_address: &str,
        trajectories: Vec<Trajectory>,
    ) -> Result<SendTrajectoriesResponse, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_client(server_address).await?;

        let request = Request::new(SendTrajectoriesRequest {
            client_id: self.client_id.clone(),
            trajectories,
        });

        let response = timeout(
            Duration::from_secs(30),
            self.client.as_mut().unwrap().send_trajectories(request),
        )
        .await??;

        Ok(response.into_inner())
    }

    fn convert_relayrl_to_proto_trajectory(&self, traj: &RelayRLTrajectory) -> Trajectory {
        let actions: Vec<Action> = traj
            .actions
            .iter()
            .map(|action| {
                let mut data = HashMap::new();

                // Convert additional data if present using proper serialization
                if let Some(action_data) = &action.data {
                    for (key, value) in action_data {
                        if let Ok(serialized) = serde_json::to_vec(value) {
                            data.insert(key.clone(), serialized);
                        }
                    }
                }

                Action {
                    obs: action
                        .obs
                        .as_ref()
                        .map_or_else(Vec::new, |tensor_data| tensor_data.data.clone()),
                    action: action
                        .act
                        .as_ref()
                        .map_or_else(Vec::new, |tensor_data| tensor_data.data.clone()),
                    mask: action
                        .mask
                        .as_ref()
                        .map_or_else(Vec::new, |tensor_data| tensor_data.data.clone()),
                    reward: action.rew,
                    data,
                    done: action.done,
                }
            })
            .collect();

        Trajectory {
            actions,
            version: self.current_version.load(Ordering::SeqCst) as i32,
        }
    }

    async fn initialize_algorithm_if_needed(
        &mut self,
        server_address: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.algorithm_initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.ensure_client(server_address).await?;

        // Get algorithm name from config
        let algorithm_name = self.config.client_config.algorithm_name.clone();

        // Prepare algorithm parameters from config
        let mut algorithm_parameters = HashMap::new();

        // Add common parameters that might be needed
        algorithm_parameters.insert(
            "max_traj_length".to_string(),
            ParameterValue {
                value: Some(rl_service::parameter_value::Value::IntValue(
                    self.config.transport_config.max_traj_length as i32,
                )),
            },
        );

        let request = Request::new(InitRequest {
            client_id: self.client_id.clone(),
            algorithm_name,

            algorithm_parameters,
        });

        let response = timeout(
            Duration::from_secs(30),
            self.client.as_mut().unwrap().init_algorithm(request),
        )
        .await??;

        let init_response = response.into_inner();
        if init_response.is_success {
            self.algorithm_initialized.store(true, Ordering::SeqCst);
            println!(
                "[TonicClient] Algorithm initialized successfully: {}",
                init_response.message
            );
        } else {
            return Err(
                format!("Failed to initialize algorithm: {}", init_response.message).into(),
            );
        }

        Ok(())
    }

    /// Helper method to send an RelayRLTrajectory directly (for compatibility with internal usage)
    pub async fn send_relayrl_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
        training_server_address: &str,
    ) -> Result<(), String> {
        let mut client = Self::new(&self.config);

        // Ensure algorithm is initialized
        if let Err(e) = client
            .initialize_algorithm_if_needed(training_server_address)
            .await
        {
            return Err(format!("Failed to initialize algorithm: {}", e));
        }

        let proto_trajectory = client.convert_relayrl_to_proto_trajectory(&trajectory);

        match client
            .send_trajectories_with_timeout(training_server_address, vec![proto_trajectory])
            .await
        {
            Ok(response) => {
                if response.model_updated {
                    println!(
                        "[TonicClient] Trajectory sent successfully, model updated to version {}",
                        response.new_version
                    );
                    client
                        .current_version
                        .store(response.new_version as i64, Ordering::SeqCst);
                } else {
                    println!("[TonicClient] Trajectory sent successfully, no model update");
                }
                Ok(())
            }
            Err(e) => {
                let error_msg = format!("Failed to send trajectory: {}", e);
                eprintln!("[TonicClient] {}", error_msg);
                Err(error_msg)
            }
        }
    }
}

#[cfg(feature = "grpc_network")]
#[async_trait]
impl AsyncClientTransport for TonicClient {
    /// Helper method to perform initial handshake and return whether a model was received
    async fn initial_model_handshake(
        &self,
        training_server_address: &str,
    ) -> Result<Option<CModule>, String> {
        let mut client = Self::new(&self.config);

        // Initialize algorithm first
        if let Err(e) = client
            .initialize_algorithm_if_needed(training_server_address)
            .await
        {
            return Err(format!("Failed to initialize algorithm: {}", e));
        }

        match client
            .get_model_with_timeout(training_server_address, 0)
            .await
        {
            Ok(model_response) => {
                let current_version = client.current_version.load(Ordering::SeqCst);
                if model_response.version as i64 > current_version {
                    println!(
                        "[TonicClient] Model handshake successful, received version {}",
                        model_response.version
                    );
                    client
                        .current_version
                        .store(model_response.version as i64, Ordering::SeqCst);

                    // Validate and deserialize the model if available
                    if !model_response.model_state.is_empty() {
                        match deserialize_model(model_response.model_state) {
                            Ok(model) => {
                                // Validate the model
                                validate_model(&model);
                                println!("[TonicClient] Model validated and ready for use");

                                Ok(Some(model))
                            }
                            Err(e) => Err(format!("Failed to deserialize model: {}", e)),
                        }
                    } else {
                        println!("[TonicClient] No model data available yet");
                        Ok(None)
                    }
                } else {
                    println!("[TonicClient] Model version up to date");
                    Ok(None)
                }
            }
            Err(e) => Err(format!("Model handshake failed: {}", e)),
        }
    }

    async fn send_traj_to_server(
        &self,
        trajectory: rl_service::Trajectory,
        training_server_address: &str,
    ) {
        let mut client = Self::new(&self.config);

        // Ensure algorithm is initialized
        if let Err(e) = client
            .initialize_algorithm_if_needed(training_server_address)
            .await
        {
            eprintln!(
                "[TonicClient] Failed to initialize algorithm before sending trajectory: {}",
                e
            );
            return;
        }

        // Convert from the old proto format to new format
        let proto_trajectory = rl_service::Trajectory {
            actions: trajectory
                .actions
                .into_iter()
                .map(|action| Action {
                    obs: action.obs,
                    action: action.action,
                    mask: action.mask,
                    reward: action.reward,
                    data: action.data,
                    done: action.done,
                })
                .collect(),
            version: client.current_version.load(Ordering::SeqCst) as i32,
        };

        match client
            .send_trajectories_with_timeout(training_server_address, vec![proto_trajectory])
            .await
        {
            Ok(response) => {
                if response.model_updated {
                    println!(
                        "[TonicClient] Trajectory sent successfully, model updated to version {}",
                        response.new_version
                    );
                    client
                        .current_version
                        .store(response.new_version as i64, Ordering::SeqCst);
                } else {
                    println!("[TonicClient] Trajectory sent successfully, no model update");
                }
            }
            Err(e) => {
                eprintln!("[TonicClient] Failed to send trajectory: {}", e);
            }
        }
    }

    async fn listen_for_model(&self, model_server_address: &str) {
        let mut client = Self::new(&self.config);
        let mut polling_interval = tokio::time::interval(Duration::from_secs(5));

        // Ensure algorithm is initialized before starting to listen
        if let Err(e) = client
            .initialize_algorithm_if_needed(model_server_address)
            .await
        {
            eprintln!(
                "[TonicClient] Failed to initialize algorithm before listening: {}",
                e
            );
            return;
        }

        loop {
            polling_interval.tick().await;

            let current_version = client.current_version.load(Ordering::SeqCst);
            match client
                .get_model_with_timeout(model_server_address, current_version + 1)
                .await
            {
                Ok(model_response) => {
                    if model_response.version as i64 > current_version {
                        println!(
                            "[TonicClient] New model available, version {}",
                            model_response.version
                        );
                        client
                            .current_version
                            .store(model_response.version as i64, Ordering::SeqCst);

                        // Log model update availability - the router integration would need
                        // to be handled at a higher level since this trait doesn't provide
                        // access to the router's sender channel
                        if !model_response.model_state.is_empty() {
                            println!(
                                "[TonicClient] Model data received (size: {} bytes)",
                                model_response.model_state.len()
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[TonicClient] Model polling failed: {}", e);
                    // Back off on error
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    fn convert_relayrl_to_proto_trajectory(&self, traj: &RelayRLTrajectory) -> Trajectory {
        // Delegate to the existing implementation method
        let actions: Vec<Action> = traj
            .actions
            .iter()
            .map(|action| {
                let mut data = std::collections::HashMap::new();

                // Convert additional data if present using proper serialization
                if let Some(action_data) = &action.data {
                    for (key, value) in action_data {
                        if let Ok(serialized) = serde_json::to_vec(value) {
                            data.insert(key.clone(), serialized);
                        }
                    }
                }

                Action {
                    obs: action
                        .obs
                        .as_ref()
                        .map_or_else(Vec::new, |tensor_data| tensor_data.data.clone()),
                    action: action
                        .act
                        .as_ref()
                        .map_or_else(Vec::new, |tensor_data| tensor_data.data.clone()),
                    mask: action
                        .mask
                        .as_ref()
                        .map_or_else(Vec::new, |tensor_data| tensor_data.data.clone()),
                    reward: action.rew,
                    data,
                    done: action.done,
                }
            })
            .collect();

        Trajectory {
            actions,
            version: self
                .current_version
                .load(std::sync::atomic::Ordering::SeqCst) as i32,
        }
    }

    async fn send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), String> {
        // TODO: implement
        Ok(())
    }

    async fn send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), String> {
        // TODO: implement
        Ok(())
    }
}
