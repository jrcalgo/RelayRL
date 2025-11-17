#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::AsyncClientTransport;
#[cfg(feature = "grpc_network")]
use crate::utilities::configuration::ClientConfigLoader;
#[cfg(feature = "grpc_network")]
use crate::utilities::orchestration::tonic_utils::relayrl_encoded_trajectory_to_grpc_encoded_trajectory;
#[cfg(feature = "grpc_network")]
use async_trait::async_trait;
#[cfg(feature = "grpc_network")]
use relayrl_types::types::data::trajectory::EncodedTrajectory;
#[cfg(feature = "grpc_network")]
use relayrl_types::types::model::utils::{
    deserialize_model_module, serialize_model_module, validate_module,
};
#[cfg(feature = "grpc_network")]
use relayrl_types::types::model::{HotReloadableModel, ModelModule};
#[cfg(feature = "grpc_network")]
use std::collections::HashMap;
#[cfg(feature = "grpc_network")]
use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};
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
    EncodedAction as GrpcEncodedAction, EncodedTrajectory as GrpcEncodedTrajectory,
    GetModelRequest, InitRequest, ModelResponse, ParameterValue, SendTrajectoriesRequest,
    SendTrajectoriesResponse, rl_service_client::RlServiceClient,
};

use crate::network::client::runtime::transport::TransportError;
use burn_tensor::backend::Backend;
use relayrl_types::types::data::tensor::BackendMatcher;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TonicClientError {
    #[error("Client connection error: {0}")]
    ClientConnectionError(String),
    #[error("Client not initialized: {0}")]
    ClientNotInitializedError(String),
    #[error("Algorithm initialization error: {0}")]
    AlgorithmInitializationError(String),
    #[error("Send trajectory error: {0}")]
    SendTrajectoryError(String),
    #[error("Request timeout: {0}")]
    TimeoutError(String),
    #[error("gRPC error: {0}")]
    GrpcError(String),
}

#[cfg(feature = "grpc_network")]
impl From<tonic::Status> for TonicClientError {
    fn from(status: tonic::Status) -> Self {
        TonicClientError::GrpcError(status.to_string())
    }
}

#[cfg(feature = "grpc_network")]
impl From<tokio::time::error::Elapsed> for TonicClientError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        TonicClientError::TimeoutError("Request timed out".to_string())
    }
}

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

    async fn ensure_client(&mut self, server_address: &str) -> Result<(), TonicClientError> {
        if self.client.is_none() {
            let channel = Channel::from_shared(format!("http://{}", server_address))
                .map_err(|e| TonicClientError::ClientConnectionError(e.to_string()))?
                .connect()
                .await
                .map_err(|e| TonicClientError::ClientConnectionError(e.to_string()))?;
            self.client = Some(RlServiceClient::new(channel));
        }
        Ok(())
    }

    async fn get_model_with_timeout(
        &mut self,
        server_address: &str,
        expected_version: i64,
    ) -> Result<ModelResponse, TonicClientError> {
        self.ensure_client(server_address).await?;

        let current_version = self.current_version.load(Ordering::SeqCst);
        let request = Request::new(GetModelRequest {
            client_id: self.client_id.clone(),
            client_version: current_version as i32,
            expected_version: expected_version as i32,
        });

        let response = timeout(
            Duration::from_secs(30),
            self.client
                .as_mut()
                .ok_or_else(|| {
                    TonicClientError::ClientNotInitializedError(
                        "Client not initialized".to_string(),
                    )
                })?
                .get_model(request),
        )
        .await??;

        Ok(response.into_inner())
    }

    async fn send_trajectories_with_timeout(
        &mut self,
        server_address: &str,
        trajectories: Vec<GrpcEncodedTrajectory>,
    ) -> Result<SendTrajectoriesResponse, TonicClientError> {
        self.ensure_client(server_address).await?;

        let request = Request::new(SendTrajectoriesRequest {
            client_id: self.client_id.clone(),
            trajectories,
        });

        let response = timeout(
            Duration::from_secs(30),
            self.client
                .as_mut()
                .ok_or_else(|| {
                    TonicClientError::ClientNotInitializedError(
                        "Client not initialized".to_string(),
                    )
                })?
                .send_trajectories(request),
        )
        .await??;

        Ok(response.into_inner())
    }

    async fn initialize_algorithm_if_needed(
        &mut self,
        server_address: &str,
    ) -> Result<(), TonicClientError> {
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
            self.client
                .as_mut()
                .ok_or_else(|| {
                    TonicClientError::ClientNotInitializedError(
                        "Client not initialized".to_string(),
                    )
                })?
                .init_algorithm(request),
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
            return Err(TonicClientError::AlgorithmInitializationError(format!(
                "Failed to initialize algorithm: {}",
                init_response.message
            )));
        }

        Ok(())
    }

    /// Helper method to send an RelayRLTrajectory directly (for compatibility with internal usage)
    pub async fn send_relayrl_trajectory(
        &self,
        encoded_trajectory: EncodedTrajectory,
        training_server_address: &str,
    ) -> Result<(), TonicClientError> {
        let mut client = Self::new(&self.config);

        // Ensure algorithm is initialized
        if let Err(e) = client
            .initialize_algorithm_if_needed(training_server_address)
            .await
        {
            return Err(TonicClientError::AlgorithmInitializationError(format!(
                "Failed to initialize algorithm: {}",
                e
            )));
        }

        let proto_trajectory =
            relayrl_encoded_trajectory_to_grpc_encoded_trajectory(encoded_trajectory);

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
            Err(e) => Err(TonicClientError::SendTrajectoryError(format!(
                "Failed to send trajectory: {}",
                e
            ))),
        }
    }
}

#[cfg(feature = "grpc_network")]
#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientTransport<B> for TonicClient {
    /// Helper method to perform initial handshake and return whether a model was received
    async fn initial_model_handshake(
        &self,
        training_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        let mut client = Self::new(&self.config);

        // Initialize algorithm first
        if let Err(e) = client
            .initialize_algorithm_if_needed(training_server_address)
            .await
        {
            return Err(TransportError::ModelHandshakeError(format!(
                "Failed to initialize algorithm: {}",
                e
            )));
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
                        // Device parameter is not used in deserialize_model_module (prefixed with _)
                        let device = relayrl_types::types::data::tensor::DeviceType::Cpu;
                        match deserialize_model_module::<B>(model_response.model_state, device) {
                            Ok(model) => {
                                // Validate the model - it gets dimensions from the model itself
                                if let Err(e) = validate_module::<B>(&model) {
                                    return Err(TransportError::ModelHandshakeError(format!(
                                        "Failed to validate model: {:?}",
                                        e
                                    )));
                                }
                                println!("[TonicClient] Model validated and ready for use");
                                Ok(Some(model))
                            }
                            Err(e) => Err(TransportError::ModelHandshakeError(format!(
                                "Failed to deserialize model: {:?}",
                                e
                            ))),
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
            Err(e) => Err(TransportError::ModelHandshakeError(format!(
                "Model handshake failed: {}",
                e
            ))),
        }
    }

    async fn send_traj_to_server(
        &self,
        encoded_trajectory: EncodedTrajectory,
        training_server_address: &str,
    ) -> Result<(), TransportError> {
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
            return Err(TransportError::SendTrajError(format!(
                "Failed to initialize algorithm before sending trajectory: {}",
                e
            )));
        }

        // Convert from the old proto format to new format
        let proto_trajectory =
            relayrl_encoded_trajectory_to_grpc_encoded_trajectory(encoded_trajectory);

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
                eprintln!(
                    "[TonicClient] Failed to send trajectory: {:?}",
                    e.to_string()
                );
                return Err(TransportError::SendTrajError(format!(
                    "Failed to send trajectory: {:?}",
                    e.to_string()
                )));
            }
        }
        Ok(())
    }

    async fn listen_for_model(&self, model_server_address: &str) -> Result<(), TransportError> {
        let mut client = Self::new(&self.config);
        let mut polling_interval = tokio::time::interval(Duration::from_secs(5));

        // Ensure algorithm is initialized before starting to listen
        if let Err(e) = client
            .initialize_algorithm_if_needed(model_server_address)
            .await
        {
            eprintln!(
                "[TonicClient] Failed to initialize algorithm before listening: {}",
                e.to_string()
            );
            return Err(TransportError::ListenForModelError(format!(
                "Failed to initialize algorithm before listening: {}",
                e.to_string()
            )));
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
                    eprintln!("[TonicClient] Model polling failed: {:?}", e.to_string());
                    // Back off on error
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    async fn send_scaling_warning(
        &self,
        operation: ScalingOperation,
    ) -> Result<(), TransportError> {
        // TODO: implement
        Ok(())
    }

    async fn send_scaling_complete(
        &self,
        operation: ScalingOperation,
    ) -> Result<(), TransportError> {
        // TODO: implement
        Ok(())
    }
}
