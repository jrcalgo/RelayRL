use crate::network::client::runtime::coordination::coordinator::{
    ClientCoordinator, ClientInterface, CoordinatorError,
};
use crate::network::client::runtime::coordination::scale_manager::AlgorithmArgs;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::{HyperparameterArgs, TransportType};
use crate::prelude::config::ClientConfigLoader;
use crate::utilities::configuration::Algorithm;

use thiserror::Error;

use burn_tensor::{Tensor, backend::Backend};
use relayrl_types::Hyperparams;
use relayrl_types::types::data::action::{CodecConfig, RelayRLAction};
use relayrl_types::types::data::tensor::{
    AnyBurnTensor, BackendMatcher, DeviceType, SupportedTensorBackend,
};
use relayrl_types::types::model::{HotReloadableModel, ModelModule};

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("Inference server mode disabled: {0}")]
    InferenceServerModeDisabled(String),
    #[error("Inference server mode enabled: {0}")]
    InferenceServerModeEnabled(String),
    #[error(transparent)]
    CoordinatorError(#[from] CoordinatorError),
    #[error("Backend mismatch: {0}")]
    BackendMismatchError(String),
    #[error("Noop router scale: {0}")]
    NoopRouterScale(String),
    #[error("Noop actor count: {0}")]
    NoopActorCount(String),
    #[error("Invalid inference mode: {0}")]
    InvalidInferenceMode(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActorInferenceMode {
    Local,
    Server,
}

impl Default for ActorInferenceMode {
    fn default() -> Self {
        Self::Local
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DatabaseTypeParams {
    Sqlite(SqliteParams),
    PostgreSQL(PostgreSQLParams),
}

// TODO: Add actual Sqlite params
#[derive(Debug, Clone, PartialEq)]
pub struct SqliteParams {
    pub path: PathBuf,
}

// TODO: Add actual PostgreSQL params
#[derive(Debug, Clone, PartialEq)]
pub struct PostgreSQLParams {
    pub connection: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryRecordMode {
    Local,
    Database(DatabaseTypeParams),
    Hybrid(DatabaseTypeParams),
    Disabled,
}

impl Default for TrajectoryRecordMode {
    fn default() -> Self {
        Self::Disabled
    }
}

/// TODO: Add architecture support for independent/shared model inference/training requests to server(s).
#[derive(Debug, Clone, PartialEq)]
pub enum ActorServerModelMode {
    Independent,
    Shared,
    Disabled,
}

impl Default for ActorServerModelMode {
    fn default() -> Self {
        Self::Independent
    }
}

#[derive(Debug, Clone)]
pub struct ClientCapabilities {
    pub local_inference: bool,
    pub server_inference: bool,
    pub inference_server_mode: ActorServerModelMode,
    pub local_trajectory_recording: bool,
    pub database_trajectory_recording: bool,
    pub training_server_mode: ActorServerModelMode,
    pub db_params: Option<DatabaseTypeParams>,
}

impl ClientCapabilities {
    pub fn trajectory_recording_enabled(&self) -> bool {
        self.local_trajectory_recording || self.database_trajectory_recording
    }
}

pub struct ClientModes {
    pub actor_inference_mode: ActorInferenceMode,
    pub inference_server_mode: ActorServerModelMode,
    pub training_server_mode: ActorServerModelMode,
    pub trajectory_recording_mode: TrajectoryRecordMode,
}

impl Default for ClientModes {
    fn default() -> Self {
        Self {
            actor_inference_mode: ActorInferenceMode::Local,
            trajectory_recording_mode: TrajectoryRecordMode::Disabled,
            inference_server_mode: ActorServerModelMode::Disabled,
            training_server_mode: ActorServerModelMode::Independent,
        }
    }
}

impl ClientModes {
    pub fn validate_modes(&self) -> Result<(), ClientError> {
        if self.actor_inference_mode == ActorInferenceMode::Server
            && self.inference_server_mode == ActorServerModelMode::Disabled
        {
            return Err(ClientError::InvalidInferenceMode(
                "Inference server mode disabled for server-side inference: {:?}".to_string(),
            ));
        }
        if self.actor_inference_mode == ActorInferenceMode::Local
            && self.inference_server_mode != ActorServerModelMode::Disabled
        {
            return Err(ClientError::InvalidInferenceMode(
                "Inference server mode enabled for client-side inference: {:?}".to_string(),
            ));
        }
        Ok(())
    }

    pub fn capabilities(&self) -> Result<ClientCapabilities, ClientError> {
        self.validate_modes()?;

        let (local_inference, server_inference) = match self.actor_inference_mode {
            ActorInferenceMode::Local => (true, false),
            ActorInferenceMode::Server => (false, true),
        };

        let (local_trajectory_recording, database_trajectory_recording) =
            match &self.trajectory_recording_mode {
                TrajectoryRecordMode::Local => (true, false),
                TrajectoryRecordMode::Database(_) => (false, true),
                TrajectoryRecordMode::Hybrid(_) => (true, true),
                TrajectoryRecordMode::Disabled => (false, false),
            };

        let db_params: Option<DatabaseTypeParams> = match &self.trajectory_recording_mode {
            TrajectoryRecordMode::Database(params) => Some(params.clone()),
            TrajectoryRecordMode::Hybrid(params) => Some(params.clone()),
            _ => None,
        };

        Ok(ClientCapabilities {
            local_inference,
            server_inference,
            inference_server_mode: self.inference_server_mode.clone(),
            local_trajectory_recording,
            database_trajectory_recording,
            training_server_mode: self.training_server_mode.clone(),
            db_params,
        })
    }
}

pub struct AgentStartParameters<B: Backend + BackendMatcher<Backend = B>> {
    pub algorithm_args: AlgorithmArgs,
    pub actor_count: u32,
    pub router_scale: u32,
    pub default_device: DeviceType,
    pub default_model: Option<ModelModule<B>>,
    pub config_path: Option<PathBuf>,
    pub codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for AgentStartParameters<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentStartParameters")
    }
}

/// `AgentBuilder` is a builder for the `RelayRLAgent` instance and its associated `fn start()` parameters, returned as `AgentStartParameters`.
pub struct AgentBuilder<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    pub client_modes: Option<ClientModes>,
    pub transport_type: Option<TransportType>,
    pub algorithm: Option<Algorithm>,
    pub hyperparams: Option<HyperparameterArgs>,
    pub actor_count: Option<u32>,
    pub router_scale: Option<u32>,
    pub default_device: Option<DeviceType>,
    pub default_model: Option<ModelModule<B>>,
    pub config_path: Option<PathBuf>,
    pub codec: Option<CodecConfig>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    AgentBuilder<B, D_IN, D_OUT>
{
    /// Create a new builder with required transport type
    pub fn builder() -> Self {
        Self {
            client_modes: Some(ClientModes::default()),
            transport_type: Some(TransportType::default()),
            algorithm: None,
            hyperparams: None,
            actor_count: None,
            router_scale: None,
            default_device: None,
            default_model: None,
            config_path: None,
            codec: None,
        }
    }

    pub fn actor_inference_mode(mut self, actor_inference_mode: ActorInferenceMode) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.actor_inference_mode = actor_inference_mode;
        }
        self
    }

    pub fn trajectory_recording_mode(
        mut self,
        trajectory_recording_mode: TrajectoryRecordMode,
    ) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.trajectory_recording_mode = trajectory_recording_mode;
        }
        self
    }

    pub fn inference_server_mode(mut self, inference_server_mode: ActorServerModelMode) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.inference_server_mode = inference_server_mode;
        }
        self
    }

    pub fn training_server_mode(mut self, training_server_mode: ActorServerModelMode) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.training_server_mode = training_server_mode;
        }
        self
    }

    pub fn transport_type(mut self, transport_type: TransportType) -> Self {
        self.transport_type = Some(transport_type);
        self
    }

    pub fn actor_count(mut self, count: u32) -> Self {
        self.actor_count = Some(count);
        self
    }

    pub fn router_scale(mut self, count: u32) -> Self {
        self.router_scale = Some(count);
        self
    }

    pub fn default_device(mut self, device: DeviceType) -> Self {
        self.default_device = Some(device);
        self
    }

    pub fn default_model(mut self, model: ModelModule<B>) -> Self {
        self.default_model = Some(model);
        self
    }

    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = Some(algorithm);
        self
    }

    pub fn hyperparams(mut self, hyperparams: HyperparameterArgs) -> Self {
        self.hyperparams = Some(hyperparams);
        self
    }

    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn codec(mut self, codec: CodecConfig) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Build the RelayRLAgent, returning the agent object and its associated startup parameters
    pub async fn build(
        self,
    ) -> Result<(RelayRLAgent<B, D_IN, D_OUT>, AgentStartParameters<B>), ClientError> {
        // Initialize agent object
        let agent: RelayRLAgent<B, D_IN, D_OUT> = RelayRLAgent::new(
            self.transport_type.unwrap_or(TransportType::ZMQ),
            self.client_modes.unwrap_or(ClientModes::default()),
        )?;

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            algorithm_args: AlgorithmArgs {
                algorithm: self.algorithm.unwrap_or(Algorithm::ConfigInit),
                hyperparams: self.hyperparams,
            },
            actor_count: self.actor_count.unwrap_or(1),
            router_scale: self.router_scale.unwrap_or(1),
            default_device: self.default_device.unwrap_or_default(),
            default_model: self.default_model,
            config_path: self.config_path,
            codec: self.codec.unwrap_or_default(),
        };

        Ok((agent, startup_params))
    }
}

/// `RelayRLAgent` is the client entry point for the RelayRL Framework.
///
/// Functions as a thin facade over the `ClientCoordinator`, providing a clean public API for the client runtime.
///
/// - `coordinator`: The `ClientCoordinator`` instance for managing the client runtime.
/// - `supported_backend`: The supported tensor backend for the client runtime.
pub struct RelayRLAgent<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    coordinator: ClientCoordinator<B, D_IN, D_OUT>,
    supported_backend: SupportedTensorBackend,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    std::fmt::Debug for RelayRLAgent<B, D_IN, D_OUT>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RLAgent")
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RelayRLAgent<B, D_IN, D_OUT>
{
    /// Creates a new RelayRLAgent instance using runtime invariant parameters.
    ///
    /// - `transport_type`:
    ///   - `TransportType::GRPC`: Use gRPC (HTTP/2-based) transport. Best for:
    ///     - Cross-language interoperability
    ///     - HTTP-friendly deployments (load balancers, proxies, etc.)
    ///
    ///   - `TransportType::ZMQ`: Use ZeroMQ transport. Best for:
    ///     - High-frequency, low-latency messaging
    ///     - Minimal protocol overhead
    ///
    /// - `client_modes`:
    ///
    ///     - `actor_inference_mode`:
    ///         - If `ActorInferenceMode::ClientSide`: Model inference occurs locally.
    ///         - If `ActorInferenceMode::ServerSide`: Model inference occurs on RelayRLInferenceServer.
    ///         - If `ActorInferenceMode::Hybrid`: Model inference primarily occurs locally and falls back to RelayRLInferenceServer based on load or local failure.
    ///
    ///     - `trajectory_write_mode`:
    ///         - If `TrajectoryWriteMode::ClientSide`: Trajectories are written to the local file system.
    ///         - If `TrajectoryWriteMode::ServerSide`: Trajectories are written to the RelayRLTrajectoryServer.
    ///         - If `TrajectoryWriteMode::Disabled`: Trajectory writing is disabled.
    ///
    ///     - `inference_server_mode`:
    ///         - If `ActorServerMode::Independent`: Each runtime actor will have its own model inference instance on the RelayRLInferenceServer.
    ///         - If `ActorServerMode::Shared`: All runtime actors will share the same model inference instance on the RelayRLInferenceServer.
    ///         - If `ActorServerMode::Disabled`: Model inference via the RelayRLInferenceServer is disabled.
    ///
    ///     - `training_server_mode`:
    ///         - If `ActorServerMode::Independent`: Each runtime actor will have its own algorithm training instance on the RelayRLTrainingServer.
    ///         - If `ActorServerMode::Shared`: All runtime actors will share the same algorithm training instance on the RelayRLTrainingServer.
    ///         - If `ActorServerMode::Disabled`: Algorithm training via the RelayRLTrainingServer is disabled.
    ///
    /// Panics if the inference mode and the inference server mode are misconfigured / incompatible with one another.
    ///
    /// To avoid a panic, ensure your parameters are not as follows:
    ///   - `_inference_mode == ActorInferenceMode::ServerSide` and `_inference_server_mode == ActorServerMode::Disabled`: the inference server mode is disabled for server-side inference.
    ///   - `_inference_mode == ActorInferenceMode::Hybrid` and `_inference_server_mode == ActorServerMode::Disabled`: the inference server mode is disabled for hybrid inference.
    ///   - `_inference_mode == ActorInferenceMode::ClientSide` and `_inference_server_mode != ActorServerMode::Disabled`: the inference server mode is enabled for client-side inference.
    pub fn new(
        transport_type: TransportType,
        client_modes: ClientModes,
    ) -> Result<Self, ClientError> {
        let capabilities = client_modes.capabilities()?;
        Ok(Self {
            coordinator: ClientCoordinator::<B, D_IN, D_OUT>::new(transport_type, capabilities)?,
            supported_backend: SupportedTensorBackend::default(),
        })
    }

    /// Start the client runtime process with the specified parameters
    ///
    /// - `actor_count`: Number of runtime actors to spawn on startup.
    ///
    /// - `hyperparams`: Hyperparameters for the specified algorithm.
    ///
    /// - `router_scale`: Number of routers for sending messages from this Agent API / from the transport receiver to the corresponding runtime actors
    ///
    /// - `default_device`: Default device for tensor ops for each runtime actor.
    ///
    /// - `default_model`: Default model for each runtime actor.
    ///   - If `Some(ModelModule<B>)`: each runtime actor will start out using this model. If training is enabled, the model will be hot reloaded based on the model version from the server.
    ///   - If `None`:
    ///     - If training is enabled, the runtime actor will do a model handshake with the training server to get the default model.
    ///     - If training is disabled, the client will fail to start.
    ///
    /// - `algorithm_name`: (For now) A single algorithm type (e.g. "REINFORCE", "PPO", "TD3", etc.) for all runtime actors.
    ///
    /// - `config_path`: Path to the client configuration JSON file.
    ///   - If `Some(PathBuf)`: the client will attempt to load the configuration from the specified path.
    ///   - If `None`: the client will attempt to load from the default path (./client_config.json)
    ///   - If there is **no config at the default path**, the client will create a new one with default values.
    ///
    /// - `codec`:
    ///
    /// If `AgentBuilder` was used to create the agent object, the returned `AgentStartParameters` can be used to start the runtime
    pub async fn start(
        mut self,
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        router_scale: u32,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), ClientError> {
        self.coordinator
            ._start(
                algorithm_args,
                actor_count,
                router_scale,
                default_device,
                default_model,
                config_path,
                codec,
            )
            .await
            .map_err(Into::into)
    }

    /// Scale the agent's actor throughput by load balancing actors across message-passing routers
    ///
    /// Takes `router_scale`: `i32` arg and converts to `u32` for internal operations.
    ///
    /// If the `router_scale < 0`: scale down by the absolute value of the routers.
    ///
    /// If the `router_scale > 0`: scale up by the value of the routers.
    ///
    /// If `routers == 0`: do nothing and return error.
    ///
    pub async fn scale_throughput(&mut self, router_scale: i32) -> Result<(), ClientError> {
        match router_scale {
            add if router_scale > 0 => {
                self.coordinator._scale_out(add as u32).await?;
                Ok(())
            }
            remove if router_scale < 0 => {
                self.coordinator._scale_in(remove.unsigned_abs()).await?;
                Ok(())
            }
            _ => Err(ClientError::NoopRouterScale(
                "Noop router scale: `router_scale` set to zero".to_string(),
            )),
        }
    }

    /// Shut down the Agent's client runtime components
    ///
    /// This requests all client runtime componenets (managers, actors, routers, transport, etc.) to conclude their operations and gracefully shut down.
    pub async fn shutdown(&mut self) -> Result<(), ClientError> {
        self.coordinator._shutdown().await?;
        Ok(())
    }

    /// Request actions from the specified actor IDs (if they exist)
    ///
    /// This will send the action request to the specified actor instances and return the action responses
    pub async fn request_action(
        &self,
        ids: Vec<Uuid>,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, ClientError> {
        match B::matches_backend(&self.supported_backend) {
            true => {
                let result = self
                    .coordinator
                    ._request_action(ids, observation.clone(), mask.clone(), reward)
                    .await?;
                Ok(result)
            }
            false => Err(ClientError::BackendMismatchError(
                "Backend mismatch".to_string(),
            )),
        }
    }

    /// Flags the last action for the specified actor IDs (if they exist)
    ///
    /// Appends a RelayRLAction with the done flag set to `true` and the specified reward (if any) to the actor's current trajectory
    pub async fn flag_last_action(
        &self,
        ids: Vec<Uuid>,
        reward: Option<f32>,
    ) -> Result<(), ClientError> {
        self.coordinator._flag_last_action(ids, reward).await?;
        Ok(())
    }

    /// Retrieves the model version for each actor ID listed (if instance IDs exist)
    ///
    /// Returns a vector of tuples where each tuple is (Actor ID, Current Model Version).
    pub async fn get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, ClientError> {
        Ok(self.coordinator._get_model_version(ids).await?)
    }

    /// Collect runtime statistics and save to a JSON file
    ///
    /// Returns the path to the statistics file containing:
    /// - Actor count and IDs
    /// - Model versions per actor
    /// - Runtime configuration
    /// - Timestamp
    ///
    /// TODO: I want to rewrite this to pipe stats from actor, managers, coordinator, routers, and transport layers into a single JSON object/map.
    pub fn runtime_statistics(&self) -> Result<PathBuf, ClientError> {
        use std::fs::File;
        use std::io::Write;
        use std::time::SystemTime;

        // Create statistics directory if it doesn't exist
        let stats_dir = std::path::Path::new("./runtime_stats");
        if !stats_dir.exists() {
            std::fs::create_dir_all(stats_dir).map_err(|e| {
                ClientError::CoordinatorError(
                    crate::network::client::runtime::coordination::coordinator::CoordinatorError::ConfigError(
                        crate::network::client::runtime::coordination::coordinator::ClientConfigError::InvalidValue(
                            format!("Failed to create stats directory: {}", e)
                        )
                    )
                )
            })?;
        }

        // Generate unique filename with timestamp
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| {
                ClientError::CoordinatorError(
                    crate::network::client::runtime::coordination::coordinator::CoordinatorError::ConfigError(
                        crate::network::client::runtime::coordination::coordinator::ClientConfigError::InvalidValue(
                            format!("System time error: {}", e)
                        )
                    )
                )
            })?
            .as_secs();

        let stats_path = stats_dir.join(format!("runtime_stats_{}.json", timestamp));

        let backend_name = match B::matches_backend(&self.supported_backend) {
            true => "ndarray",
            false => "tch",
        };

        // Collect basic statistics (in a real implementation, this would be more comprehensive)
        let stats_json = serde_json::json!({
            "timestamp": timestamp,
            "backend": backend_name,
            "input_dimensions": D_IN,
            "output_dimensions": D_OUT,
        });

        // Write to file
        let mut file = File::create(&stats_path).map_err(|e| {
            ClientError::CoordinatorError(
                crate::network::client::runtime::coordination::coordinator::CoordinatorError::ConfigError(
                    crate::network::client::runtime::coordination::coordinator::ClientConfigError::InvalidValue(
                        format!("Failed to create stats file: {}", e)
                    )
                )
            )
        })?;

        file.write_all(stats_json.to_string().as_bytes()).map_err(|e| {
            ClientError::CoordinatorError(
                crate::network::client::runtime::coordination::coordinator::CoordinatorError::ConfigError(
                    crate::network::client::runtime::coordination::coordinator::ClientConfigError::InvalidValue(
                        format!("Failed to write stats: {}", e)
                    )
                )
            )
        })?;

        Ok(stats_path)
    }

    pub async fn get_config(&self) -> Result<ClientConfigLoader, ClientError> {
        Ok(self.coordinator._get_config().await?)
    }

    pub async fn set_config(&self, config: ClientConfigLoader) -> Result<(), ClientError> {
        self.coordinator._set_config(config).await?;
        Ok(())
    }
}

/// Actor management trait using boxed futures
pub trait RelayRLAgentActors<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>
{
    fn new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn new_actors(
        &mut self,
        count: u32,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn remove_actor(
        &mut self,
        id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn get_actors(
        &self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<(Vec<Uuid>, Vec<Arc<JoinHandle<()>>>), ClientError>>
                + Send
                + '_,
        >,
    >;
    fn set_actor_id(
        &mut self,
        current_id: Uuid,
        new_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RelayRLAgentActors<B, D_IN, D_OUT> for RelayRLAgent<B, D_IN, D_OUT>
{
    /// Creates a new actor instance on the specified device with the specified model
    fn new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator
                ._new_actor(device, default_model, true)
                .await?;
            Ok(())
        })
    }

    /// Creates `n` new actor instances on the specified device with the specified model
    fn new_actors(
        &mut self,
        count: u32,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        if count == 0 {
            return Box::pin(async move {
                Err(ClientError::NoopActorCount(
                    "Noop actor count: `count` set to zero".to_string(),
                ))
            });
        }
        Box::pin(async move {
            for _ in 0..count {
                self.coordinator
                    ._new_actor(device.clone(), default_model.clone(), false)
                    .await?;
            }
            self.coordinator._send_client_ids_to_server().await?;
            Ok(())
        })
    }

    /// Removes the actor instance with the specified ID from the current Agent instance
    fn remove_actor(
        &mut self,
        id: ActorUuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator._remove_actor(id).await?;
            Ok(())
        })
    }

    /// Retrieves the current actor instances and their associated join handles from the Agent instance
    fn get_actors(
        &self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), ClientError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let actors = self.coordinator._get_actors().await?;
            Ok(actors)
        })
    }

    /// Sets the ID of the actor instance with the specified current ID to the new ID
    /// .ok_or("[ClientFilter] Actor not found".to_string())
    /// This will update the actor instance's ID in the Agent's coordinator state manager
    fn set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator._set_actor_id(current_id, new_id).await?;
            Ok(())
        })
    }
}
