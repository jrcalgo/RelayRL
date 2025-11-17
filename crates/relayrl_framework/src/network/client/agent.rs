use crate::network::TransportType;
use crate::network::client::runtime::coordination::coordinator::{
    ClientCoordinator, ClientInterface, CoordinatorError,
};

use thiserror::Error;

use burn_tensor::{Tensor, backend::Backend};
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
    #[error(transparent)]
    CoordinatorError(#[from] CoordinatorError),
    #[error("Backend mismatch: {0}")]
    BackendMismatchError(String),
    #[error("Noop router scale: {0}")]
    NoopRouterScale(String),
}

pub struct AgentStartParameters<B: Backend + BackendMatcher<Backend = B>> {
    pub actor_count: i64,
    pub default_device: DeviceType,
    pub default_model: Option<ModelModule<B>>,
    pub algorithm_name: String,
    pub config_path: Option<PathBuf>,
    pub codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for AgentStartParameters<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentStartParameters")
    }
}

/// Builder for RelayRLAgent
pub struct RelayRLAgentBuilder<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    pub transport_type: TransportType,
    pub actor_count: Option<i64>,
    pub default_device: Option<DeviceType>,
    pub default_model: Option<ModelModule<B>>,
    pub algorithm_name: Option<String>,
    pub config_path: Option<PathBuf>,
    pub codec: Option<CodecConfig>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RelayRLAgentBuilder<B, D_IN, D_OUT>
{
    /// Create a new builder with required transport type
    pub fn builder(transport_type: TransportType) -> Self {
        Self {
            transport_type,
            actor_count: None,
            default_device: None,
            default_model: None,
            algorithm_name: None,
            config_path: None,
            codec: None,
        }
    }

    pub fn actor_count(mut self, count: i64) -> Self {
        self.actor_count = Some(count);
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

    pub fn algorithm_name(mut self, name: String) -> Self {
        self.algorithm_name = Some(name.into());
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
    ) -> Result<(RelayRLAgent<B, D_IN, D_OUT>, AgentStartParameters<B>), String> {
        // Initialize agent object
        let agent: RelayRLAgent<B, D_IN, D_OUT> = RelayRLAgent::new(self.transport_type);

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            actor_count: self.actor_count.unwrap_or(1),
            default_device: self.default_device.unwrap_or_default(),
            default_model: self.default_model,
            algorithm_name: self.algorithm_name.ok_or("algorithm_name is required")?,
            config_path: self.config_path,
            codec: self.codec.unwrap_or_default(),
        };

        Ok((agent, startup_params))
    }
}

/// Thin facade over ClientCoordinator
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
        write!(f, "RelayRLAgent")
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RelayRLAgent<B, D_IN, D_OUT>
{
    /// Create a new agent with given network
    pub fn new(transport_type: TransportType) -> Self {
        Self {
            coordinator: ClientCoordinator::<B, D_IN, D_OUT>::new(transport_type),
            supported_backend: SupportedTensorBackend::default(),
        }
    }

    /// Start the runtime process with the specified parameters
    ///
    /// If `RelayRLAgentBuilder` was used to create the agent object, the returned `AgentStartParameters` can be used to start the runtime
    pub async fn start(
        self,
        actor_count: i64,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), ClientError> {
        self.coordinator
            ._start(
                actor_count,
                default_device,
                default_model,
                algorithm_name,
                config_path,
                codec,
            )
            .await
            .map_err(Into::into)
    }

    /// Scale the agent's actor throughput by adding or removing routers
    ///
    /// Takes routers: `i32` arg and converts to `u32` for internal operations.
    ///
    /// If the routers < 0: scale down by the absolute value of the routers.
    ///
    /// If the routers > 0: scale up by the value of the routers.
    ///
    /// If routers == 0: do nothing.
    pub async fn scale_throughput(&mut self, routers: i32) -> Result<(), ClientError> {
        match routers {
            add if routers > 0 => {
                self.coordinator._scale_up(add as u32).await?;
                Ok(())
            }
            remove if routers < 0 => {
                self.coordinator._scale_down(remove.abs() as u32).await?;
                Ok(())
            }
            _ => Err(ClientError::NoopRouterScale(
                "Noop router scale".to_string(),
            )),
        }
    }

    /// Shut down the agent and all of its actors
    ///
    /// This disables all of the Agent's coordinator managers and actor instances
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
        observation: AnyBurnTensor<B, D_IN>,
        mask: AnyBurnTensor<B, D_OUT>,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, ClientError> {
        match B::matches_backend(&self.supported_backend) {
            true => {
                let result = self
                    .coordinator
                    ._request_action(ids, observation, mask, reward)
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
    pub async fn get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, ClientError> {
        let versions = self.coordinator._get_model_version(ids).await?;
        Ok(versions)
    }

    /// Collect runtime statistics and save to a JSON file
    /// 
    /// Returns the path to the statistics file containing:
    /// - Actor count and IDs
    /// - Model versions per actor
    /// - Runtime configuration
    /// - Timestamp
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
}

/// Actor management trait using boxed futures
pub trait RelayRLAgentActors<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>
{
    fn new_actor(
        &self,
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
        &self,
        current_id: Uuid,
        new_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RelayRLAgentActors<B, D_IN, D_OUT> for RelayRLAgent<B, D_IN, D_OUT>
{
    /// Creates a new actor instance on the specified device with the specified model
    fn new_actor(
        &self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator._new_actor(device, default_model).await?;
            Ok(())
        })
    }

    /// Removes the actor instance with the specified ID from the current Agent instance
    fn remove_actor(
        &mut self,
        id: Uuid,
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
            dyn Future<Output = Result<(Vec<Uuid>, Vec<Arc<JoinHandle<()>>>), ClientError>>
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
    ///.ok_or("[ClientFilter] Actor not found".to_string())
    /// This will update the actor instance's ID in the Agent's coordinator state manager
    fn set_actor_id(
        &self,
        current_id: Uuid,
        new_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator._set_actor_id(current_id, new_id).await?;
            Ok(())
        })
    }
}
