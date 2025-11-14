use crate::network::TransportType;
use crate::network::client::runtime::coordination::coordinator::{
    ClientCoordinator, ClientInterface,
};

use burn_tensor::{Tensor, backend::Backend};
use relayrl_types::types::data::action::{CodecConfig, RelayRLAction};
use relayrl_types::types::model::{HotReloadableModel, ModelModule};
use relayrl_types::types::data::tensor::{BackendMatcher, DeviceType, SupportedTensorBackend, AnyBurnTensor};

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::task::JoinHandle;
use uuid::Uuid;

pub struct AgentStartParameters<B: Backend + BackendMatcher<Backend = B>> {
    actor_count: i64,
    default_device: DeviceType,
    default_model: Option<ModelModule<B>>,
    algorithm_name: String,
    config_path: Option<PathBuf>,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for AgentStartParameters<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentStartParameters")
    }
}

/// Builder for RelayRLAgent
pub struct RelayRLAgentBuilder<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> {
    transport_type: TransportType,
    actor_count: Option<i64>,
    default_device: Option<DeviceType>,
    default_model: Option<ModelModule<B>>,
    algorithm_name: Option<String>,
    config_path: Option<PathBuf>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> RelayRLAgentBuilder<B, D_IN, D_OUT> {
    /// Create a new builder with required transport type
    pub fn builder(transport_type: TransportType) -> Self {
        Self {
            transport_type,
            actor_count: None,
            default_device: None,
            default_model: None,
            algorithm_name: None,
            config_path: None,
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

    /// Build and start the RelayRLAgent, returning a running instance
    pub async fn build(self) -> Result<(RelayRLAgent<B, D_IN, D_OUT>, AgentStartParameters<B>), String> {
        // Initialize agent object
        let agent: RelayRLAgent<B, D_IN, D_OUT> = RelayRLAgent::new(self.transport_type);

        // Tuple parameters
        let startup_params = AgentStartParameters::<B> {
            actor_count: self.actor_count.unwrap_or(1),
            default_device: self.default_device.unwrap_or(DeviceType::default()),
            default_model: self.default_model,
            algorithm_name: self.algorithm_name.ok_or("algorithm_name is required")?,
            config_path: self.config_path,
        };

        Ok((agent, startup_params))
    }
}

/// Thin facade over ClientCoordinator
pub struct RelayRLAgent<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> {
    coordinator: ClientCoordinator<B, D_IN, D_OUT>,
    supported_backend: SupportedTensorBackend,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> std::fmt::Debug for RelayRLAgent<B, D_IN, D_OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RelayRLAgent")
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> RelayRLAgent<B, D_IN, D_OUT> {
    /// Create a new agent with given network
    pub fn new(transport_type: TransportType) -> Self {
        Self {
            coordinator: ClientCoordinator::<B, D_IN, D_OUT>::new(transport_type),
            supported_backend: SupportedTensorBackend::default(),
        }
    }

    /// Start the agent with all parameters
    pub async fn start(
        self,
        actor_count: i64,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) {
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
    }

    pub async fn scale_throughput(&mut self, routers: i32) {
        match routers {
            add if routers > 0 => {
                self.coordinator._scale_up(add as u32).await;
            }
            remove if routers < 0 => {
                self.coordinator._scale_down(remove.abs() as u32).await;
            }
            _ => {
                eprintln!("[RelayRLAgent] No change; zero routers requested.");
            }
        }
    }

    /// Shut down the agent
    pub async fn shutdown(&mut self) {
        self.coordinator._shutdown().await;
    }

    /// Request actions from actors
    pub async fn request_action(
        &self,
        ids: Vec<Uuid>,
        observation: AnyBurnTensor<B, D_IN>,
        mask: AnyBurnTensor<B, D_OUT>,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, String> {
        match B::matches_backend(&self.supported_backend) {
            true => {
                self.coordinator
                    ._request_action(ids, observation, mask, reward)
                    .await
            }
            false => Err("Backend mismatch".to_string()),
        }
    }

    /// Flag the last action
    pub async fn flag_last_action(&self, ids: Vec<Uuid>, reward: Option<f32>) {
        self.coordinator._flag_last_action(ids, reward).await;
    }

    /// Get model versions
    pub async fn get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, String> {
        self.coordinator._get_model_version(ids).await
    }

    /// Collect runtime statistics
    pub fn runtime_statistics(&self) -> PathBuf {
        // TODO: implement metrics dump
        PathBuf::new()
    }
}

/// Actor management trait using boxed futures
pub trait RelayRLAgentActors<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> {
    fn new_actor(
        &self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
    fn remove_actor(
        &mut self,
        id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>>;
    fn get_actors(
        &self,
    ) -> Pin<Box<dyn Future<Output = (Vec<Uuid>, Vec<Arc<JoinHandle<()>>>)> + Send + '_>>;
    fn set_actor_id(
        &self,
        current_id: Uuid,
        new_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>>;
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> RelayRLAgentActors<B, D_IN, D_OUT> for RelayRLAgent<B, D_IN, D_OUT> {
    fn new_actor(
        &self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.coordinator._new_actor(device, default_model).await;
        })
    }

    fn remove_actor(
        &mut self,
        id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move {
            let _ = &mut self.coordinator._remove_actor(id).await;
            Ok(())
        })
    }

    fn get_actors(
        &self,
    ) -> Pin<Box<dyn Future<Output = (Vec<Uuid>, Vec<Arc<JoinHandle<()>>>)> + Send + '_>> {
        Box::pin(async move {
            self.coordinator
                ._get_actors()
                .await
                .expect("Failed to get actors")
        })
    }

    fn set_actor_id(
        &self,
        current_id: Uuid,
        new_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move { self.coordinator._set_actor_id(current_id, new_id).await })
    }
}
