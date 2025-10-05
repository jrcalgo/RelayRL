use crate::network::HotReloadableModel;
use crate::network::TransportType;
use crate::network::client::runtime::coordination::coordinator::{
    ClientCoordinator, ClientInterface,
};
use crate::types::action::RelayRLAction;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tch::{CModule, Device, Tensor};
use tokio::task::JoinHandle;
use uuid::Uuid;

pub struct AgentStartParameters {
    actor_count: i64,
    default_device: Device,
    default_model: Option<CModule>,
    algorithm_name: String,
    config_path: Option<PathBuf>,
}

/// Builder for RelayRLAgent
pub struct RelayRLAgentBuilder {
    transport_type: TransportType,
    actor_count: Option<i64>,
    default_device: Option<Device>,
    default_model: Option<CModule>,
    algorithm_name: Option<String>,
    config_path: Option<PathBuf>,
}

impl RelayRLAgentBuilder {
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

    pub fn default_device(mut self, device: Device) -> Self {
        self.default_device = Some(device);
        self
    }

    pub fn default_model(mut self, model: CModule) -> Self {
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
    pub async fn build(self) -> Result<(RelayRLAgent, AgentStartParameters), String> {
        // Initialize agent object
        let agent: RelayRLAgent = RelayRLAgent::new(self.transport_type);

        // Tuple parameters
        let startup_params = AgentStartParameters {
            actor_count: self.actor_count.unwrap_or(1),
            default_device: self.default_device.unwrap_or(Device::Cpu),
            default_model: self.default_model,
            algorithm_name: self.algorithm_name.ok_or("algorithm_name is required")?,
            config_path: self.config_path,
        };

        Ok((agent, startup_params))
    }
}

/// Thin facade over ClientCoordinator
pub struct RelayRLAgent {
    coordinator: ClientCoordinator,
}

impl RelayRLAgent {
    /// Create a new agent with given network
    pub fn new(transport_type: TransportType) -> Self {
        Self {
            coordinator: ClientCoordinator::new(transport_type),
        }
    }

    /// Start the agent with all parameters
    pub async fn start(
        self,
        actor_count: i64,
        default_device: Device,
        default_model: Option<CModule>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
    ) {
        self.coordinator
            ._start(
                actor_count,
                default_device,
                default_model,
                algorithm_name,
                config_path,
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
        observation: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, String> {
        self.coordinator
            ._request_for_action(ids, observation, mask, reward)
            .await
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
pub trait RelayRLAgentActors {
    fn new_actor(
        &self,
        device: Device,
        default_model: Option<HotReloadableModel>,
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

impl RelayRLAgentActors for RelayRLAgent {
    fn new_actor(
        &self,
        device: Device,
        default_model: Option<HotReloadableModel>,
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
            &mut self.coordinator._remove_actor(id).await;
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
