use crate::get_or_create_client_config_json_path;
use crate::network::{random_uuid, TransportType};
use crate::network::client::runtime::actor::ActorEntity;
use crate::network::client::runtime::coordination::lifecycle_manager::LifeCycleManager;
use crate::network::client::runtime::coordination::metrics_manager::MetricsManager;
use crate::network::client::runtime::coordination::scale_manager::ScaleManager;
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::{
    ClientExternalReceiver, ClientExternalSender, ClientFilter, RoutedMessage, RoutedPayload,
    RoutingProtocol,
};
use tokio::sync::RwLock;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::transport::{client_transport_factory, TransportClient};
use crate::resolve_client_config_json_path;
use crate::sys_utils::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
use crate::sys_utils::observability::logging::builder::LoggingBuilder;
use crate::types::action::RL4SysAction;
use tokio::task::JoinHandle;
use std::path::PathBuf;
use std::sync::Arc;
use rand::Rng;
use tch::{CModule, Device, Tensor};
use tokio::fs;
use tokio::sync::oneshot;
use uuid::Uuid;

pub(crate) const CHANNEL_THROUGHPUT: usize = 100_000;

pub trait ClientInterface {
    fn new(transport_type: TransportType) -> Self;
    async fn _start(
        self,
        actor_count: i64,
        default_device: Device,
        default_model: Option<CModule>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
    );
    async fn _shutdown(&mut self);
    async fn _restart(
        self,
        actor_count: i64,
        default_device: Device,
        default_model: Option<CModule>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
    );
    async fn _new_actor(&self, device: Device, default_model: Option<CModule>);
    async fn _remove_actor(&mut self, id: Uuid);
    async fn _get_actors(&self) -> (Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>) ;
    async fn _set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), String>;
    async fn _request_for_action(
        &self,
        ids: Vec<Uuid>,
        observation: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RL4SysAction>)>, String>;
    async fn _flag_last_action(&self, ids: Vec<Uuid>, reward: Option<f32>);
    async fn _get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, String>;
    async fn _scale_up(&mut self, router_add: i32);
    async fn _scale_down(&mut self, router_remove: i32);
}

pub struct CoordinatorParams {
    logger: LoggingBuilder,
    transport: Arc<TransportClient>,
    lifecycle: LifeCycleManager,
    state: Arc<RwLock<StateManager>>,
    scaling: ScaleManager,
    metrics: MetricsManager,
}

pub struct ClientCoordinator {
    transport_type: TransportType,
    runtime_params: Option<CoordinatorParams>,
}

impl ClientInterface for ClientCoordinator {
    fn new(transport_type: TransportType) -> Self {
        Self {
            transport_type,
            runtime_params: None,
        }
    }

    async fn _start(
        mut self,
        actor_count: i64,
        default_device: Device,
        default_model: Option<CModule>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
    ) {
        let logger = LoggingBuilder::new();

        let config_path: PathBuf = match config_path {
            Some(path) => path,
            None => {
                match DEFAULT_CLIENT_CONFIG_PATH.clone() {
                    Some(path) => path,
                    None => {
                        eprintln!("[Coordinator] No config path provided and default config path not found...");
                        std::process::exit(1);
                    }
                }
            },
        };

        let config_loader: ClientConfigLoader = ClientConfigLoader::load_config(&config_path);

        let shared_transport = Arc::new(client_transport_factory(self.transport_type, &config_loader));

        let lifecycle = LifeCycleManager::new(config_loader.to_owned());

        let shared_state_config: Arc<ClientConfigLoader> = lifecycle.get_active_config();
        let (mut state, global_bus_rx, rx_from_actor) =
            StateManager::new(shared_state_config, default_model);

        let shared_scaling_config: Arc<ClientConfigLoader> = lifecycle.get_active_config();

        let (training_server_address, agent_listener_address) = match self.transport_type {
            TransportType::ZMQ => {
                (config_loader.transport_config.training_server_address.host.clone() + ":" + &config_loader.transport_config.training_server_address.port.to_string(), config_loader.transport_config.agent_listener_address.host.clone() + ":" + &config_loader.transport_config.agent_listener_address.port.to_string())
            }
            TransportType::GRPC => {
                (config_loader.transport_config.training_server_address.host.clone() + ":" + &config_loader.transport_config.training_server_address.port.to_string(), config_loader.transport_config.agent_listener_address.host.clone() + ":" + &config_loader.transport_config.agent_listener_address.port.to_string())
            }
        };
        
        let shared_state = Arc::from(state);
        let scaling = ScaleManager::new(shared_state.clone(), Arc::clone(&shared_scaling_config), shared_transport.clone(), rx_from_actor, agent_listener_address, training_server_address);
        scaling.__scale_up(1, global_bus_rx, shared_state.clone()).await;

        if actor_count > 0 {
            for _ in 0..actor_count {
                let actor_config = config_loader.clone();

                let pid: u32 = std::process::id();
                let pid_bytes = pid.to_be_bytes();

                let mut pid_buf = [0u8; 16];
                pid_buf[..4].copy_from_slice(&pid_bytes);

                let id: Uuid = Uuid::new_v8(pid_buf);

                let actor_model_path = Some(PathBuf::from(
                    actor_config
                        .transport_config.local_model_path
                        .clone()
                        .to_str()
                        .unwrap()
                        .to_string()
                        + &*String::from(id),
                ));
                shared_state.__new_actor(id, default_device, actor_model_path, shared_transport.clone()).await;
            }
        }

        let metrics = MetricsManager::new();

        self.runtime_params = Some(CoordinatorParams {
            logger,
            transport: shared_transport,
            lifecycle,
            state: Arc::new(RwLock::new(shared_state)),
            scaling,
            metrics,
        });
    }

    async fn _shutdown(&mut self) {
        let runtime_params = self.runtime_params.take().unwrap();
        runtime_params.lifecycle._shutdown();
    }

    async fn _restart(
        mut self,
        actor_count: i64,
        default_device: Device,
        default_model: Option<CModule>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
    ) {
        self._shutdown().await;
        self._start(
            actor_count,
            default_device,
            default_model,
            algorithm_name,
            config_path,
        )
        .await;
    }

    async fn _new_actor(&self, device: Device, default_model: Option<CModule>) {
        match &self.runtime_params {
            Some(params) => {
                let actor_config: Arc<_> = params.lifecycle.get_active_config();

                let pid: u32 = std::process::id();
                let pid_bytes = pid.to_be_bytes();

                let mut pid_buf = [0u8; 16];
                pid_buf[..4].copy_from_slice(&pid_bytes);

                let id: Uuid = Uuid::new_v8(pid_buf);

                let actor_model_path = Some(PathBuf::from(
                    actor_config
                        .client_model_path
                        .clone()
                        .to_str()
                        .unwrap()
                        .to_string()
                        + &*String::from(id),
                ));
                params.state.read().await.__new_actor(id, default_device, actor_model_path, params.transport.clone()).await;
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }

    async fn _remove_actor(&mut self, id: Uuid) {
        match &self.runtime_params {
            Some(params) => {
                if let Ok(mut state) = params.state.write().await {
                    state.__remove_actor(id);
                }
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }

    async fn _get_actors(&self) -> (Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>) {
        match &self.runtime_params {
            Some(params) => {
                if let Ok(state) = params.state.read().await {
                    state.__get_actors()
                }
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }

    async fn _set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), String> {
        match &self.runtime_params {
            Some(params) => {
                params.state.__set_actor_id(current_id, new_id)
            }
            None => {
                return Err("[Coordinator] No runtime instance to _shutdown...".to_string());
            }
        }
    }

    async fn _request_for_action(
        &self,
        ids: Vec<Uuid>,
        observation: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RL4SysAction>)>, String> {
        match &self.runtime_params {
            Some(_) => {
                let mut actions = Vec::with_capacity(ids.len());
                for id in ids {
                    let (resp_tx, resp_rx) = oneshot::channel::<Arc<RL4SysAction>>();

                    let mut obs_tensor = Tensor::zeros_like(&observation);
                    obs_tensor.copy_(&observation);
                    let mut mask_tensor: Tensor = Tensor::zeros_like(&mask);
                    mask_tensor.copy_(&mask);

                    let action_request_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::RequestInference,
                        payload: RoutedPayload::RequestInference {
                            observation: obs_tensor,
                            mask: mask_tensor,
                            reward,
                            reply_to: resp_tx,
                        },
                    };

                    let runtime_params = self.runtime_params.as_ref().unwrap();
                    if let sender = runtime_params.scaling.tx_to_router.clone() {
                        let _ = sender.send(action_request_message).await;
                    }

                    match resp_rx.await.map_err(|e| e.to_string()) {
                        Ok(action) => actions.push((id, action)),
                        Err(e) => return Err(e),
                    }
                }
                Ok(actions)
            }
            None => Err("[Coordinator] No runtime instance to _shutdown..."
                .parse()
                .unwrap()),
        }
    }

    async fn _flag_last_action(&self, ids: Vec<Uuid>, reward: Option<f32>) {
        match &self.runtime_params {
            Some(params) => {
                for id in ids {
                    let reward: f32 = reward.unwrap_or(0.0);
                    let flag_last_action_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::FlagLastInference,
                        payload: RoutedPayload::FlagLastInference { reward },
                    };

                    if let sender = params.scaling.tx_to_router.clone() {
                        let _ = sender.send(flag_last_action_message).await;
                    }
                }
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }

    async fn _get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, String> {
        match &self.runtime_params {
            Some(_) => {
                let mut versions = Vec::with_capacity(ids.len());
                for id in ids {
                    let (resp_tx, resp_rx) = oneshot::channel::<i64>();

                    let model_version_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::ModelVersion,
                        payload: RoutedPayload::ModelVersion { reply_to: resp_tx },
                    };

                    let runtime_params = self.runtime_params.as_ref().unwrap();
                    if let sender = runtime_params.state.read().await.tx_to_router.clone() {
                        let _ = sender.send(model_version_message).await;
                    };

                    match resp_rx.await.map_err(|e| e.to_string()) {
                        Ok(model_version) => versions.push((id, model_version)),
                        Err(e) => return Err(e),
                    }
                }
                Ok(versions)
            }
            None => Err("[Coordinator] No runtime instance to _shutdown...".to_string()),
        }
    }

    async fn _scale_up(&mut self, router_add: i32) {
        match &mut self.runtime_params {
            Some(params) => {
                let global_bus_rx = params.state.read().await.global_bus_rx.clone();
                let shared_state = params.state.read().await.clone();
                params.scaling.__scale_up(router_add, global_bus_rx, shared_state).await;
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }

    async fn _scale_down(&mut self, router_remove: i32) {
        match &mut self.runtime_params {
            Some(params) => {
                params.scaling.__scale_down(router_remove).await;
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }
}
