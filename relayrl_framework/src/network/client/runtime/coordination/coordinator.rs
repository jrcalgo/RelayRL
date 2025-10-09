use crate::network::HotReloadableModel;
use crate::network::TransportType;
use crate::network::client::runtime::coordination::lifecycle_manager::LifeCycleManager;
use crate::network::client::runtime::coordination::scale_manager::ScaleManager;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::transport::{TransportClient, client_transport_factory};
use crate::types::action::RelayRLAction;
use crate::utilities::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
use crate::utilities::observability;
use crate::utilities::observability::logging::builder::LoggingBuilder;
use crate::utilities::observability::metrics::MetricsManager;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tch::{CModule, Device, Tensor};
use tokio::sync::RwLock;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use uuid::Uuid;

pub(crate) const CHANNEL_THROUGHPUT: usize = 256_000;

pub trait ClientInterface {
    fn new(transport_type: TransportType) -> Self;
    async fn _start(
        self,
        actor_count: i64,
        default_device: Device,
        default_model: Option<CModule>,
        _algorithm_name: String,
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
    async fn _new_actor(&self, device: Device, default_model: Option<HotReloadableModel>);
    async fn _remove_actor(&mut self, id: Uuid);
    async fn _get_actors(&self) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), String>;
    async fn _set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), String>;
    async fn _request_for_action(
        &self,
        ids: Vec<Uuid>,
        observation: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, String>;
    async fn _flag_last_action(&self, ids: Vec<Uuid>, reward: Option<f32>);
    async fn _get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, String>;
    async fn _scale_up(&mut self, router_add: u32);
    async fn _scale_down(&mut self, router_remove: u32);
}

pub struct CoordinatorParams {
    logger: LoggingBuilder,
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
        _algorithm_name: String,
        config_path: Option<PathBuf>,
    ) {
        let logger = LoggingBuilder::new();

        let config_path: PathBuf = match config_path {
            Some(path) => path,
            None => match DEFAULT_CLIENT_CONFIG_PATH.clone() {
                Some(path) => path,
                None => {
                    eprintln!(
                        "[Coordinator] No config path provided and default config path not found..."
                    );
                    std::process::exit(1);
                }
            },
        };

        let config_loader: ClientConfigLoader = ClientConfigLoader::load_config(&config_path);

        let transport: TransportClient =
            client_transport_factory(self.transport_type, &config_loader);

        let lifecycle: LifeCycleManager = LifeCycleManager::new(config_loader.to_owned());
        lifecycle.spawn_loop();

        let shared_state_config: Arc<RwLock<ClientConfigLoader>> = lifecycle.get_active_config();

        let reloadable_default_model = match default_model {
            Some(model) => Some(
                HotReloadableModel::new_from_model(model, default_device)
                    .await
                    .expect("[Coordinator] Failed to create reloadable default model..."),
            ),
            None => None,
        };

        let (state, global_bus_rx, rx_from_actor) =
            StateManager::new(shared_state_config, reloadable_default_model.clone());

        let shared_scaling_config: Arc<RwLock<ClientConfigLoader>> = lifecycle.get_active_config();

        let (training_server_address, agent_listener_address) = match self.transport_type {
            TransportType::ZMQ => (
                config_loader
                    .transport_config
                    .training_server_address
                    .host
                    .clone()
                    + ":"
                    + &config_loader
                        .transport_config
                        .training_server_address
                        .port
                        .to_string(),
                config_loader
                    .transport_config
                    .agent_listener_address
                    .host
                    .clone()
                    + ":"
                    + &config_loader
                        .transport_config
                        .agent_listener_address
                        .port
                        .to_string(),
            ),
            TransportType::GRPC => (
                config_loader
                    .transport_config
                    .training_server_address
                    .host
                    .clone()
                    + ":"
                    + &config_loader
                        .transport_config
                        .training_server_address
                        .port
                        .to_string(),
                config_loader
                    .transport_config
                    .agent_listener_address
                    .host
                    .clone()
                    + ":"
                    + &config_loader
                        .transport_config
                        .agent_listener_address
                        .port
                        .to_string(),
            ),
        };

        let shared_global_bus_rx = Arc::new(RwLock::new(global_bus_rx));
        let shared_state = Arc::from(RwLock::new(state));
        // Subscribe to lifecycle shutdown and propagate to all actors
        StateManager::spawn_shutdown_watcher(shared_state.clone(), lifecycle.clone());
        let mut scaling = ScaleManager::new(
            shared_state.clone(),
            Arc::clone(&shared_scaling_config),
            shared_global_bus_rx,
            transport,
            rx_from_actor,
            agent_listener_address,
            training_server_address,
        )
        .with_lifecycle(lifecycle.clone());
        scaling.__scale_up(1).await;

        if actor_count > 0 {
            for _ in 0..actor_count {
                Self::_new_actor(&self, default_device, reloadable_default_model.clone()).await;
            }
        }

        let metrics = observability::init_observability();

        self.runtime_params = Some(CoordinatorParams {
            logger,
            lifecycle,
            state: shared_state,
            scaling,
            metrics,
        });
    }

    async fn _shutdown(&mut self) {
        if let Some(mut runtime_params) = self.runtime_params.take() {
            runtime_params.lifecycle._shutdown();

            runtime_params
                .state
                .read()
                .await
                .__shutdown_all_actors()
                .await;

            let router_count: u32 = runtime_params
                .scaling
                .runtime_params
                .as_ref()
                .map(|m| m.len() as u32)
                .unwrap_or(0);
            if router_count > 0 {
                runtime_params.scaling.__scale_down(router_count).await;
            }
        }
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

    async fn _new_actor(&self, device: Device, default_model: Option<HotReloadableModel>) {
        match &self.runtime_params {
            Some(params) => {
                let pid: u32 = std::process::id();
                let pid_bytes: [u8; 4] = pid.to_be_bytes();

                let mut pid_buf: [u8; 16] = [0u8; 16];
                pid_buf[..4].copy_from_slice(&pid_bytes);

                let id: Uuid = Uuid::new_v8(pid_buf);

                params.metrics.record_counter("actors_created", 1, &[]);

                params
                    .state
                    .write()
                    .await
                    .__new_actor(
                        id,
                        device,
                        default_model,
                        params.scaling.shared_transport.clone(),
                    )
                    .await;
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _new_actor...");
            }
        }
    }

    async fn _remove_actor(&mut self, id: Uuid) {
        match &self.runtime_params {
            Some(params) => {
                params.metrics.record_counter("actors_removed", 1, &[]);
                params.state.write().await.__remove_actor(id);
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _remove_actor...");
            }
        }
    }

    async fn _get_actors(&self) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), String> {
        match &self.runtime_params {
            Some(params) => Ok(params.state.read().await.__get_actors()?),
            None => {
                Err("[Coordinator] No runtime parameter instance to _get_actors...".to_string())
            }
        }
    }

    async fn _set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), String> {
        match &self.runtime_params {
            Some(params) => params
                .state
                .write()
                .await
                .__set_actor_id(current_id, new_id),
            None => Err("[Coordinator] No runtime instance to _shutdown...".to_string()),
        }
    }

    async fn _request_for_action(
        &self,
        ids: Vec<Uuid>,
        observation: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, String> {
        match &self.runtime_params {
            Some(params) => {
                let start_time = Instant::now();
                let num_ids = ids.len() as u64;
                let mut actions = Vec::with_capacity(ids.len());

                let router_runtime_params = self
                    .runtime_params
                    .as_ref()
                    .expect("No runtime params")
                    .scaling
                    .runtime_params
                    .as_ref()
                    .expect("No scaling runtime params");
                for id in ids {
                    if !router_runtime_params.contains_key(&id) {
                        continue;
                    }

                    let (resp_tx, resp_rx) = oneshot::channel::<Arc<RelayRLAction>>();

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

                    let sender = router_runtime_params
                        .get(&id)
                        .expect("No router runtime params")
                        .tx_to_router
                        .clone();
                    let _ = sender.send(action_request_message).await;

                    match resp_rx.await.map_err(|e| e.to_string()) {
                        Ok(action) => actions.push((id, action)),
                        Err(e) => return Err(e),
                    }
                }

                let duration = start_time.elapsed().as_secs_f64();
                params
                    .metrics
                    .record_histogram("action_request_latency", duration, &[]);
                params
                    .metrics
                    .record_counter("action_requests", num_ids, &[]);

                Ok(actions)
            }
            None => Err("[Coordinator] No runtime instance to _shutdown..."
                .parse()
                .unwrap()),
        }
    }

    async fn _flag_last_action(&self, ids: Vec<Uuid>, reward: Option<f32>) {
        match &self.runtime_params {
            Some(_) => {
                let router_runtime_params = self
                    .runtime_params
                    .as_ref()
                    .expect("No runtime params")
                    .scaling
                    .runtime_params
                    .as_ref()
                    .expect("No scaling runtime params");

                for id in ids {
                    if !router_runtime_params.contains_key(&id) {
                        continue;
                    }

                    let reward: f32 = reward.unwrap_or(0.0);
                    let flag_last_action_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::FlagLastInference,
                        payload: RoutedPayload::FlagLastInference { reward },
                    };

                    let sender = router_runtime_params
                        .get(&id)
                        .expect("No router runtime params")
                        .tx_to_router
                        .clone();
                    let _ = sender.send(flag_last_action_message).await;
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
                let router_runtime_params = self
                    .runtime_params
                    .as_ref()
                    .expect("No runtime params")
                    .scaling
                    .runtime_params
                    .as_ref()
                    .expect("No scaling runtime params");

                for id in ids {
                    if !router_runtime_params.contains_key(&id) {
                        continue;
                    }

                    let (resp_tx, resp_rx) = oneshot::channel::<i64>();

                    let model_version_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::ModelVersion,
                        payload: RoutedPayload::ModelVersion { reply_to: resp_tx },
                    };

                    let sender = router_runtime_params
                        .get(&id)
                        .expect("No router runtime params")
                        .tx_to_router
                        .clone();
                    let _ = sender.send(model_version_message).await;

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

    async fn _scale_up(&mut self, router_add: u32) {
        match &mut self.runtime_params {
            Some(params) => {
                params.scaling.__scale_up(router_add).await;
            }
            None => {
                eprintln!("[Coordinator] No runtime instance to _shutdown...");
            }
        }
    }

    async fn _scale_down(&mut self, router_remove: u32) {
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
