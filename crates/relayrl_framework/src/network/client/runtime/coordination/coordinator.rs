use crate::network::TransportType;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::scale_manager::{
    ScaleManager, ScaleManagerError,
};
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::{
    StateManager, StateManagerError,
};
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
use crate::network::client::runtime::transport::{TransportClient, client_transport_factory};
use crate::utilities::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
use crate::utilities::observability;
use crate::utilities::observability::logging::builder::LoggingBuilder;
use crate::utilities::observability::metrics::MetricsManager;

use thiserror::Error;

use burn_tensor::{Tensor, backend::Backend};
use relayrl_types::prelude::DeviceType;
use relayrl_types::types::data::action::{CodecConfig, RelayRLAction};
use relayrl_types::types::data::tensor::{AnyBurnTensor, BackendMatcher, TensorData};
use relayrl_types::types::model::{HotReloadableModel, ModelModule};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use uuid::Uuid;

pub(crate) const CHANNEL_THROUGHPUT: usize = 256_000;

/// Logging subsystem errors
#[derive(Debug, Error)]
pub enum LoggingError {
    #[error("Failed to initialize logging: {0}")]
    InitializationError(String),
    #[error("Failed to configure logger: {0}")]
    ConfigurationError(String),
}

/// Metrics subsystem errors
#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("Failed to initialize metrics: {0}")]
    InitializationError(String),
    #[error("Failed to record metric: {0}")]
    RecordError(String),
}

/// Client configuration errors
#[derive(Debug, Error)]
pub enum ClientConfigError {
    #[error("Config file not found: {0}")]
    NotFound(String),
    #[error("Failed to parse config: {0}")]
    ParseError(String),
    #[error("Invalid config value: {0}")]
    InvalidValue(String),
}

impl From<String> for ClientConfigError {
    fn from(e: String) -> Self {
        ClientConfigError::InvalidValue(e)
    }
}

#[derive(Debug, Error)]
pub enum CoordinatorError {
    #[error(transparent)]
    ScaleManagerError(#[from] ScaleManagerError),
    #[error(transparent)]
    StateManagerError(#[from] StateManagerError),
    #[error(transparent)]
    LifeCycleManagerError(#[from] LifeCycleManagerError),
    #[error(transparent)]
    LoggingError(#[from] LoggingError),
    #[error(transparent)]
    MetricsError(#[from] MetricsError),
    #[error(transparent)]
    ConfigError(#[from] ClientConfigError),
}

pub trait ClientInterface<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>
{
    fn new(transport_type: TransportType) -> Self;
    async fn _start(
        self,
        actor_count: i64,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        _algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), CoordinatorError>;
    async fn _shutdown(&mut self) -> Result<(), CoordinatorError>;
    async fn _restart(
        self,
        actor_count: i64,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), CoordinatorError>;
    async fn _new_actor(
        &self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Result<(), CoordinatorError>;
    async fn _remove_actor(&mut self, id: Uuid) -> Result<(), CoordinatorError>;
    async fn _get_actors(
        &self,
    ) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), CoordinatorError>;
    async fn _set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), CoordinatorError>;
    async fn _request_action(
        &self,
        ids: Vec<Uuid>,
        observation: AnyBurnTensor<B, D_IN>,
        mask: AnyBurnTensor<B, D_OUT>,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, CoordinatorError>;
    async fn _flag_last_action(
        &self,
        ids: Vec<Uuid>,
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError>;
    async fn _get_model_version(
        &self,
        ids: Vec<Uuid>,
    ) -> Result<Vec<(Uuid, i64)>, CoordinatorError>;
    async fn _scale_up(&mut self, router_add: u32) -> Result<(), CoordinatorError>;
    async fn _scale_down(&mut self, router_remove: u32) -> Result<(), CoordinatorError>;
}

pub struct CoordinatorParams<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    logger: LoggingBuilder,
    lifecycle: LifeCycleManager,
    state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    scaling: ScaleManager<B, D_IN, D_OUT>,
    metrics: MetricsManager,
}

pub struct ClientCoordinator<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    transport_type: TransportType,
    runtime_params: Option<CoordinatorParams<B, D_IN, D_OUT>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientInterface<B, D_IN, D_OUT> for ClientCoordinator<B, D_IN, D_OUT>
{
    fn new(transport_type: TransportType) -> Self {
        Self {
            transport_type,
            runtime_params: None,
        }
    }

    async fn _start(
        mut self,
        actor_count: i64,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        _algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), CoordinatorError> {
        let logger = LoggingBuilder::new();

        let config_path: PathBuf = match config_path {
            Some(path) => path,
            None => match DEFAULT_CLIENT_CONFIG_PATH.clone() {
                Some(path) => path,
                None => return Err(CoordinatorError::ConfigError(ClientConfigError::NotFound(
                    "[Coordinator] No config path provided and default config path not found..."
                        .to_string(),
                ))),
            },
        };

        let config_loader: ClientConfigLoader = ClientConfigLoader::load_config(&config_path);

        let transport: TransportClient<B> =
            client_transport_factory(self.transport_type, &config_loader);

        let lifecycle: LifeCycleManager = LifeCycleManager::new(config_loader.to_owned());
        lifecycle.spawn_loop();

        let shared_state_config: Arc<RwLock<ClientConfigLoader>> = lifecycle.get_active_config();

        let (state, global_bus_rx, rx_from_actor) =
            StateManager::new(shared_state_config, default_model.clone());

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

        let shared_global_bus_rx: Arc<RwLock<tokio::sync::mpsc::Receiver<RoutedMessage>>> =
            Arc::new(RwLock::new(global_bus_rx));
        let shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> = Arc::from(RwLock::new(state));
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
            codec,
        )
        .with_lifecycle(lifecycle.clone());
        if let Err(e) = scaling.__scale_up(1).await {
            return Err(CoordinatorError::ScaleManagerError(e));
        }

        if actor_count > 0 {
            for _ in 0..actor_count {
                Self::_new_actor(&self, default_device.clone(), default_model.clone()).await?;
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
        Ok(())
    }

    async fn _shutdown(&mut self) -> Result<(), CoordinatorError> {
        if let Some(mut runtime_params) = self.runtime_params.take() {
            if let Err(e) = runtime_params.lifecycle._shutdown() {
                return Err(CoordinatorError::LifeCycleManagerError(e));
            }

            StateManager::<B, D_IN, D_OUT>::__shutdown_all_actors(
                &*runtime_params.state.read().await,
            );

            let router_count: u32 = runtime_params
                .scaling
                .runtime_params
                .as_ref()
                .map(|m| m.len() as u32)
                .unwrap_or(0);
            if router_count > 0 {
                if let Err(e) = runtime_params.scaling.__scale_down(router_count).await {
                    return Err(CoordinatorError::ScaleManagerError(e));
                }
            }
        }
        Ok(())
    }

    async fn _restart(
        mut self,
        actor_count: i64,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), CoordinatorError> {
        self._shutdown().await?;
        self._start(
            actor_count,
            default_device,
            default_model,
            algorithm_name,
            config_path,
            codec,
        )
        .await?;
        Ok(())
    }

    async fn _new_actor(
        &self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Result<(), CoordinatorError> {
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
                    .await?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::NewActorError(
                    "[Coordinator] No runtime instance to _new_actor...".to_string(),
                ),
            )),
        }
    }

    async fn _remove_actor(&mut self, id: Uuid) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.metrics.record_counter("actors_removed", 1, &[]);
                params
                    .state
                    .write()
                    .await
                    .__remove_actor(id)
                    .map_err(CoordinatorError::from)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::RemoveActorError(
                    "[Coordinator] No runtime instance to _remove_actor...".to_string(),
                ),
            )),
        }
    }

    async fn _get_actors(
        &self,
    ) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let actors =
                    StateManager::<B, D_IN, D_OUT>::__get_actors(&*params.state.read().await)?;
                Ok(actors)
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetActorsError(
                    "[Coordinator] No runtime parameter instance to _get_actors...".to_string(),
                ),
            )),
        }
    }

    async fn _set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                StateManager::<B, D_IN, D_OUT>::__set_actor_id(
                    &*params.state.write().await,
                    current_id,
                    new_id,
                )?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetActorIdError(
                    "[Coordinator] No runtime instance to _shutdown...".to_string(),
                ),
            )),
        }
    }

    async fn _request_action(
        &self,
        ids: Vec<Uuid>,
        observation: AnyBurnTensor<B, D_IN>,
        mask: AnyBurnTensor<B, D_OUT>,
        reward: f32,
    ) -> Result<Vec<(Uuid, Arc<RelayRLAction>)>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let start_time = Instant::now();
                let num_ids = ids.len() as u64;
                let mut actions = Vec::with_capacity(ids.len());

                let router_runtime_params = self
                    .runtime_params
                    .as_ref()
                    .ok_or(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No runtime params".to_string(),
                        ),
                    ))?
                    .scaling
                    .runtime_params
                    .as_ref()
                    .ok_or(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No scaling runtime params".to_string(),
                        ),
                    ))?;

                for id in ids {
                    if !router_runtime_params.contains_key(&id) {
                        continue;
                    }

                    let (resp_tx, resp_rx) = oneshot::channel::<Arc<RelayRLAction>>();

                    let action_request_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::RequestInference,
                        payload: RoutedPayload::RequestInference(Box::new(InferenceRequest {
                            observation: Box::new(observation.clone()),
                            mask: Box::new(mask.clone()),
                            reward,
                            reply_to: resp_tx,
                        })),
                    };

                    let sender = router_runtime_params
                        .get(&id)
                        .ok_or(CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] No router runtime params".to_string(),
                            ),
                        ))?
                        .tx_to_router
                        .clone();

                    if let Err(e) = sender
                        .send(action_request_message)
                        .await
                        .map_err(|e| e.to_string())
                    {
                        return Err(CoordinatorError::ScaleManagerError(
                            ScaleManagerError::SendActionRequestError(e),
                        ));
                    }

                    match resp_rx.await.map_err(|e| e.to_string()) {
                        Ok(action) => actions.push((id, action)),
                        Err(e) => {
                            return Err(CoordinatorError::ScaleManagerError(
                                ScaleManagerError::ReceiveActionResponseError(e),
                            ));
                        }
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
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to _shutdown...".to_string(),
                ),
            )),
        }
    }

    async fn _flag_last_action(
        &self,
        ids: Vec<Uuid>,
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(_) => {
                let router_runtime_params: &dashmap::DashMap<
                    Uuid,
                    super::scale_manager::RouterRuntimeParams,
                > = self
                    .runtime_params
                    .as_ref()
                    .ok_or(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No runtime params".to_string(),
                        ),
                    ))?
                    .scaling
                    .runtime_params
                    .as_ref()
                    .ok_or(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No scaling runtime params".to_string(),
                        ),
                    ))?;

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
                        .ok_or(CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] No router runtime params".to_string(),
                            ),
                        ))?
                        .tx_to_router
                        .clone();
                    if let Err(e) = sender
                        .send(flag_last_action_message)
                        .await
                        .map_err(|e| e.to_string())
                    {
                        return Err(CoordinatorError::ScaleManagerError(
                            ScaleManagerError::SendFlagLastActionMessageError(e),
                        ));
                    }
                }
                Ok(())
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to _flag_last_action...".to_string(),
                ),
            )),
        }
    }

    async fn _get_model_version(
        &self,
        ids: Vec<Uuid>,
    ) -> Result<Vec<(Uuid, i64)>, CoordinatorError> {
        match &self.runtime_params {
            Some(_) => {
                let mut versions = Vec::with_capacity(ids.len());
                let router_runtime_params = self
                    .runtime_params
                    .as_ref()
                    .ok_or(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No runtime params".to_string(),
                        ),
                    ))?
                    .scaling
                    .runtime_params
                    .as_ref()
                    .ok_or(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No scaling runtime params".to_string(),
                        ),
                    ))?;

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
                        .ok_or(CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] No router runtime params".to_string(),
                            ),
                        ))?
                        .tx_to_router
                        .clone();
                    if let Err(e) = sender
                        .send(model_version_message)
                        .await
                        .map_err(|e| e.to_string())
                    {
                        return Err(CoordinatorError::ScaleManagerError(
                            ScaleManagerError::SendModelVersionMessageError(e),
                        ));
                    }

                    match resp_rx.await.map_err(|e| e.to_string()) {
                        Ok(model_version) => versions.push((id, model_version)),
                        Err(e) => {
                            return Err(CoordinatorError::ScaleManagerError(
                                ScaleManagerError::ReceiveModelVersionResponseError(e),
                            ));
                        }
                    }
                }
                Ok(versions)
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to _get_model_version...".to_string(),
                ),
            )),
        }
    }

    async fn _scale_up(&mut self, router_add: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                return params
                    .scaling
                    .__scale_up(router_add)
                    .await
                    .map_err(|e| CoordinatorError::ScaleManagerError(e));
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to _shutdown...".to_string(),
                ),
            )),
        }
    }

    async fn _scale_down(&mut self, router_remove: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                params.scaling.__scale_down(router_remove).await?;
                Ok(())
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to _shutdown...".to_string(),
                ),
            )),
        }
    }
}
