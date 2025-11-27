use crate::network::TransportType;
use crate::network::UuidPoolError;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::scale_manager::ScaleManagerUuid;
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
use crate::network::client::runtime::transport::{
    TransportClient, TransportError, client_transport_factory,
};
use crate::network::random_uuid;
use crate::network::remove_uuid_from_pool;
use crate::utilities::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
use crate::utilities::misc_utils::ServerAddresses;
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
    TransportError(#[from] TransportError),
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
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
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
        actor_count: u32,
        scale: u32,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        _algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), CoordinatorError>;
    async fn _shutdown(&mut self) -> Result<(), CoordinatorError>;
    async fn _restart(
        self,
        actor_count: u32,
        scale: u32,
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
    async fn _remove_actor(&mut self, id: ActorUuid) -> Result<(), CoordinatorError>;
    async fn _get_actors(
        &self,
    ) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), CoordinatorError>;
    async fn _set_actor_id(
        &self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError>;
    async fn _request_action(
        &self,
        ids: Vec<ActorUuid>,
        observation: AnyBurnTensor<B, D_IN>,
        mask: AnyBurnTensor<B, D_OUT>,
        reward: f32,
    ) -> Result<Vec<(ActorUuid, Arc<RelayRLAction>)>, CoordinatorError>;
    async fn _flag_last_action(
        &self,
        ids: Vec<ActorUuid>,
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError>;
    async fn _get_model_version(
        &self,
        ids: Vec<ActorUuid>,
    ) -> Result<Vec<(ActorUuid, i64)>, CoordinatorError>;
    async fn _scale_up(&mut self, router_add: u32) -> Result<(), CoordinatorError>;
    async fn _scale_down(&mut self, router_remove: u32) -> Result<(), CoordinatorError>;
    async fn _get_config(&self) -> Result<ClientConfigLoader, CoordinatorError>;
    async fn _set_config(&self, config: ClientConfigLoader) -> Result<(), CoordinatorError>;
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
        actor_count: u32,
        router_scale: u32,
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

        let lifecycle: LifeCycleManager =
            LifeCycleManager::new(config_loader.to_owned(), config_path, self.transport_type);
        lifecycle.spawn_loop();

        let shared_state_server_addresses: Arc<RwLock<ServerAddresses>> =
            lifecycle.get_server_addresses();
        let shared_state_config: Arc<RwLock<ClientConfigLoader>> = lifecycle.get_active_config();
        let (state, global_dispatcher_rx) = StateManager::new(
            shared_state_config,
            shared_state_server_addresses,
            default_model.clone(),
        );

        let shared_scaling_server_addresses: Arc<RwLock<ServerAddresses>> =
            lifecycle.get_server_addresses();
        let transport: TransportClient<B> = client_transport_factory(self.transport_type)
            .map_err(|e| CoordinatorError::TransportError(e))?;
        let shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> = Arc::from(RwLock::new(state));
        let scaling_identity: ScaleManagerUuid = random_uuid("scale_manager", 2, 100, 0)
            .map_err(|e| CoordinatorError::UuidPoolError(e))?;
        // Subscribe to lifecycle shutdown and propagate to all actors
        StateManager::spawn_shutdown_watcher(shared_state.clone(), lifecycle.clone());
        let mut scaling = ScaleManager::new(
            scaling_identity,
            shared_state.clone(),
            global_dispatcher_rx,
            transport,
            shared_scaling_server_addresses,
            codec,
            Some(lifecycle.clone()),
        );

        if let Err(e) = scaling.__scale_up(router_scale).await {
            return Err(CoordinatorError::ScaleManagerError(e));
        }

        if actor_count > 0 {
            for _ in 1..=actor_count {
                Self::_new_actor(&self, default_device.clone(), default_model.clone()).await?;
            }
        }

        let metrics: MetricsManager = observability::init_observability();

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

            remove_uuid_from_pool("scale_manager", &runtime_params.scaling.scaling_id)
                .map_err(|e| CoordinatorError::UuidPoolError(e))?;
        }
        Ok(())
    }

    async fn _restart(
        mut self,
        actor_count: u32,
        router_scale: u32,
        default_device: DeviceType,
        default_model: Option<ModelModule<B>>,
        algorithm_name: String,
        config_path: Option<PathBuf>,
        codec: Option<CodecConfig>,
    ) -> Result<(), CoordinatorError> {
        self._shutdown().await?;
        self._start(
            actor_count,
            router_scale,
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

                let actor_id: Uuid =
                    random_uuid("actor", pid_buf.into_iter().sum::<u8>() as u32, 100, 0)
                        .map_err(|e| CoordinatorError::UuidPoolError(e))?;

                params.metrics.record_counter("actors_created", 1, &[]);

                // Get router runtime params to assign actor to a router
                let router_runtime_params =
                    params.scaling.runtime_params.as_ref().ok_or_else(|| {
                        CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] No routers available for actor assignment"
                                    .to_string(),
                            ),
                        )
                    })?;

                // Round-robin assignment: pick a router based on current actor count
                let router_ids: Vec<Uuid> =
                    router_runtime_params.iter().map(|r| *r.key()).collect();
                if router_ids.is_empty() {
                    return Err(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No routers available".to_string(),
                        ),
                    ));
                }

                let actor_count: usize = params.state.read().await.actor_inboxes.len();
                let router_id: Uuid = router_ids[actor_count % router_ids.len()];

                // Get the router's sender_tx
                let sender_tx = router_runtime_params
                    .get(&router_id)
                    .ok_or_else(|| {
                        CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] Router not found".to_string(),
                            ),
                        )
                    })?
                    .sender_tx
                    .clone();

                params
                    .state
                    .write()
                    .await
                    .__new_actor(
                        actor_id,
                        router_id,
                        device,
                        default_model,
                        params.scaling.shared_transport.clone(),
                        sender_tx,
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

    async fn _remove_actor(&mut self, id: ActorUuid) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.metrics.record_counter("actors_removed", 1, &[]);
                params
                    .state
                    .write()
                    .await
                    .__remove_actor(id)
                    .map_err(CoordinatorError::from)?;
                remove_uuid_from_pool("actor", &id)
                    .map_err(|e| CoordinatorError::UuidPoolError(e))?;
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

    async fn _set_actor_id(
        &self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError> {
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
        ids: Vec<ActorUuid>,
        observation: AnyBurnTensor<B, D_IN>,
        mask: AnyBurnTensor<B, D_OUT>,
        reward: f32,
    ) -> Result<Vec<(ActorUuid, Arc<RelayRLAction>)>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let start_time: Instant = Instant::now();
                let num_ids: u64 = ids.len() as u64;
                let mut actions: Vec<(Uuid, Arc<RelayRLAction>)> = Vec::with_capacity(ids.len());

                // Extract router runtime params with clear error messages
                let router_runtime_params: &dashmap::DashMap<
                    Uuid,
                    super::scale_manager::RouterRuntimeParams,
                > = {
                    let runtime_params = self.runtime_params.as_ref().ok_or_else(|| {
                        CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] No runtime params".to_string(),
                            ),
                        )
                    })?;

                    runtime_params
                        .scaling
                        .runtime_params
                        .as_ref()
                        .ok_or_else(|| {
                            CoordinatorError::ScaleManagerError(
                                ScaleManagerError::GetRouterRuntimeParamsError(
                                    "[Coordinator] No scaling runtime params".to_string(),
                                ),
                            )
                        })?
                };

                // Get the global dispatcher sender
                let global_dispatcher_tx = params.state.read().await.global_dispatcher_tx.clone();

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

                    if let Err(e) = global_dispatcher_tx
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

                let duration: f64 = start_time.elapsed().as_secs_f64();
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
        ids: Vec<ActorUuid>,
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let global_dispatcher_tx = params.state.read().await.global_dispatcher_tx.clone();

                for id in ids {
                    let reward: f32 = reward.unwrap_or(0.0);
                    let flag_last_action_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::FlagLastInference,
                        payload: RoutedPayload::FlagLastInference { reward },
                    };

                    if let Err(e) = global_dispatcher_tx
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
        ids: Vec<ActorUuid>,
    ) -> Result<Vec<(Uuid, i64)>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let mut versions = Vec::with_capacity(ids.len());
                let global_dispatcher_tx = params.state.read().await.global_dispatcher_tx.clone();

                for id in ids {
                    let (resp_tx, resp_rx) = oneshot::channel::<i64>();

                    let model_version_message = RoutedMessage {
                        actor_id: id,
                        protocol: RoutingProtocol::ModelVersion,
                        payload: RoutedPayload::ModelVersion { reply_to: resp_tx },
                    };

                    if let Err(e) = global_dispatcher_tx
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

    async fn _get_config(&self) -> Result<ClientConfigLoader, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => Ok(params.lifecycle.get_active_config().read().await.clone()),
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetConfigError(
                    "[Coordinator] No runtime instance to _get_config...".to_string(),
                ),
            )),
        }
    }

    async fn _set_config(&self, config: ClientConfigLoader) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.lifecycle.set_active_config(&config).await?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetConfigError(
                    "[Coordinator] No runtime instance to _set_config...".to_string(),
                ),
            )),
        }
    }
}
