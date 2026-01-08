use crate::network::HyperparameterArgs;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::TransportType;
use crate::network::client::agent::ActorInferenceMode;
use crate::network::client::agent::ActorServerModelMode;
use crate::network::client::agent::FormattedTrajectoryFileParams;
use crate::network::client::agent::{ClientCapabilities, ClientModes};
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::scale_manager::ScaleManagerUuid;
use crate::network::client::runtime::coordination::scale_manager::{
    AlgorithmArgs, ScaleManager, ScaleManagerError,
};
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::{
    StateManager, StateManagerError,
};
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::data::transport::{
    DispatcherConfig, DispatcherError, ScalingDispatcher, TrainingDispatcher, TransportClient,
    TransportError, client_transport_factory,
};
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
use crate::utilities::configuration::{Algorithm, ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
use crate::utilities::observability;
#[cfg(feature = "logging")]
use crate::utilities::observability::logging::builder::LoggingBuilder;
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::MetricsManager;

use thiserror::Error;

use burn_tensor::{Tensor, backend::Backend};

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::{clear_all, clear_context, get, remove, reserve_with};
use relayrl_types::prelude::DeviceType;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use relayrl_types::types::data::action::CodecConfig;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::{AnyBurnTensor, BackendMatcher, TensorData};
use relayrl_types::types::model::ModelModule;

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
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[error(transparent)]
    TransportError(#[from] TransportError),
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[error(transparent)]
    DispatcherError(#[from] DispatcherError),
    #[error(transparent)]
    ScaleManagerError(#[from] ScaleManagerError),
    #[error(transparent)]
    StateManagerError(#[from] StateManagerError),
    #[error(transparent)]
    LifeCycleManagerError(#[from] LifeCycleManagerError),
    #[cfg(feature = "logging")]
    #[error(transparent)]
    LoggingError(#[from] LoggingError),
    #[cfg(feature = "metrics")]
    #[error(transparent)]
    MetricsError(#[from] MetricsError),
    #[error(transparent)]
    ConfigError(#[from] ClientConfigError),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error("No runtime instance to send client IDs to server...")]
    NoRuntimeInstanceError,
}

pub trait ClientInterface<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>
{
    fn new(
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        transport_type: TransportType,
        client_capabilities: ClientCapabilities,
    ) -> Result<Self, CoordinatorError>
    where
        Self: Sized;
    async fn _start(
        &mut self,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        default_model: Option<ModelModule<B>>,
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError>;
    async fn _shutdown(&mut self) -> Result<(), CoordinatorError>;
    async fn _restart(
        &mut self,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        default_model: Option<ModelModule<B>>,
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError>;
    async fn _new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        send_id: bool,
    ) -> Result<(), CoordinatorError>;
    async fn _remove_actor(&mut self, id: ActorUuid) -> Result<(), CoordinatorError>;
    async fn _get_actors(
        &self,
    ) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), CoordinatorError>;
    async fn _set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError>;
    async fn _request_action(
        &self,
        ids: Vec<ActorUuid>,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
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
    async fn _scale_out(&mut self, router_add: u32) -> Result<(), CoordinatorError>;
    async fn _scale_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError>;
    async fn _get_config(&self) -> Result<ClientConfigLoader, CoordinatorError>;
    async fn _set_config_path(&self, config_path: PathBuf) -> Result<(), CoordinatorError>;
}

pub struct CoordinatorParams<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    #[cfg(feature = "logging")]
    pub(crate) logger: LoggingBuilder,
    #[cfg(feature = "metrics")]
    pub(crate) metrics: MetricsManager,
    pub(crate) lifecycle: LifeCycleManager,
    pub(crate) shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    pub(crate) scaling: ScaleManager<B, D_IN, D_OUT>,
}

pub struct ClientCoordinator<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    transport_type: TransportType,
    client_capabilities: Arc<ClientCapabilities>,
    pub(crate) runtime_params: Option<CoordinatorParams<B, D_IN, D_OUT>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientCoordinator<B, D_IN, D_OUT>
{
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub(crate) async fn _send_client_ids_to_server(
        &self,
        client_ids: Vec<(String, Uuid)>,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => params
                .scaling
                ._send_client_ids_to_server(client_ids)
                .await
                .map_err(CoordinatorError::from),
            None => Err(CoordinatorError::NoRuntimeInstanceError),
        }?;
        Ok(())
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientInterface<B, D_IN, D_OUT> for ClientCoordinator<B, D_IN, D_OUT>
{
    fn new(
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        transport_type: TransportType,
        client_capabilities: ClientCapabilities,
    ) -> Result<Self, CoordinatorError> {
        Ok(Self {
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            transport_type,
            client_capabilities: Arc::new(client_capabilities),
            runtime_params: None,
        })
    }

    async fn _start(
        &mut self,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        router_scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        default_model: Option<ModelModule<B>>,
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError> {
        #[cfg(feature = "logging")]
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

        let lifecycle: LifeCycleManager = LifeCycleManager::new(
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            algorithm_args.to_owned(),
            config_loader.to_owned(),
            config_path,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            self.transport_type,
        );
        lifecycle.spawn_loop();

        let shared_client_capabilities = self.client_capabilities.clone();
        let shared_max_traj_length = lifecycle.get_max_traj_length();

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_state_server_addresses: Arc<RwLock<ServerAddresses>> =
            lifecycle.get_server_addresses();
        let shared_local_model_path: Arc<RwLock<PathBuf>> = lifecycle.get_local_model_path();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let state_default_model = default_model.clone();
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        let state_default_model: Option<ModelModule<B>> = Some(default_model.clone());

        let (state, global_dispatcher_rx) = StateManager::new(
            shared_client_capabilities.clone(),
            shared_max_traj_length,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_state_server_addresses,
            shared_local_model_path,
            state_default_model
        );

        let shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> = Arc::from(RwLock::new(state));
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_scaling_server_addresses: Arc<RwLock<ServerAddresses>> =
            lifecycle.get_server_addresses();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_algorithm_args: Arc<AlgorithmArgs> = lifecycle.get_algorithm_args();
        let shared_trajectory_file_output: Arc<RwLock<FormattedTrajectoryFileParams>> =
            lifecycle.get_trajectory_file_output();

        // Create transport and wrap in Arc for sharing across dispatchers
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let transport: TransportClient<B> = client_transport_factory(self.transport_type)
            .map_err(|e| CoordinatorError::TransportError(e))?;
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_transport: Arc<TransportClient<B>> = Arc::new(transport);

        // Create dispatchers with reliability layer (retry, circuit breaker, backpressure)
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let scaling_dispatcher = Arc::new(
            ScalingDispatcher::with_default_config(shared_transport.clone())
                .map_err(CoordinatorError::DispatcherError)?,
        );
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let training_dispatcher = Arc::new(TrainingDispatcher::with_default_config(
            shared_transport.clone(),
        ));

        let mut scaling = ScaleManager::new(
            shared_client_capabilities,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_algorithm_args,
            shared_state.clone(),
            global_dispatcher_rx,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_transport,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            scaling_dispatcher,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            training_dispatcher,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_scaling_server_addresses,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            codec,
            lifecycle.clone(),
        )
        .map_err(|e| CoordinatorError::ScaleManagerError(e))?;

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        if let Err(e) = scaling.__scale_out(router_scale, false).await {
            return Err(CoordinatorError::ScaleManagerError(e));
        }

        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        if let Err(e) = scaling.__scale_in(router_scale).await {
            return Err(CoordinatorError::ScaleManagerError(e));
        }

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let actor_default_model: Option<ModelModule<B>> = default_model.clone();
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        let actor_default_model: Option<ModelModule<B>> = Some(default_model.clone());
        if actor_count > 0 {
            for _ in 1..=actor_count {
                Self::_new_actor(self, default_device.clone(), actor_default_model.clone(), false)
                    .await?;
            }
        } else {
            return Err(CoordinatorError::StateManagerError(
                StateManagerError::NewActorError(
                    "[Coordinator] No actors to create...".to_string(),
                ),
            ));
        }

        #[cfg(feature = "metrics")]
        let metrics: MetricsManager = observability::init_observability();

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let client_ids: Vec<(String, Uuid)> = {
            let actor_pairs = get("actor").map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to get actor pairs: {}",
                    e
                ))
            })?;
            let scale_manager_pairs = get("scale_manager").map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to get scale manager pairs: {}",
                    e
                ))
            })?;
            let external_sender_pairs = get("external_sender").map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to get external sender pairs: {}",
                    e
                ))
            })?;
            let zmq_transport_client_pairs = get("zmq_transport_client").map_err(|e| {
                TransportError::SendClientIdsToServerError(format!(
                    "Failed to get zmq transport client pairs: {}",
                    e
                ))
            })?;
            actor_pairs
                .iter()
                .chain(scale_manager_pairs.iter())
                .chain(external_sender_pairs.iter())
                .chain(zmq_transport_client_pairs.iter())
                .map(|(id, name)| (id.clone(), name.clone()))
                .collect::<Vec<(String, Uuid)>>()
        };

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        scaling._send_client_ids_to_server(client_ids).await?;

        self.runtime_params = Some(CoordinatorParams {
            #[cfg(feature = "logging")]
            logger,
            #[cfg(feature = "metrics")]
            metrics,
            lifecycle,
            shared_state,
            scaling,
        });
        Ok(())
    }

    async fn _shutdown(&mut self) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                // Sends a shutdown RoutedMessage to all actors, which flushes current trajectory to the server and then aborts the actor's message loop task
                params
                    .shared_state
                    .write()
                    .await
                    .__shutdown_all_actors()
                    .await?;

                // the following will trigger shutdown tx/rx for all scalable router nodes in the runtime (router receivers, router senders, central filters)
                // + the single router dispatcher task (the dispatcher informs the actors to shutdown via their inboxes)
                if let Err(e) = params.lifecycle._shutdown() {
                    return Err(CoordinatorError::LifeCycleManagerError(e));
                }

                // shutdown transport client components (sockets, etc.)
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                match &*params.scaling.transport {
                    #[cfg(feature = "async_transport")]
                    TransportClient::Async(async_tr) => async_tr.shutdown().await?,
                    #[cfg(feature = "sync_transport")]
                    TransportClient::Sync(sync_tr) => sync_tr
                        .shutdown()
                        .map_err(|e| CoordinatorError::TransportError(e))?,
                }

                // joins router dispatcher and scales down all routers, pretty redundant but just in case
                params.scaling.clear_runtime_components().await?;

                // inform server that the client is being shutdown and to remove all actor-related data from server runtime
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                params.scaling._send_shutdown_signal_to_server().await?;

                // drain the UUID pool to ensure all UUIDs are removed from the pool
                clear_all().map_err(|e| CoordinatorError::UuidPoolError(e))?;

                params
                    .shared_state
                    .write()
                    .await
                    .clear_runtime_components()
                    .await?;
            }
            None => {
                return Err(CoordinatorError::NoRuntimeInstanceError);
            }
        }

        // if the above shutdown operations were successful, remove the runtime parameters from memory
        if let Some(_) = &mut self.runtime_params {
            let _ = self.runtime_params.take(); // sets the runtime parameters to None
        }
        Ok(())
    }

    async fn _restart(
        &mut self,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        router_scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        default_model: Option<ModelModule<B>>,
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError> {
        self._shutdown().await?;
        self._start(
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            algorithm_args,
            actor_count,
            router_scale,
            default_device,
            default_model,
            config_path,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            codec,
        )
        .await?;
        Ok(())
    }

    async fn _new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        send_id: bool,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let actor_id: Uuid = reserve_with("actor", 117, 100)
                    .map_err(|e| CoordinatorError::UuidPoolError(e))?;

                #[cfg(feature = "metrics")]
                params.metrics.record_counter("actors_created", 1, &[]);

                // Get router runtime params
                let router_runtime_params =
                    params.scaling.runtime_params.as_ref().ok_or_else(|| {
                        CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] No routers available for actor assignment"
                                    .to_string(),
                            ),
                        )
                    })?;

                // Round-robin assignment
                let router_ids: Vec<Uuid> =
                    router_runtime_params.iter().map(|r| *r.key()).collect();
                if router_ids.is_empty() {
                    return Err(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No routers available".to_string(),
                        ),
                    ));
                }

                let actor_count: usize = params.shared_state.read().await.actor_inboxes.len();
                let router_id: Uuid = router_ids[actor_count % router_ids.len()];

                // Get the router's sender_tx
                let trajectory_buffer_tx = router_runtime_params
                    .get(&router_id)
                    .ok_or_else(|| {
                        CoordinatorError::ScaleManagerError(
                            ScaleManagerError::GetRouterRuntimeParamsError(
                                "[Coordinator] Router not found".to_string(),
                            ),
                        )
                    })?
                    .trajectory_buffer_tx
                    .clone();

                params
                    .shared_state
                    .write()
                    .await
                    .__new_actor(
                        actor_id.clone(),
                        router_id,
                        device,
                        default_model,
                        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                        params.scaling.transport.clone(),
                        trajectory_buffer_tx,
                    )
                    .await?;
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                if send_id {
                    params
                        .scaling
                        ._send_client_ids_to_server(vec![("actor".to_string(), actor_id)])
                        .await?;
                }

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
                #[cfg(feature = "metrics")]
                params.metrics.record_counter("actors_removed", 1, &[]);
                params
                    .shared_state
                    .write()
                    .await
                    .__remove_actor(id)
                    .map_err(CoordinatorError::from)?;
                remove("actor", id).map_err(|e| CoordinatorError::UuidPoolError(e))?;
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
                let actors = StateManager::<B, D_IN, D_OUT>::__get_actors(
                    &*params.shared_state.read().await,
                )?;
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
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                StateManager::<B, D_IN, D_OUT>::__set_actor_id(
                    &*params.shared_state.write().await,
                    current_id,
                    new_id,
                )?;
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                let actor_ids = get("actor").map_err(CoordinatorError::from)?;
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                params.scaling._send_client_ids_to_server(actor_ids).await?;

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
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
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
                let global_dispatcher_tx = params
                    .shared_state
                    .read()
                    .await
                    .global_dispatcher_tx
                    .clone();

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
                #[cfg(feature = "metrics")]
                params
                    .metrics
                    .record_histogram("action_request_latency", duration, &[]);
                #[cfg(feature = "metrics")]
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
                let global_dispatcher_tx = params
                    .shared_state
                    .read()
                    .await
                    .global_dispatcher_tx
                    .clone();

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
                let global_dispatcher_tx = params
                    .shared_state
                    .read()
                    .await
                    .global_dispatcher_tx
                    .clone();

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

    async fn _scale_out(&mut self, router_add: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                return params
                    .scaling
                    .__scale_out(router_add, true)
                    .await
                    .map_err(CoordinatorError::ScaleManagerError);
                #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
                return params
                    .scaling
                    .__scale_out(router_add)
                    .await
                    .map_err(CoordinatorError::ScaleManagerError);
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to _shutdown...".to_string(),
                ),
            )),
        }
    }

    async fn _scale_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                params.scaling.__scale_in(router_remove, true).await?;
                #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
                params.scaling.__scale_in(router_remove).await?;
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
            Some(params) => Ok(ClientConfigLoader::load_config(
                &params.lifecycle.get_config_path(),
            )),
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetConfigError(
                    "[Coordinator] No runtime instance to _get_config...".to_string(),
                ),
            )),
        }
    }

    async fn _set_config_path(&self, config_path: PathBuf) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.lifecycle._handle_config_change(config_path).await?;
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
