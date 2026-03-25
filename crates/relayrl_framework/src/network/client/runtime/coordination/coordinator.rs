#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::HyperparameterArgs;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::agent::{InferenceAddressesArgs, TrainingAddressesArgs, AlgorithmArgs};
use crate::network::client::agent::{
    ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::scale_manager::
    ProcessInitFlag;
use crate::network::client::runtime::coordination::scale_manager::{
    ScaleManager, ScaleManagerError,
};
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::{
    StateManager, StateManagerError,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    InferenceDispatcher, ScalingDispatcher, TrainingDispatcher,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::{
    ClientTransportInterface, TransportError, client_transport_factory,
};
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
use crate::prelude::config::TransportConfigParams;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::Algorithm;
use crate::utilities::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
#[cfg(feature = "logging")]
use crate::utilities::observability::logging::builder::LoggingBuilder;
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::MetricsManager;

use thiserror::Error;

use burn_tensor::backend::Backend;

use active_uuid_registry::{NamespaceString, ContextString, registry_uuid::Uuid, UuidPoolError};
use active_uuid_registry::interface::{
    clear_namespace, get_context_entries, get_namespace_entries, remove_id, remove_namespace,
    reserve_id_with, reserve_namespace,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher, TensorData};
use relayrl_types::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::DeviceType;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

pub(crate) const CHANNEL_THROUGHPUT: usize = 256_000;

/// Logging subsystem errors
#[derive(Debug, Error)]
#[cfg(feature = "logging")]
pub enum LoggingError {
    #[error("Failed to initialize logging: {0}")]
    InitializationError(String),
    #[error("Failed to configure logger: {0}")]
    ConfigurationError(String),
}

/// Metrics subsystem errors
#[derive(Debug, Error)]
#[cfg(feature = "metrics")]
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
    #[error("Client modes are invalid: {0}")]
    InvalidClientModesError(String),
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error(transparent)]
    TransportError(#[from] TransportError),
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
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: TransportType,
        client_modes: ClientModes,
    ) -> Self
    where
        Self: Sized;
    async fn start(
        &mut self,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] default_model: Option<
            ModelModule<B>,
        >,
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError>;
    async fn shutdown(&mut self) -> Result<(), CoordinatorError>;
    async fn restart(
        &mut self,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] default_model: Option<
            ModelModule<B>,
        >,
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError>;
    async fn new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_id: bool,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        send_algorithm_init: bool,
    ) -> Result<Uuid, CoordinatorError>;
    async fn remove_actor(
        &mut self,
        id: ActorUuid,
        send_ids: bool,
    ) -> Result<(), CoordinatorError>;
    async fn set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError>;
    async fn request_action(
        &self,
        ids: Vec<ActorUuid>,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
        reward: f32,
    ) -> Result<Vec<(ActorUuid, Arc<RelayRLAction>)>, CoordinatorError>;
    async fn flag_last_action(
        &self,
        ids: Vec<ActorUuid>,
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError>;
    async fn get_model_version(
        &self,
        ids: Vec<ActorUuid>,
    ) -> Result<Vec<(ActorUuid, i64)>, CoordinatorError>;
    async fn scale_out(&mut self, router_add: u32) -> Result<(), CoordinatorError>;
    async fn scale_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError>;
    async fn get_config(&self) -> Result<ClientConfigLoader, CoordinatorError>;
    async fn set_config_path(&self, config_path: PathBuf) -> Result<(), CoordinatorError>;
}

pub struct CoordinatorParams<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    pub(crate) client_namespace: Arc<str>,
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
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    transport_type: TransportType,
    pub(crate) client_modes: Arc<ClientModes>,
    pub(crate) runtime_params: Option<CoordinatorParams<B, D_IN, D_OUT>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientCoordinator<B, D_IN, D_OUT>
{
    /// Transparent helper function used by the agent API for calling into the runtime to send client IDs to the server
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn send_client_ids_to_server(
        &self,
        client_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        replace_context: bool,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => params
                .scaling
                .send_client_ids_to_server(client_entries, replace_context)
                .await
                .map_err(CoordinatorError::from),
            None => Err(CoordinatorError::NoRuntimeInstanceError),
        }?;

        Ok(())
    }

    /// Transparent helper function used by the agent API for calling into the runtime to send an algorithm init request to the server
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn send_algorithm_init_request(
        &mut self,
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
    ) -> Result<(), CoordinatorError> {
        match self.runtime_params.as_mut() {
            Some(params) => params
                .scaling
                .send_process_init_request(
                    actor_entries,
                    ProcessInitFlag::<B>::TrainingAlgorithmInit,
                )
                .await
                .map_err(CoordinatorError::from),
            None => Err(CoordinatorError::NoRuntimeInstanceError),
        }?;

        Ok(())
    }

    /// Transparent helper function used by the agent API for calling into the runtime to send an inference model init request to the server
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn send_inference_model_init_request(
        &mut self,
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        default_model: Option<ModelModule<B>>,
    ) -> Result<(), CoordinatorError> {
        match self.runtime_params.as_mut() {
            Some(params) => params
                .scaling
                .send_process_init_request(
                    actor_entries,
                    ProcessInitFlag::<B>::InferenceModelInit(default_model),
                )
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
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: TransportType,
        client_modes: ClientModes,
    ) -> Self {
        Self {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_type,
            client_modes: Arc::new(client_modes),
            runtime_params: None,
        }
    }

    async fn start(
        &mut self,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        router_scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] default_model: Option<
            ModelModule<B>,
        >,
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError> {
        let client_namespace: Arc<str> = Arc::from(format!(
            "{}-{}",
            crate::network::CLIENT_NAMESPACE_PREFIX,
            Uuid::new_v4()
        ));

        clear_namespace(client_namespace.as_ref()); // for this agent runtime, ensure no overlapping namespace exists in uuid registry/entire process
        reserve_namespace(client_namespace.as_ref());

        #[cfg(feature = "logging")]
        let logger = LoggingBuilder::new();

        #[cfg(feature = "metrics")]
        let metrics: MetricsManager = observability::init_observability();

        let shared_client_modes: Arc<ClientModes> = self.client_modes.clone();

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

        let mut config_loader: ClientConfigLoader = ClientConfigLoader::load_config(&config_path);

        let lifecycle: LifeCycleManager = LifeCycleManager::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args.to_owned(),
            &config_loader,
            config_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.transport_type,
        );

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        /// if args are set in client mode init config, set lifecycle manager server addresses while keeping unchanged config values
        {
            let inference_address_args = if let ActorInferenceMode::Server(server_params) =
                &shared_client_modes.actor_inference_mode
            {
                server_params.inference_addresses.clone()
            } else {
                None
            };

            let training_address_args = match &shared_client_modes.actor_training_data_mode {
                ActorTrainingDataMode::Online(server_params) => {
                    server_params.training_addresses.clone()
                }
                ActorTrainingDataMode::Hybrid(server_params, _) => {
                    server_params.training_addresses.clone()
                }
                ActorTrainingDataMode::Disabled | ActorTrainingDataMode::Offline(_) => None,
            };

            if inference_address_args.is_some() || training_address_args.is_some() {
                let transport_params_for_packing: &mut TransportConfigParams =
                    &mut config_loader.transport_config;

                if let Some(inference_addresses) = inference_address_args {
                    match &self.transport_type {
                        #[cfg(feature = "nats-transport")]
                        TransportType::NATS => {
                            if let Some(inference_server_address) = match inference_addresses {
                                #[cfg(feature = "nats-transport")]
                                InferenceAddressesArgs::NATS(params) => params.clone(),
                                #[cfg(feature = "zmq-transport")]
                                InferenceAddressesArgs::ZMQ(_) => None,
                            } {
                                transport_params_for_packing.nats_addresses.inference_server_address = inference_server_address;
                            }
                        }
                        #[cfg(feature = "zmq-transport")]
                        TransportType::ZMQ => {
                            if let Some(inference_server_address) = match inference_addresses {
                                #[cfg(feature = "nats-transport")]
                                InferenceAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                InferenceAddressesArgs::ZMQ(ref params) => params.inference_server_address.clone(),
                            } {
                                transport_params_for_packing.zmq_addresses.inference_addresses.inference_server_address = inference_server_address;
                            } 

                            if let Some(inference_scaling_server_address) = match inference_addresses {
                                #[cfg(feature = "nats-transport")]
                                InferenceAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                InferenceAddressesArgs::ZMQ(ref params) => params.inference_scaling_server_address.clone(),
                            } {
                                transport_params_for_packing.zmq_addresses.inference_addresses.inference_scaling_server_address = inference_scaling_server_address;
                            }
                        }
                    }
                }

                if let Some(training_addresses) = training_address_args {
                    match &self.transport_type {
                        #[cfg(feature = "nats-transport")]
                        TransportType::NATS => {
                            if let Some(training_server_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(params) => params.clone(),
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(_) => None,
                            } {
                                transport_params_for_packing.nats_addresses.training_server_address = training_server_address;
                            }
                        }
                        #[cfg(feature = "zmq-transport")]
                        TransportType::ZMQ => {
                            if let Some(agent_listener_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => params.agent_listener_address.clone(),
                            } {
                                transport_params_for_packing.zmq_addresses.training_addresses.agent_listener_address = agent_listener_address;
                            }

                            if let Some(model_server_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => params.model_server_address.clone(),
                            } {
                                transport_params_for_packing.zmq_addresses.training_addresses.model_server_address = model_server_address;
                            }

                            if let Some(trajectory_server_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => params.trajectory_server_address.clone(),
                            } {
                                transport_params_for_packing.zmq_addresses.training_addresses.trajectory_server_address = trajectory_server_address;
                            }

                            if let Some(training_scaling_server_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => params.training_scaling_server_address.clone(),
                            } {
                                transport_params_for_packing.zmq_addresses.training_addresses.training_scaling_server_address = training_scaling_server_address;
                            }
                        }
                    }
                }

                lifecycle
                    .set_transport_addresses(&transport_params_for_packing, &self.transport_type)
                    .await?;
            }
        }

        {
            /// if args are set in client mode init config, set lifecycle manager trajectory file path
            let local_trajectory_file_params = match &shared_client_modes.actor_training_data_mode {
                ActorTrainingDataMode::Offline(file_params) => match file_params {
                    Some(params) => Some(params),
                    None => None,
                },
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorTrainingDataMode::Hybrid(_, file_params) => match file_params {
                    Some(params) => Some(params),
                    None => None,
                },
                _ => None,
            };

            if let Some(file_params) = local_trajectory_file_params {
                lifecycle.set_trajectory_file_path(&file_params).await?;
            }
        }

        lifecycle.spawn_loop();

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let (inference_dispatcher, scaling_dispatcher, training_dispatcher) = {
            // Create transport and wrap in Arc for sharing across dispatchers
            let transport: ClientTransportInterface<B> = client_transport_factory(
                self.transport_type,
                client_namespace.clone(),
                shared_client_modes.clone(),
            ).await
            .map_err(|e| CoordinatorError::TransportError(e))?;

            let shared_transport: Arc<ClientTransportInterface<B>> = Arc::new(transport);

            let (inference_dispatcher, mut scaling_dispatcher) =
                match shared_client_modes.actor_inference_mode {
                    ActorInferenceMode::Server(_) => (
                        Some(Arc::new(InferenceDispatcher::<B>::new(
                            shared_transport.clone(),
                        ))),
                        Some(Arc::new(ScalingDispatcher::<B>::new(
                            shared_transport.clone(),
                        ))),
                    ),
                    ActorInferenceMode::Local(_) => (None, None),
                };

            let training_dispatcher = match shared_client_modes.actor_training_data_mode {
                ActorTrainingDataMode::Disabled | ActorTrainingDataMode::Offline(_) => None,
                _ => {
                    scaling_dispatcher = Some(Arc::new(ScalingDispatcher::<B>::new(
                        shared_transport.clone(),
                    )));
                    Some(Arc::new(TrainingDispatcher::<B>::new(
                        shared_transport.clone(),
                    )))
                }
            };

            (
                inference_dispatcher,
                scaling_dispatcher,
                training_dispatcher,
            )
        };

        {
            let shared_max_traj_length = lifecycle.get_max_traj_length();
            
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let shared_transport_addresses = if let ActorInferenceMode::Server(_) =
                shared_client_modes.actor_inference_mode
            {
                Some(lifecycle.get_transport_addresses())
            } else if let ActorTrainingDataMode::Online(_) | ActorTrainingDataMode::Hybrid(_, _) =
                shared_client_modes.actor_training_data_mode
            {
                Some(lifecycle.get_transport_addresses())
            } else {
                None
            };

            let (state, global_dispatcher_rx) = {
                let shared_local_model_path = lifecycle.get_local_model_path();

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                let state_default_model = default_model.clone();
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                let state_default_model: Option<ModelModule<B>> = Some(default_model.clone());

                StateManager::new(
                    client_namespace.clone(),
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    inference_dispatcher.clone(),
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    training_dispatcher.clone(),
                    shared_client_modes.clone(),
                    shared_max_traj_length,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    shared_transport_addresses.clone(),
                    shared_local_model_path,
                    state_default_model,
                )
            };

            let shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                Arc::from(RwLock::new(state));

            let mut scaling = {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                let shared_algorithm_args = lifecycle.get_algorithm_args();

                ScaleManager::new(
                    client_namespace.clone(),
                    shared_client_modes,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    shared_algorithm_args,
                    shared_state.clone(),
                    global_dispatcher_rx,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    scaling_dispatcher,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    training_dispatcher,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    shared_transport_addresses.clone(),
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    codec,
                    lifecycle.clone(),
                )
                .map_err(|e| CoordinatorError::ScaleManagerError(e))?
            };

            self.runtime_params = Some(CoordinatorParams {
                client_namespace,
                #[cfg(feature = "logging")]
                logger,
                #[cfg(feature = "metrics")]
                metrics,
                lifecycle,
                shared_state,
                scaling,
            });
        }

        if let Some(params) = self.runtime_params.as_mut() {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if let Err(e) = params.scaling.scale_out(router_scale, false).await {
                return Err(CoordinatorError::ScaleManagerError(e));
            }
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            if let Err(e) = params.scaling.scale_out(router_scale).await {
                return Err(CoordinatorError::ScaleManagerError(e));
            }
        }

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let actor_default_model: Option<ModelModule<B>> = default_model.clone();
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        let actor_default_model: Option<ModelModule<B>> = Some(default_model.clone());
        if actor_count > 0 {
            for _ in 1..=actor_count {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                Self::new_actor(
                    self,
                    default_device.clone(),
                    actor_default_model.clone(),
                    false,
                    false,
                )
                .await?;
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                Self::new_actor(self, default_device.clone(), actor_default_model.clone()).await?;
            }
        } else {
            println!(
                "[Coordinator] RelayRLAgent started with no actors: either restart or add actors to the runtime!"
            );
        }

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let Some(params) = self.runtime_params.as_mut() {
            let client_entries: Vec<(NamespaceString, ContextString, Uuid)> =
                get_namespace_entries(params.client_namespace.as_ref())
                    .map_err(CoordinatorError::from)?;
            params
                .scaling
                .send_client_ids_to_server(client_entries, true)
                .await?;

            let actor_entries = get_context_entries(
                params.client_namespace.as_ref(),
                crate::network::ACTOR_CONTEXT,
            )?;
            params
                .scaling
                .send_process_init_request(
                    actor_entries,
                    ProcessInitFlag::<B>::TrainingAlgorithmInit,
                )
                .await?;
        }

        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                // Sends a shutdown RoutedMessage to all actors, which flushes current trajectory to the server and then aborts the actor's message loop task
                params
                    .shared_state
                    .write()
                    .await
                    .shutdown_all_actors()
                    .await?;

                // inform server(s) that the client is being shutdown and to remove all actor-related data from server runtime
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                params.scaling.send_shutdown_signal_to_server().await?;

                // shutdown transport client components (sockets, etc.)
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                match &params.scaling.scaling_dispatcher {
                    Some(dispatcher) => dispatcher.shutdown_transport().await?,
                    None => (),
                }

                // the following will trigger shutdown tx/rx for all scalable router nodes in the runtime (router receivers, router senders, central filters)
                // + the single router dispatcher task (the dispatcher informs the actors to shutdown via their inboxes)
                if let Err(e) = params.lifecycle.shutdown() {
                    return Err(CoordinatorError::LifeCycleManagerError(e));
                }

                // joins router dispatcher and scales down all routers, pretty redundant but just in case
                params.scaling.clear_runtime_components().await?;

                // drain the UUID pool to ensure all UUIDs are removed from the pool for the client namespace
                remove_namespace(params.client_namespace.as_ref());

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

    async fn restart(
        &mut self,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        algorithm_args: AlgorithmArgs,
        actor_count: u32,
        router_scale: u32,
        default_device: DeviceType,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] default_model: Option<
            ModelModule<B>,
        >,
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        default_model: ModelModule<B>,
        config_path: Option<PathBuf>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: Option<
            CodecConfig,
        >,
    ) -> Result<(), CoordinatorError> {
        self.shutdown().await?;
        self.start(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args,
            actor_count,
            router_scale,
            default_device,
            default_model,
            config_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            codec,
        )
        .await?;
        Ok(())
    }

    async fn new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_id: bool,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        send_algorithm_init: bool,
    ) -> Result<Uuid, CoordinatorError> {
        match self.runtime_params.as_mut() {
            Some(params) => {
                let actor_id: Uuid = reserve_id_with(
                    params.client_namespace.as_ref(),
                    crate::network::ACTOR_CONTEXT,
                    117,
                    100,
                )
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
                let router_namespaces: Vec<RouterNamespace> = router_runtime_params
                    .iter()
                    .map(|r| r.key().clone())
                    .collect();
                if router_namespaces.is_empty() {
                    return Err(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No routers available".to_string(),
                        ),
                    ));
                }

                let actor_count: usize = params.shared_state.read().await.actor_inboxes.len();
                let router_namespace: RouterNamespace =
                    router_namespaces[actor_count % router_namespaces.len()].clone();

                // Get the router's sender_tx
                let trajectory_buffer_tx = router_runtime_params
                    .get(&router_namespace)
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
                    .new_actor(
                        actor_id.clone(),
                        router_namespace,
                        device,
                        default_model,
                        trajectory_buffer_tx,
                    )
                    .await?;

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                {
                    if send_id {
                        let actor_entry = vec![(
                            params.client_namespace.to_string(),
                            crate::network::ACTOR_CONTEXT.to_string(),
                            actor_id,
                        )];

                        params
                            .scaling
                            .send_client_ids_to_server(actor_entry.clone(), false)
                            .await?;

                        if send_algorithm_init {
                            params
                                .scaling
                                .send_process_init_request(
                                    actor_entry,
                                    ProcessInitFlag::<B>::TrainingAlgorithmInit,
                                )
                                .await?;
                        }
                    }
                }

                Ok(actor_id)
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::NewActorError(
                    "[Coordinator] No runtime instance to new_actor...".to_string(),
                ),
            )),
        }
    }

    async fn remove_actor(
        &mut self,
        id: ActorUuid,
        send_ids: bool,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                params.metrics.record_counter("actors_removed", 1, &[]);
                params
                    .shared_state
                    .write()
                    .await
                    .remove_actor(id)
                    .map_err(CoordinatorError::from)?;
                // remove the actor id from the namespace/context since we're removing the actor
                remove_id(
                    params.client_namespace.as_ref(),
                    crate::network::ACTOR_CONTEXT,
                    id,
                )
                .map_err(|e| CoordinatorError::UuidPoolError(e))?;

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if send_ids {
                    let actor_entries = get_context_entries(
                        params.client_namespace.as_ref(),
                        crate::network::ACTOR_CONTEXT,
                    )?;
                    params
                        .scaling
                        .send_client_ids_to_server(actor_entries, true)
                        .await?;
                }

                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::RemoveActorError(
                    "[Coordinator] No runtime instance to remove_actor...".to_string(),
                ),
            )),
        }
    }

    async fn set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                StateManager::<B, D_IN, D_OUT>::set_actor_id(
                    &*params.shared_state.write().await,
                    current_id,
                    new_id,
                )?;

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                {
                    let actor_ids = get_context_entries(
                        params.client_namespace.as_ref(),
                        crate::network::ACTOR_CONTEXT,
                    )?;
                    // send all actor ids to the server since all we do here is replace an id with another one
                    params
                        .scaling
                        .send_client_ids_to_server(actor_ids, true)
                        .await?;
                }

                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetActorIdError(
                    "[Coordinator] No runtime instance to set_actor_id...".to_string(),
                ),
            )),
        }
    }

    async fn request_action(
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
                    RouterNamespace,
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
                    let has_router = params
                        .shared_state
                        .read()
                        .await
                        .actor_router_addresses
                        .contains_key(&id);
                    if !has_router {
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
                    "[Coordinator] No runtime instance to request_action...".to_string(),
                ),
            )),
        }
    }

    async fn flag_last_action(
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
                    "[Coordinator] No runtime instance to flag_last_action...".to_string(),
                ),
            )),
        }
    }

    async fn get_model_version(
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
                    "[Coordinator] No runtime instance to get_model_version...".to_string(),
                ),
            )),
        }
    }

    async fn scale_out(&mut self, router_add: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                return params
                    .scaling
                    .scale_out(router_add, true)
                    .await
                    .map_err(CoordinatorError::ScaleManagerError);
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                return params
                    .scaling
                    .scale_out(router_add)
                    .await
                    .map_err(CoordinatorError::ScaleManagerError);
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to scale_out...".to_string(),
                ),
            )),
        }
    }

    async fn scale_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                params.scaling.scale_in(router_remove, true).await?;
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                params.scaling.scale_in(router_remove).await?;
                Ok(())
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to scale_in...".to_string(),
                ),
            )),
        }
    }

    async fn get_config(&self) -> Result<ClientConfigLoader, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => Ok(ClientConfigLoader::load_config(
                &params.lifecycle.get_config_path(),
            )),
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetConfigError(
                    "[Coordinator] No runtime instance to get_config...".to_string(),
                ),
            )),
        }
    }

    async fn set_config_path(&self, config_path: PathBuf) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.lifecycle.handle_config_change(config_path).await?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetConfigError(
                    "[Coordinator] No runtime instance to set_config_path...".to_string(),
                ),
            )),
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use active_uuid_registry::registry_uuid::Uuid;
    use burn_ndarray::NdArray;
    use std::path::PathBuf;
    use std::sync::Arc;

    type TestBackend = NdArray<f32>;

    fn make_coordinator() -> ClientCoordinator<TestBackend, 4, 1> {
        ClientCoordinator::<TestBackend, 4, 1>::new(#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] TransportType::default(), ClientModes::default())
    }

    #[test]
    fn from_string_yields_invalid_value() {
        let err = ClientConfigError::from("bad input".to_string());
        assert!(matches!(err, ClientConfigError::InvalidValue(ref s) if s == "bad input"));
    }

    #[test]
    fn new_has_no_runtime_params() {
        let coordinator = make_coordinator();
        assert!(coordinator.runtime_params.is_none());
    }

    #[tokio::test]
    async fn remove_actor_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let result = c.remove_actor(Uuid::new_v4(), false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn set_actor_id_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let result = c.set_actor_id(Uuid::new_v4(), Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn flag_last_action_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.flag_last_action(vec![], None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_model_version_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.get_model_version(vec![]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn scale_out_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let result = c.scale_out(1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn scale_in_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let result = c.scale_in(1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_config_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.get_config().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn set_config_path_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.set_config_path(PathBuf::new()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn shutdown_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let result = c.shutdown().await;
        assert!(result.is_err());
    }
}