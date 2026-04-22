//! Client runtime coordinator.
//!
//! This module owns top-level orchestration for the client runtime: configuration loading,
//! lifecycle management, actor state, router scaling, and the public request path exposed through
//! `RelayRLAgent`.

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
use crate::network::client::agent::{ActorInferenceMode, ActorTrainingDataMode, ClientModes};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::agent::{AlgorithmArgs, InferenceAddressesArgs, TrainingAddressesArgs};
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifecycleManager, LifecycleManagerError,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::scale_manager::ProcessInitFlag;
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
use crate::network::client::runtime::coordination::scale_manager::{
    ScaleManager, ScaleManagerError,
};
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::{
    StateManager, StateManagerError,
};
use crate::network::client::runtime::data::environments::vec_env::IntoAnyTensorKind;
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
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::TransportConfigParams;
use crate::utilities::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
#[cfg(feature = "logging")]
use crate::utilities::observability::logging::*;
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::*;

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::{get_context_entries, get_namespace_entries};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::{ContextString, NamespaceString};

use thiserror::Error;

use burn_tensor::{BasicOps, backend::Backend};

use active_uuid_registry::interface::{
    clear_namespace, remove_namespace, reserve_id_with, reserve_namespace,
};
use active_uuid_registry::{UuidPoolError, registry_uuid::Uuid};
use relayrl_env_trait::traits::Environment;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::ModelModule;
use relayrl_types::model::utils::serialize_model_module;
use relayrl_types::prelude::tensor::burn::TensorKind;
use relayrl_types::prelude::tensor::relayrl::DeviceType;

use dashmap::DashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "metrics")]
use std::time::Instant;

use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

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
#[allow(clippy::enum_variant_names)]
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
    LifecycleManagerError(#[from] LifecycleManagerError),
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

pub(crate) trait ClientInterface<
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    async fn update_model(
        &self,
        model: ModelModule<B>,
        actor_ids: Option<Vec<ActorUuid>>,
    ) -> Result<(), CoordinatorError>;
    async fn get_model_version(
        &self,
        ids: Vec<ActorUuid>,
    ) -> Result<Vec<(ActorUuid, i64)>, CoordinatorError>;
    async fn get_trajectory_memory(
        &self,
    ) -> Result<Arc<DashMap<Uuid, Vec<Arc<RelayRLTrajectory>>>>, CoordinatorError>;
    async fn scale_out(&mut self, router_add: u32) -> Result<(), CoordinatorError>;
    async fn scale_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError>;
    async fn get_config(&self) -> Result<ClientConfigLoader, CoordinatorError>;
    async fn set_config_path(&self, config_path: PathBuf) -> Result<(), CoordinatorError>;
}

pub(crate) trait ClientActors<B: Backend + BackendMatcher<Backend = B>> {
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
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), CoordinatorError>;
    async fn set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError>;
}

pub(crate) trait ClientEnvironments<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
>
{
    async fn run_env(&self, actor_id: ActorUuid, step_count: usize)
    -> Result<(), CoordinatorError>;
    async fn set_env(
        &mut self,
        actor_id: ActorUuid,
        env: Box<dyn Environment<B, D_IN, D_OUT, KindIn, KindOut>>,
        count: u32,
    ) -> Result<(), CoordinatorError>;
    async fn remove_env(&mut self, actor_id: ActorUuid) -> Result<(), CoordinatorError>;
    async fn get_env_count(&self, actor_id: ActorUuid) -> Result<u32, CoordinatorError>;
    async fn increase_env_count(
        &mut self,
        actor_id: ActorUuid,
        count: u32,
    ) -> Result<(), CoordinatorError>;
    async fn decrease_env_count(
        &mut self,
        actor_id: ActorUuid,
        count: u32,
    ) -> Result<(), CoordinatorError>;
}

// ===== Coordinator state =====

pub struct CoordinatorParams<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    pub(crate) client_namespace: Arc<str>,
    #[cfg(feature = "metrics")]
    pub(crate) metrics: MetricsManager,
    pub(crate) lifecycle: LifecycleManager,
    pub(crate) shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    pub(crate) scaling: ScaleManager<B, D_IN, D_OUT>,
}

pub struct ClientCoordinator<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
> {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    transport_type: TransportType,
    pub(crate) client_modes: Arc<ClientModes>,
    pub(crate) runtime_params: Option<CoordinatorParams<B, D_IN, D_OUT>>,
    _phantom: PhantomData<(KindIn, KindOut)>,
}

// ===== Internal helpers =====

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
> ClientCoordinator<B, D_IN, D_OUT, KindIn, KindOut>
{
    async fn request_model_versions(
        global_dispatcher_tx: Sender<RoutedMessage>,
        ids: Vec<ActorUuid>,
    ) -> Result<Vec<(Uuid, i64)>, CoordinatorError> {
        let mut versions = Vec::with_capacity(ids.len());

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

    async fn dispatch_model_updates(
        global_dispatcher_tx: Sender<RoutedMessage>,
        target_actor_ids: Vec<ActorUuid>,
        model_bytes: Vec<u8>,
    ) -> Result<(), CoordinatorError> {
        let model_versions =
            Self::request_model_versions(global_dispatcher_tx.clone(), target_actor_ids).await?;

        for (actor_id, current_version) in model_versions {
            let next_version = if current_version < 0 {
                0
            } else {
                current_version + 1
            };
            let model_update_message = RoutedMessage {
                actor_id,
                protocol: RoutingProtocol::ModelUpdate,
                payload: RoutedPayload::ModelUpdate {
                    model_bytes: model_bytes.clone(),
                    version: next_version,
                },
            };

            if let Err(e) = global_dispatcher_tx
                .send(model_update_message)
                .await
                .map_err(|e| e.to_string())
            {
                return Err(CoordinatorError::ScaleManagerError(
                    ScaleManagerError::SendModelUpdateMessageError(e),
                ));
            }
        }

        Ok(())
    }

    async fn prepare_model_update_dispatch(
        &self,
        actor_ids: Option<&[ActorUuid]>,
    ) -> Result<
        Option<(Sender<RoutedMessage>, Vec<ActorUuid>, Arc<RwLock<PathBuf>>)>,
        CoordinatorError,
    > {
        match &self.runtime_params {
            Some(params) => match &self.client_modes.actor_inference_mode {
                ActorInferenceMode::Local(_) => {
                    let local_model_path = params.lifecycle.get_local_model_path();
                    let (global_dispatcher_tx, target_actor_ids) = {
                        let shared_state = params.shared_state.read().await;
                        (
                            shared_state.global_dispatcher_tx.clone(),
                            shared_state.model_update_dispatch_targets_for_subset(actor_ids),
                        )
                    };

                    Ok(Some((
                        global_dispatcher_tx,
                        target_actor_ids,
                        local_model_path,
                    )))
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::ServerOverflow(_, _) => {
                    // Experimental: local-client-triggered model updates are not implemented for
                    // server overflow inference in `0.5.0-beta`.
                    Ok(None)
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::Server(_) => {
                    // Experimental: local-client-triggered model updates are not implemented for
                    // server inference in `0.5.0-beta`.
                    Ok(None)
                }
            },
            None => Err(CoordinatorError::NoRuntimeInstanceError),
        }
    }

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

// ===== Client interface implementation =====

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
> ClientInterface<B, D_IN, D_OUT> for ClientCoordinator<B, D_IN, D_OUT, KindIn, KindOut>
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
            _phantom: PhantomData,
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
        #[cfg(feature = "logging")]
        init_logging();

        let client_namespace: Arc<str> = Arc::from(format!(
            "{}-{}",
            crate::network::CLIENT_NAMESPACE_PREFIX,
            Uuid::new_v4()
        ));

        clear_namespace(client_namespace.as_ref()); // for this agent runtime, ensure no overlapping namespace exists in uuid registry/entire process
        reserve_namespace(client_namespace.as_ref());

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

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let mut config_loader: ClientConfigLoader = ClientConfigLoader::load_config(&config_path);
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        let config_loader: ClientConfigLoader = ClientConfigLoader::load_config(&config_path);

        let lifecycle: LifecycleManager = LifecycleManager::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args.to_owned(),
            &config_loader,
            config_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.transport_type,
        );

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        // if args are set in client mode init config, set lifecycle manager server addresses while keeping unchanged config values
        {
            let inference_address_args =
                if let ActorInferenceMode::Server(server_params)
                | ActorInferenceMode::ServerOverflow(_, server_params) =
                    &shared_client_modes.actor_inference_mode
                {
                    server_params.inference_addresses.clone()
                } else {
                    None
                };

            let training_address_args = match &shared_client_modes.actor_training_data_mode {
                ActorTrainingDataMode::Online(server_params)
                | ActorTrainingDataMode::OnlineWithFiles(server_params, _)
                | ActorTrainingDataMode::OnlineWithMemory(server_params) => {
                    server_params.training_addresses.clone()
                }
                ActorTrainingDataMode::Disabled | ActorTrainingDataMode::OfflineWithFiles(_) => {
                    None
                }
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
                                transport_params_for_packing
                                    .nats_addresses
                                    .inference_server_address = inference_server_address;
                            }
                        }
                        #[cfg(feature = "zmq-transport")]
                        TransportType::ZMQ => {
                            if let Some(inference_server_address) = match inference_addresses {
                                #[cfg(feature = "nats-transport")]
                                InferenceAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                InferenceAddressesArgs::ZMQ(ref params) => {
                                    params.inference_server_address.clone()
                                }
                            } {
                                transport_params_for_packing
                                    .zmq_addresses
                                    .inference_addresses
                                    .inference_server_address = inference_server_address;
                            }

                            if let Some(inference_scaling_server_address) =
                                match inference_addresses {
                                    #[cfg(feature = "nats-transport")]
                                    InferenceAddressesArgs::NATS(_) => None,
                                    #[cfg(feature = "zmq-transport")]
                                    InferenceAddressesArgs::ZMQ(ref params) => {
                                        params.inference_scaling_server_address.clone()
                                    }
                                }
                            {
                                transport_params_for_packing
                                    .zmq_addresses
                                    .inference_addresses
                                    .inference_scaling_server_address =
                                    inference_scaling_server_address;
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
                                transport_params_for_packing
                                    .nats_addresses
                                    .training_server_address = training_server_address;
                            }
                        }
                        #[cfg(feature = "zmq-transport")]
                        TransportType::ZMQ => {
                            if let Some(agent_listener_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => {
                                    params.agent_listener_address.clone()
                                }
                            } {
                                transport_params_for_packing
                                    .zmq_addresses
                                    .training_addresses
                                    .agent_listener_address = agent_listener_address;
                            }

                            if let Some(model_server_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => {
                                    params.model_server_address.clone()
                                }
                            } {
                                transport_params_for_packing
                                    .zmq_addresses
                                    .training_addresses
                                    .model_server_address = model_server_address;
                            }

                            if let Some(trajectory_server_address) = match training_addresses {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => {
                                    params.trajectory_server_address.clone()
                                }
                            } {
                                transport_params_for_packing
                                    .zmq_addresses
                                    .training_addresses
                                    .trajectory_server_address = trajectory_server_address;
                            }

                            if let Some(training_scaling_server_address) = match training_addresses
                            {
                                #[cfg(feature = "nats-transport")]
                                TrainingAddressesArgs::NATS(_) => None,
                                #[cfg(feature = "zmq-transport")]
                                TrainingAddressesArgs::ZMQ(ref params) => {
                                    params.training_scaling_server_address.clone()
                                }
                            } {
                                transport_params_for_packing
                                    .zmq_addresses
                                    .training_addresses
                                    .training_scaling_server_address =
                                    training_scaling_server_address;
                            }
                        }
                    }
                }

                lifecycle
                    .set_transport_addresses(transport_params_for_packing, &self.transport_type)
                    .await?;
            }
        }

        {
            // if args are set in client mode init config, set lifecycle manager trajectory file path
            let local_trajectory_file_params = match &shared_client_modes.actor_training_data_mode {
                ActorTrainingDataMode::OfflineWithFiles(Some(params)) => Some(params),
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorTrainingDataMode::OnlineWithFiles(_, Some(params)) => Some(params),
                _ => None,
            };

            if let Some(file_params) = local_trajectory_file_params {
                lifecycle.set_trajectory_file_path(file_params).await?;
            }
        }

        lifecycle.spawn_loop();

        #[cfg(feature = "metrics")]
        let metrics = {
            let metrics_args = lifecycle.get_metrics_args();
            init_metrics(metrics_args).await
        };

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let (inference_dispatcher, scaling_dispatcher, training_dispatcher) = {
            // Create transport and wrap in Arc for sharing across dispatchers
            let transport: ClientTransportInterface<B> = client_transport_factory(
                self.transport_type,
                client_namespace.clone(),
                shared_client_modes.clone(),
            )
            .await
            .map_err(CoordinatorError::from)?;

            let shared_transport: Arc<ClientTransportInterface<B>> = Arc::new(transport);

            let (inference_dispatcher, mut scaling_dispatcher) =
                match shared_client_modes.actor_inference_mode {
                    ActorInferenceMode::Server(_) | ActorInferenceMode::ServerOverflow(_, _) => (
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
                ActorTrainingDataMode::Disabled | ActorTrainingDataMode::OfflineWithFiles(_) => {
                    None
                }
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
            let shared_transport_addresses = if let ActorInferenceMode::Server(_)
            | ActorInferenceMode::ServerOverflow(_, _) =
                shared_client_modes.actor_inference_mode
            {
                Some(lifecycle.get_transport_addresses())
            } else if let ActorTrainingDataMode::Online(_)
            | ActorTrainingDataMode::OnlineWithFiles(_, _)
            | ActorTrainingDataMode::OnlineWithMemory(_) =
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
                    #[cfg(feature = "metrics")]
                    metrics.clone(),
                )
            };

            let shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                Arc::from(RwLock::new(state));

            let scaling = {
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
                    #[cfg(feature = "metrics")]
                    metrics.clone(),
                    lifecycle.clone(),
                )
                .await
                .map_err(CoordinatorError::from)?
            };

            self.runtime_params = Some(CoordinatorParams {
                client_namespace,
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
            log::warn!(
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
                if let Some(dispatcher) = &params.scaling.scaling_dispatcher {
                    dispatcher.shutdown_transport().await?;
                }

                // the following will trigger shutdown tx/rx for all scalable router nodes in the runtime (router receivers, router senders, central filters)
                // + the single router dispatcher task (the dispatcher informs the actors to shutdown via their inboxes)
                params.lifecycle.shutdown();

                // Ensure all scalable router tasks are drained before state teardown completes.
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
        if self.runtime_params.is_some() {
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

    async fn request_action(
        &self,
        ids: Vec<ActorUuid>,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
        reward: f32,
    ) -> Result<Vec<(ActorUuid, Arc<RelayRLAction>)>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let (start_time, num_ids) = (Instant::now(), ids.len() as u64);

                // Extract router runtime params with clear error messages
                let _router_runtime_params: &dashmap::DashMap<
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
                let (global_dispatcher_tx, valid_ids) = {
                    let shared_state = params.shared_state.read().await;
                    let global_dispatcher_tx = shared_state.global_dispatcher_tx.clone();
                    let valid_ids = ids
                        .into_iter()
                        .filter(|id| {
                            shared_state
                                .shared_router_state
                                .actor_router_addresses
                                .contains_key(id)
                        })
                        .collect::<Vec<_>>();
                    (global_dispatcher_tx, valid_ids)
                };

                let mut pending = Vec::with_capacity(valid_ids.len());

                for id in &valid_ids {
                    let (resp_tx, resp_rx) = oneshot::channel::<Arc<RelayRLAction>>();

                    let action_request_message = RoutedMessage {
                        actor_id: id.clone(),
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

                    pending.push((*id, resp_rx));
                }

                let mut join_set = tokio::task::JoinSet::<
                    Result<(Uuid, Arc<RelayRLAction>), CoordinatorError>,
                >::new();

                for (id, rx) in pending {
                    join_set.spawn(async move {
                        let action = rx.await.map_err(|e| {
                            CoordinatorError::ScaleManagerError(
                                ScaleManagerError::ReceiveActionResponseError(e.to_string()),
                            )
                        })?;
                        Ok::<(Uuid, Arc<RelayRLAction>), CoordinatorError>((id, action))
                    });
                }

                let mut actions: Vec<(Uuid, Arc<RelayRLAction>)> =
                    Vec::with_capacity(valid_ids.len());
                while let Some(join_result) = join_set.join_next().await {
                    let pair = join_result.map_err(|e| {
                        CoordinatorError::ScaleManagerError(
                            ScaleManagerError::ReceiveActionResponseError(e.to_string()),
                        )
                    })??;
                    actions.push(pair);
                }

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("action_request_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("action_requests", num_ids, &[])
                        .await;
                }

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
                #[cfg(feature = "metrics")]
                let (start_time, num_ids) = (Instant::now(), ids.len() as u64);

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
                        payload: RoutedPayload::FlagLastInference {
                            reward,
                            env_id: None,
                            env_label: None,
                        },
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

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("flag_last_action_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("flag_last_action_calls", num_ids, &[])
                        .await;
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

    async fn update_model(
        &self,
        model: ModelModule<B>,
        actor_ids: Option<Vec<ActorUuid>>,
    ) -> Result<(), CoordinatorError> {
        let Some((global_dispatcher_tx, target_actor_ids, local_model_path)) = self
            .prepare_model_update_dispatch(actor_ids.as_deref())
            .await?
        else {
            return Ok(());
        };

        if target_actor_ids.is_empty() {
            return Ok(());
        }

        let serialization_dir = {
            let model_path = local_model_path.read().await.clone();
            model_path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .map(|parent| parent.to_path_buf())
                .unwrap_or_else(std::env::temp_dir)
        };
        std::fs::create_dir_all(&serialization_dir).map_err(|e| {
            CoordinatorError::ConfigError(ClientConfigError::InvalidValue(format!(
                "Failed to create model serialization directory '{}': {}",
                serialization_dir.display(),
                e
            )))
        })?;

        let model_bytes = serialize_model_module(&model, serialization_dir);
        Self::dispatch_model_updates(global_dispatcher_tx, target_actor_ids, model_bytes).await
    }

    async fn get_model_version(
        &self,
        ids: Vec<ActorUuid>,
    ) -> Result<Vec<(Uuid, i64)>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let global_dispatcher_tx = params
                    .shared_state
                    .read()
                    .await
                    .global_dispatcher_tx
                    .clone();
                Self::request_model_versions(global_dispatcher_tx, ids).await
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to get_model_version...".to_string(),
                ),
            )),
        }
    }

    async fn get_trajectory_memory(
        &self,
    ) -> Result<Arc<DashMap<Uuid, Vec<Arc<RelayRLTrajectory>>>>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                if let Some(mut shared_trajectory_memory) =
                    params.scaling.shared_trajectory_memory.clone()
                {
                    Ok(std::mem::replace(
                        &mut shared_trajectory_memory,
                        Arc::new(DashMap::new()),
                    ))
                } else {
                    Err(CoordinatorError::ScaleManagerError(
                        ScaleManagerError::TrajectoryMemoryNotFoundError(
                            "[Coordinator] Trajectory memory not found".to_string(),
                        ),
                    ))
                }
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to get_trajectory_memory...".to_string(),
                ),
            )),
        }
    }

    async fn scale_out(&mut self, router_add: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let result = {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    {
                        params
                            .scaling
                            .scale_out(router_add, true)
                            .await
                            .map_err(CoordinatorError::from)
                    }

                    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                    {
                        params
                            .scaling
                            .scale_out(router_add)
                            .await
                            .map_err(CoordinatorError::from)
                    }
                };

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("scale_out_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("scale_out_calls", 1, &[])
                        .await;
                }

                result
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
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let result = {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    {
                        params
                            .scaling
                            .scale_in(router_remove, true)
                            .await
                            .map_err(CoordinatorError::from)
                    }

                    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                    {
                        params
                            .scaling
                            .scale_in(router_remove)
                            .await
                            .map_err(CoordinatorError::from)
                    }
                };

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("scale_in_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("scale_in_calls", 1, &[])
                        .await;
                }

                result
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

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
> ClientActors<B> for ClientCoordinator<B, D_IN, D_OUT, KindIn, KindOut>
{
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
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let actor_id: Uuid = reserve_id_with(
                    params.client_namespace.as_ref(),
                    crate::network::ACTOR_CONTEXT,
                    117,
                    100,
                )
                .map_err(CoordinatorError::from)?;

                #[cfg(feature = "metrics")]
                params
                    .metrics
                    .record_counter("actors_created", 1, &[])
                    .await;

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

                let actor_count: usize = params
                    .shared_state
                    .read()
                    .await
                    .shared_router_state
                    .actor_inboxes
                    .len();
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
                        actor_id,
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

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("new_actor_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("new_actor_calls", 1, &[])
                        .await;
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
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                params
                    .shared_state
                    .write()
                    .await
                    .remove_actor(id)
                    .map_err(CoordinatorError::from)?;

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

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("remove_actor_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("remove_actor_calls", 1, &[])
                        .await;
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
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

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

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("set_actor_id_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("set_actor_id_calls", 1, &[])
                        .await;
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
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync + 'static,
    KindOut: TensorKind<B> + BasicOps<B> + Send + Sync + 'static,
> ClientEnvironments<B, D_IN, D_OUT, KindIn, KindOut>
    for ClientCoordinator<B, D_IN, D_OUT, KindIn, KindOut>
{
    async fn run_env(
        &self,
        actor_id: ActorUuid,
        step_count: usize,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .write()
                    .await
                    .run_env(actor_id, step_count)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::StepEnvError(
                    "[Coordinator] No runtime instance to step_env...".to_string(),
                ),
            )),
        }
    }

    async fn set_env(
        &mut self,
        actor_id: ActorUuid,
        env: Box<dyn Environment<B, D_IN, D_OUT, KindIn, KindOut>>,
        count: u32,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .write()
                    .await
                    .set_env(actor_id, env, count)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetEnvError(
                    "[Coordinator] No runtime instance to set_env...".to_string(),
                ),
            )),
        }
    }

    async fn get_env_count(&self, actor_id: ActorUuid) -> Result<u32, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => params
                .shared_state
                .read()
                .await
                .get_env_count(actor_id)
                .map_err(CoordinatorError::from),
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetEnvCountError(
                    "[Coordinator] No runtime instance to get_env_count...".to_string(),
                ),
            )),
        }
    }

    async fn increase_env_count(
        &mut self,
        actor_id: ActorUuid,
        count: u32,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .write()
                    .await
                    .increase_env_count(actor_id, count)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::IncreaseEnvCountError(
                    "[Coordinator] No runtime instance to increase_env_count...".to_string(),
                ),
            )),
        }
    }

    async fn decrease_env_count(
        &mut self,
        actor_id: ActorUuid,
        count: u32,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .write()
                    .await
                    .decrease_env_count(actor_id, count)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::DecreaseEnvCountError(
                    "[Coordinator] No runtime instance to decrease_env_count...".to_string(),
                ),
            )),
        }
    }

    async fn remove_env(&mut self, actor_id: ActorUuid) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.shared_state.write().await.remove_env(actor_id)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::RemoveEnvError(
                    "[Coordinator] No runtime instance to remove_env...".to_string(),
                ),
            )),
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    use crate::network::client::agent::InferenceParams;
    use crate::network::client::agent::{
        ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode,
    };
    use crate::network::client::runtime::coordination::lifecycle_manager::LifecycleManager;
    use crate::utilities::configuration::ClientConfigLoader;
    use active_uuid_registry::interface::{clear_namespace, reserve_namespace};
    use active_uuid_registry::registry_uuid::Uuid;
    use burn_ndarray::NdArray;
    use relayrl_types::data::tensor::DeviceType;
    use std::path::PathBuf;
    use burn_tensor::Float;
    use tokio::sync::mpsc::{self, error::TryRecvError};

    type TestBackend = NdArray<f32>;
    type TestKind = Float;

    fn make_coordinator() -> ClientCoordinator<TestBackend, 4, 1, TestKind, TestKind> {
        ClientCoordinator::<TestBackend, 4, 1, TestKind, TestKind>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            ClientModes::default(),
        )
    }

    fn make_lifecycle_manager() -> LifecycleManager {
        use std::io::Write;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(tmp, "{{}}").expect("write temp config");
        let config = ClientConfigLoader::load_config(&tmp.path().to_path_buf());
        let lifecycle = LifecycleManager::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            AlgorithmArgs::default(),
            &config,
            tmp.path().to_path_buf(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
        );
        drop(tmp);
        lifecycle
    }

    #[cfg(feature = "metrics")]
    fn test_metrics() -> MetricsManager {
        MetricsManager::new(
            Arc::new(RwLock::new(("test-coordinator".to_string(), String::new()))),
            ("test-coordinator".to_string(), String::new()),
            None,
        )
    }

    async fn make_runtime_coordinator(
        client_modes: ClientModes,
    ) -> (
        ClientCoordinator<TestBackend, 4, 1, TestKind, TestKind>,
        Arc<RwLock<StateManager<TestBackend, 4, 1>>>,
        tokio::sync::mpsc::Receiver<RoutedMessage>,
    ) {
        let client_namespace: Arc<str> = Arc::from(format!("test-coordinator-{}", Uuid::new_v4()));
        clear_namespace(client_namespace.as_ref());
        reserve_namespace(client_namespace.as_ref());

        let lifecycle = make_lifecycle_manager();
        *lifecycle.get_local_model_path().write().await = PathBuf::new();
        let shared_client_modes = Arc::new(client_modes.clone());
        let (state, global_dispatcher_rx) = StateManager::<TestBackend, 4, 1>::new(
            client_namespace.clone(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            shared_client_modes.clone(),
            lifecycle.get_max_traj_length(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            lifecycle.get_local_model_path(),
            None,
            #[cfg(feature = "metrics")]
            test_metrics(),
        );
        let shared_state = Arc::new(RwLock::new(state));
        let (dummy_tx, dummy_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        let scaling = ScaleManager::new(
            client_namespace.clone(),
            shared_client_modes,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            lifecycle.get_algorithm_args(),
            shared_state.clone(),
            dummy_rx,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(feature = "metrics")]
            test_metrics(),
            lifecycle.clone(),
        )
        .await
        .unwrap();
        drop(dummy_tx);

        let mut coordinator = ClientCoordinator::<TestBackend, 4, 1, TestKind, TestKind>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            client_modes,
        );
        coordinator.runtime_params = Some(CoordinatorParams {
            client_namespace,
            #[cfg(feature = "metrics")]
            metrics: test_metrics(),
            lifecycle,
            shared_state: shared_state.clone(),
            scaling,
        });

        (coordinator, shared_state, global_dispatcher_rx)
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
        let result = c
            .remove_actor(
                Uuid::new_v4(),
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                false,
            )
            .await;
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
    async fn prepare_model_update_dispatch_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.prepare_model_update_dispatch(None).await;
        assert!(result.is_err());
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[tokio::test]
    async fn prepare_model_update_dispatch_server_mode_returns_none() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Server(InferenceParams::default()),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        };
        let (coordinator, _shared_state, mut global_dispatcher_rx) =
            make_runtime_coordinator(client_modes).await;

        let result = coordinator.prepare_model_update_dispatch(None).await;

        assert!(matches!(result, Ok(None)));
        assert!(matches!(
            global_dispatcher_rx.try_recv(),
            Err(TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn prepare_model_update_dispatch_subset_filters_requested_actor_ids() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        };
        let (coordinator, shared_state, _global_dispatcher_rx) =
            make_runtime_coordinator(client_modes).await;
        let actor_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let unknown_actor_id = Uuid::new_v4();

        {
            let mut shared_state = shared_state.write().await;
            let (tx_to_buffer, _buffer_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
            for actor_id in &actor_ids {
                shared_state
                    .new_actor(
                        *actor_id,
                        Arc::from("router-a"),
                        DeviceType::Cpu,
                        None,
                        tx_to_buffer.clone(),
                    )
                    .await
                    .unwrap();
            }
        }

        let requested_actor_ids = vec![actor_ids[2], unknown_actor_id, actor_ids[0], actor_ids[2]];
        let (_global_dispatcher_tx, target_actor_ids, _local_model_path) = coordinator
            .prepare_model_update_dispatch(Some(&requested_actor_ids))
            .await
            .unwrap()
            .unwrap();

        let mut expected_target_actor_ids = vec![actor_ids[0], actor_ids[2]];
        expected_target_actor_ids.sort_by_key(|actor_id| actor_id.to_string());

        assert_eq!(target_actor_ids, expected_target_actor_ids);
    }

    #[tokio::test]
    async fn dispatch_model_updates_sends_expected_targets_and_versions() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        };
        let (coordinator, shared_state, mut global_dispatcher_rx) =
            make_runtime_coordinator(client_modes).await;
        let actor_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let current_versions = vec![
            (actor_ids[0], 0_i64),
            (actor_ids[1], 4_i64),
            (actor_ids[2], -1_i64),
        ];

        {
            let mut shared_state = shared_state.write().await;
            let (tx_to_buffer, _buffer_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
            for actor_id in &actor_ids {
                shared_state
                    .new_actor(
                        *actor_id,
                        Arc::from("router-a"),
                        DeviceType::Cpu,
                        None,
                        tx_to_buffer.clone(),
                    )
                    .await
                    .unwrap();
            }
        }

        let (captured_updates_tx, captured_updates_rx) =
            oneshot::channel::<Vec<(Uuid, i64, usize)>>();
        let expected_update_count = actor_ids.len();
        tokio::spawn(async move {
            let mut captured_updates = Vec::new();

            while let Some(message) = global_dispatcher_rx.recv().await {
                match message.payload {
                    RoutedPayload::ModelVersion { reply_to } => {
                        let current_version = current_versions
                            .iter()
                            .find(|(actor_id, _)| *actor_id == message.actor_id)
                            .map(|(_, version)| *version)
                            .unwrap();
                        let _ = reply_to.send(current_version);
                    }
                    RoutedPayload::ModelUpdate {
                        model_bytes,
                        version,
                    } => {
                        captured_updates.push((message.actor_id, version, model_bytes.len()));
                        if captured_updates.len() == expected_update_count {
                            let _ = captured_updates_tx.send(captured_updates);
                            break;
                        }
                    }
                    _ => {}
                }
            }
        });

        let (global_dispatcher_tx, target_actor_ids, _local_model_path) = coordinator
            .prepare_model_update_dispatch(None)
            .await
            .unwrap()
            .unwrap();
        ClientCoordinator::<TestBackend, 4, 1, TestKind, TestKind>::dispatch_model_updates(
            global_dispatcher_tx,
            target_actor_ids,
            vec![1, 2, 3],
        )
        .await
        .unwrap();
        let mut captured_updates = captured_updates_rx.await.unwrap();
        captured_updates.sort_by_key(|(actor_id, _, _)| actor_id.to_string());

        let mut expected_updates = vec![
            (actor_ids[0], 1_i64),
            (actor_ids[1], 5_i64),
            (actor_ids[2], 0_i64),
        ];
        expected_updates.sort_by_key(|(actor_id, _)| actor_id.to_string());

        assert_eq!(captured_updates.len(), expected_updates.len());
        for ((actor_id, version, model_bytes_len), (expected_actor_id, expected_version)) in
            captured_updates.iter().zip(expected_updates.iter())
        {
            assert_eq!(actor_id, expected_actor_id);
            assert_eq!(version, expected_version);
            assert!(
                *model_bytes_len > 0,
                "serialized model bytes should not be empty"
            );
        }
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
