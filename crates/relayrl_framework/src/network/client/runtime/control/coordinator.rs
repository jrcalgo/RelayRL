//! Client runtime coordinator.
//!
//! This module owns top-level orchestration for the client runtime: configuration loading,
//! lifecycle management, actor state, router scaling, and the public request path exposed through
//! `RelayRLAgent`.

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportMode;
use crate::network::client::agent::{ActorDataMode, ActorInferenceMode, ActorInfo, ClientModes};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::agent::{
    AlgorithmInitArgs, DefaultHyperparameterArgs, InferenceAddressesArgs, TrainingAddressesArgs,
};
use crate::network::client::runtime::actor::{
    ActorDTypes, ActorError, ActorShape, ErasedActorRuntime,
};
use crate::network::client::runtime::control::lifecycle_manager::{
    LifecycleManager, LifecycleManagerError,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::control::scale_manager::ProcessInitFlag;
use crate::network::client::runtime::control::scale_manager::RouterNamespace;
use crate::network::client::runtime::control::scale_manager::{ScaleManager, ScaleManagerError};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::control::state_manager::SharedRouterState;
use crate::network::client::runtime::control::state_manager::{ActorUuid, NameTag};
use crate::network::client::runtime::control::state_manager::{StateManager, StateManagerError};
use crate::network::client::runtime::data::router::{
    ControlPayload, RoutedMessage, RoutingProtocol,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::router::{DataPayload, InferenceRequest};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::sinks::transport_sink::transport_dispatcher::{
    InferenceDispatcher, ScalingDispatcher, TrainingDispatcher,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::sinks::transport_sink::{
    ClientTransportInterface, TransportError, client_transport_factory,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::TransportConfigParams;
use crate::utilities::configuration::{ClientConfigLoader, DEFAULT_CLIENT_CONFIG_PATH};
use crate::utilities::observability::logging::*;
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::*;

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::{get_context_entries, get_namespace_entries};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::{ContextString, NamespaceString};

use thiserror::Error;

use burn_tensor::backend::Backend;

use active_uuid_registry::interface::reserve_owned_namespace;
use active_uuid_registry::{OwnedNamespace, UuidPoolError, registry_uuid::Uuid};
use relayrl_algorithms::prelude::nn::NeuralNetwork;
use relayrl_algorithms::prelude::ppo::trainer::PPOTrainerSpec;
use relayrl_env_trait::traits::Environment;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::utils::serialize_model_module;
use relayrl_types::model::{ModelMetadata, ModelModule};
use relayrl_types::prelude::tensor::burn::{
    BasicOps, Bool, Float, Int, Numeric, Tensor, TensorKind,
};
use relayrl_types::prelude::tensor::relayrl::{
    BoolBurnTensor, DType, DeviceType, FloatBurnTensor, IntBurnTensor,
};

use dashmap::DashMap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "metrics")]
use std::time::Instant;

use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

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
    #[error(
        "Actor {actor_id} dimension mismatch: expected D_IN={expected_d_in}, D_OUT={expected_d_out}; actual D_IN={actual_d_in}, D_OUT={actual_d_out}"
    )]
    ActorShapeMismatch {
        actor_id: ActorUuid,
        expected_d_in: usize,
        expected_d_out: usize,
        actual_d_in: usize,
        actual_d_out: usize,
    },
    #[error(
        "dimension mismatch: expected D_IN={expected_d_in}, D_OUT={expected_d_out}; actual D_IN={actual_d_in}, D_OUT={actual_d_out}"
    )]
    ModelShapeMismatch {
        expected_d_in: usize,
        expected_d_out: usize,
        actual_d_in: usize,
        actual_d_out: usize,
    },
}

pub trait ToAnyBurnTensor<B: Backend + BackendMatcher<Backend = B>, const D: usize> {
    fn to_any_burn_tensor(self, dtype: DType) -> AnyBurnTensor<B, D>;
}

impl<B: Backend + BackendMatcher<Backend = B>, const D: usize> ToAnyBurnTensor<B, D>
    for Tensor<B, D, Float>
{
    fn to_any_burn_tensor(self, dtype: DType) -> AnyBurnTensor<B, D> {
        AnyBurnTensor::Float(FloatBurnTensor {
            tensor: Arc::new(self),
            dtype,
        })
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D: usize> ToAnyBurnTensor<B, D>
    for Tensor<B, D, Int>
{
    fn to_any_burn_tensor(self, dtype: DType) -> AnyBurnTensor<B, D> {
        AnyBurnTensor::Int(IntBurnTensor {
            tensor: Arc::new(self),
            dtype,
        })
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D: usize> ToAnyBurnTensor<B, D>
    for Tensor<B, D, Bool>
{
    fn to_any_burn_tensor(self, dtype: DType) -> AnyBurnTensor<B, D> {
        AnyBurnTensor::Bool(BoolBurnTensor {
            tensor: Arc::new(self),
            dtype,
        })
    }
}

/// Drained trajectory snapshots are keyed by stable [`ActorUuid`] rather than by [`ActorInfo`]:
/// callers select actors to drain via `&[ActorInfo]`, but the returned map is a one-shot copy that
/// must not be affected by a concurrent id rename on the live actor handle.
pub(crate) type DrainedCacheResult =
    Result<Option<HashMap<ActorUuid, Vec<Arc<RelayRLTrajectory>>>>, CoordinatorError>;

pub(crate) trait ClientInterface<B: Backend + BackendMatcher<Backend = B>>:
    ClientStart<B>
{
    fn new(
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: TransportMode,
        client_modes: ClientModes,
    ) -> Self
    where
        Self: Sized;
    #[allow(clippy::too_many_arguments)]
    async fn start(
        &mut self,
        data_routers: u32,
        data_buffer_size: usize,
        default_model: Option<ModelModule<B>>,
        config_path: Option<PathBuf>,
        config_polling_seconds: Option<u64>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
    ) -> Result<(), CoordinatorError>;
    async fn shutdown(&mut self) -> DrainedCacheResult;
    #[allow(clippy::too_many_arguments)]
    async fn restart(
        &mut self,
        data_routers: u32,
        data_buffer_size: usize,
        default_model: Option<ModelModule<B>>,
        config_path: Option<PathBuf>,
        config_polling_seconds: Option<u64>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
    ) -> Result<(), CoordinatorError>;
    async fn request_actions<
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B> + 'static,
        KindOut: TensorKind<B> + 'static,
    >(
        &self,
        actors: &[ActorInfo],
        observation: Tensor<B, D_IN, KindIn>,
        mask: Option<Tensor<B, D_OUT, KindOut>>,
        reward: f32,
    ) -> Result<Vec<(ActorInfo, Arc<RelayRLAction>)>, CoordinatorError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>;
    async fn flag_last_actions(
        &self,
        actors: &[ActorInfo],
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError>;
    async fn scale_routers_out(&mut self, router_add: u32) -> Result<(), CoordinatorError>;
    async fn scale_routers_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError>;
    async fn scale_data_buffers(&mut self, new_size: usize) -> Result<(), CoordinatorError>;
    async fn get_config(&self) -> Result<ClientConfigLoader, CoordinatorError>;
    async fn set_config_path(&self, config_path: PathBuf) -> Result<(), CoordinatorError>;
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) trait ClientStart<B: Backend + BackendMatcher<Backend = B>>:
    LifecycleStart<B> + TransportStart<B> + CoreRuntimeStart<B>
{
}

#[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
pub(crate) trait ClientStart<B: Backend + BackendMatcher<Backend = B>>:
    LifecycleStart<B> + CoreRuntimeStart<B>
{
}

pub(crate) trait LifecycleStart<B: Backend + BackendMatcher<Backend = B>> {
    fn build_lifecycle_manager(
        &mut self,
        config_path: Option<PathBuf>,
        config_update_polling_seconds: Option<u64>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
    ) -> Result<(LifecycleManager, ClientConfigLoader), CoordinatorError>;
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn set_lifecycle_addresses(
        &self,
        lifecycle: &LifecycleManager,
        shared_client_modes: &Arc<ClientModes>,
        config_loader: &mut ClientConfigLoader,
    ) -> Result<(), CoordinatorError>;
    async fn set_lifecycle_traj_file_path(
        lifecycle: &LifecycleManager,
        shared_client_modes: &Arc<ClientModes>,
    ) -> Result<(), CoordinatorError>;
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
type TransportDispatchers<B> = (
    Option<Arc<InferenceDispatcher<B>>>,
    Option<Arc<ScalingDispatcher<B>>>,
    Option<Arc<TrainingDispatcher<B>>>,
);

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) trait TransportStart<B: Backend + BackendMatcher<Backend = B>> {
    async fn build_transport_dispatchers(
        &self,
        shared_client_modes: &Arc<ClientModes>,
        client_namespace: &ClientNamespace,
    ) -> Result<TransportDispatchers<B>, CoordinatorError>;
}

pub(crate) trait CoreRuntimeStart<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(feature = "metrics")]
    async fn initialize_metrics(lifecycle: &LifecycleManager) -> MetricsManager;
    async fn build_state_and_scale_managers(
        &mut self,
        client_namespace: ClientNamespace,
        shared_client_modes: Arc<ClientModes>,
        lifecycle: LifecycleManager,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        scaling_dispatcher: Option<Arc<ScalingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        default_model: Option<ModelModule<B>>,
        data_buffer_size: usize,
    ) -> Result<(), CoordinatorError>;
    async fn set_inference_path(&mut self);
    async fn initialize_data_routers(&mut self, data_routers: u32) -> Result<(), CoordinatorError>;
}

pub(crate) trait ClientActors<B: Backend + BackendMatcher<Backend = B>> {
    async fn new_actor<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        device: DeviceType,
        max_traj_length: usize,
        nametag: Option<NameTag>,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        algorithm_args: AlgorithmInitArgs,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_id: bool,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        send_algorithm_init: bool,
    ) -> Result<ActorInfo, CoordinatorError>;
    async fn remove_actor(
        &mut self,
        actor: &ActorInfo,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), CoordinatorError>;
    async fn resolve_new_nametag(
        &self,
        nametag: Option<&str>,
        actor_count: u32,
    ) -> Result<Option<Vec<NameTag>>, CoordinatorError>;
    async fn get_actor(&self, id: ActorUuid) -> Result<ActorInfo, CoordinatorError>;
    async fn get_all_actors(&self) -> Result<Vec<ActorInfo>, CoordinatorError>;
    async fn get_actors_by_rank<const D_IN: usize, const D_OUT: usize>(
        &self,
    ) -> Result<Vec<ActorInfo>, CoordinatorError>;
    async fn get_actors_by_tag(
        &self,
        nametag: Option<&str>,
    ) -> Result<Vec<ActorInfo>, CoordinatorError>;
    async fn set_actor_id(
        &mut self,
        actor: &ActorInfo,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError>;
    async fn set_actor_nametag(
        &mut self,
        actor: &ActorInfo,
        new_nametag: Option<&str>,
    ) -> Result<(), CoordinatorError>;
    async fn update_models<const D_IN: usize, const D_OUT: usize>(
        &self,
        specific_actors: Option<&[ActorInfo]>,
        model: ModelModule<B>,
    ) -> Result<(), CoordinatorError>;
    async fn get_model_versions(
        &self,
        actors: &[ActorInfo],
    ) -> Result<Vec<(ActorInfo, i64)>, CoordinatorError>;
    fn drain_trajectory_caches(&self, actors: &[ActorInfo]) -> DrainedCacheResult;
}

pub(crate) trait ClientEnvironments<B: Backend + BackendMatcher<Backend = B>> {
    async fn run_env_eval(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
    ) -> Result<(), CoordinatorError>;
    async fn run_env_with_ppo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Clone + Send + 'static,
    >(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, CoordinatorError>
    where
        B: Default + Send + Sync + 'static;
    /// TODO: implement this :)
    #[allow(unused)]
    async fn run_env_with_ippo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    >(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, CoordinatorError>
    where
        B: Default + Send + Sync + 'static;
    /// TODO: implement this :)
    #[allow(unused)]
    async fn run_env_with_mappo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    >(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, CoordinatorError>
    where
        B: Default + Send + Sync + 'static;
    async fn set_env(
        &mut self,
        actor: &ActorInfo,
        env: Box<dyn Environment>,
        count: u32,
    ) -> Result<(), CoordinatorError>;
    async fn remove_env(&mut self, actor: &ActorInfo) -> Result<(), CoordinatorError>;
    async fn get_env_count(&self, actor: &ActorInfo) -> Result<u32, CoordinatorError>;
    async fn increase_env_count(
        &mut self,
        actor: &ActorInfo,
        count: u32,
    ) -> Result<(), CoordinatorError>;
    async fn decrease_env_count(
        &mut self,
        actor: &ActorInfo,
        count: u32,
    ) -> Result<(), CoordinatorError>;
}

// ===== Coordinator state =====

pub(crate) enum InferencePathParams<B: Backend + BackendMatcher<Backend = B>> {
    Local {
        local_runtimes: Arc<DashMap<ActorUuid, Arc<dyn ErasedActorRuntime<B>>>>,
    },
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    Network {
        filter_channels: Arc<DashMap<RouterNamespace, Sender<RoutedMessage>>>,
        shared_router_state: Arc<SharedRouterState>,
        global_dispatcher_tx: Sender<RoutedMessage>,
    },
}

pub struct CoordinatorParams<B: Backend + BackendMatcher<Backend = B>> {
    pub(crate) client_namespace: ClientNamespace,
    #[cfg(feature = "metrics")]
    pub(crate) metrics: MetricsManager,
    pub(crate) lifecycle: LifecycleManager,
    pub(crate) shared_state: Arc<RwLock<StateManager<B>>>,
    pub(crate) scaling: ScaleManager<B>,
}

pub struct ClientCoordinator<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    transport_type: TransportMode,
    pub(crate) client_modes: Arc<ClientModes>,
    pub(crate) runtime_params: Option<CoordinatorParams<B>>,
    inference_path_params: Option<InferencePathParams<B>>,
}

// ===== Internal helpers =====

impl<B: Backend + BackendMatcher<Backend = B>> ClientCoordinator<B> {
    async fn request_model_versions(
        global_dispatcher_tx: Sender<RoutedMessage>,
        actors: &[ActorInfo],
    ) -> Result<Vec<(ActorInfo, i64)>, CoordinatorError> {
        let mut versions = Vec::with_capacity(actors.len());

        for actor in actors {
            let (resp_tx, resp_rx) = oneshot::channel::<i64>();

            let model_version_message = RoutedMessage {
                actor_id: actor.id(),
                protocol: RoutingProtocol::Control(ControlPayload::ModelVersion {
                    reply_to: resp_tx,
                }),
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
                Ok(model_version) => versions.push((actor.clone(), model_version)),
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
        target_actors: &[ActorInfo],
        model_bytes: Vec<u8>,
    ) -> Result<(), CoordinatorError> {
        let model_versions =
            Self::request_model_versions(global_dispatcher_tx.clone(), target_actors).await?;

        for (actor_info, current_version) in model_versions {
            let next_version = if current_version < 0 {
                0
            } else {
                current_version + 1
            };
            let model_update_message = RoutedMessage {
                actor_id: actor_info.id(),
                protocol: RoutingProtocol::Control(ControlPayload::ModelUpdate {
                    model_bytes: model_bytes.clone(),
                    version: next_version,
                }),
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

    async fn prepare_model_update_dispatch<const D_IN: usize, const D_OUT: usize>(
        &self,
        actors: Option<&[ActorInfo]>,
        metadata: &ModelMetadata,
    ) -> Result<
        Option<(Sender<RoutedMessage>, Vec<ActorInfo>, Arc<RwLock<PathBuf>>)>,
        CoordinatorError,
    > {
        match &self.runtime_params {
            Some(params) => match &self.client_modes.actor_inference_mode {
                ActorInferenceMode::Client(_) => {
                    let local_model_path = params.lifecycle.get_local_model_path();
                    let valid_actors = self
                        .verify_model_ranks_against_actors::<D_IN, D_OUT>(actors, metadata)
                        .await?;

                    if valid_actors.is_some() {
                        let (global_dispatcher_tx, target_actors) = {
                            let shared_state = params.shared_state.read().await;
                            (
                                shared_state.global_dispatcher_tx.clone(),
                                shared_state.model_update_dispatch_targets_for_subset(
                                    valid_actors.as_deref(),
                                ),
                            )
                        };

                        Ok(Some((
                            global_dispatcher_tx,
                            target_actors,
                            local_model_path,
                        )))
                    } else {
                        Ok(None)
                    }
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::ClientFallback(_, _) => {
                    // Experimental: local-client-triggered model updates are not implemented for
                    // server overflow inference in `0.5.0`.
                    Ok(None)
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::Server(_) => {
                    // Experimental: local-client-triggered model updates are not implemented for
                    // server inference in `0.5.0`.
                    Ok(None)
                }
            },
            None => Err(CoordinatorError::NoRuntimeInstanceError),
        }
    }

    async fn verify_model_ranks_against_actors<const D_IN: usize, const D_OUT: usize>(
        &self,
        actors: Option<&[ActorInfo]>,
        metadata: &ModelMetadata,
    ) -> Result<Option<Vec<ActorInfo>>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                {
                    let model_ranks = (metadata.input_shape.len(), metadata.output_shape.len());
                    if model_ranks.0 != D_IN || model_ranks.1 != D_OUT {
                        return Err(CoordinatorError::ModelShapeMismatch {
                            expected_d_in: D_IN,
                            expected_d_out: D_OUT,
                            actual_d_in: model_ranks.0,
                            actual_d_out: model_ranks.1,
                        });
                    }
                }

                let actor_runtimes = &params.shared_state.read().await.actor_runtime_handles;

                Ok(match actors {
                    Some(infos) => {
                        // Skip actors not in the registry (they may have been removed),
                        // but return None (abort) if a known actor has mismatched ranks.
                        let collected = infos.iter()
                            .filter_map(|actor| {
                                let runtime_entry = actor_runtimes.get(&actor.id());
                                match runtime_entry {
                                    Some(entry) if entry.actor_shape().d_in != D_IN || entry.actor_shape().d_out != D_OUT => {
                                        log::error!("[Coordinator] Actor {}'s ranks did not match rank generics: Actor ({}, {}), Generics ({}, {})", actor.id(), entry.actor_shape().d_in, entry.actor_shape().d_out, D_IN, D_OUT);
                                        None
                                    }
                                    Some(_) => Some(actor.clone()),
                                    None => None, // silently skip
                                }
                            })
                            .collect::<Vec<ActorInfo>>();

                        if collected.is_empty() {
                            None
                        } else {
                            Some(collected)
                        }
                    }
                    None => {
                        let mut collected = Vec::new();
                        for runtime_entry in actor_runtimes.iter() {
                            let actor_shape = runtime_entry.actor_shape();
                            if actor_shape.d_in != D_IN || actor_shape.d_out != D_OUT {
                                log::error!(
                                    "[Coordinator] Actor {}'s ranks did not match rank generics: Actor ({}, {}), Generics ({}, {})",
                                    runtime_entry.key(),
                                    actor_shape.d_in,
                                    actor_shape.d_out,
                                    D_IN,
                                    D_OUT
                                );
                                return Ok(None);
                            }
                            match runtime_entry.get_actor_info() {
                                Ok(actor_info) => collected.push(actor_info),
                                Err(e) => {
                                    log::error!("{}", e);
                                }
                            }
                        }
                        if collected.is_empty() {
                            None
                        } else {
                            Some(collected)
                        }
                    }
                })
            }
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
        algorithm_args: AlgorithmInitArgs,
    ) -> Result<(), CoordinatorError> {
        match self.runtime_params.as_mut() {
            Some(params) => params
                .scaling
                .send_process_init_request(
                    actor_entries,
                    ProcessInitFlag::<B>::TrainingAlgorithmInit(algorithm_args),
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

pub(crate) struct ClientNamespace {
    handle: OwnedNamespace,
    namespace: Arc<str>,
}

impl ClientNamespace {
    pub(crate) fn new(handle: OwnedNamespace, namespace: Arc<str>) -> Self {
        Self { handle, namespace }
    }

    /// Returns a cheap clone of the underlying namespace string for read-only/logging APIs.
    pub(crate) fn as_arc(&self) -> Arc<str> {
        self.namespace.clone()
    }

    pub(crate) fn reserve_id(&self, context: &str) -> Result<Uuid, UuidPoolError> {
        self.handle.reserve_id(context)
    }

    pub(crate) fn reserve_id_with(
        &self,
        context: &str,
        base: u32,
        max_retries: usize,
    ) -> Result<Uuid, UuidPoolError> {
        self.handle.reserve_id_with(context, base, max_retries)
    }

    pub(crate) fn add_id(&self, context: &str, id: Uuid) -> Result<(), UuidPoolError> {
        self.handle.add_id(context, id)
    }

    pub(crate) fn remove_id(&self, context: &str, id: Uuid) -> Result<(), UuidPoolError> {
        self.handle.remove_id(context, id)
    }

    pub(crate) fn replace_id(
        &self,
        context: &str,
        old: Uuid,
        new: Uuid,
    ) -> Result<(), UuidPoolError> {
        self.handle.replace_id(context, old, new)
    }

    /// Explicitly releases ownership of the namespace via the stored handle.
    pub(crate) fn remove(self) -> Result<(), UuidPoolError> {
        self.handle.remove()
    }
}

impl Clone for ClientNamespace {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            namespace: self.namespace.clone(),
        }
    }
}

impl AsRef<str> for ClientNamespace {
    fn as_ref(&self) -> &str {
        &self.namespace
    }
}

impl std::fmt::Display for ClientNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.namespace)
    }
}

// ===== Client interface implementation =====

impl<B: Backend + BackendMatcher<Backend = B>> ClientInterface<B> for ClientCoordinator<B> {
    fn new(
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: TransportMode,
        client_modes: ClientModes,
    ) -> Self {
        Self {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_type,
            client_modes: Arc::new(client_modes),
            runtime_params: None,
            inference_path_params: None,
        }
    }

    async fn start(
        &mut self,
        data_routers: u32,
        data_buffer_size: usize,
        default_model: Option<ModelModule<B>>,
        config_path: Option<PathBuf>,
        config_polling_seconds: Option<u64>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
    ) -> Result<(), CoordinatorError> {
        // initializes with default settings, used when caller wants the `RelayRLAgent` to initialize the log4rs logging backend
        #[cfg(feature = "logging-init")]
        init_logging();

        // initialization and reservation process for UUID registry (used for component tracking internally and as a source of truth for distributed consistency).
        // the public api caller will interact only with actor-related UUIDs in this namespace.
        // the returned `ClientNamespace` keeps the `OwnedNamespace` handle alive for the full
        // runtime lifetime, so subsequent writes into this namespace must go through it.
        let client_namespace: ClientNamespace = {
            let mut namespace = format!(
                "{}-{}",
                crate::network::CLIENT_NAMESPACE_PREFIX,
                Uuid::new_v4()
            );
            // for this agent runtime, ensure no overlapping namespace exists in uuid registry/entire process
            let namespace_handle = {
                loop {
                    match reserve_owned_namespace(&namespace) {
                        Ok(handle) => break handle,
                        Err(e) => {
                            log::error!(
                                "[Coordinator] Failed to reserve namespace {}: {}",
                                namespace,
                                e
                            );
                            log::info!(
                                "[Coordinator] Retrying to reserve namespace as {}_#: {}",
                                namespace,
                                e
                            );

                            namespace = format!("{}_#", namespace);
                        }
                    }
                }
            };

            ClientNamespace::new(namespace_handle, Arc::from(namespace))
        };

        // shared across runtime components for internal consistency
        let shared_client_modes: Arc<ClientModes> = self.client_modes.clone();

        // builds `LifeCycleManager` in-memory
        let (lifecycle, mut _config_loader) = self.build_lifecycle_manager(
            config_path,
            config_polling_seconds,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters,
        )?;

        // if args are set in client mode init config, set lifecycle manager server addresses while keeping unchanged config values
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        self.set_lifecycle_addresses(&lifecycle, &shared_client_modes, &mut _config_loader)
            .await?;

        // if args are set in client mode init config, set lifecycle manager trajectory file path for local file data sink config values
        ClientCoordinator::<B>::set_lifecycle_traj_file_path(&lifecycle, &shared_client_modes)
            .await?;

        // begins lifecycle file watching operations
        lifecycle.spawn_loop();

        // based on config
        #[cfg(feature = "metrics")]
        let metrics: MetricsManager = ClientCoordinator::<B>::initialize_metrics(&lifecycle).await;

        // builds dispatchers necessary for performing each type of transport-related operation (server inference, scaling op consistency, server training)
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let (inference_dispatcher, scaling_dispatcher, training_dispatcher) = self
            .build_transport_dispatchers(&shared_client_modes, &client_namespace)
            .await?;

        // creates a new state manager and scale manager
        self.build_state_and_scale_managers(
            client_namespace,
            shared_client_modes,
            lifecycle,
            #[cfg(feature = "metrics")]
            metrics,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            inference_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            scaling_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            training_dispatcher,
            default_model,
            data_buffer_size,
        )
        .await?;

        // determines if inference path should be local or externally sourced via transport layer
        self.set_inference_path().await;

        // by using the scale manager, scale up to as many `data_routers`` are specified in arg
        self.initialize_data_routers(data_routers).await?;

        Ok(())
    }

    async fn shutdown(&mut self) -> DrainedCacheResult {
        let shutdown_result = match &mut self.runtime_params {
            Some(params) => {
                // Sends a shutdown RoutedMessage to all actors, which flushes current trajectory to the buffers and then aborts the actor's message loop task
                let actor_ids = params
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

                // the following will trigger shutdown tx/rx for all scalable router nodes in the runtime (the receiver, filters, and buffers)
                // + the single router dispatcher task (the dispatcher informs the actors to shutdown via their inboxes)
                params.lifecycle.shutdown();

                let maybe_traj_cache =
                    if let Some(traj_cache) = &mut params.scaling.shared_traj_cache {
                        match traj_cache.drain(&actor_ids) {
                            Ok(traj_map) => Some(traj_map),
                            Err((traj_map, invalid_ids)) => {
                                log::error!(
                                    "[Coordinator] Failed to drain trajectory cache: {:?}",
                                    invalid_ids
                                );
                                traj_map
                            }
                        }
                    } else {
                        None
                    };

                // Ensure all scalable router tasks are drained before state teardown completes.
                params.scaling.clear_runtime_components().await?;

                // drain the UUID pool to ensure all UUIDs are removed from the pool for the client namespace.
                // uses a clone of the owned handle: StateManager/ScaleManager still hold their own
                // clones at this point, but only local caches (no registry writes) are touched below.
                if let Err(e) = params.client_namespace.clone().remove() {
                    log::error!(
                        "[Coordinator] Failed to remove owned client namespace: {}",
                        e
                    );
                }

                // removes all actor-related
                params
                    .shared_state
                    .write()
                    .await
                    .clear_runtime_components()
                    .await?;

                // by this point, `RelayRLAgent` should be reset back to default

                Ok(maybe_traj_cache)
            }
            None => Err(CoordinatorError::NoRuntimeInstanceError),
        };

        // if the above shutdown operations were successful, remove the runtime parameters from memory
        if self.runtime_params.is_some() {
            let _ = self.runtime_params.take(); // sets the runtime parameters to None
        }

        shutdown_result
    }

    async fn restart(
        &mut self,
        data_routers: u32,
        data_buffer_size: usize,
        default_model: Option<ModelModule<B>>,
        config_path: Option<PathBuf>,
        config_polling_seconds: Option<u64>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
    ) -> Result<(), CoordinatorError> {
        self.shutdown().await?;
        self.start(
            data_routers,
            data_buffer_size,
            default_model,
            config_path,
            config_polling_seconds,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters,
        )
        .await?;
        Ok(())
    }

    async fn request_actions<
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B> + 'static,
        KindOut: TensorKind<B> + 'static,
    >(
        &self,
        actors: &[ActorInfo],
        observation: Tensor<B, D_IN, KindIn>,
        mask: Option<Tensor<B, D_OUT, KindOut>>,
        reward: f32,
    ) -> Result<Vec<(ActorInfo, Arc<RelayRLAction>)>, CoordinatorError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>,
    {
        match self.runtime_params {
            Some(_) => {
                let inference_path = self.inference_path_params.as_ref().ok_or_else(|| {
                    CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No runtime instance to request_action...".to_string(),
                        ),
                    )
                })?;

                #[cfg(feature = "metrics")]
                let (start_time, num_ids) = (Instant::now(), actors.len() as u64);

                let actions = match inference_path {
                    InferencePathParams::Local { local_runtimes } => {
                        // Zero async task boundaries — call ActorRuntime directly.
                        let mut results = Vec::with_capacity(actors.len());
                        let expected_shape = ActorShape {
                            d_in: D_IN,
                            d_out: D_OUT,
                        };
                        for actor in actors {
                            let Some(runtime) = local_runtimes
                                .get(&actor.id())
                                .map(|r| Arc::clone(r.value()))
                            else {
                                continue;
                            };
                            let actual_shape = runtime.actor_shape();
                            if actual_shape != expected_shape {
                                return Err(CoordinatorError::ActorShapeMismatch {
                                    actor_id: actor.id(),
                                    expected_d_in: expected_shape.d_in,
                                    expected_d_out: expected_shape.d_out,
                                    actual_d_in: actual_shape.d_in,
                                    actual_d_out: actual_shape.d_out,
                                });
                            }
                            let ActorDTypes {
                                dtype_in,
                                dtype_out,
                            } = runtime.current_model_dtypes().map_err(|e| {
                                CoordinatorError::StateManagerError(
                                    StateManagerError::InferenceRequestError(e.to_string()),
                                )
                            })?;
                            let obs_tensor: Arc<AnyBurnTensor<B, D_IN>> =
                                Arc::new(observation.to_owned().to_any_burn_tensor(dtype_in));
                            let mask_tensor: Option<Arc<AnyBurnTensor<B, D_OUT>>> =
                                mask.as_ref().map(|tensor| {
                                    Arc::new(tensor.to_owned().to_any_burn_tensor(dtype_out))
                                });

                            let action = runtime
                                .request_inference_erased(
                                    Box::new(obs_tensor.clone()),
                                    Box::new(mask_tensor.clone()),
                                    reward,
                                )
                                .await
                                .map_err(|e| {
                                    CoordinatorError::StateManagerError(
                                        StateManagerError::InferenceRequestError(e.to_string()),
                                    )
                                })?;
                            results.push((actor.clone(), Arc::new(action)));
                        }
                        results
                    }
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    InferencePathParams::Network {
                        filter_channels,
                        shared_router_state,
                        global_dispatcher_tx,
                    } => {
                        let mut pending = Vec::with_capacity(actors.len());
                        for actor in actors {
                            let Some(ns) = shared_router_state
                                .actor_routes
                                .get(&actor.id())
                                .and_then(|r| r.router_namespace.clone())
                            else {
                                continue;
                            };

                            let (resp_tx, resp_rx) = oneshot::channel::<Arc<RelayRLAction>>();
                            let msg = RoutedMessage {
                                actor_id: actor.id(),
                                protocol: RoutingProtocol::Data(DataPayload::RequestInference(
                                    Box::new(InferenceRequest {
                                        observation: Box::new(observation.clone()),
                                        mask: Box::new(mask.clone()),
                                        reward,
                                        reply_to: resp_tx,
                                    }),
                                )),
                            };

                            if let Some(filter_tx) = filter_channels.get(&ns) {
                                match filter_tx.send(msg).await {
                                    Ok(()) => {}
                                    Err(e) => {
                                        global_dispatcher_tx.send(e.0).await.map_err(|e| {
                                            CoordinatorError::ScaleManagerError(
                                                ScaleManagerError::SendActionRequestError(
                                                    e.to_string(),
                                                ),
                                            )
                                        })?;
                                    }
                                }
                            } else {
                                global_dispatcher_tx.send(msg).await.map_err(|e| {
                                    CoordinatorError::ScaleManagerError(
                                        ScaleManagerError::SendActionRequestError(e.to_string()),
                                    )
                                })?;
                            }
                            pending.push((actor.clone(), resp_rx));
                        }

                        let pending_len = pending.len();
                        let mut join_set = tokio::task::JoinSet::<
                            Result<(ActorInfo, Arc<RelayRLAction>), CoordinatorError>,
                        >::new();
                        for (actor, rx) in pending {
                            join_set.spawn(async move {
                                let action = rx.await.map_err(|e| {
                                    CoordinatorError::ScaleManagerError(
                                        ScaleManagerError::ReceiveActionResponseError(
                                            e.to_string(),
                                        ),
                                    )
                                })?;
                                Ok::<(ActorInfo, Arc<RelayRLAction>), CoordinatorError>((
                                    actor, action,
                                ))
                            });
                        }

                        let mut results: Vec<(ActorInfo, Arc<RelayRLAction>)> =
                            Vec::with_capacity(pending_len);
                        while let Some(join_result) = join_set.join_next().await {
                            let pair = join_result.map_err(|e| {
                                CoordinatorError::ScaleManagerError(
                                    ScaleManagerError::ReceiveActionResponseError(e.to_string()),
                                )
                            })??;
                            results.push(pair);
                        }
                        results
                    }
                };

                #[cfg(feature = "metrics")]
                if let Some(params) = &self.runtime_params {
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
                    "[Coordinator] No runtime instance to request_actions...".to_string(),
                ),
            )),
        }
    }

    async fn flag_last_actions(
        &self,
        actors: &[ActorInfo],
        reward: Option<f32>,
    ) -> Result<(), CoordinatorError> {
        match self.runtime_params {
            Some(_) => {
                let inference_path = self.inference_path_params.as_ref().ok_or_else(|| {
                    CoordinatorError::ScaleManagerError(
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[Coordinator] No runtime instance to flag_last_action...".to_string(),
                        ),
                    )
                })?;

                #[cfg(feature = "metrics")]
                let (start_time, num_ids) = (Instant::now(), actors.len() as u64);

                let reward_val: f32 = reward.unwrap_or(0.0);
                match inference_path {
                    InferencePathParams::Local { local_runtimes } => {
                        for actor in actors {
                            let Some(runtime) = local_runtimes
                                .get(&actor.id())
                                .map(|r| Arc::clone(r.value()))
                            else {
                                continue;
                            };
                            runtime
                                .flag_last_action_erased(reward_val, None, None, false)
                                .await
                                .map_err(|e| {
                                    CoordinatorError::StateManagerError(
                                        StateManagerError::InferenceRequestError(e.to_string()),
                                    )
                                })?;
                        }
                    }
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    InferencePathParams::Network {
                        filter_channels,
                        shared_router_state,
                        global_dispatcher_tx,
                    } => {
                        for actor in actors {
                            let Some(ns) = shared_router_state
                                .actor_routes
                                .get(&actor.id())
                                .and_then(|r| r.router_namespace.clone())
                            else {
                                continue;
                            };

                            let msg = RoutedMessage {
                                actor_id: actor.id(),
                                protocol: RoutingProtocol::Data(DataPayload::FlagLastAction {
                                    reward: reward_val,
                                    env_id: None,
                                    env_label: None,
                                }),
                            };

                            if let Some(filter_tx) = filter_channels.get(&ns) {
                                match filter_tx.send(msg).await {
                                    Ok(()) => {}
                                    Err(e) => {
                                        global_dispatcher_tx.send(e.0).await.map_err(|_| {
                                            CoordinatorError::ScaleManagerError(
                                                ScaleManagerError::SendFlagLastActionMessageError(
                                                    format!(
                                                        "Hot dispatch failed for actor {}",
                                                        actor.id()
                                                    ),
                                                ),
                                            )
                                        })?;
                                    }
                                }
                            } else {
                                global_dispatcher_tx.send(msg).await.map_err(|_| {
                                    CoordinatorError::ScaleManagerError(
                                        ScaleManagerError::SendFlagLastActionMessageError(format!(
                                            "Hot dispatch failed for actor {}",
                                            actor.id()
                                        )),
                                    )
                                })?;
                            }
                        }
                    }
                }

                #[cfg(feature = "metrics")]
                if let Some(params) = &self.runtime_params {
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

    async fn scale_routers_out(&mut self, router_add: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let result = {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    {
                        params
                            .scaling
                            .scale_routers_out(router_add, true)
                            .await
                            .map_err(CoordinatorError::from)
                    }

                    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                    {
                        params
                            .scaling
                            .scale_routers_out(router_add)
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

    async fn scale_routers_in(&mut self, router_remove: u32) -> Result<(), CoordinatorError> {
        match &mut self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let result = {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    {
                        params
                            .scaling
                            .scale_routers_in(router_remove, true)
                            .await
                            .map_err(CoordinatorError::from)
                    }

                    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                    {
                        params
                            .scaling
                            .scale_routers_in(router_remove)
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

    async fn scale_data_buffers(&mut self, new_size: usize) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .scaling
                    .shared_buffer_size
                    .swap(new_size, Ordering::SeqCst);
                Ok(())
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to scale_data_buffers...".to_string(),
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

// Start traits for ClientCoordinator::start() and their associated operations as functions

impl<B: Backend + BackendMatcher<Backend = B>> ClientStart<B> for ClientCoordinator<B> {}

impl<B: Backend + BackendMatcher<Backend = B>> LifecycleStart<B> for ClientCoordinator<B> {
    fn build_lifecycle_manager(
        &mut self,
        config_path: Option<PathBuf>,
        config_polling_seconds: Option<u64>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
    ) -> Result<(LifecycleManager, ClientConfigLoader), CoordinatorError> {
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
            default_hyperparameters,
            &config_loader,
            config_path,
            config_polling_seconds,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.transport_type,
        );
        Ok((lifecycle, config_loader))
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn set_lifecycle_addresses(
        &self,
        lifecycle: &LifecycleManager,
        shared_client_modes: &Arc<ClientModes>,
        config_loader: &mut ClientConfigLoader,
    ) -> Result<(), CoordinatorError> {
        let inference_address_args = if let ActorInferenceMode::Server(server_params)
        | ActorInferenceMode::ClientFallback(_, server_params) =
            &shared_client_modes.actor_inference_mode
        {
            server_params.inference_addresses.clone()
        } else {
            None
        };

        let training_address_args = match &shared_client_modes.actor_data_mode {
            ActorDataMode::Online(server_params)
            | ActorDataMode::OnlineWithFiles(server_params, _)
            | ActorDataMode::OnlineWithCache(server_params, _)
            | ActorDataMode::OnlineWithFilesAndCache(server_params, ..) => {
                server_params.training_addresses.clone()
            }
            ActorDataMode::Disabled
            | ActorDataMode::OfflineWithFiles(_)
            | ActorDataMode::OfflineWithCache(_)
            | ActorDataMode::OfflineWithFilesAndCache(..) => None,
        };

        if inference_address_args.is_some() || training_address_args.is_some() {
            let transport_params_for_packing: &mut TransportConfigParams =
                &mut config_loader.transport_config;

            if let Some(inference_addresses) = inference_address_args {
                match &self.transport_type {
                    #[cfg(feature = "nats-transport")]
                    TransportMode::NATS => {
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
                    TransportMode::ZMQ => {
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

                        if let Some(inference_scaling_server_address) = match inference_addresses {
                            #[cfg(feature = "nats-transport")]
                            InferenceAddressesArgs::NATS(_) => None,
                            #[cfg(feature = "zmq-transport")]
                            InferenceAddressesArgs::ZMQ(ref params) => {
                                params.inference_scaling_server_address.clone()
                            }
                        } {
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
                    TransportMode::NATS => {
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
                    TransportMode::ZMQ => {
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

                        if let Some(training_scaling_server_address) = match training_addresses {
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
                                .training_scaling_server_address = training_scaling_server_address;
                        }
                    }
                }
            }

            lifecycle
                .set_transport_addresses(transport_params_for_packing, &self.transport_type)
                .await
                .map_err(CoordinatorError::from)?
        }

        Ok(())
    }

    async fn set_lifecycle_traj_file_path(
        lifecycle: &LifecycleManager,
        shared_client_modes: &Arc<ClientModes>,
    ) -> Result<(), CoordinatorError> {
        // if args are set in client mode init config, set lifecycle manager trajectory file path
        let local_trajectory_file_params = match &shared_client_modes.actor_data_mode {
            ActorDataMode::OfflineWithFiles(Some(params))
            | ActorDataMode::OfflineWithFilesAndCache(Some(params), _) => Some(params),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            ActorDataMode::OnlineWithFiles(_, Some(params))
            | ActorDataMode::OnlineWithFilesAndCache(_, Some(params), _) => Some(params),
            _ => None,
        };

        if let Some(file_params) = local_trajectory_file_params {
            return lifecycle
                .set_trajectory_file_path(file_params)
                .await
                .map_err(CoordinatorError::from);
        }

        Ok(())
    }
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
impl<B: Backend + BackendMatcher<Backend = B>> TransportStart<B> for ClientCoordinator<B> {
    async fn build_transport_dispatchers(
        &self,
        shared_client_modes: &Arc<ClientModes>,
        client_namespace: &ClientNamespace,
    ) -> Result<TransportDispatchers<B>, CoordinatorError> {
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
                ActorInferenceMode::Server(_) | ActorInferenceMode::ClientFallback(_, _) => (
                    Some(Arc::new(InferenceDispatcher::<B>::new(
                        shared_transport.clone(),
                    ))),
                    Some(Arc::new(ScalingDispatcher::<B>::new(
                        shared_transport.clone(),
                    ))),
                ),
                ActorInferenceMode::Client(_) => (None, None),
            };

        let training_dispatcher = match shared_client_modes.actor_data_mode {
            ActorDataMode::Disabled | ActorDataMode::OfflineWithFiles(_) => None,
            _ => {
                scaling_dispatcher = Some(Arc::new(ScalingDispatcher::<B>::new(
                    shared_transport.clone(),
                )));
                Some(Arc::new(TrainingDispatcher::<B>::new(
                    shared_transport.clone(),
                )))
            }
        };

        Ok((
            inference_dispatcher,
            scaling_dispatcher,
            training_dispatcher,
        ) as TransportDispatchers<B>)
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> CoreRuntimeStart<B> for ClientCoordinator<B> {
    #[cfg(feature = "metrics")]
    async fn initialize_metrics(lifecycle: &LifecycleManager) -> MetricsManager {
        let metrics_args = lifecycle.get_metrics_args();
        init_metrics(metrics_args).await
    }

    async fn build_state_and_scale_managers(
        &mut self,
        client_namespace: ClientNamespace,
        shared_client_modes: Arc<ClientModes>,
        lifecycle: LifecycleManager,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        scaling_dispatcher: Option<Arc<ScalingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        default_model: Option<ModelModule<B>>,
        data_buffer_size: usize,
    ) -> Result<(), CoordinatorError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_transport_addresses = if let ActorInferenceMode::Server(_)
        | ActorInferenceMode::ClientFallback(..) =
            shared_client_modes.actor_inference_mode
        {
            Some(lifecycle.get_transport_addresses())
        } else if let ActorDataMode::Online(_)
        | ActorDataMode::OnlineWithFiles(..)
        | ActorDataMode::OnlineWithCache(..) = shared_client_modes.actor_data_mode
        {
            Some(lifecycle.get_transport_addresses())
        } else {
            None
        };
        let (state, global_dispatcher_rx) = {
            let shared_local_model_path = lifecycle.get_local_model_path();

            let state_default_model = default_model.clone();

            StateManager::new(
                client_namespace.clone(),
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                inference_dispatcher.clone(),
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                training_dispatcher.clone(),
                shared_client_modes.clone(),
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_transport_addresses.clone(),
                shared_local_model_path,
                state_default_model,
                #[cfg(feature = "metrics")]
                metrics.clone(),
            )
        };
        let shared_state: Arc<RwLock<StateManager<B>>> = Arc::from(RwLock::new(state));
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let training_codec = match &shared_client_modes.actor_data_mode {
            ActorDataMode::Online(params) => params.codec.clone(),
            ActorDataMode::OnlineWithFiles(params, _) => params.codec.clone(),
            ActorDataMode::OnlineWithCache(params, _) => params.codec.clone(),
            ActorDataMode::OnlineWithFilesAndCache(params, ..) => params.codec.clone(),
            _ => None,
        };
        let scaling = ScaleManager::new(
            client_namespace.clone(),
            data_buffer_size,
            shared_client_modes,
            shared_state.clone(),
            global_dispatcher_rx,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            scaling_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            training_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_transport_addresses.clone(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            training_codec,
            #[cfg(feature = "metrics")]
            metrics.clone(),
            lifecycle.clone(),
        )
        .await
        .map_err(CoordinatorError::from)?;

        self.runtime_params = Some(CoordinatorParams {
            client_namespace,
            #[cfg(feature = "metrics")]
            metrics,
            lifecycle,
            shared_state,
            scaling,
        });
        Ok(())
    }

    async fn set_inference_path(&mut self) {
        if let Some(params) = self.runtime_params.as_ref() {
            let is_local_inference = matches!(
                self.client_modes.actor_inference_mode,
                ActorInferenceMode::Client(_)
            );

            self.inference_path_params = Some(if is_local_inference {
                InferencePathParams::Local {
                    local_runtimes: {
                        let state_guard = params.shared_state.read().await;
                        state_guard.actor_runtime_handles.clone()
                    },
                }
            } else {
                // this path only executes if zmq or nats transport feature flags are enabled
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                {
                    let (filter_channels, shared_router_state, global_dispatcher_tx) = {
                        let state_guard = params.shared_state.read().await;
                        (
                            params.scaling.router_filter_channels.clone(),
                            state_guard.shared_router_state.clone(),
                            state_guard.global_dispatcher_tx.clone(),
                        )
                    };
                    InferencePathParams::Network {
                        filter_channels,
                        shared_router_state,
                        global_dispatcher_tx,
                    }
                }
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                {
                    unreachable!()
                }
            });
        }
    }

    async fn initialize_data_routers(&mut self, data_routers: u32) -> Result<(), CoordinatorError> {
        if let Some(params) = self.runtime_params.as_mut() {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            params
                .scaling
                .scale_routers_out(data_routers, false)
                .await
                .map_err(CoordinatorError::from)?;

            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            params
                .scaling
                .scale_routers_out(data_routers)
                .await
                .map_err(CoordinatorError::from)?;
        }

        Ok(())
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientActors<B> for ClientCoordinator<B> {
    async fn new_actor<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        device: DeviceType,
        max_traj_length: usize,
        nametag: Option<NameTag>,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        algorithm_args: AlgorithmInitArgs,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_id: bool,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        send_algorithm_init: bool,
    ) -> Result<ActorInfo, CoordinatorError> {
        match self.runtime_params.as_mut() {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let actor_id: Uuid = params
                    .client_namespace
                    .reserve_id_with(crate::network::ACTOR_CONTEXT, 117, 100)
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
                    .actor_routes
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

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                let initialized_algorithm_args = match algorithm_args {
                    AlgorithmInitArgs::PPO(None) => AlgorithmInitArgs::PPO(Some(
                        params
                            .lifecycle
                            .get_ppo_hyperparameters()
                            .read()
                            .await
                            .clone(),
                    )),
                    AlgorithmInitArgs::IPPO(None) => AlgorithmInitArgs::IPPO(Some(
                        params
                            .lifecycle
                            .get_ippo_hyperparameters()
                            .read()
                            .await
                            .clone(),
                    )),
                    AlgorithmInitArgs::MAPPO(None) => AlgorithmInitArgs::MAPPO(Some(
                        params
                            .lifecycle
                            .get_mappo_hyperparameters()
                            .read()
                            .await
                            .clone(),
                    )),
                    _ => algorithm_args,
                };

                params
                    .shared_state
                    .write()
                    .await
                    .new_actor::<D_IN, D_OUT>(
                        actor_id,
                        router_namespace,
                        device,
                        max_traj_length,
                        nametag,
                        default_model,
                        trajectory_buffer_tx,
                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        initialized_algorithm_args.clone(),
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
                                    ProcessInitFlag::<B>::TrainingAlgorithmInit(
                                        initialized_algorithm_args,
                                    ),
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

                let actor_info = {
                    let actors = &params.shared_state.read().await.actor_runtime_handles;
                    match actors.get(&actor_id) {
                        Some(runtime) => Ok(runtime.value().get_actor_info().map_err(|e| {
                            CoordinatorError::StateManagerError(StateManagerError::ActorError(e))
                        })?),
                        None => Err(CoordinatorError::StateManagerError(
                            StateManagerError::NewActorError(format!(
                                "[Coordinator] new_actor() cannot retrieve nametag from actor {} after initialization",
                                actor_id
                            )),
                        )),
                    }
                }?;

                Ok(actor_info)
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::NewActorError(
                    "[Coordinator] No runtime instance to new_actor...".to_string(),
                ),
            )),
        }
    }

    /// Resolves a user-supplied nametag string into `actor_count` concrete [`NameTag`]s.
    ///
    /// `None` passes through as `None` (no tag requested). For `Some(tag)`, this scans every
    /// live actor's current nametag for matches on `tag` and continues the `duplicate` counter
    /// from one past the highest duplicate already in use, so repeated calls with the same tag
    /// never collide with existing actors.
    async fn resolve_new_nametag(
        &self,
        nametag: Option<&str>,
        actor_count: u32,
    ) -> Result<Option<Vec<NameTag>>, CoordinatorError> {
        let Some(tag) = nametag else {
            return Ok(None);
        };

        if actor_count == 0 {
            return Ok(Some(Vec::new()));
        }

        let next_duplicate = match &self.runtime_params {
            Some(params) => {
                let actors = &params.shared_state.read().await.actor_runtime_handles;
                actors
                    .iter()
                    .filter_map(|runtime| match runtime.get_actor_nametag() {
                        Ok(Some(existing)) if existing.tag == tag => Some(existing.duplicate),
                        _ => None,
                    })
                    .max()
                    .map_or(0, |max_duplicate| max_duplicate + 1)
            }
            None => 0,
        };

        Ok(Some(
            (0..actor_count as usize)
                .map(|offset| NameTag {
                    tag: tag.to_string(),
                    duplicate: next_duplicate + offset,
                })
                .collect(),
        ))
    }

    async fn remove_actor(
        &mut self,
        actor: &ActorInfo,
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
                    .remove_actor(actor.id())
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

    async fn get_actor(&self, id: ActorUuid) -> Result<ActorInfo, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let actors = &params.shared_state.read().await.actor_runtime_handles;

                let runtime = actors.get(&id).ok_or_else(|| {
                    CoordinatorError::StateManagerError(StateManagerError::GetActorsError(format!(
                        "[Coordinator] Actor {} not found",
                        id
                    )))
                })?;

                runtime.get_actor_info().map_err(|e| {
                    CoordinatorError::StateManagerError(StateManagerError::ActorError(e))
                })
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetActorsError(
                    "[Coordinator] No runtime instance to get_actor_info_by_id...".to_string(),
                ),
            )),
        }
    }

    async fn get_all_actors(&self) -> Result<Vec<ActorInfo>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let actors = &params.shared_state.read().await.actor_runtime_handles;

                Ok(actors
                    .iter()
                    .filter_map(|runtime| match runtime.get_actor_info() {
                        Ok(actor_info) => Some(actor_info),
                        Err(e) => {
                            log::error!("{}", e);
                            None
                        }
                    })
                    .collect())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetActorsError(
                    "[Coordinator] No runtime instance to get_actor_info...".to_string(),
                ),
            )),
        }
    }

    async fn get_actors_by_rank<const D_IN: usize, const D_OUT: usize>(
        &self,
    ) -> Result<Vec<ActorInfo>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let actors = &params.shared_state.read().await.actor_runtime_handles;

                let valid_actor_info = actors
                    .iter()
                    .filter_map(|runtime| {
                        let actor_shape = runtime.actor_shape();

                        if actor_shape.d_in == D_IN && actor_shape.d_out == D_OUT {
                            match runtime.get_actor_info() {
                                Ok(actor_info) => Some(actor_info),
                                Err(e) => {
                                    log::error!("{}", e);
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                Ok(valid_actor_info)
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetActorsError(
                    "[Coordinator] No runtime instance to get_actor_info_by_rank...".to_string(),
                ),
            )),
        }
    }

    /// Returns every live actor whose current nametag string matches `nametag`, regardless of
    /// duplicate index. `None` returns every actor with no nametag set.
    async fn get_actors_by_tag(
        &self,
        nametag: Option<&str>,
    ) -> Result<Vec<ActorInfo>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let actors = &params.shared_state.read().await.actor_runtime_handles;

                let valid_actor_info = actors
                    .iter()
                    .filter_map(|runtime| match runtime.get_actor_info() {
                        Ok(actor_info) => {
                            let matches = match (nametag, actor_info.nametag_arc()) {
                                (Some(wanted), Some(existing)) => existing.tag == wanted,
                                (None, None) => true,
                                _ => false,
                            };
                            matches.then_some(actor_info)
                        }
                        Err(e) => {
                            log::error!("{}", e);
                            None
                        }
                    })
                    .collect();

                Ok(valid_actor_info)
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::GetActorsError(
                    "[Coordinator] No runtime instance to get_actor_info_by_tag...".to_string(),
                ),
            )),
        }
    }

    async fn set_actor_id(
        &mut self,
        current_actor: &ActorInfo,
        new_id: ActorUuid,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                // `StateManager::set_actor_id` moves the id-keyed maps and then updates the
                // actor's shared identity slot, so `current_actor` (and every other clone of
                // it) observes the new id without any local field assignment here.
                StateManager::<B>::set_actor_id(
                    &*params.shared_state.write().await,
                    current_actor.id(),
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

    async fn set_actor_nametag(
        &mut self,
        actor: &ActorInfo,
        new_nametag: Option<&str>,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                #[cfg(feature = "metrics")]
                let start_time = Instant::now();

                let nametag: Option<NameTag> = {
                    let nametags = self.resolve_new_nametag(new_nametag, 1).await?;
                    nametags.map(|tags| tags[0].clone())
                };

                // `StateManager::set_actor_nametag` writes straight into the actor's shared
                // nametag slot, so `actor` (and every other clone of it) observes the new tag
                // without any local field assignment here.
                StateManager::<B>::set_actor_nametag(
                    &*params.shared_state.write().await,
                    actor.id(),
                    nametag,
                )?;

                #[cfg(feature = "metrics")]
                {
                    let duration: f64 = start_time.elapsed().as_secs_f64();
                    params
                        .metrics
                        .record_histogram("set_actor_nametag_latency", duration, &[])
                        .await;
                    params
                        .metrics
                        .record_counter("set_actor_nametag_calls", 1, &[])
                        .await;
                }

                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetActorNameTagError(
                    "[Coordinator] No runtime instance to set_actor_nametag...".to_string(),
                ),
            )),
        }
    }

    async fn update_models<const D_IN: usize, const D_OUT: usize>(
        &self,
        specific_actors: Option<&[ActorInfo]>,
        model: ModelModule<B>,
    ) -> Result<(), CoordinatorError> {
        match self.runtime_params {
            Some(_) => {
                let Some((global_dispatcher_tx, target_actors, local_model_path)) = self
                    .prepare_model_update_dispatch::<D_IN, D_OUT>(
                        specific_actors.as_deref(),
                        &model.metadata,
                    )
                    .await?
                else {
                    return Ok(());
                };

                if target_actors.is_empty() {
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
                Self::dispatch_model_updates(global_dispatcher_tx, &target_actors, model_bytes)
                    .await
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetActorModelError(
                    "[Coordinator] No runtime instance to update_models...".to_string(),
                ),
            )),
        }
    }

    async fn get_model_versions(
        &self,
        actors: &[ActorInfo],
    ) -> Result<Vec<(ActorInfo, i64)>, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                let global_dispatcher_tx = params
                    .shared_state
                    .read()
                    .await
                    .global_dispatcher_tx
                    .clone();
                Self::request_model_versions(global_dispatcher_tx, actors).await
            }
            None => Err(CoordinatorError::ScaleManagerError(
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[Coordinator] No runtime instance to get_model_versions...".to_string(),
                ),
            )),
        }
    }

    fn drain_trajectory_caches(&self, actors: &[ActorInfo]) -> DrainedCacheResult {
        match &self.runtime_params {
            Some(params) => {
                if let Some(mut shared_traj_cache) = params.scaling.shared_traj_cache.clone() {
                    Ok(match shared_traj_cache.drain(actors) {
                        Ok(traj_map) => Some(traj_map),
                        Err((Some(traj_map), invalid_ids)) => {
                            log::error!(
                                "Actor IDs not found in trajectory cache: {:?}",
                                invalid_ids
                            );
                            Some(traj_map)
                        }
                        Err((None, invalid_ids)) => {
                            log::error!(
                                "All actor IDs not found in trajectory cache: {:?}",
                                invalid_ids
                            );
                            None
                        }
                    })
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
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientEnvironments<B> for ClientCoordinator<B> {
    async fn run_env_eval(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => match self.client_modes.actor_inference_mode {
                ActorInferenceMode::Client(_) => {
                    let (runtime, env_map) = {
                        let shared_state_guard = params.shared_state.read().await;
                        shared_state_guard
                            .get_run_env_handles(actor.id())
                            .map_err(CoordinatorError::from)?
                    };
                    StateManager::<B>::run_env_eval_step_loop(
                        actor.id(),
                        runtime,
                        env_map,
                        loop_iters,
                    )
                    .map_err(CoordinatorError::from)
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::Server(_) | ActorInferenceMode::ClientFallback(_, _) => {
                    unimplemented!("Not supported yet")
                }
            },
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::StepEnvError(
                    "[Coordinator] No runtime instance to step_env...".to_string(),
                ),
            )),
        }
    }

    async fn run_env_with_ppo<KindIn, KindOut, Pi>(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, CoordinatorError>
    where
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Clone + Send + 'static,
        B: Default + Send + Sync + 'static,
    {
        match &self.runtime_params {
            Some(params) => match self.client_modes.actor_inference_mode {
                ActorInferenceMode::Client(_) => {
                    let (runtime, env_map, shutdown_rx) = {
                        let shared_state_guard = params.shared_state.read().await;
                        let (runtime, env_map) = shared_state_guard
                            .get_run_env_handles(actor.id())
                            .map_err(CoordinatorError::from)?;
                        let shutdown_rx = params
                            .lifecycle
                            .subscribe_shutdown()
                            .map_err(CoordinatorError::from)?;
                        (runtime, env_map, shutdown_rx)
                    };

                    StateManager::<B>::run_env_step_loop_with_ppo::<KindIn, KindOut, Pi>(
                        actor.id(),
                        Some(shutdown_rx),
                        Arc::clone(&runtime),
                        env_map,
                        loop_iters,
                        max_traj_length,
                        trainer_spec,
                    )
                    .map_err(CoordinatorError::from)
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::Server(_) | ActorInferenceMode::ClientFallback(_, _) => {
                    unimplemented!("Not supported yet")
                }
            },
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::StepEnvError(
                    "[Coordinator] No runtime instance to step_env...".to_string(),
                ),
            )),
        }
    }

    async fn run_env_with_ippo<KindIn, KindOut, Pi>(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, CoordinatorError>
    where
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
        B: Default + Send + Sync + 'static,
    {
        match &self.runtime_params {
            Some(params) => match self.client_modes.actor_inference_mode {
                ActorInferenceMode::Client(_) => {
                    let (runtime, env_map, shutdown_rx) = {
                        let shared_state_guard = params.shared_state.read().await;
                        let (runtime, env_map) = shared_state_guard
                            .get_run_env_handles(actor.id())
                            .map_err(CoordinatorError::from)?;
                        let shutdown_rx = params
                            .lifecycle
                            .subscribe_shutdown()
                            .map_err(CoordinatorError::from)?;
                        (runtime, env_map, shutdown_rx)
                    };

                    StateManager::<B>::run_env_step_loop_with_ippo::<KindIn, KindOut, Pi>(
                        actor.id(),
                        shutdown_rx,
                        Arc::clone(&runtime),
                        env_map,
                        loop_iters,
                        max_traj_length,
                        trainer_spec,
                    )
                    .map_err(CoordinatorError::from)
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::Server(_) | ActorInferenceMode::ClientFallback(_, _) => {
                    unimplemented!("Not supported yet")
                }
            },
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::StepEnvError(
                    "[Coordinator] No runtime instance to step_env...".to_string(),
                ),
            )),
        }
    }

    async fn run_env_with_mappo<KindIn, KindOut, Pi>(
        &self,
        actor: &ActorInfo,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, CoordinatorError>
    where
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
        B: Default + Send + Sync + 'static,
    {
        match &self.runtime_params {
            Some(params) => match self.client_modes.actor_inference_mode {
                ActorInferenceMode::Client(_) => {
                    let (runtime, env_map, shutdown_rx) = {
                        let shared_state_guard = params.shared_state.read().await;
                        let (runtime, env_map) = shared_state_guard
                            .get_run_env_handles(actor.id())
                            .map_err(CoordinatorError::from)?;
                        let shutdown_rx = params
                            .lifecycle
                            .subscribe_shutdown()
                            .map_err(CoordinatorError::from)?;
                        (runtime, env_map, shutdown_rx)
                    };

                    StateManager::<B>::run_env_step_loop_with_mappo::<KindIn, KindOut, Pi>(
                        actor.id(),
                        shutdown_rx,
                        Arc::clone(&runtime),
                        env_map,
                        loop_iters,
                        max_traj_length,
                        trainer_spec,
                    )
                    .map_err(CoordinatorError::from)
                }
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                ActorInferenceMode::Server(_) | ActorInferenceMode::ClientFallback(_, _) => {
                    unimplemented!("Not supported yet")
                }
            },
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::StepEnvError(
                    "[Coordinator] No runtime instance to step_env...".to_string(),
                ),
            )),
        }
    }

    async fn set_env(
        &mut self,
        actor: &ActorInfo,
        env: Box<dyn Environment>,
        count: u32,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .read()
                    .await
                    .set_env(actor.id(), env, count)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::SetEnvError(
                    "[Coordinator] No runtime instance to set_env...".to_string(),
                ),
            )),
        }
    }

    async fn get_env_count(&self, actor: &ActorInfo) -> Result<u32, CoordinatorError> {
        match &self.runtime_params {
            Some(params) => params
                .shared_state
                .read()
                .await
                .get_env_count(actor.id())
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
        actor: &ActorInfo,
        count: u32,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .read()
                    .await
                    .increase_env_count(actor.id(), count)?;
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
        actor: &ActorInfo,
        count: u32,
    ) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params
                    .shared_state
                    .read()
                    .await
                    .decrease_env_count(actor.id(), count)?;
                Ok(())
            }
            None => Err(CoordinatorError::StateManagerError(
                StateManagerError::DecreaseEnvCountError(
                    "[Coordinator] No runtime instance to decrease_env_count...".to_string(),
                ),
            )),
        }
    }

    async fn remove_env(&mut self, actor: &ActorInfo) -> Result<(), CoordinatorError> {
        match &self.runtime_params {
            Some(params) => {
                params.shared_state.read().await.remove_env(actor.id())?;
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
        ActorDataMode, ActorInferenceMode, ClientModes, ModelMode,
    };
    use crate::network::client::runtime::control::lifecycle_manager::LifecycleManager;
    use crate::network::client::runtime::control::state_manager::ActorRoute;
    use crate::utilities::configuration::ClientConfigLoader;
    use active_uuid_registry::registry_uuid::Uuid;
    use burn_ndarray::NdArray;
    use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};
    use relayrl_types::data::action::RelayRLAction;
    use relayrl_types::data::tensor::{DType, DeviceType, NdArrayDType};
    use relayrl_types::model::{ModelFileType, ModelMetadata};
    use relayrl_types::prelude::tensor::relayrl::FloatBurnTensor;
    use std::path::PathBuf;
    use tokio::sync::mpsc::{self, error::TryRecvError};

    type TestBackend = NdArray<f32>;
    type TestKind = Float;

    fn make_coordinator() -> ClientCoordinator<TestBackend> {
        ClientCoordinator::<TestBackend>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportMode::default(),
            ClientModes::default(),
        )
    }

    fn make_model_metadata() -> ModelMetadata {
        ModelMetadata {
            model_file: "test.onnx".to_string(),
            model_type: ModelFileType::Onnx,
            input_dtype: DType::NdArray(NdArrayDType::F32),
            output_dtype: DType::NdArray(NdArrayDType::F32),
            input_shape: vec![1, 1, 1, 1],
            output_shape: vec![1],
            default_device: Some(DeviceType::Cpu),
        }
    }

    fn actor_info(id: Uuid) -> ActorInfo {
        ActorInfo::new(id, None)
    }

    fn make_lifecycle_manager() -> LifecycleManager {
        use std::io::Write;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(tmp, "{{}}").expect("write temp config");
        let config = ClientConfigLoader::load_config(&tmp.path().to_path_buf());
        let lifecycle = LifecycleManager::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            DefaultHyperparameterArgs::default(),
            &config,
            tmp.path().to_path_buf(),
            Some(1000),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportMode::default(),
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

    fn float_any_tensor(values: &[f32]) -> Arc<AnyBurnTensor<TestBackend, 4>> {
        let device = TestBackend::get_device(&DeviceType::Cpu).unwrap();
        let tensor = Tensor::<TestBackend, 4, Float>::from_data(
            BurnTensorData::new(values.to_vec(), [1, 1, 1, values.len()]),
            &device,
        );

        Arc::new(AnyBurnTensor::Float(FloatBurnTensor {
            tensor: Arc::new(tensor),
            dtype: DType::NdArray(NdArrayDType::F32),
        }))
    }

    async fn make_runtime_coordinator(
        client_modes: ClientModes,
    ) -> (
        ClientCoordinator<TestBackend>,
        Arc<RwLock<StateManager<TestBackend>>>,
        tokio::sync::mpsc::Receiver<RoutedMessage>,
    ) {
        let namespace_str = format!("test-coordinator-{}", Uuid::new_v4());
        let namespace_handle =
            reserve_owned_namespace(&namespace_str).expect("reserve owned test namespace");
        let client_namespace = ClientNamespace::new(namespace_handle, Arc::from(namespace_str));

        let lifecycle = make_lifecycle_manager();
        *lifecycle.get_local_model_path().write().await = PathBuf::new();
        let shared_client_modes = Arc::new(client_modes.clone());
        let (state, global_dispatcher_rx) = StateManager::<TestBackend>::new(
            client_namespace.clone(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            shared_client_modes.clone(),
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
            1024,
            shared_client_modes,
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

        let mut coordinator = ClientCoordinator::<TestBackend>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportMode::default(),
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

        // Build inference_path_params so that request_actions / flag_last_actions can route
        // messages through the global dispatcher in transport-feature tests.
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        {
            let (filter_channels, shared_router_state, global_dispatcher_tx) = {
                let state_guard = shared_state.read().await;
                (
                    coordinator
                        .runtime_params
                        .as_ref()
                        .unwrap()
                        .scaling
                        .router_filter_channels
                        .clone(),
                    state_guard.shared_router_state.clone(),
                    state_guard.global_dispatcher_tx.clone(),
                )
            };
            coordinator.inference_path_params = Some(InferencePathParams::Network {
                filter_channels,
                shared_router_state,
                global_dispatcher_tx,
            });
        }

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
                &actor_info(Uuid::new_v4()),
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                false,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn set_actor_id_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let actor = actor_info(Uuid::new_v4());
        let result = c.set_actor_id(&actor, Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn flag_last_action_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.flag_last_actions(&[], None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn request_action_stays_routed_through_global_dispatcher() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (mut coordinator, shared_state, mut global_dispatcher_rx) =
            make_runtime_coordinator(client_modes).await;
        coordinator
            .runtime_params
            .as_mut()
            .expect("runtime params should exist")
            .scaling
            .runtime_params = Some(dashmap::DashMap::new());
        let actor_id = Uuid::new_v4();
        let (tx_to_actor, _rx_from_actor) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        shared_state
            .write()
            .await
            .shared_router_state
            .actor_routes
            .insert(
                actor_id,
                ActorRoute {
                    router_namespace: Some(Arc::from("router-a")),
                    inbox: tx_to_actor,
                },
            );

        let responder = tokio::spawn(async move {
            let message = global_dispatcher_rx
                .recv()
                .await
                .expect("expected routed message");
            assert_eq!(message.actor_id, actor_id);
            match message.protocol {
                RoutingProtocol::Data(DataPayload::RequestInference(req)) => {
                    assert_eq!(req.reward, 0.75);
                    req.reply_to
                        .send(Arc::new(RelayRLAction::minimal(0.25, false)))
                        .expect("reply should be open");
                }
                other => panic!(
                    "expected RequestInference payload, got {:?}",
                    std::mem::discriminant(&other)
                ),
            }
            assert!(matches!(
                global_dispatcher_rx.try_recv(),
                Err(TryRecvError::Empty)
            ));
        });

        let device = TestBackend::get_device(&DeviceType::Cpu).unwrap();
        let tensor = Tensor::<TestBackend, 4, Float>::from_data(
            BurnTensorData::new(vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32], [1, 1, 1, 4]),
            &device,
        );

        let actions = coordinator
            .request_actions::<4, 1, Float, Float>(&[actor_info(actor_id)], tensor, None, 0.75)
            .await
            .unwrap();
        responder.await.unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].0.id(), actor_id);
        assert_eq!(actions[0].1.get_rew(), 0.25);
    }

    #[tokio::test]
    #[cfg(any(feature = "zmq-transport", feature = "nats-transport"))]
    async fn flag_last_action_stays_routed_through_global_dispatcher() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (coordinator, shared_state, mut global_dispatcher_rx) =
            make_runtime_coordinator(client_modes).await;
        let actor_id = Uuid::new_v4();

        // Register the actor in shared_router_state so flag_last_actions can route it.
        // Without a router_namespace in actor_routes, the code skips the actor.
        let (tx_to_actor, _rx_from_actor) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        shared_state
            .write()
            .await
            .shared_router_state
            .actor_routes
            .insert(
                actor_id,
                ActorRoute {
                    router_namespace: Some(Arc::from("router-a")),
                    inbox: tx_to_actor,
                },
            );

        coordinator
            .flag_last_actions(&[actor_info(actor_id)], Some(1.5))
            .await
            .unwrap();

        let message = global_dispatcher_rx
            .recv()
            .await
            .expect("expected routed flag-last-action message");
        assert_eq!(message.actor_id, actor_id);
        match message.protocol {
            RoutingProtocol::Data(DataPayload::FlagLastAction {
                reward,
                env_id,
                env_label,
            }) => {
                assert_eq!(reward, 1.5);
                assert_eq!(env_id, None);
                assert_eq!(env_label, None);
            }
            other => panic!(
                "expected FlagLastAction payload, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        assert!(matches!(
            global_dispatcher_rx.try_recv(),
            Err(TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn get_model_version_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c.get_model_versions(&[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn prepare_model_update_dispatch_no_runtime_returns_err() {
        let c = make_coordinator();
        let result = c
            .prepare_model_update_dispatch::<4, 1>(None, &make_model_metadata())
            .await;
        assert!(result.is_err());
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[tokio::test]
    async fn prepare_model_update_dispatch_server_mode_returns_none() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Server(InferenceParams::default()),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (coordinator, _shared_state, mut global_dispatcher_rx) =
            make_runtime_coordinator(client_modes).await;

        let result = coordinator
            .prepare_model_update_dispatch::<4, 1>(None, &make_model_metadata())
            .await;

        assert!(matches!(result, Ok(None)));
        assert!(matches!(
            global_dispatcher_rx.try_recv(),
            Err(TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn prepare_model_update_dispatch_subset_filters_requested_actor_ids() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
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
                    .new_actor::<4, 1>(
                        *actor_id,
                        Arc::from("router-a"),
                        DeviceType::Cpu,
                        100,
                        None,
                        None,
                        tx_to_buffer.clone(),
                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        AlgorithmInitArgs::default(),
                    )
                    .await
                    .unwrap();
            }
        }

        let requested_actor_ids = vec![
            actor_info(actor_ids[2]),
            actor_info(unknown_actor_id),
            actor_info(actor_ids[0]),
            actor_info(actor_ids[2]),
        ];
        let (_global_dispatcher_tx, target_actor_ids, _local_model_path) = coordinator
            .prepare_model_update_dispatch::<4, 1>(
                Some(&requested_actor_ids),
                &make_model_metadata(),
            )
            .await
            .unwrap()
            .unwrap();

        let mut expected_target_actor_ids =
            vec![actor_info(actor_ids[0]), actor_info(actor_ids[2])];
        expected_target_actor_ids.sort_by_key(|actor| actor.id().to_string());

        assert_eq!(target_actor_ids, expected_target_actor_ids);
    }

    #[tokio::test]
    async fn dispatch_model_updates_sends_expected_targets_and_versions() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
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
                    .new_actor::<4, 1>(
                        *actor_id,
                        Arc::from("router-a"),
                        DeviceType::Cpu,
                        100,
                        None,
                        None,
                        tx_to_buffer.clone(),
                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        AlgorithmInitArgs::default(),
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
                match message.protocol {
                    RoutingProtocol::Control(ControlPayload::ModelVersion { reply_to }) => {
                        let current_version = current_versions
                            .iter()
                            .find(|(actor_id, _)| *actor_id == message.actor_id)
                            .map(|(_, version)| *version)
                            .unwrap();
                        let _ = reply_to.send(current_version);
                    }
                    RoutingProtocol::Control(ControlPayload::ModelUpdate {
                        model_bytes,
                        version,
                    }) => {
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
            .prepare_model_update_dispatch::<4, 1>(None, &make_model_metadata())
            .await
            .unwrap()
            .unwrap();
        ClientCoordinator::<TestBackend>::dispatch_model_updates(
            global_dispatcher_tx,
            &target_actor_ids,
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
        let result = c.scale_routers_out(1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn scale_in_no_runtime_returns_err() {
        let mut c = make_coordinator();
        let result = c.scale_routers_in(1).await;
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

    async fn new_actor_with_nametag(
        shared_state: &Arc<RwLock<StateManager<TestBackend>>>,
        actor_id: Uuid,
        nametag: Option<NameTag>,
    ) {
        let (tx_to_buffer, _buffer_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        shared_state
            .write()
            .await
            .new_actor::<4, 1>(
                actor_id,
                Arc::from("router-a"),
                DeviceType::Cpu,
                100,
                nametag,
                None,
                tx_to_buffer,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                AlgorithmInitArgs::default(),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn get_actors_by_tag_some_returns_matching_tag_regardless_of_duplicate() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (coordinator, shared_state, _rx) = make_runtime_coordinator(client_modes).await;

        let scout_0 = Uuid::new_v4();
        let scout_1 = Uuid::new_v4();
        let untagged = Uuid::new_v4();

        new_actor_with_nametag(
            &shared_state,
            scout_0,
            Some(NameTag {
                tag: "scout".to_string(),
                duplicate: 0,
            }),
        )
        .await;
        new_actor_with_nametag(
            &shared_state,
            scout_1,
            Some(NameTag {
                tag: "scout".to_string(),
                duplicate: 1,
            }),
        )
        .await;
        new_actor_with_nametag(&shared_state, untagged, None).await;

        let mut matched: Vec<Uuid> = coordinator
            .get_actors_by_tag(Some("scout"))
            .await
            .unwrap()
            .iter()
            .map(|actor| actor.id())
            .collect();
        matched.sort_by_key(|id| id.to_string());

        let mut expected = vec![scout_0, scout_1];
        expected.sort_by_key(|id| id.to_string());

        assert_eq!(matched, expected);
    }

    #[tokio::test]
    async fn get_actors_by_tag_none_returns_untagged_actors_only() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (coordinator, shared_state, _rx) = make_runtime_coordinator(client_modes).await;

        let tagged = Uuid::new_v4();
        let untagged = Uuid::new_v4();

        new_actor_with_nametag(
            &shared_state,
            tagged,
            Some(NameTag {
                tag: "scout".to_string(),
                duplicate: 0,
            }),
        )
        .await;
        new_actor_with_nametag(&shared_state, untagged, None).await;

        let matched: Vec<Uuid> = coordinator
            .get_actors_by_tag(None)
            .await
            .unwrap()
            .iter()
            .map(|actor| actor.id())
            .collect();

        assert_eq!(matched, vec![untagged]);
    }

    #[tokio::test]
    async fn resolve_new_nametag_none_input_returns_none() {
        let c = make_coordinator();
        let result = c.resolve_new_nametag(None, 3).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn resolve_new_nametag_starts_duplicates_after_existing_max() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (coordinator, shared_state, _rx) = make_runtime_coordinator(client_modes).await;

        new_actor_with_nametag(
            &shared_state,
            Uuid::new_v4(),
            Some(NameTag {
                tag: "scout".to_string(),
                duplicate: 3,
            }),
        )
        .await;
        new_actor_with_nametag(
            &shared_state,
            Uuid::new_v4(),
            Some(NameTag {
                tag: "scout".to_string(),
                duplicate: 1,
            }),
        )
        .await;
        // A different tag's duplicate counter must not influence "scout"'s allocation.
        new_actor_with_nametag(
            &shared_state,
            Uuid::new_v4(),
            Some(NameTag {
                tag: "other".to_string(),
                duplicate: 9,
            }),
        )
        .await;

        let resolved = coordinator
            .resolve_new_nametag(Some("scout"), 2)
            .await
            .unwrap()
            .expect("Some(tag) input should resolve to Some(tags)");

        assert_eq!(
            resolved,
            vec![
                NameTag {
                    tag: "scout".to_string(),
                    duplicate: 4
                },
                NameTag {
                    tag: "scout".to_string(),
                    duplicate: 5
                },
            ]
        );
    }

    #[tokio::test]
    async fn resolve_new_nametag_unused_tag_starts_at_zero() {
        let client_modes = ClientModes {
            actor_inference_mode: ActorInferenceMode::Client(ModelMode::Independent),
            actor_data_mode: ActorDataMode::Disabled,
        };
        let (coordinator, _shared_state, _rx) = make_runtime_coordinator(client_modes).await;

        let resolved = coordinator
            .resolve_new_nametag(Some("fresh"), 1)
            .await
            .unwrap()
            .expect("Some(tag) input should resolve to Some(tags)");

        assert_eq!(
            resolved,
            vec![NameTag {
                tag: "fresh".to_string(),
                duplicate: 0
            }]
        );
    }
}
