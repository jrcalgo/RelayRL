//! Client API for starting and controlling the RelayRL client runtime.
//!
//! This module provides:
//! - `RelayRLAgent`: a thin facade over the runtime coordinator.
//! - `AgentBuilder`: ergonomic construction of an agent instance plus its startup parameters.
//! - Mode/config enums that describe inference and trajectory recording behavior.
//!
//! Beta scope in `0.5.0-beta`:
//! - Supported: the local/default client path, including local inference, actor lifecycle
//!   management, router scaling, and local trajectory writing.
//! - Experimental: transport-backed and server-backed workflows enabled by
//!   `zmq-transport` or `nats-transport`.

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::HyperparameterArgs;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
use crate::network::client::runtime::coordination::coordinator::{
    ClientActors, ClientCoordinator, ClientEnvironments, ClientInterface, CoordinatorError,
};
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::prelude::config::ClientConfigLoader;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::{Algorithm, NetworkParams};

use active_uuid_registry::UuidPoolError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::get_context_entries;
use active_uuid_registry::interface::list_ids;
use relayrl_env_trait::traits::Environment;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{
    AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
    IntBurnTensor, SupportedTensorBackend,
};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::ModelModule;
use relayrl_types::model::utils::validate_module;

use active_uuid_registry::registry_uuid::Uuid;

use burn_tensor::{BasicOps, Bool, Float, Int, Tensor, TensorKind, backend::Backend};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "metrics", feature = "logging"))]
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

/// Errors returned by the client API.
#[derive(Debug, Error)]
pub enum ClientError {
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error("Inference server mode disabled: {0}")]
    InferenceServerModeDisabled(String),
    #[error("Inference server mode enabled: {0}")]
    InferenceServerModeEnabled(String),
    #[error(transparent)]
    CoordinatorError(#[from] CoordinatorError),
    #[error("Backend mismatch: {0}")]
    BackendMismatchError(String),
    #[error("No input or output dtype set")]
    NoInputOrOutputDtypeSet(String),
    #[error("Noop router scale: {0}")]
    NoopRouterScale(String),
    #[error("Noop actor count: {0}")]
    NoopActorCount(String),
    #[error("Invalid inference mode: {0}")]
    InvalidInferenceMode(String),
    #[error("Invalid trajectory file directory: {0}")]
    InvalidTrajectoryFileDirectory(String),
    #[error("Invalid env count: {0}")]
    InvalidEnvCount(String),
    #[error("Model validation failed: {0}")]
    ModelValidationFailed(String),
    #[error("Update model is not supported: {0}")]
    ModelUpdateNotSupported(String),
}

/// Output target for runtime statistics collection.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg(any(feature = "metrics", feature = "logging"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "metrics", feature = "logging"))))]
pub enum RuntimeStatisticsReturnType {
    /// Serialize statistics to a JSON file at the given path.
    JsonFile(PathBuf),
    /// Serialize statistics to an in-memory JSON string.
    JsonString(String),
    /// Materialize a flattened view of runtime statistics.
    Hashmap(HashMap<String, String>),
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Debug, Clone)]
pub struct AlgorithmArgs {
    pub algorithm: Algorithm,
    pub hyperparams: Option<HyperparameterArgs>,
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
impl Default for AlgorithmArgs {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::ConfigInit,
            hyperparams: None,
        }
    }
}

/// Experimental ZMQ endpoints for server-backed inference workflows.
#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub struct ZmqInferenceAddressesArgs {
    pub inference_server_address: Option<NetworkParams>,
    pub inference_scaling_server_address: Option<NetworkParams>,
}

/// Experimental ZMQ endpoints for server-backed training workflows.
#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub struct ZmqTrainingAddressesArgs {
    pub agent_listener_address: Option<NetworkParams>,
    pub model_server_address: Option<NetworkParams>,
    pub trajectory_server_address: Option<NetworkParams>,
    pub training_scaling_server_address: Option<NetworkParams>,
}

/// Experimental transport address configuration for server-backed inference.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceAddressesArgs {
    #[cfg(feature = "zmq-transport")]
    ZMQ(ZmqInferenceAddressesArgs),
    #[cfg(feature = "nats-transport")]
    NATS(Option<NetworkParams>),
}

/// Experimental transport address configuration for server-backed training.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingAddressesArgs {
    #[cfg(feature = "zmq-transport")]
    ZMQ(ZmqTrainingAddressesArgs),
    #[cfg(feature = "nats-transport")]
    NATS(Option<NetworkParams>),
}

/// Experimental configuration for server-backed inference.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct InferenceParams {
    pub model_mode: ModelMode,
    pub inference_addresses: Option<InferenceAddressesArgs>,
}

/// Experimental configuration for server-backed training.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct TrainingParams {
    pub model_mode: ModelMode,
    pub training_addresses: Option<TrainingAddressesArgs>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LocalTrajectoryFileType {
    Csv,
    Arrow,
}

/// File-based trajectory recording parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalTrajectoryFileParams {
    pub directory: PathBuf,
    pub file_type: LocalTrajectoryFileType,
}

impl LocalTrajectoryFileParams {
    pub fn new(
        directory: PathBuf,
        file_type: LocalTrajectoryFileType,
    ) -> Result<Self, ClientError> {
        if directory.as_os_str().is_empty() {
            return Err(ClientError::InvalidTrajectoryFileDirectory(format!(
                "Path '{}' is empty",
                directory.display()
            )));
        }

        {
            const TOTAL_ATTEMPTS: i32 = 2;
            let mut attempts: i32 = 1;
            // Ensure the output directory exists before returning the validated parameters.
            while !directory.exists() {
                // Retry once in case the first `create_dir_all` attempt fails transiently.
                match std::fs::create_dir_all(&directory) {
                    Ok(_) => break,
                    Err(_) if attempts < TOTAL_ATTEMPTS => {
                        attempts += 1;
                        continue;
                    }
                    Err(e) => {
                        return Err(ClientError::InvalidTrajectoryFileDirectory(e.to_string()));
                    }
                }
            }
        }

        if !directory.is_dir() {
            return Err(ClientError::InvalidTrajectoryFileDirectory(format!(
                "Path is not a directory, {}",
                directory.display()
            )));
        }

        Ok(Self {
            directory,
            file_type,
        })
    }
}

impl Default for LocalTrajectoryFileParams {
    fn default() -> Self {
        Self::new(PathBuf::from("."), LocalTrajectoryFileType::Csv).unwrap_or_else(|_| {
            log::error!(
                "Failed to validate the default local trajectory directory, falling back to the current directory"
            );
            Self {
                directory: PathBuf::from("."),
                file_type: LocalTrajectoryFileType::Csv,
            }
        })
    }
}

/// Shared-model semantics are fully supported for local inference in `0.5.0-beta`.
///
/// Server-backed uses of `ModelMode` are still experimental.
#[non_exhaustive]
#[derive(Default, Debug, Clone, PartialEq)]
pub enum ModelMode {
    /// Each actor has an independent model handle.
    #[default]
    Independent,
    /// Actors on the same device share a model handle.
    Shared,
}

/// Inference mode used by runtime actors.
///
/// The local path is the beta-supported path in `0.5.0-beta`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorInferenceMode {
    /// Inference occurs locally in the local runtime actor.
    Local(ModelMode),
    /// Experimental: inference occurs on external inference server(s).
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Server(InferenceParams),
    /// Experimental: inference occurs locally for one actor, remote inference for others.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    ServerOverflow(ModelMode, InferenceParams),
}

impl Default for ActorInferenceMode {
    fn default() -> Self {
        Self::Local(ModelMode::default())
    }
}

/// Training mode used by runtime actors for training data collection and processing.
///
/// Offline local trajectory writing is part of the beta-supported local/default path.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorTrainingDataMode {
    /// Experimental: training data is sent to the server for processing.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Online(TrainingParams),
    /// Training data is recorded to a local file.
    OfflineWithFiles(Option<LocalTrajectoryFileParams>),
    /// Training data is recorded to a local memory buffer.
    OfflineWithMemory,
    /// Training data is recorded to a local file and memory buffer.
    OfflineWithFilesAndMemory(Option<LocalTrajectoryFileParams>),
    /// Experimental: training data is sent to the server and also recorded locally.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    OnlineWithFiles(TrainingParams, Option<LocalTrajectoryFileParams>),
    /// Experimental: training data is sent to the server and also recorded in memory.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    OnlineWithMemory(TrainingParams),
    /// Experimental: training data is sent to the server and also recorded in file and memory.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    OnlineWithFilesAndMemory(TrainingParams, Option<LocalTrajectoryFileParams>),
    /// Training data collection and processing is disabled
    Disabled,
}

impl Default for ActorTrainingDataMode {
    fn default() -> Self {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        return Self::Online(TrainingParams::default());
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        return Self::OfflineWithMemory;
    }
}

pub(crate) fn uses_local_file_writing(training_data_mode: &ActorTrainingDataMode) -> bool {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    return matches!(
        training_data_mode,
        ActorTrainingDataMode::OfflineWithFiles(_)
            | ActorTrainingDataMode::OfflineWithFilesAndMemory(_)
            | ActorTrainingDataMode::OnlineWithFiles(_, _)
            | ActorTrainingDataMode::OnlineWithFilesAndMemory(_, _)
    );
    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    return matches!(
        training_data_mode,
        ActorTrainingDataMode::OfflineWithFiles(_)
            | ActorTrainingDataMode::OfflineWithFilesAndMemory(_)
    );
}

pub(crate) fn uses_in_memory_data(training_data_mode: &ActorTrainingDataMode) -> bool {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    return matches!(
        training_data_mode,
        ActorTrainingDataMode::OfflineWithMemory
            | ActorTrainingDataMode::OfflineWithFilesAndMemory(_)
            | ActorTrainingDataMode::OnlineWithMemory(_)
            | ActorTrainingDataMode::OnlineWithFilesAndMemory(_, _)
    );

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    return matches!(
        training_data_mode,
        ActorTrainingDataMode::OfflineWithMemory
            | ActorTrainingDataMode::OfflineWithFilesAndMemory(_)
    );
}

/// Runtime modes consumed by the client to enable/disable functionality.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ClientModes {
    pub actor_inference_mode: ActorInferenceMode,
    pub actor_training_data_mode: ActorTrainingDataMode,
}

/// Parameters used to start a [`RelayRLAgent`].
///
/// Typically constructed via [`AgentBuilder::build`] and then passed to
/// [`RelayRLAgent::start`] or [`RelayRLAgent::restart`].
#[derive(Clone)]
pub struct AgentStartParameters<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    pub algorithm_args: AlgorithmArgs,
    pub actor_count: u32,
    pub router_scale: u32,
    pub default_device: DeviceType,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    pub default_model: Option<ModelModule<B>>,
    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[cfg_attr(
        docsrs,
        doc(cfg(not(any(feature = "nats-transport", feature = "zmq-transport"))))
    )]
    pub default_model: ModelModule<B>,
    pub config_path: Option<PathBuf>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for AgentStartParameters<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RLAgentStartParameters")
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> AgentStartParameters<B> {
    fn infer_dtypes(&self) -> (Option<DType>, Option<DType>) {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        {
            if let Some(model_module) = &self.default_model {
                return (
                    Some(model_module.metadata.input_dtype.clone()),
                    Some(model_module.metadata.output_dtype.clone()),
                );
            }

            (None, None)
        }

        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        {
            (
                Some(self.default_model.metadata.input_dtype.clone()),
                Some(self.default_model.metadata.output_dtype.clone()),
            )
        }
    }
}

/// Builder for creating a [`RelayRLAgent`] and its startup parameters.
///
/// This builder is `#[must_use]`: setters return an updated value.
///
/// # Examples
///
/// ```rust,no_run
/// use relayrl_framework::prelude::network::{AgentBuilder, RelayRLAgentActors};
/// use relayrl_framework::prelude::types::model::ModelModule;
/// use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;
/// use burn_ndarray::NdArray;
/// use burn_tensor::Float;
/// use std::path::PathBuf;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
/// let (mut agent, params) = AgentBuilder::<NdArray, 1, 1, Float, Float>::builder()
///     .default_device(DeviceType::Cpu)
///     .default_model(default_model)
///     .config_path(PathBuf::from("client_config.json"))
///     .build()
///     .await?;
///
/// agent.start(params).await?;
/// let _actor_ids = agent.get_actor_ids()?;
/// agent.shutdown().await?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct AgentBuilder<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + Send + Sync,
    KindOut: TensorKind<B> + Send + Sync,
> {
    pub client_modes: ClientModes,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub transport_type: Option<TransportType>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub algorithm_args: Option<AlgorithmArgs>,
    pub actor_count: Option<u32>,
    pub router_scale: Option<u32>,
    pub default_device: Option<DeviceType>,
    pub default_model: Option<ModelModule<B>>,
    pub config_path: Option<PathBuf>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub codec: Option<CodecConfig>,
    _phantom: PhantomData<(KindIn, KindOut)>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + Send + Sync,
    KindOut: TensorKind<B> + Send + Sync,
> AgentBuilder<B, D_IN, D_OUT, KindIn, KindOut>
{
    /// Create a new builder initialized with sensible default values.
    ///
    /// Notes:
    /// - Modes default to local inference.
    /// - Transport default to `ZMQ` when enabled by feature flags.
    pub fn builder() -> Self {
        Self {
            client_modes: ClientModes::default(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_type: Some(TransportType::default()),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args: Some(AlgorithmArgs::default()),
            actor_count: None,
            router_scale: None,
            default_device: None,
            default_model: None,
            config_path: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            codec: None,
            _phantom: PhantomData,
        }
    }

    pub fn actor_inference_mode(mut self, actor_inference_mode: ActorInferenceMode) -> Self {
        self.client_modes.actor_inference_mode = actor_inference_mode;
        self
    }

    pub fn actor_training_data_mode(
        mut self,
        actor_training_data_mode: ActorTrainingDataMode,
    ) -> Self {
        self.client_modes.actor_training_data_mode = actor_training_data_mode;
        self
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn transport_type(mut self, transport_type: TransportType) -> Self {
        self.transport_type = Some(transport_type);
        self
    }

    pub fn actor_count(mut self, count: u32) -> Self {
        self.actor_count = Some(count);
        self
    }

    pub fn router_scale(mut self, count: u32) -> Self {
        self.router_scale = Some(count);
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

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        let hyperparams = match self.algorithm_args {
            Some(args) => args.hyperparams,
            None => None,
        };
        self.algorithm_args = Some(AlgorithmArgs {
            algorithm,
            hyperparams,
        });
        self
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn hyperparams(mut self, hyperparams: HyperparameterArgs) -> Self {
        let algorithm = match self.algorithm_args {
            Some(args) => args.algorithm,
            None => Algorithm::ConfigInit,
        };

        self.algorithm_args = Some(AlgorithmArgs {
            algorithm,
            hyperparams: Some(hyperparams),
        });
        self
    }

    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn codec(mut self, codec: CodecConfig) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Build the agent facade plus its startup parameters.
    ///
    /// # Errors
    /// Returns an error if the selected modes are internally inconsistent.
    pub async fn build(
        self,
    ) -> Result<
        (
            RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>,
            AgentStartParameters<B>,
        ),
        ClientError,
    > {
        // Initialize agent object
        let agent: RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut> =
            RelayRLAgent::<B, D_IN, D_OUT, KindIn, KindOut>::new(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                self.transport_type.unwrap_or_default(),
                self.client_modes,
            );

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args: self.algorithm_args.unwrap_or_default(),
            actor_count: self.actor_count.unwrap_or(1),
            router_scale: self.router_scale.unwrap_or(1),
            default_device: self.default_device.unwrap_or_default(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_model: self.default_model,
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            default_model: self.default_model.expect(
                "AgentBuilder::build requires `default_model` for the local/default runtime",
            ),
            config_path: self.config_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            codec: self.codec.unwrap_or_default(),
        };

        Ok((agent, startup_params))
    }
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

/// Client entry point for the RelayRL framework.
///
/// `RelayRLAgent` is a thin facade over the runtime coordinator, providing a stable public API
/// for starting, scaling, and interacting with runtime actors.
///
/// In `0.5.0-beta`, the supported path is the local/default client runtime.
/// Transport-backed and server-backed flows remain experimental.
pub struct RelayRLAgent<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
> {
    coordinator: ClientCoordinator<B, D_IN, D_OUT, KindIn, KindOut>,
    supported_backend: SupportedTensorBackend,
    input_dtype: Option<DType>,
    output_dtype: Option<DType>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + Send + Sync,
    KindOut: TensorKind<B> + Send + Sync,
> std::fmt::Debug for RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RLAgent")
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + Send + Sync,
    KindOut: TensorKind<B> + Send + Sync,
> RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>
{
    /// Create a new agent facade using runtime-invariant parameters.
    ///
    /// # Errors
    /// Returns [`ClientError::InvalidInferenceMode`] if the selected [`ClientModes`] are
    /// incompatible (e.g., server inference requested while inference server mode is disabled).
    ///
    /// Returns [`ClientError::CoordinatorError`] if the runtime coordinator fails to initialize.
    pub fn new(
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: TransportType,
        client_modes: ClientModes,
    ) -> Self {
        Self {
            coordinator: ClientCoordinator::<B, D_IN, D_OUT, KindIn, KindOut>::new(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                transport_type,
                client_modes,
            ),
            supported_backend: B::get_supported_backend(),
            input_dtype: None,
            output_dtype: None,
        }
    }

    /// Start the client runtime with the specified parameters.
    ///
    /// This spawns the coordinator runtime components and (by default) creates `actor_count`
    /// runtime actors.
    ///
    /// # Errors
    /// Returns an error if startup fails (configuration, runtime init, transport init, etc).
    pub async fn start(&mut self, params: AgentStartParameters<B>) -> Result<(), ClientError> {
        let (input_dtype, output_dtype) = params.infer_dtypes();
        self.input_dtype = input_dtype;
        self.output_dtype = output_dtype;

        let AgentStartParameters {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args,
            actor_count,
            router_scale,
            default_device,
            default_model,
            config_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            codec,
        } = params;

        self.coordinator
            .start(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                algorithm_args,
                actor_count,
                router_scale,
                default_device,
                default_model,
                config_path,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                Some(codec),
            )
            .await
            .map_err(Into::<ClientError>::into)?;

        Ok(())
    }

    /// Restart the Agent's client runtime components
    ///
    /// # Errors
    /// Returns an error if restart coordination fails.
    pub async fn restart(&mut self, params: AgentStartParameters<B>) -> Result<(), ClientError> {
        let (input_dtype, output_dtype) = params.infer_dtypes();
        self.input_dtype = input_dtype;
        self.output_dtype = output_dtype;

        let AgentStartParameters {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args,
            actor_count,
            router_scale,
            default_device,
            default_model,
            config_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            codec,
        } = params;

        self.coordinator
            .restart(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                algorithm_args,
                actor_count,
                router_scale,
                default_device,
                default_model,
                config_path,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                Some(codec),
            )
            .await?;
        Ok(())
    }

    /// Gracefully shut down the Agent's client runtime components
    ///
    /// # Errors
    /// Returns an error if shutdown coordination fails.
    pub async fn shutdown(&mut self) -> Result<(), ClientError> {
        self.coordinator.shutdown().await?;
        Ok(())
    }

    /// Scale actor throughput by adjusting the number of routing workers.
    ///
    /// - `router_scale > 0`: scale out by that amount.
    /// - `router_scale < 0`: scale in by the absolute value.
    ///
    /// # Errors
    /// Returns [`ClientError::NoopRouterScale`] if `router_scale == 0`.
    pub async fn scale_throughput(&mut self, router_scale: i32) -> Result<(), ClientError> {
        match router_scale {
            add if router_scale > 0 => {
                self.coordinator.scale_out(add as u32).await?;
                Ok(())
            }
            remove if router_scale < 0 => {
                self.coordinator.scale_in(remove.unsigned_abs()).await?;
                Ok(())
            }
            _ => Err(ClientError::NoopRouterScale(
                "Noop router scale: `router_scale` set to zero in `scale_throughput()`".to_string(),
            )),
        }
    }

    /// Request actions from the specified actor IDs (if they exist)
    ///
    /// This will send the action request to the specified actor instances and return the action responses
    ///
    /// # Errors
    /// Returns [`ClientError::BackendMismatchError`] if the agent’s backend `B` does not match
    /// the configured runtime backend.
    pub async fn request_action(
        &self,
        ids: Vec<Uuid>,
        observation: Tensor<B, D_IN, KindIn>,
        mask: Option<Tensor<B, D_OUT, KindOut>>,
        reward: f32,
    ) -> Result<Vec<(ActorUuid, Arc<RelayRLAction>)>, ClientError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>,
    {
        match B::matches_backend(&self.supported_backend) {
            true => {
                if let (Some(input_dtype), Some(output_dtype)) =
                    (self.input_dtype.clone(), self.output_dtype.clone())
                {
                    let obs_tensor: Arc<AnyBurnTensor<B, D_IN>> =
                        Arc::new(observation.to_any_burn_tensor(input_dtype));
                    let mask_tensor: Option<Arc<AnyBurnTensor<B, D_OUT>>> =
                        mask.map(|tensor| Arc::new(tensor.to_any_burn_tensor(output_dtype)));

                    let result = self
                        .coordinator
                        .request_action(ids, obs_tensor, mask_tensor, reward)
                        .await?;
                    Ok(result)
                } else {
                    Err(ClientError::NoInputOrOutputDtypeSet(
                        "No input or output dtype set in agent".to_string(),
                    ))
                }
            }
            false => Err(ClientError::BackendMismatchError(
                "Backend mismatch; Tensor backends not (currently) supported by RelayRL"
                    .to_string(),
            )),
        }
    }

    /// Mark the last action as terminal (`done=true`) for the specified actor IDs (if they exist)
    ///
    /// Appends a RelayRLAction with the done flag set to `true` and the specified reward (if any) to the actor's current trajectory.
    ///
    /// # Errors
    /// Returns an error if the actor(s) do not exist or the coordinator rejects the request.
    pub async fn flag_last_action(
        &self,
        ids: Vec<Uuid>,
        reward: Option<f32>,
    ) -> Result<(), ClientError> {
        self.coordinator.flag_last_action(ids, reward).await?;
        Ok(())
    }

    /// Update the model for all actors or for the specified actor IDs (if they exist).
    ///
    /// When `actor_ids` is `Some`, only the listed actors are considered for the update.
    /// In `ModelMode::Shared`, the runtime still updates one representative actor per relevant
    /// device so each shared model handle is refreshed only once.
    pub async fn update_model(
        &self,
        model: ModelModule<B>,
        actor_ids: Option<Vec<ActorUuid>>,
    ) -> Result<(), ClientError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let ActorTrainingDataMode::Online(_)
        | ActorTrainingDataMode::OnlineWithFiles(_, _)
        | ActorTrainingDataMode::OnlineWithMemory(_) =
            self.coordinator.client_modes.actor_training_data_mode
        {
            log::warn!(
                "Updating model locally is not supported in Online or Hybrid training data modes"
            );
            return Err(ClientError::ModelUpdateNotSupported(
                "Updating model locally is not supported in Online or Hybrid training data modes"
                    .to_string(),
            ));
        }

        if let Err(e) = validate_module::<B>(&model) {
            return Err(ClientError::ModelValidationFailed(e.to_string()));
        }
        self.coordinator.update_model(model, actor_ids).await?;
        Ok(())
    }

    /// Retrieves the model version for each actor ID listed (if instance IDs exist)
    ///
    /// Returns `(ActorID, ModelVersion)` pairs.
    pub async fn get_model_version(
        &self,
        actor_ids: Vec<Uuid>,
    ) -> Result<Vec<(Uuid, i64)>, ClientError> {
        Ok(self.coordinator.get_model_version(actor_ids).await?)
    }

    pub async fn get_trajectory_memory(
        &self,
    ) -> Result<Arc<DashMap<Uuid, Vec<Arc<RelayRLTrajectory>>>>, ClientError> {
        Ok(self.coordinator.get_trajectory_memory().await?)
    }

    /// Fetch the active client configuration.
    pub async fn get_config(&self) -> Result<ClientConfigLoader, ClientError> {
        Ok(self.coordinator.get_config().await?)
    }

    /// Set the configuration path used by the runtime.
    pub async fn set_config_path(&self, config_path: PathBuf) -> Result<(), ClientError> {
        self.coordinator.set_config_path(config_path).await?;
        Ok(())
    }
}

/// Actor management trait using boxed futures
pub trait RelayRLAgentActors<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
>
{
    fn new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn new_actors(
        &mut self,
        count: u32,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn remove_actor(
        &mut self,
        id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn remove_actors(
        &mut self,
        ids: Vec<Uuid>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
    fn get_actor_ids(&mut self) -> Result<Vec<ActorUuid>, ClientError>;
    fn set_actor_id(
        &mut self,
        current_id: Uuid,
        new_id: Uuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>>;
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + Send + Sync,
    KindOut: TensorKind<B> + Send + Sync,
> RelayRLAgentActors<B, D_IN, D_OUT, KindIn, KindOut>
    for RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>
{
    /// Creates a new actor instance on the specified device with the specified model
    fn new_actor(
        &mut self,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let _ = self
                .coordinator
                .new_actor(device, default_model, true, true)
                .await?;
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            let _ = self.coordinator.new_actor(device, default_model).await?;
            Ok(())
        })
    }

    /// Creates `n` new actor instances on the specified device with the specified model
    fn new_actors(
        &mut self,
        count: u32,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        if count == 0 {
            Box::pin(async move {
                Err(ClientError::NoopActorCount(
                    "Noop actor count: `count` set to zero".to_string(),
                ))
            })
        } else if count == 1 {
            self.new_actor(device, default_model)
        } else {
            Box::pin(async move {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                let mut actor_ids: Vec<Uuid> = Vec::new();
                for _ in 0..count {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    actor_ids.push(
                        self.coordinator
                            .new_actor(device.clone(), default_model.clone(), false, false)
                            .await?,
                    );
                    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                    self.coordinator
                        .new_actor(device.clone(), default_model.clone())
                        .await?;
                }

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if let (
                    ActorTrainingDataMode::Online(_)
                    | ActorTrainingDataMode::OnlineWithFiles(_, _)
                    | ActorTrainingDataMode::OnlineWithMemory(_),
                    ActorInferenceMode::Server(_),
                ) = (
                    &self.coordinator.client_modes.actor_training_data_mode,
                    &self.coordinator.client_modes.actor_inference_mode,
                ) {
                    // sends all new actor ids to the server
                    let actor_entries = {
                        let client_namespace = self
                            .coordinator
                            .runtime_params
                            .as_ref()
                            .ok_or(ClientError::CoordinatorError(
                                CoordinatorError::NoRuntimeInstanceError,
                            ))?
                            .client_namespace
                            .as_ref();
                        get_context_entries(client_namespace, crate::network::ACTOR_CONTEXT)?
                    };

                    self.coordinator
                        .send_client_ids_to_server(actor_entries.clone(), true)
                        .await?;

                    if let ActorTrainingDataMode::Online(_)
                    | ActorTrainingDataMode::OnlineWithFiles(_, _)
                    | ActorTrainingDataMode::OnlineWithMemory(_) =
                        &self.coordinator.client_modes.actor_training_data_mode
                    {
                        self.coordinator
                            .send_algorithm_init_request(actor_entries.clone())
                            .await?;
                    }

                    if let ActorInferenceMode::Server(_) =
                        &self.coordinator.client_modes.actor_inference_mode
                    {
                        self.coordinator
                            .send_inference_model_init_request(actor_entries, default_model.clone())
                            .await?;
                    }
                }

                Ok(())
            })
        }
    }

    /// Removes the actor instance with the specified ID from the current Agent instance
    fn remove_actor(
        &mut self,
        actor_id: ActorUuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator
                .remove_actor(
                    actor_id,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    true,
                )
                .await?;
            Ok(())
        })
    }

    fn remove_actors(
        &mut self,
        actor_ids: Vec<ActorUuid>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        if actor_ids.is_empty() {
            Box::pin(async move {
                Err(ClientError::NoopActorCount(
                    "Noop actor count: `actor_ids` is empty in `remove_actors()`".to_string(),
                ))
            })
        } else if actor_ids.len() == 1 {
            self.remove_actor(actor_ids[0])
        } else {
            Box::pin(async move {
                for actor_id in actor_ids {
                    self.coordinator
                        .remove_actor(
                            actor_id,
                            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                            false,
                        )
                        .await?;
                }

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if let (
                    ActorTrainingDataMode::Online(_)
                    | ActorTrainingDataMode::OnlineWithFiles(_, _)
                    | ActorTrainingDataMode::OnlineWithMemory(_),
                    ActorInferenceMode::Server(_),
                ) = (
                    &self.coordinator.client_modes.actor_training_data_mode,
                    &self.coordinator.client_modes.actor_inference_mode,
                ) {
                    let client_actor_ids = {
                        let client_namespace = self
                            .coordinator
                            .runtime_params
                            .as_ref()
                            .ok_or(ClientError::CoordinatorError(
                                CoordinatorError::NoRuntimeInstanceError,
                            ))?
                            .client_namespace
                            .as_ref();
                        get_context_entries(client_namespace, crate::network::ACTOR_CONTEXT)?
                    };

                    self.coordinator
                        .send_client_ids_to_server(client_actor_ids, true)
                        .await?;
                }

                Ok(())
            })
        }
    }

    /// Retrieves the current actor instance IDs
    fn get_actor_ids(&mut self) -> Result<Vec<ActorUuid>, ClientError> {
        let client_namespace = self
            .coordinator
            .runtime_params
            .as_ref()
            .ok_or(ClientError::CoordinatorError(
                CoordinatorError::NoRuntimeInstanceError,
            ))?
            .client_namespace
            .as_ref();
        let actor_ids = list_ids(client_namespace, "actor");
        Ok(actor_ids)
    }

    /// Sets the ID of the actor instance with the specified current ID to the new ID
    /// .ok_or("[ClientFilter] Actor not found".to_string())
    /// This will update the actor instance's ID in the Agent's coordinator state manager
    fn set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator.set_actor_id(current_id, new_id).await?;
            Ok(())
        })
    }
}

#[allow(async_fn_in_trait)]
pub trait RelayRLActorEnv<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
>
{
    async fn run_env(&self, actor_id: ActorUuid, step_count: usize) -> Result<(), ClientError>;
    async fn set_env(
        &mut self,
        actor_id: ActorUuid,
        env: Box<dyn Environment>,
        count: u32,
    ) -> Result<(), ClientError>;
    async fn remove_env(&mut self, actor_id: ActorUuid) -> Result<(), ClientError>;
    async fn get_env_count(&self, actor_id: ActorUuid) -> Result<u32, ClientError>;
    async fn set_env_count(&mut self, actor_id: ActorUuid, count: u32) -> Result<(), ClientError>;
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + BasicOps<B> + Send + Sync + 'static,
    KindOut: TensorKind<B> + BasicOps<B> + Send + Sync + 'static,
> RelayRLActorEnv<B, D_IN, D_OUT, KindIn, KindOut>
    for RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>
{
    async fn run_env(&self, actor_id: ActorUuid, step_count: usize) -> Result<(), ClientError> {
        Ok(self.coordinator.run_env(actor_id, step_count).await?)
    }

    async fn set_env(
        &mut self,
        actor_id: ActorUuid,
        env: Box<dyn Environment>,
        count: u32,
    ) -> Result<(), ClientError> {
        Ok(self.coordinator.set_env(actor_id, env, count).await?)
    }

    async fn remove_env(&mut self, actor_id: ActorUuid) -> Result<(), ClientError> {
        Ok(self.coordinator.remove_env(actor_id).await?)
    }

    async fn set_env_count(&mut self, actor_id: ActorUuid, count: u32) -> Result<(), ClientError> {
        let current = self.coordinator.get_env_count(actor_id).await?;
        match count.cmp(&current) {
            std::cmp::Ordering::Greater => Ok(self
                .coordinator
                .increase_env_count(actor_id, count - current)
                .await?),
            std::cmp::Ordering::Less => Ok(self
                .coordinator
                .decrease_env_count(actor_id, current - count)
                .await?),
            std::cmp::Ordering::Equal => Ok(()),
        }
    }

    async fn get_env_count(&self, actor_id: ActorUuid) -> Result<u32, ClientError> {
        Ok(self.coordinator.get_env_count(actor_id).await?)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Bool, Float, Int, Tensor, TensorData};
    use relayrl_types::data::tensor::{DeviceType, NdArrayDType};
    use relayrl_types::model::{ModelFileType, ModelMetadata};
    use tch::{CModule, Device as TchDevice, Kind, Tensor as TchTensor};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn load_test_model_module() -> (tempfile::TempDir, ModelModule<TestBackend>) {
        let model_dir = tempdir().expect("tempdir should be created");
        let model_path = model_dir.path().join("test.pt");
        let metadata = ModelMetadata {
            model_file: "test.pt".to_string(),
            model_type: ModelFileType::Pt,
            input_dtype: DType::NdArray(NdArrayDType::F32),
            output_dtype: DType::NdArray(NdArrayDType::F32),
            input_shape: vec![2],
            output_shape: vec![2],
            default_device: Some(DeviceType::Cpu),
        };

        let trace_inputs = [TchTensor::zeros([2], (Kind::Float, TchDevice::Cpu))];
        let mut trace_closure =
            |inputs: &[TchTensor]| -> Vec<TchTensor> { vec![inputs[0].shallow_clone()] };
        let traced_module = CModule::create_by_tracing(
            "relayrl_test_module",
            "forward",
            &trace_inputs,
            &mut trace_closure,
        )
        .expect("TorchScript smoke module should be traceable");
        traced_module
            .save(&model_path)
            .expect("TorchScript smoke module should be written");

        metadata
            .save_to_dir(model_dir.path())
            .expect("model metadata should be written");

        let model_module = ModelModule::<TestBackend>::load_from_path(model_dir.path())
            .expect("test TorchScript payload should load through the public model API");

        (model_dir, model_module)
    }

    #[test]
    fn offline_returns_true() {
        assert!(uses_local_file_writing(
            &ActorTrainingDataMode::OfflineWithFiles(None)
        ));
    }

    #[test]
    fn disabled_returns_false() {
        assert!(!uses_local_file_writing(&ActorTrainingDataMode::Disabled));
    }

    #[test]
    fn model_mode_default_is_independent() {
        assert_eq!(ModelMode::default(), ModelMode::Independent);
    }

    #[test]
    fn actor_inference_mode_default_is_local_independent() {
        assert_eq!(
            ActorInferenceMode::default(),
            ActorInferenceMode::Local(ModelMode::Independent),
        );
    }

    #[test]
    fn client_modes_default_uses_component_defaults() {
        let modes = ClientModes::default();
        assert_eq!(modes.actor_inference_mode, ActorInferenceMode::default());
    }

    #[test]
    fn actor_count_setter_sets_field() {
        let b = AgentBuilder::<TestBackend, 4, 1, Float, Float>::builder().actor_count(5);
        assert_eq!(b.actor_count, Some(5));
    }

    #[test]
    fn router_scale_setter_sets_field() {
        let b = AgentBuilder::<TestBackend, 4, 1, Float, Float>::builder().router_scale(2);
        assert_eq!(b.router_scale, Some(2));
    }

    #[test]
    fn actor_count_does_not_change_router_scale() {
        let b = AgentBuilder::<TestBackend, 4, 1, Float, Float>::builder().actor_count(3);
        assert!(b.router_scale.is_none());
    }

    #[test]
    fn local_trajectory_file_params_new_creates_directory() {
        let tmp = tempdir().expect("tempdir should be created");
        let output_dir = tmp.path().join("nested").join("trajectories");

        let params =
            LocalTrajectoryFileParams::new(output_dir.clone(), LocalTrajectoryFileType::Arrow)
                .expect("trajectory params should create the output directory");

        assert_eq!(params.directory, output_dir);
        assert_eq!(params.file_type, LocalTrajectoryFileType::Arrow);
        assert!(params.directory.is_dir());
    }

    #[tokio::test]
    async fn build_returns_start_parameters_for_local_runtime() {
        let config_dir = tempdir().expect("tempdir should be created");
        let config_path = config_dir.path().join("client_config.json");
        let (_model_dir, default_model) = load_test_model_module();

        let (_agent, params) = AgentBuilder::<TestBackend, 1, 1, Float, Float>::builder()
            .default_model(default_model.clone())
            .config_path(config_path.clone())
            .build()
            .await
            .expect("builder should succeed with a local default model");

        assert_eq!(params.actor_count, 1);
        assert_eq!(params.router_scale, 1);
        assert_eq!(params.default_device, DeviceType::Cpu);
        assert_eq!(params.config_path, Some(config_path));
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        assert_eq!(
            params
                .default_model
                .as_ref()
                .expect("builder should preserve the provided default model")
                .metadata
                .input_dtype,
            default_model.metadata.input_dtype
        );
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        assert_eq!(
            params.default_model.metadata.output_dtype,
            default_model.metadata.output_dtype
        );
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        assert_eq!(
            params.default_model.metadata.input_dtype,
            default_model.metadata.input_dtype
        );
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        assert_eq!(
            params
                .default_model
                .as_ref()
                .expect("builder should preserve the provided default model")
                .metadata
                .output_dtype,
            default_model.metadata.output_dtype
        );
    }

    #[tokio::test]
    async fn scale_throughput_zero_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend, 4, 1, Float, Float>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            ClientModes::default(),
        );
        let result = agent.scale_throughput(0).await;
        assert!(matches!(result, Err(ClientError::NoopRouterScale(_))));
    }

    #[tokio::test]
    async fn new_actors_zero_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend, 4, 1, Float, Float>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            ClientModes::default(),
        );
        let result = agent.new_actors(0, DeviceType::Cpu, None).await;
        assert!(matches!(result, Err(ClientError::NoopActorCount(_))));
    }

    #[tokio::test]
    async fn remove_actors_empty_vec_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend, 4, 1, Float, Float>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            ClientModes::default(),
        );
        let result = agent.remove_actors(vec![]).await;
        assert!(matches!(result, Err(ClientError::NoopActorCount(_))));
    }

    #[test]
    fn float_tensor_converts_to_any_burn_tensor_float() {
        let device = NdArrayDevice::default();
        let t: Tensor<TestBackend, 1, Float> = Tensor::zeros([1], &device);
        let result = t.to_any_burn_tensor(DType::NdArray(NdArrayDType::F32));
        assert!(matches!(result, AnyBurnTensor::Float(_)));
    }

    #[test]
    fn int_tensor_converts_to_any_burn_tensor_int() {
        let device = NdArrayDevice::default();
        let data = TensorData::new(vec![0_i64], [1]);
        let t: Tensor<TestBackend, 1, Int> = Tensor::from_data(data, &device);
        let result = t.to_any_burn_tensor(DType::NdArray(NdArrayDType::I32));
        assert!(matches!(result, AnyBurnTensor::Int(_)));
    }

    #[test]
    fn bool_tensor_converts_to_any_burn_tensor_bool() {
        let device = NdArrayDevice::default();
        let float_t: Tensor<TestBackend, 1, Float> = Tensor::zeros([1], &device);
        let bool_t: Tensor<TestBackend, 1, Bool> = float_t.greater_elem(-1.0_f32);
        let result = bool_t.to_any_burn_tensor(DType::NdArray(NdArrayDType::Bool));
        assert!(matches!(result, AnyBurnTensor::Bool(_)));
    }
}
