//! Client API for starting and controlling the RelayRL client runtime.
//!
//! This module provides:
//! - `RelayRLAgent`: a thin facade over the runtime coordinator.
//! - `AgentBuilder`: ergonomic construction of an agent instance plus its startup parameters.
//! - Mode/config enums that describe inference and trajectory recording behavior.
//!
//! Transport and database layers are optional feature flags gating additional functionality.

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::HyperparameterArgs;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
use crate::network::client::runtime::coordination::coordinator::{
    ClientCoordinator, ClientInterface, CoordinatorError,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::scale_manager::AlgorithmArgs;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::prelude::config::ClientConfigLoader;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::{Algorithm, NetworkParams};

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::get_pairs;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{
    AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
    IntBurnTensor, SupportedTensorBackend,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::tensor::{NdArrayDType, TchDType};
use relayrl_types::model::ModelModule;

use burn_tensor::{Bool, Float, Int, Tensor, TensorKind, backend::Backend};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "metrics", feature = "logging"))]
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Errors returned by the client API.
#[non_exhaustive]
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

#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceAddresses {
    pub inference_server_address: Option<NetworkParams>,
    pub inference_scaling_server_address: Option<NetworkParams>,
}

#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingAddresses {
    pub agent_listener_address: Option<NetworkParams>,
    pub model_server_address: Option<NetworkParams>,
    pub trajectory_server_address: Option<NetworkParams>,
    pub training_scaling_server_address: Option<NetworkParams>,
}

#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceParams {
    pub model_mode: ModelMode,
    pub inference_addresses: Option<InferenceAddresses>,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            model_mode: ModelMode::default(),
            inference_addresses: None,
        }
    }
}

#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingParams {
    pub model_mode: ModelMode,
    pub training_addresses: Option<TrainingAddresses>,
}

#[cfg(feature = "zmq-transport")]
impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            model_mode: ModelMode::default(),
            training_addresses: None,
        }
    }
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
        if directory.to_str().as_slice().is_empty() {
            return Err(ClientError::InvalidTrajectoryFileDirectory(format!(
                "Path '{}' is empty",
                directory.display()
            )));
        }

        if directory.extension().is_some() {
            return Err(ClientError::InvalidTrajectoryFileDirectory(format!(
                "Path '{}' appears to be a file, not a directory",
                directory.display()
            )));
        }

        {
            const TOTAL_ATTEMPTS: i32 = 2;
            let mut attempts: i32 = 1;
            while !directory.exists() {
                // loop until directory exists (since create_dir_all can fail) or we've tried too many times
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

        Ok(Self {
            directory,
            file_type,
        })
    }
}

impl Default for LocalTrajectoryFileParams {
    fn default() -> Self {
        Self::new(PathBuf::from("."), LocalTrajectoryFileType::Csv).unwrap() // this will never fail
    }
}

/// TODO: Add architecture support for independent/shared model inference/training requests to server(s).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ModelMode {
    /// Each actor has an independent server-side model
    Independent,
    /// All actors share the same server-side model.
    Shared,
}

impl Default for ModelMode {
    fn default() -> Self {
        Self::Independent
    }
}

/// Inference mode used by local runtime actors.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorInferenceMode {
    /// Inference occurs locally in the local runtime actor.
    Local(ModelMode),
    /// Inference occurs on external inference server(s).
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Server(InferenceParams),
}

impl Default for ActorInferenceMode {
    fn default() -> Self {
        Self::Local(ModelMode::default())
    }
}

/// Training mode used by runtime actors for training data collection and processing
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorTrainingDataMode {
    /// Training data is sent to the server for processing
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Online(TrainingParams),
    /// Training data is recorded to local file
    Offline(Option<LocalTrajectoryFileParams>),
    /// Training data is sent to the server for processing and recorded to local file
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Hybrid(TrainingParams, Option<LocalTrajectoryFileParams>),
    /// Training data collection and processing is disabled
    Disabled,
}

impl Default for ActorTrainingDataMode {
    fn default() -> Self {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        return Self::Online(TrainingParams::default());
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        return Self::Offline(None);
    }
}

/// Runtime modes consumed by the client to enable/disable functionality.
#[derive(Debug, Clone, PartialEq)]
pub struct ClientModes {
    pub actor_inference_mode: ActorInferenceMode,
    pub actor_training_data_mode: ActorTrainingDataMode,
}

impl Default for ClientModes {
    fn default() -> Self {
        Self {
            actor_inference_mode: ActorInferenceMode::default(),
            actor_training_data_mode: ActorTrainingDataMode::default(),
        }
    }
}

/// Parameters used to start a [`RelayRLAgent`].
///
/// Typically constructed via [`AgentBuilder::build`].
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
    #[cfg(any(feature = "tch-model", feature = "onnx-model"))]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "tch-model", feature = "onnx-model"))))]
    pub default_model: Option<ModelModule<B>>,
    #[cfg(not(any(feature = "tch-model", feature = "onnx-model")))]
    #[cfg_attr(
        docsrs,
        doc(cfg(not(any(feature = "tch-model", feature = "onnx-model"))))
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

/// Builder for creating a [`RelayRLAgent`] and its startup parameters.
///
/// This builder is `#[must_use]`: setters return an updated value.
#[must_use]
pub struct AgentBuilder<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B> + Send + Sync,
    KindOut: TensorKind<B> + Send + Sync,
> {
    pub client_modes: Option<ClientModes>,
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
    #[must_use]
    pub fn builder() -> Self {
        Self {
            client_modes: Some(ClientModes::default()),
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

    #[must_use]
    pub fn actor_inference_mode(mut self, actor_inference_mode: ActorInferenceMode) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.actor_inference_mode = actor_inference_mode;
        }
        self
    }

    #[must_use]
    pub fn actor_training_data_mode(
        mut self,
        actor_training_data_mode: ActorTrainingDataMode,
    ) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.actor_training_data_mode = actor_training_data_mode;
        }
        self
    }

    #[must_use]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn transport_type(mut self, transport_type: TransportType) -> Self {
        self.transport_type = Some(transport_type);
        self
    }

    #[must_use]
    pub fn actor_count(mut self, count: u32) -> Self {
        self.actor_count = Some(count);
        self
    }

    #[must_use]
    pub fn router_scale(mut self, count: u32) -> Self {
        self.router_scale = Some(count);
        self
    }

    #[must_use]
    pub fn default_device(mut self, device: DeviceType) -> Self {
        self.default_device = Some(device);
        self
    }

    #[must_use]
    pub fn default_model(mut self, model: ModelModule<B>) -> Self {
        self.default_model = Some(model);
        self
    }

    #[must_use]
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

    #[must_use]
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

    #[must_use]
    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path.into());
        self
    }

    #[must_use]
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
        let agent: RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut> = RelayRLAgent::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.transport_type.unwrap_or(TransportType::ZMQ),
            self.client_modes.unwrap_or(ClientModes::default()),
        );

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            algorithm_args: self.algorithm_args.unwrap_or(AlgorithmArgs::default()),
            actor_count: self.actor_count.unwrap_or(1),
            router_scale: self.router_scale.unwrap_or(1),
            default_device: self.default_device.unwrap_or_default(),
            #[cfg(any(feature = "tch-model", feature = "onnx-model"))]
            default_model: self.default_model,
            #[cfg(not(any(feature = "tch-model", feature = "onnx-model")))]
            default_model: self.default_model.unwrap(), // this is guaranteed to panic if not set; without transport available, the model must be set at build time
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
pub struct RelayRLAgent<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
> {
    coordinator: ClientCoordinator<B, D_IN, D_OUT>,
    supported_backend: SupportedTensorBackend,
    input_dtype: Option<DType>,
    output_dtype: Option<DType>,
    _phantom: PhantomData<(KindIn, KindOut)>,
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
        let supported_backend = B::get_supported_backend();

        Self {
            coordinator: ClientCoordinator::<B, D_IN, D_OUT>::new(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                transport_type,
                client_modes,
            ),
            supported_backend,
            input_dtype: None,
            output_dtype: None,
            _phantom: PhantomData,
        }
    }

    /// Start the client runtime with the specified parameters.
    ///
    /// This spawns the coordinator runtime components and (by default) creates `actor_count`
    /// runtime actors.
    ///
    /// # Parameters
    /// - `algorithm_args`: Algorithm selection + optional hyperparameters.
    /// - `actor_count`: Number of runtime actors to spawn initially.
    /// - `router_scale`: Number of routing workers used to dispatch messages to actors.
    /// - `default_device`: Default device for tensor ops per actor.
    /// - `default_model`:
    ///   - `Some`: each actor starts with this model.
    ///   - `None`: the runtime may perform a server handshake to obtain a model (if enabled),
    ///     otherwise startup can fail.
    /// - `config_path`: Optional path to a client configuration file.
    ///
    /// # Errors
    /// Returns an error if startup fails (configuration, runtime init, transport init, etc).
    pub async fn start(
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
    ) -> Result<(), ClientError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let (input_dtype, output_dtype) = if let Some(ref model_module) = default_model {
            (
                Some(model_module.metadata.input_dtype.clone()),
                Some(model_module.metadata.output_dtype.clone()),
            )
        } else {
            (None, None)
        };

        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        let (input_dtype, output_dtype) = (
            Some(default_model.metadata.input_dtype.clone()),
            Some(default_model.metadata.output_dtype.clone()),
        );

        self.input_dtype = input_dtype;
        self.output_dtype = output_dtype;

        self.coordinator
            ._start(
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
            .await
            .map_err(Into::<ClientError>::into)?;

        Ok(())
    }

    /// Restart the Agent's client runtime components
    ///
    /// # Errors
    /// Returns an error if restart coordination fails.
    pub async fn restart(
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
    ) -> Result<(), ClientError> {
        self.coordinator
            ._restart(
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

    /// Gracefully shut down the Agent's client runtime components
    ///
    /// # Errors
    /// Returns an error if shutdown coordination fails.
    pub async fn shutdown(&mut self) -> Result<(), ClientError> {
        self.coordinator._shutdown().await?;
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
                self.coordinator._scale_out(add as u32).await?;
                Ok(())
            }
            remove if router_scale < 0 => {
                self.coordinator._scale_in(remove.unsigned_abs()).await?;
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
                        ._request_action(ids, obs_tensor, mask_tensor, reward)
                        .await?;
                    Ok(result)
                } else {
                    Err(ClientError::NoInputOrOutputDtypeSet(
                        "No input or output dtype set in agent".to_string(),
                    ))
                }
            }
            false => Err(ClientError::BackendMismatchError(
                "Backend mismatch".to_string(),
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
        self.coordinator._flag_last_action(ids, reward).await?;
        Ok(())
    }

    /// Retrieves the model version for each actor ID listed (if instance IDs exist)
    ///
    /// Returns `(ActorID, ModelVersion)` pairs.
    pub async fn get_model_version(&self, ids: Vec<Uuid>) -> Result<Vec<(Uuid, i64)>, ClientError> {
        Ok(self.coordinator._get_model_version(ids).await?)
    }

    /// Collect runtime statistics.
    ///
    /// Current status: not implemented.
    ///
    /// # Errors
    /// Will return an error once implemented if serialization or IO fails.
    #[deprecated(note = "Not implemented")]
    #[cfg(any(feature = "metrics", feature = "logging"))]
    pub fn runtime_statistics(
        &self,
        return_type: RuntimeStatisticsReturnType,
    ) -> Result<RuntimeStatisticsReturnType, ClientError> {
        // stand-in for actual implementation
        Ok(RuntimeStatisticsReturnType::Hashmap(HashMap::new()))
    }

    /// Fetch the active client configuration.
    pub async fn get_config(&self) -> Result<ClientConfigLoader, ClientError> {
        Ok(self.coordinator._get_config().await?)
    }

    /// Set the configuration path used by the runtime.
    pub async fn set_config_path(&self, config_path: PathBuf) -> Result<(), ClientError> {
        self.coordinator._set_config_path(config_path).await?;
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
            self.coordinator
                ._new_actor(device, default_model, true)
                .await?;
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            self.coordinator._new_actor(device, default_model).await?;
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
            return Box::pin(async move {
                Err(ClientError::NoopActorCount(
                    "Noop actor count: `count` set to zero".to_string(),
                ))
            });
        }
        Box::pin(async move {
            for _ in 0..count {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                self.coordinator
                    ._new_actor(device.clone(), default_model.clone(), false)
                    .await?;
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                self.coordinator
                    ._new_actor(device.clone(), default_model.clone())
                    .await?;
            }
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let actor_ids = get_pairs("client", "actor").map_err(ClientError::from)?;

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.coordinator
                ._send_client_ids_to_server(actor_ids)
                .await?;

            Ok(())
        })
    }

    /// Removes the actor instance with the specified ID from the current Agent instance
    fn remove_actor(
        &mut self,
        id: ActorUuid,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>> + Send + '_>> {
        Box::pin(async move {
            self.coordinator._remove_actor(id).await?;
            Ok(())
        })
    }

    /// Retrieves the current actor instance IDs
    fn get_actor_ids(&mut self) -> Result<Vec<ActorUuid>, ClientError> {
        let ids = get_pairs("client", "actor").map_err(ClientError::from)?;
        Ok(ids.iter().map(|(_, id)| id.clone()).collect())
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
            self.coordinator._set_actor_id(current_id, new_id).await?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {}
