//! Client API for starting and controlling the RelayRL client runtime.
//!
//! This module provides:
//! - `RelayRLAgent`: a thin facade over the runtime coordinator.
//! - `AgentBuilder`: ergonomic construction of an agent instance plus its startup parameters.
//! - Mode/config enums that describe inference and trajectory recording behavior.
//!
//! Transport and database layers are optional feature flags gating additional functionality.

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::HyperparameterArgs;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::TransportType;
use crate::network::client::runtime::coordination::coordinator::{
    ClientCoordinator, ClientInterface, CoordinatorError,
};
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::coordination::scale_manager::AlgorithmArgs;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::prelude::config::ClientConfigLoader;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::utilities::configuration::Algorithm;

use active_uuid_registry::UuidPoolError;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use active_uuid_registry::interface::get;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use relayrl_types::types::data::action::CodecConfig;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::{
    AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
    IntBurnTensor, SupportedTensorBackend, NdArrayDType, TchDType,
};
use relayrl_types::types::model::ModelModule;

use burn_tensor::{Bool, Float, Int, Tensor, TensorKind, backend::Backend};
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
    #[error("Noop router scale: {0}")]
    NoopRouterScale(String),
    #[error("Noop actor count: {0}")]
    NoopActorCount(String),
    #[error("Invalid inference mode: {0}")]
    InvalidInferenceMode(String),
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

/// Inference mode used by runtime actors.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorInferenceMode {
    /// Inference occurs locally in the client process.
    Local,
    /// Inference occurs on external inference server.
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    Server,
}

impl Default for ActorInferenceMode {
    fn default() -> Self {
        Self::Local
    }
}

/// File-based trajectory recording parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct FormattedTrajectoryFileParams {
    pub enabled: bool,
    pub encode: bool,
    pub path: PathBuf,
}

impl Default for FormattedTrajectoryFileParams {
    fn default() -> Self {
        Self {
            enabled: false,
            encode: false,
            path: PathBuf::from("./trajectories"),
        }
    }
}

/// Supported database backends for trajectory recording.
#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
#[derive(Debug, Clone, PartialEq)]
pub enum DatabaseTypeParams {
    Sqlite(SqliteParams),
    PostgreSQL(PostgreSQLParams),
}

// TODO: Add actual Sqlite params
#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
#[derive(Debug, Clone, PartialEq)]
pub struct SqliteParams {
    pub path: PathBuf,
}

// TODO: Add actual PostgreSQL params
#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
#[derive(Debug, Clone, PartialEq)]
pub struct PostgreSQLParams {
    pub connection: String,
}

/// Trajectory recording mode for each router buffer
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryRecordMode {
    /// Trajectories are recorded to file on local device.
    Local(FormattedTrajectoryFileParams),
    /// Trajectories are recorded to a supported database (SQLite, PostgreSQL).
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "postgres_db", feature = "sqlite_db"))))]
    Database(DatabaseTypeParams),
    /// Trajectories are recorded to a supported database and file on local device.
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "postgres_db", feature = "sqlite_db"))))]
    Hybrid(DatabaseTypeParams, FormattedTrajectoryFileParams),
    /// Trajectory recording/persistence is disabled.
    #[cfg(any(
        feature = "postgres_db",
        feature = "sqlite_db",
        feature = "async_transport",
        feature = "sync_transport"
    ))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(
            feature = "postgres_db",
            feature = "sqlite_db",
            feature = "async_transport",
            feature = "sync_transport"
        )))
    )]
    Disabled,
}

impl Default for TrajectoryRecordMode {
    fn default() -> Self {
        #[cfg(any(
            feature = "postgres_db",
            feature = "sqlite_db",
            feature = "async_transport",
            feature = "sync_transport"
        ))]
        return Self::Disabled;
        #[cfg(not(any(
            feature = "postgres_db",
            feature = "sqlite_db",
            feature = "async_transport",
            feature = "sync_transport"
        )))]
        return Self::Local(FormattedTrajectoryFileParams::default());
    }
}

/// TODO: Add architecture support for independent/shared model inference/training requests to server(s).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorServerModelMode {
    /// Each actor has an independent server-side model
    Independent,
    /// All actors share the same server-side model.
    Shared,
    /// Server-side mode disabled
    Disabled,
}

impl Default for ActorServerModelMode {
    fn default() -> Self {
        Self::Independent
    }
}

/// Contains a collection of capabilities for the client runtime.
///
/// By using `ClientModes`, you can determine the capabilities of the client runtime.
#[derive(Debug, Clone)]
pub struct ClientCapabilities {
    /// Whether local inference is enabled
    pub local_inference: bool,
    /// Whether server inference is enabled
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    pub server_inference: bool,
    /// Inference server allocation mode (when server inference is enabled)
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    pub inference_server_mode: ActorServerModelMode,
    /// Whether local trajectory recording is enabled
    pub local_trajectory_recording: bool,
    /// Whether database trajectory recording is enabled
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "postgres_db", feature = "sqlite_db"))))]
    pub database_trajectory_recording: bool,
    /// Training server allocation mode (when training server is enabled)
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    pub training_server_mode: ActorServerModelMode,
    /// Database parameters (when database trajectory recording is enabled)
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "postgres_db", feature = "sqlite_db"))))]
    pub db_params: Option<DatabaseTypeParams>,
}

impl ClientCapabilities {
    /// Returns `true` if either local or database trajectory recording is enabled.
    pub fn trajectory_recording_enabled(&self) -> bool {
        #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
        let database_trajectory_recording = self.database_trajectory_recording;
        #[cfg(not(any(feature = "postgres_db", feature = "sqlite_db")))]
        let database_trajectory_recording = false;

        self.local_trajectory_recording || database_trajectory_recording
    }
}

/// Runtime modes consumed by the client to enable/disable functionality.
///
/// Use [`ClientModes::capabilities`] to compute a concrete [`ClientCapabilities`] value.
pub struct ClientModes {
    /// Actor inference mode (local vs server)
    pub actor_inference_mode: ActorInferenceMode,
    /// Inference server model mode (independent vs shared)
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    pub inference_server_mode: ActorServerModelMode,
    /// Training server model mode (independent vs shared)
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    pub training_server_mode: ActorServerModelMode,
    /// Trajectory recording mode (local vs database vs hybrid vs disabled)
    pub trajectory_recording_mode: TrajectoryRecordMode,
}

impl Default for ClientModes {
    fn default() -> Self {
        Self {
            actor_inference_mode: ActorInferenceMode::Local,
            #[cfg(any(
                feature = "postgres_db",
                feature = "sqlite_db",
                feature = "async_transport",
                feature = "sync_transport"
            ))]
            trajectory_recording_mode: TrajectoryRecordMode::Disabled,
            #[cfg(not(any(
                feature = "postgres_db",
                feature = "sqlite_db",
                feature = "async_transport",
                feature = "sync_transport"
            )))]
            trajectory_recording_mode: TrajectoryRecordMode::Local(
                FormattedTrajectoryFileParams::default(),
            ),
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            inference_server_mode: ActorServerModelMode::Disabled,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            training_server_mode: ActorServerModelMode::Independent,
        }
    }
}

impl ClientModes {
    /// Validate internal consistency of mode selections.
    ///
    /// # Errors
    /// Returns [`ClientError::InvalidInferenceMode`] if `actor_inference_mode` and
    /// `inference_server_mode` conflict.
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn validate_modes(&self) -> Result<(), ClientError> {
        if self.actor_inference_mode == ActorInferenceMode::Server
            && self.inference_server_mode == ActorServerModelMode::Disabled
        {
            return Err(ClientError::InvalidInferenceMode(
                "Inference server mode disabled for server-side inference: {:?}".to_string(),
            ));
        }
        if self.actor_inference_mode == ActorInferenceMode::Local
            && self.inference_server_mode != ActorServerModelMode::Disabled
        {
            return Err(ClientError::InvalidInferenceMode(
                "Inference server mode enabled for client-side inference: {:?}".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute runtime capabilities from the configured modes.
    ///
    /// # Errors
    /// Returns an error if the mode combination is invalid.
    pub fn capabilities(&self) -> Result<ClientCapabilities, ClientError> {
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        self.validate_modes()?;

        let (local_inference, _server_inference) = match self.actor_inference_mode {
            ActorInferenceMode::Local => (true, false),
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            ActorInferenceMode::Server => (false, true),
        };

        let (local_trajectory_recording, _database_trajectory_recording) =
            match &self.trajectory_recording_mode {
                TrajectoryRecordMode::Local(_) => (true, false),
                #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
                TrajectoryRecordMode::Database(_) => (false, true),
                #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
                TrajectoryRecordMode::Hybrid(_, _) => (true, true),
                #[cfg(any(
                    feature = "postgres_db",
                    feature = "sqlite_db",
                    feature = "async_transport",
                    feature = "sync_transport"
                ))]
                TrajectoryRecordMode::Disabled => (false, false),
            };

        #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
        let db_params: Option<DatabaseTypeParams> = match &self.trajectory_recording_mode {
            TrajectoryRecordMode::Database(params) => Some(params.clone()),
            TrajectoryRecordMode::Hybrid(params, _) => Some(params.clone()),
            _ => None,
        };

        Ok(ClientCapabilities {
            local_inference,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            server_inference: _server_inference,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            inference_server_mode: self.inference_server_mode.clone(),
            local_trajectory_recording,
            #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
            database_trajectory_recording: _database_trajectory_recording,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            training_server_mode: self.training_server_mode.clone(),
            #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
            db_params,
        })
    }
}

/// Parameters used to start a [`RelayRLAgent`].
///
/// Typically constructed via [`AgentBuilder::build`].
pub struct AgentStartParameters<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "async_transport", feature = "sync_transport")))
    )]
    pub algorithm_args: AlgorithmArgs,
    pub actor_count: u32,
    pub router_scale: u32,
    pub default_device: DeviceType,
    pub default_model: Option<ModelModule<B>>,
    pub config_path: Option<PathBuf>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub transport_type: Option<TransportType>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub algorithm_args: Option<AlgorithmArgs>,
    pub actor_count: Option<u32>,
    pub router_scale: Option<u32>,
    pub default_device: Option<DeviceType>,
    pub default_model: Option<ModelModule<B>>,
    pub config_path: Option<PathBuf>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            transport_type: Some(TransportType::default()),
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            algorithm_args: Some(AlgorithmArgs::default()),
            actor_count: None,
            router_scale: None,
            default_device: None,
            default_model: None,
            config_path: None,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
    pub fn trajectory_recording_mode(
        mut self,
        trajectory_recording_mode: TrajectoryRecordMode,
    ) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.trajectory_recording_mode = trajectory_recording_mode;
        }
        self
    }

    #[must_use]
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn inference_server_mode(mut self, inference_server_mode: ActorServerModelMode) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.inference_server_mode = inference_server_mode;
        }
        self
    }

    #[must_use]
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn training_server_mode(mut self, training_server_mode: ActorServerModelMode) -> Self {
        if let Some(ref mut modes) = self.client_modes {
            modes.training_server_mode = training_server_mode;
        }
        self
    }

    #[must_use]
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            self.transport_type.unwrap_or(TransportType::ZMQ),
            self.client_modes.unwrap_or(ClientModes::default()),
        )?;

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            algorithm_args: self.algorithm_args.unwrap_or(AlgorithmArgs::default()),
            actor_count: self.actor_count.unwrap_or(1),
            router_scale: self.router_scale.unwrap_or(1),
            default_device: self.default_device.unwrap_or_default(),
            default_model: self.default_model,
            config_path: self.config_path,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            codec: self.codec.unwrap_or_default(),
        };

        Ok((agent, startup_params))
    }
}

trait ToAnyBurnTensor<B: Backend + BackendMatcher<Backend = B>, const D: usize> {
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
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        transport_type: TransportType,
        client_modes: ClientModes,
    ) -> Result<Self, ClientError> {
        let capabilities = client_modes.capabilities()?;
        let supported_backend = if B::matches_backend(&SupportedTensorBackend::NdArray) {
            SupportedTensorBackend::NdArray
        } else if B::matches_backend(&SupportedTensorBackend::Tch) {
            SupportedTensorBackend::Tch
        } else {
            SupportedTensorBackend::None
        };

        Ok(Self {
            coordinator: ClientCoordinator::<B, D_IN, D_OUT>::new(
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                transport_type,
                capabilities,
            )?,
            supported_backend,
            input_dtype: None,
            output_dtype: None,
            _phantom: PhantomData,
        })
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
        mut self,
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
    ) -> Result<(), ClientError> {
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let (input_dtype, output_dtype) = if let Some(ref model_module) = default_model {
            (model_module.metadata.input_dtype.clone(), model_module.metadata.output_dtype.clone())
        } else {
            let default_dtype = match &self.supported_backend {
                SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
                SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
                _ => return Err(ClientError::BackendMismatchError("Unsupported backend".to_string())),
            };
            (default_dtype.clone(), default_dtype)
        };

        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        let (input_dtype, output_dtype) = (default_model.metadata.input_dtype.clone(), default_model.metadata.output_dtype.clone());

        self.input_dtype = Some(input_dtype);
        self.output_dtype = Some(output_dtype);

        self.coordinator
            ._start(
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
            .await
            .map_err(Into::<ClientError>::into)?;

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

    /// Gracefully shut down the Agent's client runtime components
    ///
    /// # Errors
    /// Returns an error if shutdown coordination fails.
    pub async fn shutdown(&mut self) -> Result<(), ClientError> {
        self.coordinator._shutdown().await?;
        Ok(())
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
                if let (Some(input_dtype), Some(output_dtype)) = (self.input_dtype.clone(), self.output_dtype.clone()) {
                    let obs_tensor: Arc<AnyBurnTensor<B, D_IN>> =
                    Arc::new(observation.to_any_burn_tensor(input_dtype));
                    let mask_tensor: Option<Arc<AnyBurnTensor<B, D_OUT>>> = mask
                        .map(|tensor| Arc::new(tensor.to_any_burn_tensor(output_dtype)));

                    let result = self
                        .coordinator
                        ._request_action(ids, obs_tensor, mask_tensor, reward)
                        .await?;
                    Ok(result)
                } else {
                    Err(ClientError::BackendMismatchError("No input or output dtype set".to_string()))
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
    fn get_actors(
        &self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<(Vec<Uuid>, Vec<Arc<JoinHandle<()>>>), ClientError>>
                + Send
                + '_,
        >,
    >;
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
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            self.coordinator
                ._new_actor(device, default_model, true)
                .await?;
            #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
            self.coordinator
                ._new_actor(device, default_model)
                .await?;
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
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                self.coordinator
                    ._new_actor(device.clone(), default_model.clone(), false)
                    .await?;
                #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
                self.coordinator
                    ._new_actor(device.clone(), default_model.clone())
                    .await?;
            }
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            let actor_ids = get("actor").map_err(ClientError::from)?;

            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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

    /// Retrieves the current actor instances and their associated join handles from the Agent instance
    fn get_actors(
        &self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), ClientError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let actors = self.coordinator._get_actors().await?;
            Ok(actors)
        })
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
