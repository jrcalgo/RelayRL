#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::agent::TransportMode;
use crate::agent::process::{ClientError, RelayRLAgent};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::NetworkParams;

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::get_context_entries;
use relayrl_algorithms::prelude::ppo::algorithm::{IPPOParams, MAPPOParams, PPOParams};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::tensor::BackendMatcher;
use relayrl_types::model::ModelModule;

use burn_tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Hyperparameter overrides forwarded to a training server at handshake time.
///
/// When `config_default_init` is `true`, any `None` field is filled from the JSON config file or system defaults;
/// set a field to `Some(...)` to override a specific algorithm's params without touching the others.
#[derive(Debug, Clone, PartialEq)]
pub struct DefaultHyperparameterArgs {
    pub ppo: Option<PPOParams>,
    pub ippo: Option<IPPOParams>,
    pub mappo: Option<MAPPOParams>,
    // custom: Option<CustomAlgorithmParams>
    pub config_default_init: bool,
}

impl Default for DefaultHyperparameterArgs {
    fn default() -> Self {
        Self {
            ppo: None,
            ippo: None,
            mappo: None,
            config_default_init: true,
        }
    }
}

/// Algorithm identity and optional hyperparameters sent to a training server on actor init.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlgorithmInitArgs {
    PPO(Option<PPOParams>),
    IPPO(Option<IPPOParams>),
    MAPPO(Option<MAPPOParams>),
}

impl Default for AlgorithmInitArgs {
    fn default() -> Self {
        Self::PPO(None)
    }
}

impl std::fmt::Display for DefaultHyperparameterArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DefaultHyperparameterArgs {{")?;
        if let Some(ppo) = &self.ppo {
            write!(f, "ppo: {:?}", ppo)?;
        }
        if let Some(ippo) = &self.ippo {
            write!(f, "ippo: {:?}", ippo)?;
        }
        if let Some(mappo) = &self.mappo {
            write!(f, "mappo: {:?}", mappo)?;
        }
        if self.config_default_init {
            write!(f, "config_default_init: true")?;
        } else {
            write!(f, "config_default_init: false")?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

impl AlgorithmInitArgs {
    /// Returns the algorithm name as a static string (`"PPO"`, `"IPPO"`, or `"MAPPO"`).
    pub fn as_str(&self) -> &str {
        match self {
            AlgorithmInitArgs::PPO(_) => "PPO",
            AlgorithmInitArgs::IPPO(_) => "IPPO",
            AlgorithmInitArgs::MAPPO(_) => "MAPPO",
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
    pub codec: Option<CodecConfig>,
    pub inference_addresses: Option<InferenceAddressesArgs>,
}

/// Experimental configuration for server-backed training.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Default, Debug, Clone, PartialEq)]
pub struct TrainingParams {
    pub model_mode: ModelMode,
    pub default_hyperparameters: Option<DefaultHyperparameterArgs>,
    pub codec: Option<CodecConfig>,
    pub training_addresses: Option<TrainingAddressesArgs>,
}

/// Serialization format for locally written trajectory files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LocalTrajectoryFileType {
    /// Comma-separated values.
    Csv,
    /// Apache Arrow IPC format.
    Arrow,
}

/// File-based trajectory recording parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalTrajectoryFileParams {
    pub directory: PathBuf,
    pub file_type: LocalTrajectoryFileType,
}

impl LocalTrajectoryFileParams {
    /// Validates `directory` and creates it if it does not exist, then returns the params.
    ///
    /// ```ignore
    /// use std::path::PathBuf;
    /// use relayrl::network::{LocalTrajectoryFileParams, LocalTrajectoryFileType};
    ///
    /// let params = LocalTrajectoryFileParams::new(
    ///     PathBuf::from("experiment_data"),
    ///     LocalTrajectoryFileType::Arrow,
    /// )?;
    /// ```
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

/// Controls whether actors on the same device each own an independent model handle or share one.
///
/// `Independent` (default) allows actors to run genuinely different policies simultaneously.
/// `Shared` reduces memory consumption when actors on a device should always use the same weights.
/// Server-backed uses of `ModelMode` are experimental.
///
/// ```ignore
/// # use relayrl::network::{AgentBuilder, ActorInferenceMode, ModelMode};
/// # use burn_ndarray::NdArray;
/// let builder = AgentBuilder::<NdArray>::builder()
///     .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Shared));
/// ```
#[non_exhaustive]
#[derive(Default, Debug, Clone, PartialEq)]
pub enum ModelMode {
    /// Each actor has an independent model handle.
    #[default]
    Independent,
    /// Actors on the same device share a model handle.
    Shared,
}

/// Selects where actor inference occurs.
///
/// `Client` (default) runs inference locally inside each actor task; `Server` and `ClientFallback`
/// route inference to an external server and are experimental, requiring a transport feature.
///
/// ```ignore
/// # use relayrl::network::{AgentBuilder, ActorInferenceMode, ModelMode};
/// # use burn_ndarray::NdArray;
/// let (agent, params) = AgentBuilder::<NdArray>::builder()
///     .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Independent))
///     .build()
///     .await?;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorInferenceMode {
    /// Inference occurs locally in the local runtime actor.
    Client(ModelMode),
    /// Experimental: inference occurs on external inference server(s).
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Server(InferenceParams),
    /// Experimental: inference falls back to local execution when remote inference fails, for example due to network issues.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    ClientFallback(ModelMode, InferenceParams),
}

impl Default for ActorInferenceMode {
    fn default() -> Self {
        Self::Client(ModelMode::default())
    }
}

pub type TrajectoryCacheSize = usize;

/// Selects how actors record and forward trajectory data.
///
/// The `Offline*` variants write to memory and/or local files and are currently the only fully supported path.
/// The `Online*` variants stream data to a training server and require a transport feature.
///
/// ```ignore
/// # use relayrl::network::{AgentBuilder, ActorDataMode};
/// # use burn_ndarray::NdArray;
/// let (agent, params) = AgentBuilder::<NdArray>::builder()
///     .actor_data_mode(ActorDataMode::OfflineWithFilesAndCache(None, 1000))
///     .build()
///     .await?;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ActorDataMode {
    /// Training data is recorded to a local file.
    OfflineWithFiles(Option<LocalTrajectoryFileParams>),
    /// Training data is recorded to a local memory buffer with per-actor size.
    OfflineWithCache(TrajectoryCacheSize),
    /// Training data is recorded to a local file and memory buffer with per-actor size.
    OfflineWithFilesAndCache(Option<LocalTrajectoryFileParams>, TrajectoryCacheSize),
    /// Experimental: training data is sent to the server for processing.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    Online(TrainingParams),
    /// Experimental: training data is sent to the server and also recorded locally.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    OnlineWithFiles(TrainingParams, Option<LocalTrajectoryFileParams>),
    /// Experimental: training data is sent to the server and also recorded in memory with per-actor size.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    OnlineWithCache(TrainingParams, TrajectoryCacheSize),
    /// Experimental: training data is sent to the server and also recorded in file and memory with per-actor size.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(feature = "nats-transport", feature = "zmq-transport")))
    )]
    OnlineWithFilesAndCache(
        TrainingParams,
        Option<LocalTrajectoryFileParams>,
        TrajectoryCacheSize,
    ),
    /// Training data collection and processing is disabled
    Disabled,
}

impl Default for ActorDataMode {
    fn default() -> Self {
        Self::OfflineWithCache(1000)
    }
}

pub(crate) fn uses_local_file_writing(training_data_mode: &ActorDataMode) -> bool {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    return matches!(
        training_data_mode,
        ActorDataMode::OfflineWithFiles(_)
            | ActorDataMode::OfflineWithFilesAndCache(..)
            | ActorDataMode::OnlineWithFiles(..)
            | ActorDataMode::OnlineWithFilesAndCache(..)
    );
    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    return matches!(
        training_data_mode,
        ActorDataMode::OfflineWithFiles(_) | ActorDataMode::OfflineWithFilesAndCache(..)
    );
}

pub(crate) fn uses_trajectory_cache(training_data_mode: &ActorDataMode) -> bool {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    return matches!(
        training_data_mode,
        ActorDataMode::OfflineWithCache(_)
            | ActorDataMode::OfflineWithFilesAndCache(..)
            | ActorDataMode::OnlineWithCache(..)
            | ActorDataMode::OnlineWithFilesAndCache(..)
    );

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    return matches!(
        training_data_mode,
        ActorDataMode::OfflineWithCache(_) | ActorDataMode::OfflineWithFilesAndCache(..)
    );
}

/// Active inference and data-collection modes applied across all runtime actors.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ClientModes {
    pub actor_inference_mode: ActorInferenceMode,
    pub actor_data_mode: ActorDataMode,
}

/// Capacity of an actor's in-memory replay buffer.
pub type ReplayBufferSize = usize;
/// Filesystem path where a trained model is saved.
pub type SaveModelPath = PathBuf;

/// Startup parameters produced by `AgentBuilder::build` and consumed by `RelayRLAgent::start` or `restart`.
///
/// ```ignore
/// # use relayrl::network::{AgentBuilder, RelayRLAgent};
/// # use burn_ndarray::NdArray;
/// let (mut agent, params) = AgentBuilder::<NdArray>::builder().build().await?;
/// agent.start(params).await?;
/// ```
#[derive(Clone)]
pub struct AgentStartParameters<B: Backend + BackendMatcher<Backend = B>> {
    pub data_routers: u32,
    pub data_buffer_size: usize,
    pub default_model: Option<ModelModule<B>>,
    pub config_polling_seconds: Option<u64>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub default_hyperparameters: DefaultHyperparameterArgs,
    pub config_path: Option<PathBuf>,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for AgentStartParameters<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentStartParameters")
    }
}

#[derive(Clone)]
pub struct AgentBuildInvariants<B: Backend + BackendMatcher<Backend = B>> {
    pub builder: AgentBuilder<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> AgentBuildInvariants<B> {
    fn with(builder: AgentBuilder<B>) -> Self {
        Self {
            builder: builder.to_owned(),
        }
    }

    /// Adjustable runtime variables for `RelayRLAgent`
    pub fn params(self) -> AgentBuildParameters<B> {
        AgentBuildParameters::<B>::with(self.builder.to_owned())
    }

    /// Sets the inference mode for all actors. Defaults to `ActorInferenceMode::Client(ModelMode::Independent)`.
    ///
    /// ```ignore
    /// # use relayrl::network::{AgentBuilder, ActorInferenceMode, ModelMode};
    /// # use burn_ndarray::NdArray;
    /// let builder = AgentBuilder::<NdArray>::builder()
    ///     .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Shared));
    /// ```
    pub fn actor_inference_mode(mut self, actor_inference_mode: ActorInferenceMode) -> Self {
        self.builder.settings.client_modes.actor_inference_mode = actor_inference_mode;
        self
    }

    /// Sets the training data collection mode for all actors. Defaults to `ActorDataMode::OfflineWithCache`.
    ///
    /// ```ignore
    /// # use relayrl::network::{AgentBuilder, ActorDataMode};
    /// # use burn_ndarray::NdArray;
    /// let builder = AgentBuilder::<NdArray>::builder()
    ///     .modes()
    ///     .actor_data_mode(ActorDataMode::OfflineWithFilesAndCache(None, 1000));
    /// ```
    pub fn actor_data_mode(mut self, actor_data_mode: ActorDataMode) -> Self {
        self.builder.settings.client_modes.actor_data_mode = actor_data_mode;
        self
    }

    /// Selects the network transport type for server-backed workflows. Requires `zmq-transport` or `nats-transport`.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn transport_mode(mut self, transport_mode: TransportMode) -> Self {
        self.builder.settings.transport_mode = Some(transport_mode);
        self
    }

    /// Runs build on the internal `AgentBuilder`
    pub async fn build(self) -> Result<(RelayRLAgent<B>, AgentStartParameters<B>), ClientError> {
        self.builder.build().await
    }
}

#[derive(Clone)]
pub struct AgentBuildParameters<B: Backend + BackendMatcher<Backend = B>> {
    pub builder: AgentBuilder<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> AgentBuildParameters<B> {
    fn with(builder: AgentBuilder<B>) -> Self {
        Self { builder }
    }

    /// Runtime invariants for `RelayRLAgent`
    ///
    pub fn modes(self) -> AgentBuildInvariants<B> {
        AgentBuildInvariants::<B>::with(self.builder.to_owned())
    }

    /// Sets the number of routing workers started alongside the coordinator. Defaults to `1`.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// let builder = AgentBuilder::<NdArray>::builder().params().data_routers(4);
    /// ```
    pub fn data_routers(mut self, count: u32) -> Self {
        self.builder.settings.data_routers = Some(count);
        self
    }

    /// Sets the trajectory buffer size for each buffer in each router. Defaults to `1024`.
    ///
    /// ```ignore
    /// let builder = AgentBuilder::<NdArray>::builder().params().data_buffer_size(10_000);
    /// ```
    pub fn data_buffer_size(mut self, size: usize) -> Self {
        self.builder.settings.data_buffer_size = Some(size);
        self
    }

    /// Provides a default model pre-loaded into each actor at startup.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use relayrl::types::model::ModelModule;
    /// # use burn_ndarray::NdArray;
    /// let model = ModelModule::<NdArray>::load_from_path("model_dir")?;
    /// let builder = AgentBuilder::<NdArray>::builder().params().default_model(model);
    /// ```
    pub fn default_model(mut self, model: ModelModule<B>) -> Self {
        self.builder.settings.default_model = Some(model);
        self
    }

    /// Overrides the config updating polling frequency (secs). When unset the value in the JSON config (default `10`) is used.
    ///
    /// ```ignore
    /// let builder = AgentBuilder::<NdArray>::builder().params().config_polling_seconds(3);
    /// ```
    pub fn config_polling_seconds(mut self, seconds: u64) -> Self {
        self.builder.settings.config_polling_seconds = Some(seconds);
        self
    }

    /// Sets the JSON config file path. Defaults to `client_config.json` in the working directory.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// # use std::path::PathBuf;
    /// let builder = AgentBuilder::<NdArray>::builder()
    ///     .params()
    ///     .config_path(PathBuf::from("my_config.json"));
    /// ```
    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.builder.settings.config_path = Some(path);
        self
    }

    /// Supplies default PPO hyperparameters forwarded to the training server. Requires a transport feature.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_ppo_params(mut self, ppo_params: PPOParams) -> Self {
        self.builder.settings.default_hyperparameters.ppo = Some(ppo_params);
        self
    }

    /// Supplies default IPPO hyperparameters forwarded to the training server. Requires a transport feature.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_ippo_params(mut self, ippo_params: IPPOParams) -> Self {
        self.builder.settings.default_hyperparameters.ippo = Some(ippo_params);
        self
    }

    /// Supplies default MAPPO hyperparameters forwarded to the training server. Requires a transport feature.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_mappo_params(mut self, mappo_params: MAPPOParams) -> Self {
        self.builder.settings.default_hyperparameters.mappo = Some(mappo_params);
        self
    }

    /// Runs build on the internal `AgentBuilder<B>`
    pub async fn build(self) -> Result<(RelayRLAgent<B>, AgentStartParameters<B>), ClientError> {
        self.builder.build().await
    }
}

#[derive(Clone)]
pub struct BuilderSettings<B: Backend + BackendMatcher<Backend = B>> {
    pub client_modes: ClientModes,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub transport_mode: Option<TransportMode>,
    pub data_routers: Option<u32>,
    pub data_buffer_size: Option<usize>,
    pub default_model: Option<ModelModule<B>>,
    pub config_polling_seconds: Option<u64>,
    pub default_hyperparameters: DefaultHyperparameterArgs,
    pub config_path: Option<PathBuf>,
}

impl<B: Backend + BackendMatcher<Backend = B>> Default for BuilderSettings<B> {
    fn default() -> Self {
        Self {
            client_modes: ClientModes::default(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_mode: None,
            data_routers: None,
            data_buffer_size: None,
            default_model: None,
            config_polling_seconds: None,
            default_hyperparameters: DefaultHyperparameterArgs::default(),
            config_path: None,
        }
    }
}

/// Fluent builder for constructing a `RelayRLAgent` and its startup parameters.
///
/// Each setter returns the updated builder; `build()` consumes it and yields `(RelayRLAgent<B>, AgentStartParameters<B>)`.
///
/// ```ignore
/// use relayrl::network::{AgentBuilder, ActorDataMode, RelayRLActors};
/// use relayrl::types::model::ModelModule;
/// use burn_ndarray::NdArray;
/// use std::path::PathBuf;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let model = ModelModule::<NdArray>::load_from_path("model_dir")?;
/// let (mut agent, params) = AgentBuilder::<NdArray>::builder()
///     .params()
///     .data_routers(2)
///     .actor_data_mode(ActorDataMode::OfflineWithCache(1000))
///     .default_model(model)
///     .config_path(PathBuf::from("client_config.json"))
///     .build()
///     .await?;
///
/// agent.start(params).await?;
/// let actor_info = agent.get_actor_info().await?;
/// agent.shutdown().await?;
/// # Ok(())
/// # }
/// ```
#[must_use = "Provides ergonomic interface for configuring runtime invariants and start parameters for RelayRLAgent"]
#[derive(Clone)]
pub struct AgentBuilder<B: Backend + BackendMatcher<Backend = B>> {
    pub settings: BuilderSettings<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> AgentBuilder<B> {
    /// Creates a new builder with default local-inference settings.
    pub fn builder() -> Self {
        Self {
            settings: BuilderSettings::<B>::default(),
        }
    }

    /// Runtime invariants for `RelayRLAgent`
    pub fn modes(self) -> AgentBuildInvariants<B> {
        AgentBuildInvariants::<B>::with(self)
    }

    /// Adjustable runtime variables for `RelayRLAgent`
    pub fn params(self) -> AgentBuildParameters<B> {
        AgentBuildParameters::<B>::with(self)
    }

    /// Consumes the builder and returns the `(RelayRLAgent, AgentStartParameters)` pair.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// let (mut agent, params) = AgentBuilder::<NdArray>::builder()
    ///     .modes()
    ///     .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Shared))
    ///     .actor_data_mode(ActorDataMode::OfflineWithCache(1024))
    ///     .params()
    ///     .data_routers(2)
    ///     .data_buffer_size(1024)
    ///     .default_model(model)
    ///     .config_polling_seconds(3)
    ///     .config_path(PathBuf::from("client_config.json"))
    ///     .build()
    ///     .await?;
    /// agent.start(params).await?;
    /// agent.shutdown().await?;
    /// ```
    pub async fn build(self) -> Result<(RelayRLAgent<B>, AgentStartParameters<B>), ClientError> {
        // Initialize agent object
        let agent: RelayRLAgent<B> = RelayRLAgent::<B>::init(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.settings.transport_mode.unwrap_or_default(),
            self.settings.client_modes,
        );

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            data_routers: self.settings.data_routers.unwrap_or(1),
            data_buffer_size: self.settings.data_buffer_size.unwrap_or(1024),
            default_model: self.settings.default_model,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters: self.settings.default_hyperparameters,
            config_polling_seconds: self.settings.config_polling_seconds,
            config_path: self.settings.config_path,
        };

        Ok((agent, startup_params))
    }
}
