#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
use crate::network::client::agent::{ClientError, RelayRLAgent};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::NetworkParams;

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::get_context_entries;
use relayrl_algorithms::prelude::ppo::algorithm::{IPPOParams, MAPPOParams, PPOParams};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::tensor::{BackendMatcher, DeviceType};
use relayrl_types::model::ModelModule;

use burn_tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Hyperparameter overrides forwarded to a training server at handshake time.
///
/// When `config_default_init` is `true`, any `None` field is filled from the JSON config file;
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

/// Selects how actors record and forward trajectory data.
///
/// The `Offline*` variants write to memory and/or local files and are currently the only fully supported path.
/// The `Online*` variants stream data to a training server and require a transport feature.
///
/// ```ignore
/// # use relayrl::network::{AgentBuilder, ActorTrainingDataMode};
/// # use burn_ndarray::NdArray;
/// let (agent, params) = AgentBuilder::<NdArray>::builder()
///     .actor_training_data_mode(ActorTrainingDataMode::OfflineWithFilesAndMemory(None))
///     .build()
///     .await?;
/// ```
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

/// Per-actor device, model, and hyperparameter defaults used when creating actors.
#[derive(Clone)]
pub struct ActorParams<B: Backend + BackendMatcher<Backend = B>> {
    pub device: DeviceType,
    pub default_model: Option<ModelModule<B>>,
    pub hyperparameters: Option<DefaultHyperparameterArgs>,
}

impl<B: Backend + BackendMatcher<Backend = B>> Default for ActorParams<B> {
    fn default() -> Self {
        Self {
            device: DeviceType::Cpu,
            default_model: None,
            hyperparameters: Some(DefaultHyperparameterArgs::default()),
        }
    }
}

/// Active inference and data-collection modes applied across all runtime actors.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ClientModes {
    pub actor_inference_mode: ActorInferenceMode,
    pub actor_training_data_mode: ActorTrainingDataMode,
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
    pub router_scale: u32,
    pub default_model: Option<ModelModule<B>>,
    pub router_buffer_size_per_actor: Option<usize>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub default_hyperparameters: DefaultHyperparameterArgs,
    pub config_path: Option<PathBuf>,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for AgentStartParameters<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RLAgentStartParameters")
    }
}

/// Fluent builder for constructing a `RelayRLAgent` and its startup parameters.
///
/// Each setter returns the updated builder; `build()` consumes it and yields `(RelayRLAgent<B>, AgentStartParameters<B>)`.
///
/// ```ignore
/// use relayrl::network::{AgentBuilder, ActorTrainingDataMode, RelayRLAgentActors};
/// use relayrl::types::model::ModelModule;
/// use burn_ndarray::NdArray;
/// use std::path::PathBuf;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let model = ModelModule::<NdArray>::load_from_path("model_dir")?;
/// let (mut agent, params) = AgentBuilder::<NdArray>::builder()
///     .default_model(model)
///     .router_scale(2)
///     .actor_training_data_mode(ActorTrainingDataMode::OfflineWithMemory)
///     .config_path(PathBuf::from("client_config.json"))
///     .build()
///     .await?;
///
/// agent.start(params).await?;
/// let ids = agent.get_actor_ids()?;
/// agent.shutdown().await?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct AgentBuilder<B: Backend + BackendMatcher<Backend = B>> {
    pub client_modes: ClientModes,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub transport_type: Option<TransportType>,
    pub router_scale: Option<u32>,
    pub default_model: Option<ModelModule<B>>,
    pub router_buffer_size_per_actor: Option<usize>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub default_hyperparameters: DefaultHyperparameterArgs,
    pub config_path: Option<PathBuf>,
}

impl<B: Backend + BackendMatcher<Backend = B>> AgentBuilder<B> {
    /// Creates a new builder with default local-inference settings.
    pub fn builder() -> Self {
        Self {
            client_modes: ClientModes::default(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_type: Some(TransportType::default()),
            router_scale: None,
            default_model: None,
            router_buffer_size_per_actor: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters: DefaultHyperparameterArgs::default(),
            config_path: None,
        }
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
        self.client_modes.actor_inference_mode = actor_inference_mode;
        self
    }

    /// Sets the training data collection mode for all actors. Defaults to `ActorTrainingDataMode::OfflineWithMemory`.
    ///
    /// ```ignore
    /// # use relayrl::network::{AgentBuilder, ActorTrainingDataMode};
    /// # use burn_ndarray::NdArray;
    /// let builder = AgentBuilder::<NdArray>::builder()
    ///     .actor_training_data_mode(ActorTrainingDataMode::OfflineWithFilesAndMemory(None));
    /// ```
    pub fn actor_training_data_mode(
        mut self,
        actor_training_data_mode: ActorTrainingDataMode,
    ) -> Self {
        self.client_modes.actor_training_data_mode = actor_training_data_mode;
        self
    }

    /// Selects the network transport type for server-backed workflows. Requires `zmq-transport` or `nats-transport`.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn transport_type(mut self, transport_type: TransportType) -> Self {
        self.transport_type = Some(transport_type);
        self
    }

    /// Sets the number of routing workers started alongside the coordinator. Defaults to `1`.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// let builder = AgentBuilder::<NdArray>::builder().router_scale(4);
    /// ```
    pub fn router_scale(mut self, count: u32) -> Self {
        self.router_scale = Some(count);
        self
    }

    /// Provides a default model pre-loaded into each actor at startup.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use relayrl::types::model::ModelModule;
    /// # use burn_ndarray::NdArray;
    /// let model = ModelModule::<NdArray>::load_from_path("model_dir")?;
    /// let builder = AgentBuilder::<NdArray>::builder().default_model(model);
    /// ```
    pub fn default_model(mut self, model: ModelModule<B>) -> Self {
        self.default_model = Some(model);
        self
    }

    /// Overrides the per-actor router channel capacity. When unset the value in the JSON config (default `1000`) is used.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// let builder = AgentBuilder::<NdArray>::builder().router_buffer_size_per_actor(2048);
    /// ```
    pub fn router_buffer_size_per_actor(mut self, size: usize) -> Self {
        self.router_buffer_size_per_actor = Some(size);
        self
    }

    /// Sets the JSON config file path. Defaults to `client_config.json` in the working directory.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// # use std::path::PathBuf;
    /// let builder = AgentBuilder::<NdArray>::builder()
    ///     .config_path(PathBuf::from("my_config.json"));
    /// ```
    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    /// Supplies default PPO hyperparameters forwarded to the training server. Requires a transport feature.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_ppo_params(mut self, ppo_params: PPOParams) -> Self {
        self.default_hyperparameters.ppo = Some(ppo_params);
        self
    }

    /// Supplies default IPPO hyperparameters forwarded to the training server. Requires a transport feature.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_ippo_params(mut self, ippo_params: IPPOParams) -> Self {
        self.default_hyperparameters.ippo = Some(ippo_params);
        self
    }

    /// Supplies default MAPPO hyperparameters forwarded to the training server. Requires a transport feature.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_mappo_params(mut self, mappo_params: MAPPOParams) -> Self {
        self.default_hyperparameters.mappo = Some(mappo_params);
        self
    }

    /// Consumes the builder and returns the `(RelayRLAgent, AgentStartParameters)` pair.
    ///
    /// ```ignore
    /// # use relayrl::network::AgentBuilder;
    /// # use burn_ndarray::NdArray;
    /// let (mut agent, params) = AgentBuilder::<NdArray>::builder()
    ///     .router_scale(2)
    ///     .build()
    ///     .await?;
    /// agent.start(params).await?;
    /// agent.shutdown().await?;
    /// ```
    pub async fn build(self) -> Result<(RelayRLAgent<B>, AgentStartParameters<B>), ClientError> {
        // Initialize agent object
        let agent: RelayRLAgent<B> = RelayRLAgent::<B>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.transport_type.unwrap_or_default(),
            self.client_modes,
        );

        // Tuple parameters
        let startup_params: AgentStartParameters<B> = AgentStartParameters::<B> {
            router_scale: self.router_scale.unwrap_or(1),
            default_model: self.default_model,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters: self.default_hyperparameters,
            router_buffer_size_per_actor: self.router_buffer_size_per_actor,
            config_path: self.config_path,
        };

        Ok((agent, startup_params))
    }
}
