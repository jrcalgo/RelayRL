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

/// Runtime modes consumed by the client to enable/disable functionality.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ClientModes {
    pub actor_inference_mode: ActorInferenceMode,
    pub actor_training_data_mode: ActorTrainingDataMode,
}

pub type ReplayBufferSize = usize;
pub type SaveModelPath = PathBuf;

/// Parameters used to start a [`RelayRLAgent`].
///
/// Typically constructed via [`AgentBuilder::build`] and then passed to
/// [`RelayRLAgent::start`] or [`RelayRLAgent::restart`].
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
/// let (mut agent, params) = AgentBuilder::<NdArray>::builder()
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
            router_scale: None,
            default_model: None,
            router_buffer_size_per_actor: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters: DefaultHyperparameterArgs::default(),
            config_path: None,
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

    pub fn router_scale(mut self, count: u32) -> Self {
        self.router_scale = Some(count);
        self
    }

    pub fn default_model(mut self, model: ModelModule<B>) -> Self {
        self.default_model = Some(model);
        self
    }

    pub fn router_buffer_size_per_actor(mut self, size: usize) -> Self {
        self.router_buffer_size_per_actor = Some(size);
        self
    }

    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_ppo_params(mut self, ppo_params: PPOParams) -> Self {
        self.default_hyperparameters.ppo = Some(ppo_params);
        self
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_ippo_params(mut self, ippo_params: IPPOParams) -> Self {
        self.default_hyperparameters.ippo = Some(ippo_params);
        self
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn default_mappo_params(mut self, mappo_params: MAPPOParams) -> Self {
        self.default_hyperparameters.mappo = Some(mappo_params);
        self
    }

    /// Build the agent facade plus its startup parameters.
    ///
    /// # Errors
    /// Returns an error if the selected modes are internally inconsistent.
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
