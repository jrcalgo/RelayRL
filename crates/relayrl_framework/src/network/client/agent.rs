//! Client API for starting and controlling the RelayRL client runtime.
//!
//! This module provides:
//! - `RelayRLAgent`: a thin facade over the runtime coordinator.
//! - `AgentBuilder`: ergonomic construction of an agent instance plus its startup parameters.
//! - Mode/config enums that describe inference and trajectory recording behavior.

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportMode;
#[cfg(feature = "zmq-transport")]
pub use crate::network::client::builder::ZmqTrainingAddressesArgs;
pub use crate::network::client::builder::{
    ActorDataMode, ActorInferenceMode, AgentBuilder, AgentStartParameters, AlgorithmInitArgs,
    ClientModes, DefaultHyperparameterArgs, LocalTrajectoryFileParams, LocalTrajectoryFileType,
    ModelMode, ReplayBufferSize, SaveModelPath,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub use crate::network::client::builder::{InferenceAddressesArgs, TrainingAddressesArgs};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub use crate::network::client::builder::{InferenceParams, TrainingParams};
pub(crate) use crate::network::client::builder::{uses_local_file_writing, uses_trajectory_cache};
pub use crate::network::client::runtime::actor::ActorInfo;
use crate::network::client::runtime::control::coordinator::{
    ClientActors, ClientCoordinator, ClientEnvironments, ClientInterface, CoordinatorError,
    ToAnyBurnTensor,
};
use crate::network::client::runtime::control::state_manager::ActorUuid;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::control::state_manager::StateManagerError;
use crate::prelude::utilities::config::ClientConfigLoader;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::NetworkParams;

use active_uuid_registry::UuidPoolError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::get_context_entries;
use relayrl_algorithms::prelude::nn::NeuralNetwork;
use relayrl_algorithms::prelude::ppo::trainer::PPOTrainerSpec;
use relayrl_env_trait::traits::Environment;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{BackendMatcher, DeviceType};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::ModelModule;
use relayrl_types::model::utils::validate_module;

use active_uuid_registry::registry_uuid::Uuid;

use async_trait::async_trait;
use burn_tensor::{BasicOps, Numeric, Tensor, TensorKind, backend::Backend};
use dashmap::DashSet;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;

/// Errors returned by the client-facing API.
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
    /// Returned when `scale_throughput(0)` is called.
    #[error("Noop router scale: {0}")]
    NoopRouterScale(String),
    /// Returned when `new_actors(0, ...)` or `remove_actors([])` is called.
    #[error("Noop actor count: {0}")]
    NoopActorCount(String),
    #[error("Invalid data parameters: {0}")]
    InvalidDataParams(String),
    #[error("Invalid inference mode: {0}")]
    InvalidInferenceMode(String),
    #[error("Invalid trajectory file directory: {0}")]
    InvalidTrajectoryFileDirectory(String),
    #[error("Invalid env count: {0}")]
    InvalidEnvCount(String),
    #[error("Model validation failed: {0}")]
    ModelValidationFailed(String),
    /// Returned by `update_model` when the agent is in an `Online*` training data mode.
    #[error("Update model is not supported: {0}")]
    ModelUpdateNotSupported(String),
    /// Returned when a second `run_env_*` call is made for an actor already running a loop.
    #[error("Run env is already active for actor {0}")]
    RunEnvActive(String),
}

/// Client entry point for the RelayRL framework.
///
/// `RelayRLAgent` is a thin facade over the runtime coordinator, providing a stable public API
/// for starting, scaling, and interacting with runtime actors.
pub struct RelayRLAgent<B: Backend + BackendMatcher<Backend = B>> {
    coordinator: ClientCoordinator<B>,
    run_env_active_flags: DashSet<Uuid>,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for RelayRLAgent<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RLAgent")
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> RelayRLAgent<B> {
    /// Creates a new agent from runtime-invariant configuration args; prefer `AgentBuilder` for ergonomic construction.
    ///
    /// ```ignore
    /// let agent = RelayRLAgent::<NdArray>::init(ClientModes::default());
    /// ```
    pub fn init(
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_mode: TransportMode,
        client_modes: ClientModes,
    ) -> Self {
        Self {
            coordinator: ClientCoordinator::<B>::new(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                transport_mode,
                client_modes,
            ),
            run_env_active_flags: DashSet::new(),
        }
    }

    /// Starts the coordinator, managers, data routers, and supporting runtime tasks described by `params`.
    ///
    /// ```ignore
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let (mut agent, params) = AgentBuilder::<NdArray>::builder().build().await?;
    /// agent.start(params).await?;
    /// # Ok(()) }
    /// ```
    pub async fn start(&mut self, params: AgentStartParameters<B>) -> Result<(), ClientError> {
        let AgentStartParameters {
            data_routers,
            data_buffer_size,
            default_model,
            config_path,
            config_polling_seconds,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters,
        } = params;

        if data_routers == 0 {
            return Err(ClientError::InvalidDataParams(
                "Invalid argument; data_routers is set to zero in `start()`".to_string(),
            ));
        } else if data_buffer_size == 0 {
            return Err(ClientError::InvalidDataParams(
                "Invalid argument; data_buffer_size is set to zero".to_string(),
            ));
        }

        self.coordinator
            .start(
                data_routers,
                data_buffer_size,
                default_model,
                config_path,
                config_polling_seconds,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                default_hyperparameters,
            )
            .await
            .map_err(Into::<ClientError>::into)?;

        Ok(())
    }

    /// Tears down and reinitializes the runtime without destroying the agent handle.
    ///
    /// ```ignore
    /// # async fn run(mut agent: RelayRLAgent<NdArray>, params: AgentStartParameters<NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// agent.restart(params).await?;
    /// # Ok(()) }
    /// ```
    pub async fn restart(&mut self, params: AgentStartParameters<B>) -> Result<(), ClientError> {
        let AgentStartParameters {
            data_routers,
            data_buffer_size,
            default_model,
            config_path,
            config_polling_seconds,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters,
        } = params;

        if data_routers == 0 {
            return Err(ClientError::InvalidDataParams(
                "Invalid argument; data_routers is set to zero".to_string(),
            ));
        } else if data_buffer_size == 0 {
            return Err(ClientError::InvalidDataParams(
                "Invalid argument; data_buffer_size is set to zero".to_string(),
            ));
        }

        self.coordinator
            .restart(
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

    /// Gracefully shuts down all runtime components without destroying the agent handle.
    ///
    /// The returned map, if any, is keyed by each actor's stable [`ActorUuid`] rather than its
    /// `ActorInfo` handle: it is a one-shot snapshot of whatever trajectories were still buffered
    /// at shutdown, taken after every actor's runtime handle is gone, so a stable id key is both
    /// sufficient and immune to any in-flight rename.
    ///
    /// ```ignore
    /// # async fn run(mut agent: RelayRLAgent<NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// agent.shutdown().await?;
    /// # Ok(()) }
    /// ```
    pub async fn shutdown(
        &mut self,
    ) -> Result<Option<HashMap<ActorUuid, Vec<Arc<RelayRLTrajectory>>>>, ClientError> {
        let traj_map = self.coordinator.shutdown().await?;
        Ok(traj_map)
    }

    /// Adjusts the routing/buffer worker pool live. Positive values add workers; negative values remove them.
    ///
    /// Adds/removes internal message filters and trajectory data buffers used for sink exfil.
    ///
    /// If total runtime `data routers`:
    /// - `== actor count`: assignment ratio of 1:1.
    /// - `> actor count`: assignment ratio of 1:1, with excess routers left idle.
    /// - `< actor count`: actors assigned as evenly as possible across routers.
    ///
    /// ```ignore
    /// # async fn run(mut agent: RelayRLAgent<bNdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// agent.scale_data_routers(2).await?;   // add two routing workers
    /// agent.scale_data_routers(-1).await?;  // remove one
    /// # Ok(()) }
    /// ```
    pub async fn scale_data_routers(&mut self, adjustment: i32) -> Result<(), ClientError> {
        match adjustment {
            add if adjustment > 0 => {
                self.coordinator.scale_routers_out(add as u32).await?;
                Ok(())
            }
            remove if adjustment < 0 => {
                self.coordinator
                    .scale_routers_in(remove.unsigned_abs())
                    .await?;
                Ok(())
            }
            _ => Err(ClientError::NoopRouterScale(
                "Noop router scale: `data_routers` set to zero in `scale_data_routers()`"
                    .to_string(),
            )),
        }
    }

    /// Scales trajectory capacity of router data buffers.
    ///
    /// Used for altering the trajectory queue and semaphore permit capacities.
    ///
    /// ```ignore
    /// # async fn
    /// agent.scale_data_buffers(100_000).await?;
    /// ```
    pub async fn scale_data_buffers(&mut self, new_size: usize) -> Result<(), ClientError> {
        if new_size == 0 {
            return Err(ClientError::InvalidDataParams(
                "Invalid argument; new_size is set to zero".to_string(),
            ));
        }
        Ok(self.coordinator.scale_data_buffers(new_size).await?)
    }

    /// Reads and returns the current `ClientConfigLoader` from the watched config file.
    pub async fn get_config(&self) -> Result<ClientConfigLoader, ClientError> {
        Ok(self.coordinator.get_config().await?)
    }

    /// Applies the config at `config_path` immediately and updates the active runtime settings.
    pub async fn set_config_path(&self, config_path: PathBuf) -> Result<(), ClientError> {
        self.coordinator.set_config_path(config_path).await?;
        Ok(())
    }
}

/// Provides actor lifecycle management for a `RelayRLAgent`.
///
/// ```ignore
/// # use relayrl::network::{RelayRLActors, AgentBuilder};
/// # use burn_ndarray::NdArray;
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let (mut agent, params) = AgentBuilder::<NdArray>::builder().build().await?;
/// agent.start(params).await?;
/// let ids = agent.new_actors::<2, 2>(4, DeviceType::Cpu, 1_000, None).await?;
/// agent.remove_actors(ids).await?;
/// # Ok(()) }
/// ```
#[async_trait]
pub trait RelayRLActors<B: Backend + BackendMatcher<Backend = B>> {
    /// Creates one actor on `device` with a trajectory buffer of `max_traj_length` steps.
    ///
    /// `D_IN` and `D_OUT` declare the observation and action tensor ranks for this actor.
    async fn new_actor<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        device: DeviceType,
        max_traj_length: usize,
        nametag: Option<&str>,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<ActorInfo, ClientError>;

    /// Creates `count` actors; equivalent to calling `new_actor` that many times.
    async fn new_actors<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        count: u32,
        device: DeviceType,
        max_traj_length: usize,
        nametag: Option<&str>,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<Vec<ActorInfo>, ClientError>;

    /// Aborts an actor's task and frees all associated resources.
    async fn remove_actor(&mut self, actor: &ActorInfo) -> Result<(), ClientError>;

    /// Removes multiple actors; equivalent to calling `remove_actor` for each.
    async fn remove_actors(&mut self, actors: &[ActorInfo]) -> Result<(), ClientError>;

    /// Returns the `ActorInfo` of the specified actor.
    async fn get_actor(&self, id: ActorUuid) -> Result<ActorInfo, ClientError>;

    /// Returns the `ActorInfo`s of all live actors from the namespaced registry.
    async fn get_all_actors(&self) -> Result<Vec<ActorInfo>, ClientError>;

    /// Returns the `ActorInfo`s of all live actors that match the specified D_IN, D_OUT qualifications
    async fn get_actors_by_rank<const D_IN: usize, const D_OUT: usize>(
        &self,
    ) -> Result<Vec<ActorInfo>, ClientError>;

    /// Returns the `ActorInfo`s of all live actors that match the specified nametag.
    async fn get_actors_by_tag(&self, nametag: Option<&str>)
    -> Result<Vec<ActorInfo>, ClientError>;

    /// Renames a live actor's ID in place; its task and inbox are preserved.
    ///
    /// `actor` observes the new id afterward (and so does every other clone of it), since the
    /// id lives in a shared slot rather than being copied into each `ActorInfo` handle.
    async fn set_actor_id(
        &mut self,
        actor: &ActorInfo,
        new_id: ActorUuid,
    ) -> Result<(), ClientError>;

    /// Renames a live actor's nametag in place; useful for tracking.
    ///
    /// `actor` observes the new nametag afterward (and so does every other clone of it), since
    /// the nametag lives in a shared slot rather than being copied into each `ActorInfo` handle.
    async fn set_actor_nametag(
        &mut self,
        actor: &ActorInfo,
        new_nametag: Option<&str>,
    ) -> Result<(), ClientError>;

    /// Hot-swaps the model into the specified actors (or all actors when `specific_actor_ids` is `None`).
    ///
    /// In `ModelMode::Shared`, one representative actor per device is updated so each shared handle
    /// is refreshed exactly once. Rejected with `ModelUpdateNotSupported` under `Online*` data modes.
    ///
    /// This method verifies model-actor compatibility using the `D_IN`, `D_OUT` generics. If:
    ///  - a) the model metadata ranks do not match the generic args, return error.
    ///  - b) an actor id does not match the model metadata and generic args, log error and continue.
    ///  - c) all actor ids do not match the model metadata and generic args, return error.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<burn_ndarray::NdArray>, new_model: ModelModule<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// let ids = agent.get_actor_ids()?;
    /// // Swap into actors 0 and 2 only; actor 1 keeps the previous policy.
    /// agent.update_model<2, 1>(Some(vec![ids[0], ids[2]].as_slice()), new_model).await?;
    /// let versions = agent.get_model_version(vec![ids[0], ids[2]].as_slice()).await?;
    /// # Ok(()) }
    /// ```
    async fn update_models<const D_IN: usize, const D_OUT: usize>(
        &self,
        specific_actors: Option<&[ActorInfo]>,
        model: ModelModule<B>,
    ) -> Result<(), ClientError>;

    /// Returns `(ActorUuid, swap_count)` pairs reflecting how many times each actor's model has been hot-swapped.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// let ids = agent.get_actor_ids()?;
    /// let versions = agent.get_model_versions(ids).await?;
    /// # Ok(()) }
    /// ```
    async fn get_model_versions(
        &self,
        actors: &[ActorInfo],
    ) -> Result<Vec<(ActorInfo, i64)>, ClientError>;

    /// Returns a shared view of all in-memory trajectories collected across the specified actors.
    ///
    /// Only populated under `...WithMemory` or `...WithFilesAndMemory` data modes.
    ///
    /// The returned map, if any, is keyed by each actor's stable [`ActorUuid`] rather than its
    /// `ActorInfo` handle: it is a one-shot snapshot copied out of the shared cache at the moment
    /// of the call, so a stable id key keeps lookups valid even if one of the selected actors is
    /// renamed via `set_actor_id` afterward.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// let cache = agent.drain_trajectory_cache().await?;
    /// for (actor_id, trajectories) in cache.iter() {
    ///     println!("actor {:?} has {} trajectories", actor_id, trajectories.len());
    /// }
    /// # Ok(()) }
    /// ```
    fn drain_trajectory_caches(
        &self,
        actors: &[ActorInfo],
    ) -> Option<HashMap<ActorUuid, Vec<Arc<RelayRLTrajectory>>>>;
}

#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> RelayRLActors<B> for RelayRLAgent<B> {
    async fn new_actor<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        device: DeviceType,
        max_traj_length: usize,
        nametag: Option<&str>,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<ActorInfo, ClientError> {
        let actor_nametag = self
            .coordinator
            .resolve_new_nametag(nametag, 1)
            .await?
            .map(|tags| tags[0].clone());

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let actor_info = self
            .coordinator
            .new_actor::<D_IN, D_OUT>(
                device,
                max_traj_length,
                actor_nametag,
                default_model,
                algorithm_args.unwrap_or_default(),
                true,
                true,
            )
            .await?;
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        let actor_info = self
            .coordinator
            .new_actor::<D_IN, D_OUT>(device, max_traj_length, actor_nametag, default_model)
            .await?;
        Ok(actor_info)
    }

    async fn new_actors<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        count: u32,
        device: DeviceType,
        max_traj_length: usize,
        nametag: Option<&str>,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<Vec<ActorInfo>, ClientError> {
        if count == 0 {
            Err(ClientError::NoopActorCount(
                "Noop actor count: `count` set to zero".to_string(),
            ))
        } else if count == 1 {
            Ok(vec![
                self.new_actor::<D_IN, D_OUT>(
                    device,
                    max_traj_length,
                    nametag,
                    default_model,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    algorithm_args,
                )
                .await?,
            ])
        } else {
            let mut actor_info: Vec<ActorInfo> = Vec::new();
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let algorithm_args = algorithm_args.unwrap_or_default();

            let nametags = self.coordinator.resolve_new_nametag(nametag, count).await?;
            for i in 0..count {
                let actor_nametag = nametags.as_ref().map(|tags| tags[i as usize].clone());

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                actor_info.push(
                    self.coordinator
                        .new_actor::<D_IN, D_OUT>(
                            device.clone(),
                            max_traj_length,
                            actor_nametag,
                            default_model.clone(),
                            algorithm_args.clone(),
                            false,
                            false,
                        )
                        .await?,
                );
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                actor_info.push({
                    self.coordinator
                        .new_actor::<D_IN, D_OUT>(
                            device.clone(),
                            max_traj_length,
                            actor_nametag,
                            default_model.clone(),
                        )
                        .await?
                });
            }

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if let (
                ActorDataMode::Online(_)
                | ActorDataMode::OnlineWithFiles(..)
                | ActorDataMode::OnlineWithCache(..),
                ActorInferenceMode::Server(_),
            ) = (
                &self.coordinator.client_modes.actor_data_mode,
                &self.coordinator.client_modes.actor_inference_mode,
            ) {
                // sends all new actor ids to the server
                let actor_entries: Vec<(String, String, Uuid)> = {
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

                let resolved_algorithm_args: AlgorithmInitArgs = {
                    let some_relevant_actor_id = actor_entries[0].2;
                    let state_read = self
                        .coordinator
                        .runtime_params
                        .as_ref()
                        .ok_or(ClientError::CoordinatorError(
                            CoordinatorError::NoRuntimeInstanceError,
                        ))?
                        .shared_state
                        .read()
                        .await;
                    let actor_runtime_handle = state_read
                        .actor_runtime_handles
                        .get(&some_relevant_actor_id)
                        .ok_or(ClientError::CoordinatorError(
                            CoordinatorError::NoRuntimeInstanceError,
                        ))?;
                    actor_runtime_handle
                        .value()
                        .current_algorithm_args()
                        .map_err(|e| {
                            ClientError::CoordinatorError(CoordinatorError::StateManagerError(
                                StateManagerError::from(e),
                            ))
                        })?
                };

                self.coordinator
                    .send_client_ids_to_server(actor_entries.clone(), true)
                    .await?;

                if let ActorDataMode::Online(_)
                | ActorDataMode::OnlineWithFiles(..)
                | ActorDataMode::OnlineWithCache(..) =
                    &self.coordinator.client_modes.actor_data_mode
                {
                    self.coordinator
                        .send_algorithm_init_request(actor_entries.clone(), resolved_algorithm_args)
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

            Ok(actor_info)
        }
    }

    async fn remove_actor(&mut self, actor: &ActorInfo) -> Result<(), ClientError> {
        self.coordinator
            .remove_actor(
                actor,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                true,
            )
            .await?;
        Ok(())
    }

    async fn remove_actors(&mut self, actors: &[ActorInfo]) -> Result<(), ClientError> {
        if actors.is_empty() {
            Err(ClientError::NoopActorCount(
                "Noop actor count: `actors` is empty in `remove_actors()`".to_string(),
            ))
        } else if actors.len() == 1 {
            self.remove_actor(&actors[0]).await
        } else {
            for actor in actors {
                self.coordinator
                    .remove_actor(
                        actor,
                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        false,
                    )
                    .await?;
            }

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if let (
                ActorDataMode::Online(_)
                | ActorDataMode::OnlineWithFiles(..)
                | ActorDataMode::OnlineWithCache(..),
                ActorInferenceMode::Server(_),
            ) = (
                &self.coordinator.client_modes.actor_data_mode,
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
        }
    }

    async fn get_actor(&self, id: ActorUuid) -> Result<ActorInfo, ClientError> {
        self.coordinator
            .get_actor(id)
            .await
            .map_err(ClientError::from)
    }

    async fn get_all_actors(&self) -> Result<Vec<ActorInfo>, ClientError> {
        self.coordinator
            .get_all_actors()
            .await
            .map_err(ClientError::from)
    }

    async fn get_actors_by_rank<const D_IN: usize, const D_OUT: usize>(
        &self,
    ) -> Result<Vec<ActorInfo>, ClientError> {
        self.coordinator
            .get_actors_by_rank::<D_IN, D_OUT>()
            .await
            .map_err(ClientError::from)
    }

    async fn get_actors_by_tag(
        &self,
        nametag: Option<&str>,
    ) -> Result<Vec<ActorInfo>, ClientError> {
        self.coordinator
            .get_actors_by_tag(nametag)
            .await
            .map_err(ClientError::from)
    }

    async fn set_actor_id(
        &mut self,
        actor: &ActorInfo,
        new_id: ActorUuid,
    ) -> Result<(), ClientError> {
        self.coordinator.set_actor_id(actor, new_id).await?;
        Ok(())
    }

    async fn set_actor_nametag(
        &mut self,
        actor: &ActorInfo,
        new_nametag: Option<&str>,
    ) -> Result<(), ClientError> {
        self.coordinator
            .set_actor_nametag(actor, new_nametag)
            .await?;
        Ok(())
    }

    async fn update_models<const D_IN: usize, const D_OUT: usize>(
        &self,
        specific_actors: Option<&[ActorInfo]>,
        model: ModelModule<B>,
    ) -> Result<(), ClientError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let ActorDataMode::Online(_)
        | ActorDataMode::OnlineWithFiles(..)
        | ActorDataMode::OnlineWithCache(..) = self.coordinator.client_modes.actor_data_mode
        {
            log::warn!("Updating model locally is not supported in Online training data modes");
            return Err(ClientError::ModelUpdateNotSupported(
                "Updating model locally is not supported in Online training data modes".to_string(),
            ));
        }

        if let Err(e) = validate_module::<B>(&model) {
            return Err(ClientError::ModelValidationFailed(e.to_string()));
        }
        self.coordinator
            .update_models::<D_IN, D_OUT>(specific_actors, model)
            .await?;
        Ok(())
    }

    async fn get_model_versions(
        &self,
        actors: &[ActorInfo],
    ) -> Result<Vec<(ActorInfo, i64)>, ClientError> {
        Ok(self.coordinator.get_model_versions(actors).await?)
    }

    fn drain_trajectory_caches(
        &self,
        actors: &[ActorInfo],
    ) -> Option<HashMap<ActorUuid, Vec<Arc<RelayRLTrajectory>>>> {
        match self.coordinator.drain_trajectory_caches(actors) {
            Ok(traj_map) => traj_map,
            Err(e) => {
                log::error!("Failed to drain trajectory caches: {:?}", e);
                None
            }
        }
    }
}

#[allow(async_fn_in_trait)]
pub trait RelayRLStepDriven<B: Backend + BackendMatcher<Backend = B>> {
    /// Sends an observation to the specified actor and returns its action.
    ///
    /// `D_IN` and `D_OUT` are the observation and action tensor ranks and must match those used when
    /// the actor was created. Returns the action.
    ///
    /// Use `request_actions` to perform inference and retrieve `RelayRLAction`s for multiple actors.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let id = agent.get_actor_ids()?[0];
    /// let obs = Tensor::<NdArray, 2, Float>::zeros([1, 8], &Default::default());
    /// let action = agent.request_action(id, obs, None, 0.0).await?;
    /// # Ok(()) }
    /// ```
    async fn request_action<
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B> + 'static,
        KindOut: TensorKind<B> + 'static,
    >(
        &self,
        actor: &ActorInfo,
        observation: Tensor<B, D_IN, KindIn>,
        mask: Option<Tensor<B, D_OUT, KindOut>>,
        reward: f32,
    ) -> Result<Arc<RelayRLAction>, ClientError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>;

    /// Sends an observation to the specified actors and returns their actions.
    ///
    /// `D_IN` and `D_OUT` are the observation and action tensor ranks and must match those used when
    /// the actors were created. Returns one `(ActorUuid, RelayRLAction)` per valid id in `ids`.
    ///
    /// Use `request_action` to perform inference and retrieve a `RelayRLAction` for a single actor.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let ids = agent.get_actor_ids()?;
    /// let obs = Tensor::<NdArray, 2, Float>::zeros([1, 8], &Default::default());
    /// let actions = agent.request_action(ids.clone(), obs, None, 0.0).await?;
    /// # Ok(()) }
    /// ```
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
    ) -> Result<Vec<(ActorInfo, Arc<RelayRLAction>)>, ClientError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>;

    /// Appends a terminal action (`done=true`) to the specified actor's current trajectory, signalling episode end.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// let id = agent.get_actor_ids()?[0];
    /// agent.flag_last_action(id, Some(1.0)).await?;
    /// # Ok(()) }
    /// ```
    async fn flag_last_action(
        &self,
        actor: &ActorInfo,
        reward: Option<f32>,
    ) -> Result<(), ClientError>;

    /// Appends a terminal action (`done=true`) to each named actor's current trajectory, signalling episode end.
    ///
    /// ```ignore
    /// # async fn run(agent: &RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
    /// let ids = agent.get_actor_ids()?;
    /// agent.flag_last_actions(ids, Some(1.0)).await?;
    /// # Ok(()) }
    /// ```
    async fn flag_last_actions(
        &self,
        actors: &[ActorInfo],
        reward: Option<f32>,
    ) -> Result<(), ClientError>;
}

impl<B: Backend + BackendMatcher<Backend = B>> RelayRLStepDriven<B> for RelayRLAgent<B> {
    async fn request_action<
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B> + 'static,
        KindOut: TensorKind<B> + 'static,
    >(
        &self,
        actor: &ActorInfo,
        observation: Tensor<B, D_IN, KindIn>,
        mask: Option<Tensor<B, D_OUT, KindOut>>,
        reward: f32,
    ) -> Result<Arc<RelayRLAction>, ClientError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>,
    {
        let actions = self
            .request_actions(std::slice::from_ref(actor), observation, mask, reward)
            .await?;
        Ok(actions[0].1.clone())
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
    ) -> Result<Vec<(ActorInfo, Arc<RelayRLAction>)>, ClientError>
    where
        Tensor<B, D_IN, KindIn>: ToAnyBurnTensor<B, D_IN>,
        Tensor<B, D_OUT, KindOut>: ToAnyBurnTensor<B, D_OUT>,
    {
        Ok(self
            .coordinator
            .request_actions(actors, observation, mask, reward)
            .await?)
    }

    async fn flag_last_action(
        &self,
        actor: &ActorInfo,
        reward: Option<f32>,
    ) -> Result<(), ClientError> {
        self.flag_last_actions(std::slice::from_ref(actor), reward)
            .await?;
        Ok(())
    }

    async fn flag_last_actions(
        &self,
        actors: &[ActorInfo],
        reward: Option<f32>,
    ) -> Result<(), ClientError> {
        self.coordinator.flag_last_actions(actors, reward).await?;
        Ok(())
    }
}

/// Environment-driven execution and management for a `RelayRLAgent`.
///
/// Bind an environment to an actor with `set_env`, then drive rollouts with `run_env_eval` or
/// `run_env_with_ppo`. When `count` (in `set_env`) is `>= 8`, Rayon data parallelism is used
/// across env copies; below 8 they are stepped sequentially.
///
/// ```ignore
/// # async fn run(mut agent: RelayRLAgent<burn_ndarray::NdArray>, env: Box<dyn Environment>) -> Result<(), Box<dyn std::error::Error>> {
/// use relayrl::network::RelayRLBatchEnv;
/// let ids = agent.get_actor_ids()?;
/// agent.set_env(ids[0], env, 16).await?;
/// agent.run_env_eval(ids[0], 1_000).await?;
/// agent.remove_env(ids[0]).await?;
/// # Ok(()) }
/// ```
#[allow(async_fn_in_trait)]
pub trait RelayRLBatchEnv<B: Backend + BackendMatcher<Backend = B>> {
    /// Runs `loop_iters` evaluation steps on the bound environment without applying any training update.
    async fn run_env_eval(&self, actor: &ActorInfo, loop_iters: usize) -> Result<(), ClientError>;

    /// Runs a single-agent PPO training rollout on the bound environment for `loop_iters` steps.
    ///
    /// `max_traj_length` sets the trajectory buffer size. Only one `run_env_*` loop may be active
    /// per actor at a time; a second call returns `ClientError::RunEnvActive`.
    ///
    /// Running
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
    ) -> Result<ModelModule<B>, ClientError>;

    // Runs an independent PPO (IPPO) training rollout; coming soon.
    // async fn run_env_with_ippo<
    //     KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
    //     KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
    //     Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    // >(
    //     &self,
    //     actor_id: ActorUuid,
    //     loop_iters: usize,
    //     max_traj_length: usize,
    //     trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    // ) -> Result<ModelModule<B>, ClientError>;
    // /// Runs a multi-agent PPO (MAPPO) training rollout; coming soon.
    // async fn run_env_with_mappo<
    //     KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
    //     KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
    //     Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    // >(
    //     &self,
    //     actor_id: ActorUuid,
    //     loop_iters: usize,
    //     max_traj_length: usize,
    //     trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    // ) -> Result<ModelModule<B>, ClientError>;

    /// Binds `env` to the actor and associates `count` logical env copies with it.
    async fn set_env(
        &mut self,
        actor: &ActorInfo,
        env: Box<dyn Environment>,
        count: u32,
    ) -> Result<(), ClientError>;

    /// Removes the bound environment from the actor.
    async fn remove_env(&mut self, actor: &ActorInfo) -> Result<(), ClientError>;

    /// Returns the current number of env copies bound to the actor.
    async fn get_env_count(&self, actor: &ActorInfo) -> Result<u32, ClientError>;

    /// Adjusts the env copy count live without rebinding the environment.
    async fn set_env_count(&mut self, actor: &ActorInfo, count: u32) -> Result<(), ClientError>;
}

impl<B: Backend + BackendMatcher<Backend = B>> RelayRLBatchEnv<B> for RelayRLAgent<B> {
    async fn run_env_eval(&self, actor: &ActorInfo, loop_iters: usize) -> Result<(), ClientError> {
        if !self.run_env_active_flags.insert(actor.id()) {
            return Err(ClientError::RunEnvActive(format!(
                "run_env is already active for actor {}",
                actor.id()
            )));
        }
        let result = self
            .coordinator
            .run_env_eval(actor, loop_iters)
            .await
            .map_err(ClientError::from);
        self.run_env_active_flags.remove(&actor.id());
        result
    }

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
    ) -> Result<ModelModule<B>, ClientError> {
        if !self.run_env_active_flags.insert(actor.id()) {
            return Err(ClientError::RunEnvActive(format!(
                "run_env is already active for actor {}",
                actor.id()
            )));
        }
        let result = self
            .coordinator
            .run_env_with_ppo::<KindIn, KindOut, Pi>(
                actor,
                loop_iters,
                max_traj_length,
                trainer_spec,
            )
            .await
            .map_err(ClientError::from);
        self.run_env_active_flags.remove(&actor.id());
        result
    }

    // async fn run_env_with_ippo<
    //     KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
    //     KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
    //     Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    // >(
    //     &self,
    //     actor_id: ActorUuid,
    //     loop_iters: usize,
    //     max_traj_length: usize,
    //     trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    // ) -> Result<ModelModule<B>, ClientError> {
    //     if !self.run_env_active_flags.insert(actor_id) {
    //         return Err(ClientError::RunEnvActive(format!(
    //             "run_env is already active for actor {}",
    //             actor_id
    //         )));
    //     }
    //     let result = self
    //         .coordinator
    //         .run_env_with_ippo::<KindIn, KindOut, Pi>(
    //             actor_id,
    //             loop_iters,
    //             max_traj_length,
    //             trainer_spec,
    //         )
    //         .await
    //         .map_err(ClientError::from);
    //     self.run_env_active_flags.remove(&actor_id);
    //     result
    // }

    // async fn run_env_with_mappo<
    //     KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
    //     KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + 'static,
    //     Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    // >(
    //     &self,
    //     actor_id: ActorUuid,
    //     loop_iters: usize,
    //     max_traj_length: usize,
    //     trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    // ) -> Result<ModelModule<B>, ClientError> {
    //     if !self.run_env_active_flags.insert(actor_id) {
    //         return Err(ClientError::RunEnvActive(format!(
    //             "run_env is already active for actor {}",
    //             actor_id
    //         )));
    //     }
    //     let result = self
    //         .coordinator
    //         .run_env_with_mappo::<KindIn, KindOut, Pi>(
    //             actor_id,
    //             loop_iters,
    //             max_traj_length,
    //             trainer_spec,
    //         )
    //         .await
    //         .map_err(ClientError::from);
    //     self.run_env_active_flags.remove(&actor_id);
    //     result
    // }

    async fn set_env(
        &mut self,
        actor: &ActorInfo,
        env: Box<dyn Environment>,
        count: u32,
    ) -> Result<(), ClientError> {
        Ok(self.coordinator.set_env(actor, env, count).await?)
    }

    async fn remove_env(&mut self, actor: &ActorInfo) -> Result<(), ClientError> {
        Ok(self.coordinator.remove_env(actor).await?)
    }

    async fn set_env_count(&mut self, actor: &ActorInfo, count: u32) -> Result<(), ClientError> {
        let current = self.coordinator.get_env_count(actor).await?;
        match count.cmp(&current) {
            std::cmp::Ordering::Greater => Ok(self
                .coordinator
                .increase_env_count(actor, count - current)
                .await?),
            std::cmp::Ordering::Less => Ok(self
                .coordinator
                .decrease_env_count(actor, current - count)
                .await?),
            std::cmp::Ordering::Equal => Ok(()),
        }
    }

    async fn get_env_count(&self, actor: &ActorInfo) -> Result<u32, ClientError> {
        Ok(self.coordinator.get_env_count(actor).await?)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Bool, Float, Int, Tensor, TensorData};
    use relayrl_types::data::tensor::{AnyBurnTensor, DType, DeviceType, NdArrayDType};
    use relayrl_types::model::{ModelError, ModelFileType, ModelMetadata};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;
    const TEST_ONNX_IDENTITY: &[u8] = &[
        // thank you chat, i did not want to generate this manually whatsoever
        0x08, 0x07, 0x12, 0x0d, 0x72, 0x65, 0x6c, 0x61, 0x79, 0x72, 0x6c, 0x2d, 0x74, 0x65, 0x73,
        0x74, 0x73, 0x3a, 0x67, 0x0a, 0x23, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x06,
        0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x1a, 0x08, 0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74,
        0x79, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x15, 0x72, 0x65,
        0x6c, 0x61, 0x79, 0x72, 0x6c, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x69, 0x64, 0x65, 0x6e,
        0x74, 0x69, 0x74, 0x79, 0x5a, 0x13, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x0a,
        0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x02, 0x62, 0x14, 0x0a, 0x06, 0x6f,
        0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02,
        0x08, 0x02, 0x42, 0x02, 0x10, 0x0d,
    ];

    fn load_test_model_module() -> Result<(tempfile::TempDir, ModelModule<TestBackend>), ModelError>
    {
        let model_dir = tempdir().expect("tempdir should be created");
        let metadata = ModelMetadata {
            model_file: "test.onnx".to_string(),
            model_type: ModelFileType::Onnx,
            input_dtype: DType::NdArray(NdArrayDType::F32),
            output_dtype: DType::NdArray(NdArrayDType::F32),
            input_shape: vec![2],
            output_shape: vec![2],
            default_device: Some(DeviceType::Cpu),
        };

        let model_module =
            ModelModule::<TestBackend>::from_onnx_bytes(TEST_ONNX_IDENTITY.to_vec(), metadata)?;

        Ok((model_dir, model_module))
    }

    #[test]
    fn offline_returns_true() {
        assert!(uses_local_file_writing(&ActorDataMode::OfflineWithFiles(
            None
        )));
    }

    #[test]
    fn disabled_returns_false() {
        assert!(!uses_local_file_writing(&ActorDataMode::Disabled));
    }

    #[test]
    fn model_mode_default_is_independent() {
        assert_eq!(ModelMode::default(), ModelMode::Independent);
    }

    #[test]
    fn actor_inference_mode_default_is_client_independent() {
        assert_eq!(
            ActorInferenceMode::default(),
            ActorInferenceMode::Client(ModelMode::Independent),
        );
    }

    #[test]
    fn client_modes_default_uses_component_defaults() {
        let modes = ClientModes::default();
        assert_eq!(modes.actor_inference_mode, ActorInferenceMode::default());
    }

    #[test]
    fn data_routers_setter_sets_field() {
        let b = AgentBuilder::<TestBackend>::builder()
            .params()
            .data_routers(2);
        assert_eq!(b.builder.settings.data_routers, Some(2));
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
        let (_model_dir, default_model) = match load_test_model_module() {
            Ok(model) => model,
            Err(err) => {
                eprintln!("skipping ONNX model test because ONNX Runtime is unavailable: {err}");
                return;
            }
        };

        let (_agent, params) = AgentBuilder::<TestBackend>::builder()
            .params()
            .default_model(default_model.clone())
            .config_path(config_path.clone())
            .build()
            .await
            .expect("builder should succeed with a local default model");

        assert_eq!(params.data_routers, 1);
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
            params
                .default_model
                .as_ref()
                .expect("builder should preserve the provided default model")
                .metadata
                .output_dtype,
            default_model.metadata.output_dtype
        );
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        assert_eq!(
            params
                .default_model
                .as_ref()
                .expect("builder should preserve the provided default model")
                .metadata
                .input_dtype,
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
    async fn scale_routers_zero_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend>::init(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportMode::default(),
            ClientModes::default(),
        );
        let result = agent.scale_data_routers(0).await;
        assert!(matches!(result, Err(ClientError::NoopRouterScale(_))));
    }

    #[tokio::test]
    async fn new_actors_zero_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend>::init(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportMode::default(),
            ClientModes::default(),
        );
        let result = agent
            .new_actors::<4, 1>(
                0,
                DeviceType::Cpu,
                0usize,
                None,
                None,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                None,
            )
            .await;
        assert!(matches!(result, Err(ClientError::NoopActorCount(_))));
    }

    #[tokio::test]
    async fn remove_actors_empty_vec_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend>::init(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportMode::default(),
            ClientModes::default(),
        );
        let actors: Vec<ActorInfo> = vec![];
        let result = agent.remove_actors(&actors).await;
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
