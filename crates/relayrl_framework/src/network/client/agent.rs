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

use crate::network::HyperparameterArgs;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
#[cfg(feature = "zmq-transport")]
pub use crate::network::client::builder::ZmqTrainingAddressesArgs;
pub use crate::network::client::builder::{
    ActorInferenceMode, ActorParams, ActorTrainingDataMode, AgentBuilder, AgentStartParameters,
    AlgorithmInitArgs, ClientModes, DefaultHyperparameterArgs, LocalTrajectoryFileParams,
    LocalTrajectoryFileType, ModelMode, ReplayBufferSize, SaveModelPath,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub use crate::network::client::builder::{InferenceAddressesArgs, TrainingAddressesArgs};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub use crate::network::client::builder::{InferenceParams, TrainingParams};
pub(crate) use crate::network::client::builder::{uses_in_memory_data, uses_local_file_writing};
use crate::network::client::runtime::coordination::coordinator::{
    ClientActors, ClientCoordinator, ClientEnvironments, ClientInterface, CoordinatorError,
    ToAnyBurnTensor,
};
use crate::network::client::runtime::coordination::state_manager::{ActorUuid, StateManagerError};
use crate::prelude::config::ClientConfigLoader;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::NetworkParams;

use active_uuid_registry::UuidPoolError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use active_uuid_registry::interface::get_context_entries;
use active_uuid_registry::interface::list_ids;
use relayrl_algorithms::prelude::nn::NeuralNetwork;
use relayrl_algorithms::prelude::ppo::algorithm::{IPPOParams, MAPPOParams, PPOParams};
use relayrl_algorithms::prelude::ppo::trainer::PPOTrainerSpec;
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

use async_trait::async_trait;
use burn_tensor::{BasicOps, Bool, Float, Int, Numeric, Tensor, TensorKind, backend::Backend};
use dashmap::{DashMap, DashSet};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "metrics", feature = "logging"))]
use std::collections::HashMap;
use std::future::Future;
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
    #[error("Run env is already active for actor {0}")]
    RunEnvActive(String),
}

/// Client entry point for the RelayRL framework.
///
/// `RelayRLAgent` is a thin facade over the runtime coordinator, providing a stable public API
/// for starting, scaling, and interacting with runtime actors.
///
/// In `0.5.0-beta`, the supported path is the local/default client runtime.
/// Transport-backed and server-backed flows remain experimental.
pub struct RelayRLAgent<B: Backend + BackendMatcher<Backend = B>> {
    coordinator: ClientCoordinator<B>,
    supported_backend: SupportedTensorBackend,
    run_env_active_flags: DashSet<Uuid>,
}

impl<B: Backend + BackendMatcher<Backend = B>> std::fmt::Debug for RelayRLAgent<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RLAgent")
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> RelayRLAgent<B> {
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
            coordinator: ClientCoordinator::<B>::new(
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                transport_type,
                client_modes,
            ),
            supported_backend: B::get_supported_backend(),
            run_env_active_flags: DashSet::new(),
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
        let AgentStartParameters {
            router_scale,
            default_model,
            config_path,
            router_buffer_size_per_actor,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters,
        } = params;

        self.coordinator
            .start(
                router_scale,
                default_model,
                config_path,
                router_buffer_size_per_actor,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                default_hyperparameters,
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
        let AgentStartParameters {
            router_scale,
            default_model,
            config_path,
            router_buffer_size_per_actor,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters,
        } = params;

        self.coordinator
            .restart(
                router_scale,
                default_model,
                config_path,
                router_buffer_size_per_actor,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                default_hyperparameters,
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
    pub async fn request_action<
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B> + 'static,
        KindOut: TensorKind<B> + 'static,
    >(
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
                let result = self
                    .coordinator
                    .request_action(ids, observation, mask, reward)
                    .await?;
                Ok(result)
            }
            false => Err(ClientError::BackendMismatchError(
                "Backend mismatch; Some tensor backends are not (currently) supported by RelayRL"
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
            log::warn!("Updating model locally is not supported in Online training data modes");
            return Err(ClientError::ModelUpdateNotSupported(
                "Updating model locally is not supported in Online training data modes".to_string(),
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
        actor_ids: Vec<ActorUuid>,
    ) -> Result<Vec<(ActorUuid, i64)>, ClientError> {
        Ok(self.coordinator.get_model_version(actor_ids).await?)
    }

    pub async fn get_trajectory_memory(
        &self,
    ) -> Result<Arc<DashMap<ActorUuid, Vec<Arc<RelayRLTrajectory>>>>, ClientError> {
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
#[async_trait]
pub trait RelayRLAgentActors<B: Backend + BackendMatcher<Backend = B>> {
    async fn new_actor<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        device: DeviceType,
        max_traj_length: usize,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<(), ClientError>;
    async fn new_actors<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        count: u32,
        device: DeviceType,
        max_traj_length: usize,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<(), ClientError>;
    async fn remove_actor(&mut self, id: Uuid) -> Result<(), ClientError>;
    async fn remove_actors(&mut self, ids: Vec<Uuid>) -> Result<(), ClientError>;
    fn get_actor_ids(&mut self) -> Result<Vec<ActorUuid>, ClientError>;
    async fn set_actor_id(&mut self, current_id: Uuid, new_id: Uuid) -> Result<(), ClientError>;
}

#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> RelayRLAgentActors<B> for RelayRLAgent<B> {
    /// Creates a new actor instance on the specified device with the specified model
    async fn new_actor<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        device: DeviceType,
        max_traj_length: usize,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<(), ClientError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let _ = self
            .coordinator
            .new_actor::<D_IN, D_OUT>(
                device,
                max_traj_length,
                default_model,
                algorithm_args.unwrap_or_default(),
                true,
                true,
            )
            .await?;
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        let _ = self
            .coordinator
            .new_actor::<D_IN, D_OUT>(device, max_traj_length, default_model)
            .await?;
        Ok(())
    }

    /// Creates `n` new actor instances on the specified device with the specified model
    async fn new_actors<const D_IN: usize, const D_OUT: usize>(
        &mut self,
        count: u32,
        device: DeviceType,
        max_traj_length: usize,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] algorithm_args: Option<
            AlgorithmInitArgs,
        >,
    ) -> Result<(), ClientError> {
        if count == 0 {
            Err(ClientError::NoopActorCount(
                "Noop actor count: `count` set to zero".to_string(),
            ))
        } else if count == 1 {
            self.new_actor::<D_IN, D_OUT>(
                device,
                max_traj_length,
                default_model,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                algorithm_args,
            )
            .await
        } else {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let mut actor_ids: Vec<Uuid> = Vec::new();
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let algorithm_args = algorithm_args.unwrap_or_default();
            for _ in 0..count {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                actor_ids.push(
                    self.coordinator
                        .new_actor::<D_IN, D_OUT>(
                            device.clone(),
                            max_traj_length,
                            default_model.clone(),
                            algorithm_args.clone(),
                            false,
                            false,
                        )
                        .await?,
                );
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                self.coordinator
                    .new_actor::<D_IN, D_OUT>(
                        device.clone(),
                        max_traj_length,
                        default_model.clone(),
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

                if let ActorTrainingDataMode::Online(_)
                | ActorTrainingDataMode::OnlineWithFiles(_, _)
                | ActorTrainingDataMode::OnlineWithMemory(_) =
                    &self.coordinator.client_modes.actor_training_data_mode
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

            Ok(())
        }
    }

    /// Removes the actor instance with the specified ID from the current Agent instance
    async fn remove_actor(&mut self, actor_id: ActorUuid) -> Result<(), ClientError> {
        self.coordinator
            .remove_actor(
                actor_id,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                true,
            )
            .await?;
        Ok(())
    }

    async fn remove_actors(&mut self, actor_ids: Vec<ActorUuid>) -> Result<(), ClientError> {
        if actor_ids.is_empty() {
            Err(ClientError::NoopActorCount(
                "Noop actor count: `actor_ids` is empty in `remove_actors()`".to_string(),
            ))
        } else if actor_ids.len() == 1 {
            self.remove_actor(actor_ids[0]).await
        } else {
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
    async fn set_actor_id(
        &mut self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), ClientError> {
        self.coordinator.set_actor_id(current_id, new_id).await?;
        Ok(())
    }
}

#[allow(async_fn_in_trait)]
pub trait RelayRLActorEnv<B: Backend + BackendMatcher<Backend = B>> {
    async fn run_env_eval(&self, actor_id: ActorUuid, loop_iters: usize)
    -> Result<(), ClientError>;
    async fn run_env_with_ppo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + Default + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + Default + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Clone + Send + Default + 'static,
    >(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), ClientError>;
    async fn run_env_with_ippo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + Default + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + Default + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + Default + 'static,
    >(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), ClientError>;
    async fn run_env_with_mappo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + Default + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + Default + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + Default + 'static,
    >(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), ClientError>;
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

impl<B: Backend + BackendMatcher<Backend = B>> RelayRLActorEnv<B> for RelayRLAgent<B> {
    async fn run_env_eval(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
    ) -> Result<(), ClientError> {
        if !self.run_env_active_flags.insert(actor_id) {
            return Err(ClientError::RunEnvActive(format!(
                "run_env is already active for actor {}",
                actor_id
            )));
        }
        let result = self
            .coordinator
            .run_env_eval(actor_id, loop_iters)
            .await
            .map_err(ClientError::from);
        self.run_env_active_flags.remove(&actor_id);
        result
    }

    async fn run_env_with_ppo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + Default + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + Default + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Clone + Send + Default + 'static,
    >(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), ClientError> {
        if !self.run_env_active_flags.insert(actor_id) {
            return Err(ClientError::RunEnvActive(format!(
                "run_env is already active for actor {}",
                actor_id
            )));
        }
        let result = self
            .coordinator
            .run_env_with_ppo::<KindIn, KindOut, Pi>(
                actor_id,
                loop_iters,
                max_traj_length,
                trainer_spec,
            )
            .await
            .map_err(ClientError::from);
        self.run_env_active_flags.remove(&actor_id);
        result
    }

    async fn run_env_with_ippo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + Default + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + Default + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + Default + 'static,
    >(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), ClientError> {
        if !self.run_env_active_flags.insert(actor_id) {
            return Err(ClientError::RunEnvActive(format!(
                "run_env is already active for actor {}",
                actor_id
            )));
        }
        let result = self
            .coordinator
            .run_env_with_ippo::<KindIn, KindOut, Pi>(
                actor_id,
                loop_iters,
                max_traj_length,
                trainer_spec,
            )
            .await
            .map_err(ClientError::from);
        self.run_env_active_flags.remove(&actor_id);
        result
    }

    async fn run_env_with_mappo<
        KindIn: TensorKind<B> + BasicOps<B> + Send + Default + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Numeric<B> + Send + Default + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + Default + 'static,
    >(
        &self,
        actor_id: ActorUuid,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), ClientError> {
        if !self.run_env_active_flags.insert(actor_id) {
            return Err(ClientError::RunEnvActive(format!(
                "run_env is already active for actor {}",
                actor_id
            )));
        }
        let result = self
            .coordinator
            .run_env_with_mappo::<KindIn, KindOut, Pi>(
                actor_id,
                loop_iters,
                max_traj_length,
                trainer_spec,
            )
            .await
            .map_err(ClientError::from);
        self.run_env_active_flags.remove(&actor_id);
        result
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
    fn router_scale_setter_sets_field() {
        let b = AgentBuilder::<TestBackend>::builder().router_scale(2);
        assert_eq!(b.router_scale, Some(2));
    }

    #[test]
    fn actor_count_does_not_change_router_scale() {
        let b = AgentBuilder::<TestBackend>::builder();
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

        let (_agent, params) = AgentBuilder::<TestBackend>::builder()
            .default_model(default_model.clone())
            .config_path(config_path.clone())
            .build()
            .await
            .expect("builder should succeed with a local default model");

        assert_eq!(params.router_scale, 1);
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
    async fn scale_throughput_zero_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            ClientModes::default(),
        );
        let result = agent.scale_throughput(0).await;
        assert!(matches!(result, Err(ClientError::NoopRouterScale(_))));
    }

    #[tokio::test]
    async fn new_actors_zero_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend>::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            ClientModes::default(),
        );
        let result = agent
            .new_actors::<4, 1>(
                0,
                DeviceType::Cpu,
                0usize,
                None,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                None,
            )
            .await;
        assert!(matches!(result, Err(ClientError::NoopActorCount(_))));
    }

    #[tokio::test]
    async fn remove_actors_empty_vec_returns_noop_error() {
        let mut agent = RelayRLAgent::<TestBackend>::new(
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
