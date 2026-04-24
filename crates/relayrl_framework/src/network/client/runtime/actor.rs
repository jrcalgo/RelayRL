//! Runtime actor implementation.
//!
//! Actors own local inference state, trajectory assembly, and the message-handling loop for the
//! client runtime. Transport-backed server inference paths remain experimental in `0.5.0-beta`.

use crate::network::client::agent::{ActorInferenceMode, ClientModes};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::TransportError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    InferenceDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::router::{
    BatchedInferenceRequest, InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::MetricsManager;

use active_uuid_registry::registry_uuid::Uuid;
use arc_swap::ArcSwapOption;
use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{BackendMatcher, DeviceType};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::utils::{deserialize_model_module, validate_module};
use relayrl_types::model::{HotReloadableModel, ModelError, ModelModule};
use relayrl_types::prelude::tensor::relayrl::AnyBurnTensor;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "metrics")]
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;
use tokio::sync::{Mutex, RwLock};

use burn_tensor::backend::Backend;
use thiserror::Error;

/// Shared handle to a hot-reloadable model.
///
/// The outer `Arc<ArcSwapOption<...>>` enables two ownership modes:
/// - **Independent**: each actor holds its own `Arc`, wrapping its own model.
/// - **Shared**: all actors on the same device hold a clone of the *same* `Arc`, so
///   a snapshot swap through any one actor (handshake / model update) is immediately visible
///   to every other actor that shares it.
pub(crate) type LocalModelHandle<B> = Arc<ArcSwapOption<HotReloadableModel<B>>>;

struct ActorTrajectoryState {
    current_traj: RelayRLTrajectory,
    current_episode: u64,
    per_env_trajs: HashMap<Uuid, RelayRLTrajectory>,
    per_env_episodes: HashMap<Uuid, u64>,
    per_env_labels: HashMap<Uuid, String>,
}

impl ActorTrajectoryState {
    fn new(max_traj_length: usize) -> Self {
        Self {
            current_traj: RelayRLTrajectory::new(max_traj_length),
            current_episode: 0,
            per_env_trajs: HashMap::new(),
            per_env_episodes: HashMap::new(),
            per_env_labels: HashMap::new(),
        }
    }

    fn ensure_env_trajectory(
        &mut self,
        actor_id: ActorUuid,
        env_id: Uuid,
        env_label: impl Into<String>,
        max_traj_length: usize,
    ) -> &mut RelayRLTrajectory {
        let label = env_label.into();
        self.per_env_labels
            .entry(env_id)
            .or_insert_with(|| label.clone());
        self.per_env_trajs.entry(env_id).or_insert_with(|| {
            RelayRLTrajectory::with_metadata(
                max_traj_length,
                Some(actor_id),
                Some(env_id),
                Some(label),
                None,
                None,
            )
        })
    }

    fn take_completed_env_trajectory(
        &mut self,
        actor_id: ActorUuid,
        env_id: Uuid,
        env_label: Option<String>,
        reward: f32,
        max_traj_length: usize,
    ) -> RelayRLTrajectory {
        let label = env_label.unwrap_or_else(|| {
            self.per_env_labels
                .get(&env_id)
                .cloned()
                .unwrap_or_else(|| env_id.to_string())
        });
        let traj = self.ensure_env_trajectory(actor_id, env_id, label.clone(), max_traj_length);
        traj.add_action(RelayRLAction::new(
            None,
            None,
            None,
            reward,
            true,
            None,
            Some(actor_id),
        ));

        let mut traj_to_send = self
            .per_env_trajs
            .insert(
                env_id,
                RelayRLTrajectory::with_metadata(
                    max_traj_length,
                    Some(actor_id),
                    Some(env_id),
                    Some(label),
                    None,
                    None,
                ),
            )
            .expect("env trajectory should exist before flush");
        let episode = self.per_env_episodes.entry(env_id).or_insert(0);
        traj_to_send.set_episode(*episode);
        *episode += 1;
        traj_to_send
    }

    fn take_completed_actor_trajectory(
        &mut self,
        actor_id: ActorUuid,
        reward: f32,
        max_traj_length: usize,
    ) -> RelayRLTrajectory {
        let last_action = RelayRLAction::new(None, None, None, reward, true, None, Some(actor_id));
        self.current_traj.add_action(last_action);

        let mut traj_to_send = std::mem::replace(
            &mut self.current_traj,
            RelayRLTrajectory::new(max_traj_length),
        );
        traj_to_send.set_episode(self.current_episode);
        self.current_episode += 1;
        traj_to_send
    }

    fn take_shutdown_trajectories(
        &mut self,
        actor_id: ActorUuid,
        max_traj_length: usize,
    ) -> Vec<RelayRLTrajectory> {
        let mut trajectories = Vec::new();

        if !self.current_traj.actions.is_empty() {
            trajectories.push(std::mem::replace(
                &mut self.current_traj,
                RelayRLTrajectory::new(max_traj_length),
            ));
        }

        let env_ids_to_flush: Vec<_> = self
            .per_env_trajs
            .iter()
            .filter_map(|(env_id, traj)| (!traj.actions.is_empty()).then_some(*env_id))
            .collect();
        for env_id in env_ids_to_flush {
            let env_label = self.per_env_labels.get(&env_id).cloned();
            trajectories.push(self.take_completed_env_trajectory(
                actor_id,
                env_id,
                env_label,
                0.0,
                max_traj_length,
            ));
        }

        trajectories
    }
}

pub(crate) struct ActorRuntime<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    actor_id: ActorUuid,
    reloadable_model: LocalModelHandle<B>,
    shared_max_traj_length: Arc<RwLock<usize>>,
    shared_tx_to_buffer: Sender<RoutedMessage>,
    trajectories: Mutex<ActorTrajectoryState>,
    #[cfg(feature = "metrics")]
    metrics: MetricsManager,
}

impl<
    B: Backend + BackendMatcher<Backend = B> + Send + Sync + 'static,
    const D_IN: usize,
    const D_OUT: usize,
> ActorRuntime<B, D_IN, D_OUT>
{
    pub(crate) async fn new(
        actor_id: ActorUuid,
        reloadable_model: LocalModelHandle<B>,
        shared_max_traj_length: Arc<RwLock<usize>>,
        shared_tx_to_buffer: Sender<RoutedMessage>,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
    ) -> Self {
        let max_traj_length = *shared_max_traj_length.read().await;
        Self {
            actor_id,
            reloadable_model,
            shared_max_traj_length,
            shared_tx_to_buffer,
            trajectories: Mutex::new(ActorTrajectoryState::new(max_traj_length)),
            #[cfg(feature = "metrics")]
            metrics,
        }
    }

    fn build_send_trajectory_message(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<RoutedMessage, ActorError> {
        let now = SystemTime::now();
        let duration = now
            .duration_since(UNIX_EPOCH)
            .map_err(|e| ActorError::SystemError(format!("Clock skew: {e}")))?;
        Ok(RoutedMessage {
            actor_id: self.actor_id,
            protocol: RoutingProtocol::SendTrajectory,
            payload: RoutedPayload::SendTrajectory {
                timestamp: (duration.as_millis(), duration.as_nanos()),
                trajectory,
            },
        })
    }

    async fn send_trajectory(&self, trajectory: RelayRLTrajectory) -> Result<(), ActorError> {
        if self.shared_tx_to_buffer.is_closed() {
            // indicates that the buffer is disabled via client mode OR is shutdown
            return Ok(());
        }
        let send_traj_msg = self.build_send_trajectory_message(trajectory)?;
        self.shared_tx_to_buffer
            .send(send_traj_msg)
            .await
            .map_err(|e| ActorError::TrajectorySendError(format!("{e:?}")))
    }

    pub(crate) async fn perform_local_inference(
        &self,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
        reward: f32,
    ) -> Result<RelayRLAction, ActorError> {
        #[cfg(feature = "metrics")]
        let start_time = Instant::now();

        let result = async {
            let action = {
                let guard = self.reloadable_model.load();
                let reloadable_model = match &*guard {
                    Some(m) => m,
                    None => {
                        return Err(ActorError::SystemError(
                            "Model not loaded/available for actor inference".to_string(),
                        ));
                    }
                };
                reloadable_model
                    .forward::<D_IN, D_OUT>(observation, mask, reward, self.actor_id)
                    .map_err(ActorError::from)?
            };

            let mut trajectories = self.trajectories.lock().await;
            trajectories.current_traj.add_action(action.clone());
            Ok(action)
        }
        .await;

        #[cfg(feature = "metrics")]
        match &result {
            Ok(_) => {
                let duration = start_time.elapsed().as_secs_f64();
                self.metrics
                    .record_histogram("actor_local_inference_latency", duration, &[])
                    .await;
                self.metrics
                    .record_counter("actor_local_inferences", 1, &[])
                    .await;
            }
            Err(_) => {
                self.metrics
                    .record_counter("actor_local_inference_failures", 1, &[])
                    .await;
            }
        }

        result
    }

    pub(crate) async fn perform_local_inference_batch(
        &self,
        env_ids: Vec<Uuid>,
        env_labels: Vec<String>,
        observations: Vec<Arc<AnyBurnTensor<B, D_IN>>>,
        masks: Vec<Option<Arc<AnyBurnTensor<B, D_OUT>>>>,
        rewards: Vec<f32>,
    ) -> Result<Vec<RelayRLAction>, ActorError> {
        #[cfg(feature = "metrics")]
        let start_time = Instant::now();

        let result = async {
            if env_ids.len() != observations.len()
                || env_labels.len() != observations.len()
                || rewards.len() != observations.len()
                || masks.len() != observations.len()
            {
                return Err(ActorError::MessageHandlingError(
                    "Batched inference payload lengths must match".to_string(),
                ));
            }

            let actions = {
                let guard = self.reloadable_model.load();
                let reloadable_model = match &*guard {
                    Some(m) => m,
                    None => {
                        return Err(ActorError::SystemError(
                            "Model not loaded/available for actor inference".to_string(),
                        ));
                    }
                };
                reloadable_model
                    .forward_batch::<D_IN, D_OUT>(&observations, &masks, &rewards, self.actor_id)
                    .map_err(ActorError::from)?
            };

            if actions.len() != env_ids.len() {
                return Err(ActorError::MessageHandlingError(format!(
                    "Batched inference returned {} actions for {} env ids",
                    actions.len(),
                    env_ids.len()
                )));
            }

            let max_traj_length = *self.shared_max_traj_length.read().await;
            let mut trajectories = self.trajectories.lock().await;
            for ((env_id, env_label), action) in env_ids
                .iter()
                .copied()
                .zip(env_labels.iter())
                .zip(actions.iter().cloned())
            {
                let traj = trajectories.ensure_env_trajectory(
                    self.actor_id,
                    env_id,
                    env_label.clone(),
                    max_traj_length,
                );
                traj.add_action(action);
            }

            Ok(actions)
        }
        .await;

        #[cfg(feature = "metrics")]
        match &result {
            Ok(_) => {
                let duration = start_time.elapsed().as_secs_f64();
                self.metrics
                    .record_histogram("actor_local_inference_latency", duration, &[])
                    .await;
                self.metrics
                    .record_counter("actor_local_inferences", 1, &[])
                    .await;
            }
            Err(_) => {
                self.metrics
                    .record_counter("actor_local_inference_failures", 1, &[])
                    .await;
            }
        }

        result
    }

    pub(crate) async fn flag_last_action(
        &self,
        reward: f32,
        env_id: Option<Uuid>,
        env_label: Option<String>,
    ) -> Result<(), ActorError> {
        #[cfg(feature = "metrics")]
        let start_time = Instant::now();

        let result = async {
            let max_traj_length = *self.shared_max_traj_length.read().await;
            let maybe_trajectory = {
                let mut trajectories = self.trajectories.lock().await;
                Some(match env_id {
                    Some(env_id) => trajectories.take_completed_env_trajectory(
                        self.actor_id,
                        env_id,
                        env_label,
                        reward,
                        max_traj_length,
                    ),
                    None => trajectories.take_completed_actor_trajectory(
                        self.actor_id,
                        reward,
                        max_traj_length,
                    ),
                })
            };

            if let Some(trajectory) = maybe_trajectory {
                self.send_trajectory(trajectory).await?;
            }

            Ok(())
        }
        .await;

        #[cfg(feature = "metrics")]
        match &result {
            Ok(()) => {
                let duration = start_time.elapsed().as_secs_f64();
                self.metrics
                    .record_histogram("actor_flag_last_action_latency", duration, &[])
                    .await;
                self.metrics
                    .record_counter("actor_flag_last_actions", 1, &[])
                    .await;
            }
            Err(_) => {
                self.metrics
                    .record_counter("actor_flag_last_action_failures", 1, &[])
                    .await;
            }
        }

        result
    }

    pub(crate) async fn flush_shutdown_trajectories(&self) -> Result<(), ActorError> {
        let max_traj_length = *self.shared_max_traj_length.read().await;
        let trajectories = {
            let mut state = self.trajectories.lock().await;
            state.take_shutdown_trajectories(self.actor_id, max_traj_length)
        };

        for trajectory in trajectories {
            self.send_trajectory(trajectory).await?;
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum ActorError {
    #[error(transparent)]
    ModelError(#[from] ModelError),
    #[error("Trajectory send failed: {0}")]
    TrajectorySendError(String),
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error("Inference request failed: {0}")]
    InferenceRequestError(String),
    #[error("Message handling failed: {0}")]
    MessageHandlingError(String),
    #[error("Type conversion failed: {0}")]
    TypeConversionError(String),
    #[error("System error: {0}")]
    SystemError(String),
    #[error(transparent)]
    UuidPoolError(#[from] active_uuid_registry::UuidPoolError),
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error(transparent)]
    TransportError(#[from] TransportError),
}

pub trait ActorEntity<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>: Send + Sync + 'static
{
    #[allow(clippy::too_many_arguments)]
    async fn new(
        client_namespace: Arc<str>,
        actor_id: ActorUuid,
        device: DeviceType,
        runtime: Arc<ActorRuntime<B, D_IN, D_OUT>>,
        shared_local_model_path: Arc<RwLock<PathBuf>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
    ) -> Self
    where
        Self: Sized;
    async fn spawn_loop(&mut self) -> Result<(), ActorError>;
    async fn initial_model_handshake(&mut self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn get_model_version(&self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn refresh_model(&self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn handle_shutdown(&mut self, _msg: RoutedMessage) -> Result<(), ActorError>;
}

/// Responsible for performing inference with an in-memory model
pub(crate) struct Actor<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    #[allow(dead_code)]
    client_namespace: Arc<str>,
    actor_id: ActorUuid,
    runtime: Arc<ActorRuntime<B, D_IN, D_OUT>>,
    shared_local_model_path: Arc<RwLock<PathBuf>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
    model_device: DeviceType,
    rx_from_router: Receiver<RoutedMessage>,
    shared_client_modes: Arc<ClientModes>,
    #[cfg(feature = "metrics")]
    metrics: MetricsManager,
}

impl<
    B: Backend + BackendMatcher<Backend = B> + Send + Sync + 'static,
    const D_IN: usize,
    const D_OUT: usize,
> Actor<B, D_IN, D_OUT>
{
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn extract_inference_request(
        msg: RoutedMessage,
    ) -> Result<
        (
            Arc<AnyBurnTensor<B, D_IN>>,
            Option<Arc<AnyBurnTensor<B, D_OUT>>>,
            f32,
            oneshot::Sender<Arc<RelayRLAction>>,
        ),
        ActorError,
    > {
        let RoutedPayload::RequestInference(req) = msg.payload else {
            return Err(ActorError::MessageHandlingError(
                "Expected RequestInference payload".to_string(),
            ));
        };

        let InferenceRequest {
            observation,
            mask,
            reward,
            reply_to,
        } = *req;

        let obs: Arc<AnyBurnTensor<B, D_IN>> = *observation
            .downcast::<Arc<AnyBurnTensor<B, D_IN>>>()
            .map_err(|_| {
                ActorError::TypeConversionError("Failed to downcast observation".into())
            })?;

        let mask: Option<Arc<AnyBurnTensor<B, D_OUT>>> = *mask
            .downcast::<Option<Arc<AnyBurnTensor<B, D_OUT>>>>()
            .map_err(|_| ActorError::TypeConversionError("Failed to downcast mask".into()))?;

        Ok((obs, mask, reward, reply_to))
    }

    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn extract_batched_inference_request(
        msg: RoutedMessage,
    ) -> Result<
        (
            Vec<Uuid>,
            Vec<String>,
            Vec<Arc<AnyBurnTensor<B, D_IN>>>,
            Vec<Option<Arc<AnyBurnTensor<B, D_OUT>>>>,
            Vec<f32>,
            oneshot::Sender<Vec<RelayRLAction>>,
        ),
        ActorError,
    > {
        let RoutedPayload::RequestInferenceBatch(req) = msg.payload else {
            return Err(ActorError::MessageHandlingError(
                "Expected RequestInferenceBatch payload".to_string(),
            ));
        };

        let BatchedInferenceRequest {
            env_ids,
            env_labels,
            observations,
            masks,
            rewards,
            reply_to,
        } = *req;

        let observations: Vec<Arc<AnyBurnTensor<B, D_IN>>> = *observations
            .downcast::<Vec<Arc<AnyBurnTensor<B, D_IN>>>>()
            .map_err(|_| {
                ActorError::TypeConversionError("Failed to downcast batched observations".into())
            })?;
        let masks: Vec<Option<Arc<AnyBurnTensor<B, D_OUT>>>> = *masks
            .downcast::<Vec<Option<Arc<AnyBurnTensor<B, D_OUT>>>>>()
            .map_err(|_| {
                ActorError::TypeConversionError("Failed to downcast batched masks".into())
            })?;

        Ok((env_ids, env_labels, observations, masks, rewards, reply_to))
    }

    #[inline(always)]
    async fn handle_inference_kind(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        match self.shared_client_modes.actor_inference_mode {
            ActorInferenceMode::Local(_) => self.perform_local_inference(msg).await,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            ActorInferenceMode::Server(_) => self.request_server_inference(msg).await,
        }
    }

    async fn perform_local_inference(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        let (obs, mask, reward, reply_to) = Self::extract_inference_request(msg)?;
        let action = self
            .runtime
            .perform_local_inference(obs, mask, reward)
            .await?;
        reply_to.send(Arc::new(action)).map_err(|e| {
            ActorError::MessageHandlingError(format!("reply_to send failed: {e:?}"))
        })?;
        Ok(())
    }

    async fn perform_local_inference_batch(
        &mut self,
        msg: RoutedMessage,
    ) -> Result<(), ActorError> {
        let (env_ids, env_labels, observations, masks, rewards, reply_to) =
            Self::extract_batched_inference_request(msg)?;
        let actions = self
            .runtime
            .perform_local_inference_batch(env_ids, env_labels, observations, masks, rewards)
            .await?;
        reply_to.send(actions).map_err(|e| {
            ActorError::MessageHandlingError(format!("reply_to send failed: {e:?}"))
        })?;
        Ok(())
    }

    /// Server inference: serialize observation (and optionally mask) and send to server.
    /// Note: if obs/mask live on GPU, you will pay a device->host copy during serialization.
    ///
    /// This path is still experimental in `0.5.0-beta`.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn request_server_inference(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        // Both the inference_kind and inference_dispatcher initializations are based on
        // the client_capabilities.server_inference flag. Thus, if server_inference is true, the inference_dispatcher will be Some
        // and inference_kind will be InferenceKind::Server. The opposite is true: see request_local_inference for the opposite case.
        // If the inference_dispatcher is None, we will use the local model.
        if let Some(inference_dispatcher) = &self.shared_inference_dispatcher {
            // we assume that the transport_addresses are available if the inference_dispatcher is Some
            let shared_transport_addresses = self
                .shared_transport_addresses
                .as_ref()
                .ok_or_else(|| ActorError::SystemError("Server addresses not available".into()))?
                .clone();

            let (obs, _mask, _reward, reply_to) = Self::extract_inference_request(msg)?;

            let obs_bytes: Vec<u8> = Vec::new();
            let _ = obs; // Experimental server inference serialization is not implemented yet.

            let actor_entry = (
                self.client_namespace.to_string(),
                crate::network::ACTOR_CONTEXT.to_string(),
                self.actor_id,
            );
            let rla = inference_dispatcher
                .send_inference_request(actor_entry, obs_bytes, shared_transport_addresses)
                .await?;

            let mut trajectories = self.runtime.trajectories.lock().await;
            trajectories.current_traj.add_action(rla.clone());
            reply_to.send(Arc::new(rla)).map_err(|e| {
                ActorError::MessageHandlingError(format!("reply_to send failed: {e:?}"))
            })?;
        } else {
            // Fall back to local inference if a server dispatcher is not available.
            return self.perform_local_inference(msg).await;
        }

        Ok(())
    }

    async fn perform_flag_last_action(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::FlagLastInference {
            reward,
            env_id,
            env_label,
        } = msg.payload
        {
            return self
                .runtime
                .flag_last_action(reward, env_id, env_label)
                .await;
        }
        Ok(())
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B> + Send + Sync + 'static,
    const D_IN: usize,
    const D_OUT: usize,
> ActorEntity<B, D_IN, D_OUT> for Actor<B, D_IN, D_OUT>
{
    async fn new(
        client_namespace: Arc<str>,
        actor_id: ActorUuid,
        device: DeviceType,
        runtime: Arc<ActorRuntime<B, D_IN, D_OUT>>,
        shared_local_model_path: Arc<RwLock<PathBuf>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
    ) -> Self
    where
        Self: Sized,
    {
        let model_init_flag = runtime.reloadable_model.load_full().is_none();
        if model_init_flag {
            log::warn!(
                "[ActorEntity] Startup model is None, initial model handshake necessitated..."
            );
        }

        let actor: Actor<B, D_IN, D_OUT> = Self {
            client_namespace,
            actor_id,
            runtime,
            shared_local_model_path,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_inference_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_training_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_transport_addresses,
            model_device: device,
            rx_from_router,
            shared_client_modes,
            #[cfg(feature = "metrics")]
            metrics,
        };

        actor
    }

    async fn spawn_loop(&mut self) -> Result<(), ActorError> {
        while let Some(msg) = self.rx_from_router.recv().await {
            match msg.protocol {
                RoutingProtocol::ModelHandshake => {
                    self.initial_model_handshake(msg).await?;
                }
                RoutingProtocol::RequestInference => {
                    self.handle_inference_kind(msg).await?;
                }
                RoutingProtocol::RequestInferenceBatch => match self
                    .shared_client_modes
                    .actor_inference_mode
                {
                    ActorInferenceMode::Local(_) => self.perform_local_inference_batch(msg).await?,
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    ActorInferenceMode::Server(_) => {
                        return Err(ActorError::SystemError(
                            "Batched server inference is not implemented".to_string(),
                        ));
                    }
                },
                RoutingProtocol::FlagLastInference => {
                    self.perform_flag_last_action(msg).await?;
                }
                RoutingProtocol::ModelVersion => {
                    self.get_model_version(msg).await?;
                }
                RoutingProtocol::ModelUpdate => {
                    self.refresh_model(msg).await?;
                }
                RoutingProtocol::Shutdown => {
                    self.handle_shutdown(msg).await?;
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }

    async fn initial_model_handshake(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelHandshake = msg.payload {
            // Fast path: skip the handshake when a model is already available locally.
            if self.runtime.reloadable_model.load_full().is_some() {
                log::warn!(
                    "[Actor {:?}] Model already available, handshake not needed",
                    self.actor_id
                );
                return Ok(());
            }

            #[cfg(feature = "metrics")]
            let start_time = Instant::now();

            let result: Result<bool, ActorError> = async {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                {
                    if let Some(training_dispatcher) = &self.shared_training_dispatcher {
                        log::info!(
                            "[Actor {:?}] Starting training model handshake",
                            self.actor_id
                        );

                        let shared_transport_addresses = self
                            .shared_transport_addresses
                            .as_ref()
                            .ok_or_else(|| {
                                ActorError::SystemError("Server addresses not available".into())
                            })?
                            .clone();

                        let actor_entry = (
                            self.client_namespace.to_string(),
                            crate::network::ACTOR_CONTEXT.to_string(),
                            self.actor_id,
                        );

                        match training_dispatcher
                            .initial_model_handshake(actor_entry, shared_transport_addresses)
                            .await
                        {
                            Ok(Some(model)) => {
                                log::info!(
                                    "[Actor {:?}] Model handshake successful, received model data",
                                    self.actor_id
                                );

                                if let Err(e) =
                                    model.save(self.shared_local_model_path.read().await.clone())
                                {
                                    log::error!(
                                        "[Actor {:?}] Failed to save model: {:?}",
                                        self.actor_id,
                                        e
                                    );
                                }

                                let model_path = self.shared_local_model_path.clone();
                                let model_device = self.model_device.clone();
                                let actor_id = self.actor_id;

                                match self.runtime.reloadable_model.load_full() {
                                    Some(existing_model) => {
                                        let version = existing_model.version() + 1;
                                        existing_model
                                            .reload_from_path(
                                                model_path.read().await.clone(),
                                                version,
                                            )
                                            .await
                                            .map_err(|e| {
                                                log::error!(
                                                    "[Actor {:?}] Failed to reload model: {:?}",
                                                    actor_id,
                                                    e
                                                );
                                                ActorError::from(e)
                                            })?;
                                    }
                                    None => {
                                        let reloadable_model = Arc::new(
                                            HotReloadableModel::<B>::new_from_module(
                                                model,
                                                model_device,
                                            )
                                            .await
                                            .map_err(ActorError::from)?,
                                        );
                                        self.runtime.reloadable_model.store(Some(reloadable_model));
                                    }
                                }

                                Ok(true)
                            }
                            _ => {
                                log::error!(
                                    "[Actor {:?}] Model handshake failed or no model update needed",
                                    self.actor_id
                                );
                                Ok(false)
                            }
                        }
                    } else {
                        log::error!(
                            "[Actor {:?}] No transport dispatcher configured for model handshake",
                            self.actor_id
                        );
                        Ok(false)
                    }
                }

                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                {
                    log::error!(
                        "[Actor {:?}] No transport dispatcher configured for model handshake",
                        self.actor_id
                    );
                    Ok(false)
                }
            }
            .await;

            #[cfg(feature = "metrics")]
            match &result {
                Ok(true) => {
                    let duration = start_time.elapsed().as_secs_f64();
                    self.metrics
                        .record_histogram("actor_model_handshake_latency", duration, &[])
                        .await;
                    self.metrics
                        .record_counter("actor_model_handshakes", 1, &[])
                        .await;
                }
                Ok(false) | Err(_) => {
                    self.metrics
                        .record_counter("actor_model_handshake_failures", 1, &[])
                        .await;
                }
            }

            return result.map(|_| ());
        }

        Ok(())
    }

    async fn get_model_version(&self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelVersion { reply_to } = msg.payload {
            let version = self
                .runtime
                .reloadable_model
                .load_full()
                .map(|model| model.version())
                .unwrap_or(-1);
            reply_to
                .send(version)
                .map_err(|e| ActorError::MessageHandlingError(format!("{:?}", e)))?;
        }

        Ok(())
    }

    async fn refresh_model(&self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelUpdate {
            model_bytes,
            version,
        } = msg.payload
        {
            #[cfg(feature = "metrics")]
            let start_time = Instant::now();

            let result: Result<bool, ActorError> = async {
                let model: Result<ModelModule<B>, ModelError> =
                    deserialize_model_module::<B>(model_bytes, self.model_device.clone());
                let model_path: PathBuf = self.shared_local_model_path.read().await.clone();

                if let Ok(ok_model) = model {
                    if let Err(e) = validate_module::<B>(&ok_model).map_err(ActorError::from) {
                        log::error!(
                            "[ActorEntity {:?}] Failed to validate model: {:?}",
                            self.actor_id,
                            e
                        );
                        return Err(e);
                    }

                    if let Err(e) = ok_model.save(&model_path).map_err(ActorError::from) {
                        log::error!(
                            "[ActorEntity {:?}] Failed to save model: {:?}",
                            self.actor_id,
                            e
                        );
                        return Err(e);
                    }

                    let model_device = self.model_device.clone();
                    match self.runtime.reloadable_model.load_full() {
                        Some(existing_model) => {
                            existing_model
                                .reload_from_module(ok_model, version)
                                .await
                                .map_err(ActorError::from)?;
                        }
                        None => {
                            let reloadable_model = Arc::new(
                                HotReloadableModel::<B>::new_from_module(ok_model, model_device)
                                    .await
                                    .map_err(ActorError::from)?,
                            );
                            self.runtime.reloadable_model.store(Some(reloadable_model));
                        }
                    }

                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            .await;

            #[cfg(feature = "metrics")]
            match &result {
                Ok(true) => {
                    let duration = start_time.elapsed().as_secs_f64();
                    self.metrics
                        .record_histogram("actor_model_refresh_latency", duration, &[])
                        .await;
                    self.metrics
                        .record_counter("actor_model_refreshes", 1, &[])
                        .await;
                }
                Ok(false) | Err(_) => {
                    self.metrics
                        .record_counter("actor_model_refresh_failures", 1, &[])
                        .await;
                }
            }

            return result.map(|_| ());
        }

        Ok(())
    }

    async fn handle_shutdown(&mut self, _msg: RoutedMessage) -> Result<(), ActorError> {
        let _ = self.runtime.flush_shutdown_trajectories().await;

        active_uuid_registry::interface::remove_id(
            self.client_namespace.as_ref(),
            crate::network::ACTOR_CONTEXT,
            self.actor_id,
        )
        .map_err(ActorError::from)?;

        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    use crate::network::client::agent::{
        ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode,
    };
    use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;

    use active_uuid_registry::registry_uuid::Uuid;
    use relayrl_types::data::tensor::{DType, DeviceType, NdArrayDType};

    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::{RwLock, mpsc, oneshot};

    use burn_ndarray::NdArray;
    use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};
    use relayrl_types::prelude::tensor::relayrl::FloatBurnTensor;

    type NdArrayBackend = NdArray<f32>;

    const D_IN: usize = 4;
    const D_OUT: usize = 1;

    fn disabled_data_mode() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn empty_onnx_model_handle() -> LocalModelHandle<NdArrayBackend> {
        Arc::new(ArcSwapOption::new(None))
    }

    fn float_any_tensor(values: &[f32]) -> Arc<AnyBurnTensor<NdArrayBackend, D_IN>> {
        let device = NdArrayBackend::get_device(&DeviceType::Cpu).unwrap();
        let tensor = Tensor::<NdArrayBackend, D_IN, Float>::from_data(
            BurnTensorData::new(values.to_vec(), [1, 1, 1, values.len()]),
            &device,
        );

        Arc::new(AnyBurnTensor::Float(FloatBurnTensor {
            tensor: Arc::new(tensor),
            dtype: DType::NdArray(NdArrayDType::F32),
        }))
    }

    async fn create_ndarray_actor(
        max_traj_length: usize,
        device: DeviceType,
    ) -> (
        Actor<NdArrayBackend, D_IN, D_OUT>,
        mpsc::Sender<RoutedMessage>,
        mpsc::Receiver<RoutedMessage>,
    ) {
        let actor_id = Uuid::new_v4();
        let (tx_to_actor, rx_from_router) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        let (tx_to_buffer, rx_from_buffer) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        let model_handle = empty_onnx_model_handle();
        let runtime = Arc::new(
            ActorRuntime::new(
                actor_id,
                model_handle,
                Arc::new(RwLock::new(max_traj_length)),
                tx_to_buffer.clone(),
                #[cfg(feature = "metrics")]
                test_metrics(),
            )
            .await,
        );

        let actor = Actor::<NdArrayBackend, D_IN, D_OUT>::new(
            Arc::from("test-actor-namespace"),
            actor_id,
            device,
            runtime,
            Arc::new(RwLock::new(PathBuf::new())),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            rx_from_router,
            disabled_data_mode(),
            #[cfg(feature = "metrics")]
            test_metrics(),
        )
        .await;

        (actor, tx_to_actor, rx_from_buffer)
    }

    fn build_msg(
        actor_id: ActorUuid,
        protocol: RoutingProtocol,
        payload: RoutedPayload,
    ) -> RoutedMessage {
        RoutedMessage {
            actor_id,
            protocol,
            payload,
        }
    }

    #[cfg(feature = "metrics")]
    fn test_metrics() -> MetricsManager {
        MetricsManager::new(
            Arc::new(RwLock::new(("test-actor".to_string(), String::new()))),
            ("test-actor".to_string(), String::new()),
            None,
        )
    }

    #[tokio::test]
    async fn spawn_loop_exits_on_channel_close() {
        let (mut actor, tx, _rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let handle = tokio::spawn(async move { actor.spawn_loop().await });

        drop(tx); // closed channel, loop exits

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("spawn_loop did not exit in time")
            .expect("join error");

        assert!(
            result.is_ok(),
            "spawn_loop did not exit in time on channel close for NdArray, CPU actor"
        );
    }

    #[tokio::test]
    async fn spawn_loop_exits_on_shutdown_message() {
        let (mut actor, tx, _rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;
        active_uuid_registry::interface::add_id(
            "test-actor-namespace",
            crate::network::ACTOR_CONTEXT,
            actor_id,
        )
        .unwrap();
        let handle = tokio::spawn(async move { actor.spawn_loop().await });

        tx.send(build_msg(
            actor_id,
            RoutingProtocol::Shutdown,
            RoutedPayload::Shutdown,
        ))
        .await
        .unwrap();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("spawn_loop did not exit on Shutdown")
            .expect("join error");

        assert!(
            result.is_ok(),
            "spawn_loop did not exit on Shutdown for NdArray, CPU actor"
        );
    }

    #[tokio::test]
    async fn get_model_version_returns_minus_one_when_no_model() {
        let (mut actor, tx, _rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;
        let handle = tokio::spawn(async move { actor.spawn_loop().await });

        let (reply_tx, reply_rx) = oneshot::channel::<i64>();
        tx.send(build_msg(
            actor_id,
            RoutingProtocol::ModelVersion,
            RoutedPayload::ModelVersion { reply_to: reply_tx },
        ))
        .await
        .unwrap();

        let version = tokio::time::timeout(tokio::time::Duration::from_millis(200), reply_rx)
            .await
            .expect("timeout waiting for model version")
            .expect("oneshot cancelled");

        assert_eq!(version, -1, "Unloaded model should report version -1");

        // Shutdown the actor
        tx.send(build_msg(
            actor_id,
            RoutingProtocol::Shutdown,
            RoutedPayload::Shutdown,
        ))
        .await
        .unwrap();
        let _ = handle.await;
    }

    #[tokio::test]
    async fn handle_shutdown_sends_trajectory_when_non_empty() {
        let (mut actor, tx, mut rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;
        let handle = tokio::spawn(async move { actor.spawn_loop().await });

        // Build trajectory via FlagLastInference (adds a terminal action)
        tx.send(build_msg(
            actor_id,
            RoutingProtocol::FlagLastInference,
            RoutedPayload::FlagLastInference {
                reward: 1.0,
                env_id: None,
                env_label: None,
            },
        ))
        .await
        .unwrap();

        // Wait for the FlagLastInference to produce a SendTrajectory
        let traj_msg = tokio::time::timeout(tokio::time::Duration::from_millis(200), rx_buf.recv())
            .await
            .expect("timeout waiting for trajectory after FlagLastInference")
            .expect("buffer rx closed");

        assert!(matches!(
            traj_msg.payload,
            RoutedPayload::SendTrajectory { .. }
        ));

        // Now send Shutdown
        tx.send(build_msg(
            actor_id,
            RoutingProtocol::Shutdown,
            RoutedPayload::Shutdown,
        ))
        .await
        .unwrap();
        let _ = handle.await;
    }

    #[tokio::test]
    async fn handle_shutdown_does_not_send_when_empty_traj() {
        let (mut actor, tx, mut rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;
        let handle = tokio::spawn(async move { actor.spawn_loop().await });

        // Send Shutdown immediately without adding any actions
        tx.send(build_msg(
            actor_id,
            RoutingProtocol::Shutdown,
            RoutedPayload::Shutdown,
        ))
        .await
        .unwrap();
        let _ = handle.await;

        // Buffer should be empty
        assert!(
            rx_buf.try_recv().is_err(),
            "Buffer should be empty after shutdown with no trajectory"
        );
    }

    #[tokio::test]
    async fn flag_last_action_appends_terminal_action_and_sends_traj() {
        let (mut actor, tx, mut rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;
        let _handle = tokio::spawn(async move { actor.spawn_loop().await });

        tx.send(build_msg(
            actor_id,
            RoutingProtocol::FlagLastInference,
            RoutedPayload::FlagLastInference {
                reward: 0.0,
                env_id: None,
                env_label: None,
            },
        ))
        .await
        .unwrap();

        let msg = tokio::time::timeout(tokio::time::Duration::from_millis(300), rx_buf.recv())
            .await
            .expect("timeout waiting for trajectory")
            .expect("buffer rx closed");

        assert!(
            matches!(msg.payload, RoutedPayload::SendTrajectory { .. }),
            "FlagLastInference should produce a SendTrajectory message"
        );
    }

    #[tokio::test]
    async fn extract_batched_inference_request_preserves_env_order() {
        let env_id_1 = Uuid::new_v4();
        let env_id_2 = Uuid::new_v4();
        let (reply_tx, reply_rx) = oneshot::channel();
        let msg = build_msg(
            Uuid::new_v4(),
            RoutingProtocol::RequestInferenceBatch,
            RoutedPayload::RequestInferenceBatch(Box::new(BatchedInferenceRequest {
                env_ids: vec![env_id_1, env_id_2],
                env_labels: vec!["env-1".to_string(), "env-2".to_string()],
                observations: Box::new(vec![
                    float_any_tensor(&[1.0, 2.0, 3.0, 4.0]),
                    float_any_tensor(&[5.0, 6.0, 7.0, 8.0]),
                ]),
                masks: Box::new(vec![None::<Arc<AnyBurnTensor<NdArrayBackend, D_OUT>>>; 2]),
                rewards: vec![1.0, 2.0],
                reply_to: reply_tx,
            })),
        );

        let (env_ids, env_labels, observations, masks, rewards, returned_reply_tx) =
            Actor::<NdArrayBackend, D_IN, D_OUT>::extract_batched_inference_request(msg).unwrap();

        assert_eq!(env_ids, vec![env_id_1, env_id_2]);
        assert_eq!(env_labels, vec!["env-1".to_string(), "env-2".to_string()]);
        assert_eq!(observations.len(), 2);
        assert_eq!(masks.len(), 2);
        assert_eq!(rewards, vec![1.0, 2.0]);
        drop(returned_reply_tx);
        drop(reply_rx);
    }

    #[tokio::test]
    async fn flag_last_action_with_env_id_flushes_only_target_env_traj() {
        let (mut actor, _tx, mut rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let env_id_1 = Uuid::new_v4();
        let env_id_2 = Uuid::new_v4();

        let mut trajectories = actor.runtime.trajectories.lock().await;
        trajectories
            .per_env_labels
            .insert(env_id_1, "env-1".to_string());
        trajectories
            .per_env_labels
            .insert(env_id_2, "env-2".to_string());
        let mut traj_1 = RelayRLTrajectory::with_metadata(
            10,
            Some(actor.actor_id),
            Some(env_id_1),
            Some("env-1".to_string()),
            None,
            None,
        );
        traj_1.add_action(RelayRLAction::minimal(0.5, false));
        let mut traj_2 = RelayRLTrajectory::with_metadata(
            10,
            Some(actor.actor_id),
            Some(env_id_2),
            Some("env-2".to_string()),
            None,
            None,
        );
        traj_2.add_action(RelayRLAction::minimal(0.25, false));
        trajectories.per_env_trajs.insert(env_id_1, traj_1);
        trajectories.per_env_trajs.insert(env_id_2, traj_2);
        drop(trajectories);

        actor
            .perform_flag_last_action(build_msg(
                actor.actor_id,
                RoutingProtocol::FlagLastInference,
                RoutedPayload::FlagLastInference {
                    reward: 1.0,
                    env_id: Some(env_id_1),
                    env_label: Some("env-1".to_string()),
                },
            ))
            .await
            .unwrap();

        let msg = rx_buf.recv().await.expect("expected env trajectory flush");
        match msg.payload {
            RoutedPayload::SendTrajectory { trajectory, .. } => {
                assert_eq!(trajectory.get_env_id(), Some(&env_id_1));
                assert_eq!(trajectory.get_env_label(), Some("env-1"));
            }
            other => panic!(
                "expected SendTrajectory payload, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        let trajectories = actor.runtime.trajectories.lock().await;
        assert!(
            trajectories
                .per_env_trajs
                .get(&env_id_1)
                .is_some_and(|trajectory| trajectory.is_empty())
        );
        assert!(
            trajectories
                .per_env_trajs
                .get(&env_id_2)
                .is_some_and(|trajectory| !trajectory.is_empty())
        );
    }

    #[tokio::test]
    async fn spawn_loop_can_run_concurrently() {
        let mut handles = Vec::new();

        for _ in 0..3 {
            let (mut actor, tx, _rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
            let actor_id = actor.actor_id;
            active_uuid_registry::interface::add_id(
                "test-actor-namespace",
                crate::network::ACTOR_CONTEXT,
                actor_id,
            )
            .unwrap();
            let h = tokio::spawn(async move { actor.spawn_loop().await });
            // Immediately shut each actor down
            tx.send(build_msg(
                actor_id,
                RoutingProtocol::Shutdown,
                RoutedPayload::Shutdown,
            ))
            .await
            .unwrap();
            handles.push(h);
        }

        for h in handles {
            let result = tokio::time::timeout(tokio::time::Duration::from_millis(500), h)
                .await
                .expect("actor did not shut down in time")
                .expect("join error");
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn trajectory_send_failure_returns_err() {
        let (mut actor, tx, rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;

        // Drop buffer receiver so send() fails
        drop(rx_buf);

        let result = actor
            .perform_flag_last_action(build_msg(
                actor_id,
                RoutingProtocol::FlagLastInference,
                RoutedPayload::FlagLastInference {
                    reward: 0.0,
                    env_id: None,
                    env_label: None,
                },
            ))
            .await;

        assert!(
            matches!(result, Err(ActorError::TrajectorySendError(_))),
            "Expected TrajectorySendError, got {:?}",
            result
        );
        drop(tx);
    }

    #[tokio::test]
    async fn model_version_reply_failure_returns_err() {
        let (actor, _tx, _rx_buf) = create_ndarray_actor(10, DeviceType::Cpu).await;
        let actor_id = actor.actor_id;

        let (reply_tx, reply_rx) = oneshot::channel::<i64>();
        // Drop the receiver side before the actor replies
        drop(reply_rx);

        let result = actor
            .get_model_version(build_msg(
                actor_id,
                RoutingProtocol::ModelVersion,
                RoutedPayload::ModelVersion { reply_to: reply_tx },
            ))
            .await;

        assert!(
            matches!(result, Err(ActorError::MessageHandlingError(_))),
            "Expected MessageHandlingError when oneshot rx is dropped, got {:?}",
            result
        );
    }
}
