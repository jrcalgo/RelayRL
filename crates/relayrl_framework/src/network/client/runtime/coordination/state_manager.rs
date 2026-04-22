//! Actor state storage and model-handle coordination.
//!
//! This module tracks actor task handles, inboxes, router assignments, and local model handles for
//! the client runtime.

use crate::network::client::agent::{ActorInferenceMode, ClientModes, ModelMode};
use crate::network::client::runtime::actor::LocalModelHandle;
use crate::network::client::runtime::actor::{Actor, ActorEntity};
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::network::client::runtime::coordination::lifecycle_manager::LifecycleManagerError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
use crate::network::client::runtime::data::environments::EnvironmentInterface;
use crate::network::client::runtime::data::environments::EnvironmentInterfaceError;
use crate::network::client::runtime::data::environments::vec_env::IntoAnyTensorKind;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    InferenceDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::MetricsManager;

use std::path::PathBuf;
use thiserror::Error;

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::{remove_id, replace_id};
use relayrl_env_trait::{Environment, EnvironmentUuid};
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher, DeviceType, TensorData};
use relayrl_types::model::{HotReloadableModel, ModelModule};
use relayrl_types::prelude::tensor::burn::TensorKind;

use active_uuid_registry::registry_uuid::Uuid;

use arc_swap::ArcSwapOption;
use burn_tensor::{BasicOps, backend::Backend};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum StateManagerError {
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error("Failed to create reloadable model: {0}")]
    FailedToCreateReloadableModelError(String),
    #[error("Actor handle not found: {0}")]
    ActorHandleNotFoundError(String),
    #[error("Actor inbox not found: {0}")]
    ActorInboxNotFoundError(String),
    #[error("Actor already taken: {0}")]
    ActorAlreadyTakenError(String),
    #[error("Subscribe shutdown failed: {0}")]
    SubscribeShutdownError(#[from] LifecycleManagerError),
    #[error("Failed to receive shutdown signal: {0}")]
    ReceiveShutdownSignalError(String),
    #[error("Shutdown all actors failed: {0}")]
    ShutdownAllActorsError(String),
    #[error("Set actor ID failed: {0}")]
    SetActorIdError(String),
    #[error("Get actors failed: {0}")]
    GetActorsError(String),
    #[error("New actor failed: {0}")]
    NewActorError(String),
    #[error("Remove actor failed: {0}")]
    RemoveActorError(String),
    #[error("Get config failed: {0}")]
    GetConfigError(String),
    #[error("Set config failed: {0}")]
    SetConfigError(String),
    #[error("Set env failed: {0}")]
    SetEnvError(String),
    #[error("Step env failed: {0}")]
    StepEnvError(String),
    #[error("Get env info failed: {0}")]
    GetEnvInfoError(String),
    #[error("Get env count failed: {0}")]
    GetEnvCountError(String),
    #[error("Increase env count failed: {0}")]
    IncreaseEnvCountError(String),
    #[error("Decrease env count failed: {0}")]
    DecreaseEnvCountError(String),
    #[error("Remove envs failed: {0}")]
    RemoveEnvError(String),
    #[error("Invalid environment kind: {0}")]
    InvalidEnvironmentKindError(String),
    #[error(transparent)]
    EnvironmentInterfaceError(#[from] EnvironmentInterfaceError),
    #[error("Tensor conversion failed: {0}")]
    TensorConversionError(String),
    #[error("Inference request failed: {0}")]
    InferenceRequestError(String),
}

pub type ActorUuid = Uuid;

pub(crate) struct SharedRouterState {
    pub(crate) actor_inboxes: DashMap<ActorUuid, Sender<RoutedMessage>>,
    pub(crate) actor_router_addresses: DashMap<ActorUuid, RouterNamespace>,
}

/// In-memory actor state management and global channel transport
pub(crate) struct StateManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    client_namespace: Arc<str>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    shared_client_modes: Arc<ClientModes>,
    shared_max_traj_length: Arc<RwLock<usize>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
    shared_local_model_path: Arc<RwLock<PathBuf>>,
    default_model: Option<ModelModule<B>>,
    shared_local_models: Vec<(DeviceType, LocalModelHandle<B>)>,
    #[cfg(feature = "metrics")]
    metrics: MetricsManager,
    pub(crate) global_dispatcher_tx: Sender<RoutedMessage>,
    pub(crate) shared_router_state: Arc<SharedRouterState>,
    actor_envs: DashMap<ActorUuid, EnvironmentInterface<B, D_IN, D_OUT>>,
    actor_handles: DashMap<ActorUuid, Arc<JoinHandle<()>>>,
    actor_devices: DashMap<ActorUuid, DeviceType>,
    pub(crate) actor_model_handles: DashMap<ActorUuid, LocalModelHandle<B>>,
    pub(crate) shared_actor_count: Arc<AtomicUsize>,
}

// ===== Construction and actor lifecycle =====

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    StateManager<B, D_IN, D_OUT>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client_namespace: Arc<str>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        shared_client_modes: Arc<ClientModes>,
        shared_max_traj_length: Arc<RwLock<usize>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        shared_local_model_path: Arc<RwLock<PathBuf>>,
        default_model: Option<ModelModule<B>>,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
    ) -> (Self, Receiver<RoutedMessage>) {
        let (global_dispatcher_tx, global_dispatcher_rx) =
            mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT * 2);
        (
            Self {
                client_namespace,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_inference_dispatcher,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_training_dispatcher,
                shared_client_modes,
                shared_max_traj_length,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_transport_addresses,
                shared_local_model_path,
                default_model,
                shared_local_models: Vec::new(),
                #[cfg(feature = "metrics")]
                metrics,
                global_dispatcher_tx,
                shared_router_state: Arc::new(SharedRouterState {
                    actor_inboxes: DashMap::new(),
                    actor_router_addresses: DashMap::new(),
                }),
                actor_envs: DashMap::new(),
                actor_handles: DashMap::new(),
                actor_devices: DashMap::new(),
                actor_model_handles: DashMap::new(),
                shared_actor_count: Arc::new(AtomicUsize::new(0)),
            },
            global_dispatcher_rx,
        )
    }

    /// Helper function to load a reloadable model from various sources
    ///
    /// Priority order:
    /// 1. Provided `default_model` parameter
    /// 2. Cached `self.default_model`
    /// 3. Config `local_model_path`
    /// 4. None
    async fn load_reloadable_model(
        &self,
        model_module: Option<ModelModule<B>>,
        device: DeviceType,
    ) -> Result<Option<HotReloadableModel<B>>, StateManagerError> {
        // Check fn param
        if let Some(model) = model_module {
            return Ok(Some(
                HotReloadableModel::<B>::new_from_module(model, device)
                    .await
                    .map_err(|_| {
                        StateManagerError::FailedToCreateReloadableModelError(
                            "[StateManager] Failed to create reloadable model from parameter"
                                .to_string(),
                        )
                    })?,
            ));
        }

        // Check cached default_model
        if let Some(model) = self.default_model.clone() {
            return Ok(Some(
                HotReloadableModel::<B>::new_from_module(model, device)
                    .await
                    .map_err(|_| {
                        StateManagerError::FailedToCreateReloadableModelError(
                            "[StateManager] Failed to create reloadable model from cache"
                                .to_string(),
                        )
                    })?,
            ));
        }

        // Try local_model_path
        let local_model_path = self.shared_local_model_path.read().await;

        if !local_model_path.to_str().unwrap_or_default().is_empty() {
            return Ok(Some(
                HotReloadableModel::<B>::new_from_path(local_model_path.as_path(), device)
                    .await
                    .map_err(|_| {
                        StateManagerError::FailedToCreateReloadableModelError(
                            "[StateManager] Failed to load model from local_model_path".to_string(),
                        )
                    })?,
            ));
        }

        // No model available
        Ok(None)
    }

    /// Returns a `(LocalModelHandle<B>, needs_handshake)` pair for a new actor on `device`.
    ///
    /// - **Independent** mode: always creates a fresh `Arc<ArcSwapOption<...>>`.
    ///   `needs_handshake` is `true` when no model could be pre-loaded.
    /// - **Shared** mode: looks up the per-device pool.  The *first* actor on a given device
    ///   creates the slot (and triggers a handshake if no model is available); every subsequent
    ///   actor on the same device reuses the existing handle with `needs_handshake = false`,
    ///   ensuring only one network round-trip happens per device.
    async fn get_or_init_model_handle(
        &mut self,
        default_model: Option<ModelModule<B>>,
        device: DeviceType,
    ) -> Result<(LocalModelHandle<B>, bool), StateManagerError> {
        match &self.shared_client_modes.actor_inference_mode {
            ActorInferenceMode::Local(ModelMode::Shared) => {
                // Reuse existing slot for this device (if any).
                if let Some(idx) = self
                    .shared_local_models
                    .iter()
                    .position(|(d, _)| d == &device)
                {
                    let handle = self.shared_local_models[idx].1.clone();
                    // Subsequent actors never trigger an additional handshake.
                    return Ok((handle, false));
                }

                // First actor for this device: create the slot.
                let reloadable = self
                    .load_reloadable_model(default_model, device.clone())
                    .await?;
                let needs_handshake = reloadable.is_none();
                let handle: LocalModelHandle<B> =
                    Arc::new(ArcSwapOption::new(reloadable.map(Arc::new)));
                self.shared_local_models.push((device, handle.clone()));
                Ok((handle, needs_handshake))
            }

            // Independent or Server-inference: every actor gets its own fresh handle.
            _ => {
                let reloadable = self
                    .load_reloadable_model(default_model, device.clone())
                    .await?;
                let needs_handshake = reloadable.is_none();
                let handle: LocalModelHandle<B> =
                    Arc::new(ArcSwapOption::new(reloadable.map(Arc::new)));
                Ok((handle, needs_handshake))
            }
        }
    }

    pub(crate) async fn new_actor(
        &mut self,
        actor_id: ActorUuid,
        router_namespace: RouterNamespace,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        tx_to_buffer: Sender<RoutedMessage>,
    ) -> Result<(), StateManagerError> {
        if self.actor_handles.contains_key(&actor_id) {
            log::warn!(
                "[StateManager] Actor ID {} already exists, replacing existing actor...",
                actor_id
            );
            self.remove_actor(actor_id)?
        }

        // Create actor inbox for receiving messages from the filter
        let (tx_to_actor, actor_inbox_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        self.shared_router_state
            .actor_inboxes
            .insert(actor_id, tx_to_actor.clone());

        let shared_local_model_path = self.shared_local_model_path.clone();
        let shared_max_traj_length = self.shared_max_traj_length.clone();
        let shared_client_modes = self.shared_client_modes.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_inference_dispatcher = self.shared_inference_dispatcher.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_training_dispatcher = self.shared_training_dispatcher.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_transport_addresses = self.shared_transport_addresses.clone();
        #[cfg(feature = "metrics")]
        let metrics = self.metrics.clone();

        let client_namespace = self.client_namespace.clone();

        let (model_handle, model_handshake_flag) = self
            .get_or_init_model_handle(default_model, device.clone())
            .await?;
        self.actor_devices.insert(actor_id, device.clone());
        self.actor_envs.insert(
            actor_id,
            EnvironmentInterface::new(self.client_namespace.clone(), device.clone()),
        );
        self.actor_model_handles
            .insert(actor_id, model_handle.clone());

        let handle: Arc<JoinHandle<()>> = Arc::new(tokio::spawn(async move {
            let mut actor: Actor<B, D_IN, D_OUT> = Actor::<B, D_IN, D_OUT>::new(
                client_namespace,
                actor_id,
                device.clone(),
                model_handle,
                shared_local_model_path,
                shared_max_traj_length,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_inference_dispatcher,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_training_dispatcher,
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                shared_transport_addresses,
                actor_inbox_rx,
                tx_to_buffer,
                shared_client_modes,
                #[cfg(feature = "metrics")]
                metrics,
            )
            .await;

            // Use the flag computed by StateManager so that, in Shared mode, only the first
            // actor per device triggers the network handshake.
            if model_handshake_flag {
                let model_handshake_ms = RoutedMessage {
                    actor_id,
                    protocol: RoutingProtocol::ModelHandshake,
                    payload: RoutedPayload::ModelHandshake,
                };
                let _ = tx_to_actor.send(model_handshake_ms).await;
            }

            if let Err(e) = actor.spawn_loop().await {
                log::error!("[StateManager] Actor {:?} loop error: {}", actor_id, e);
            }
        }));

        self.actor_handles.insert(actor_id, handle);
        self.shared_router_state
            .actor_router_addresses
            .insert(actor_id, router_namespace);

        self.shared_actor_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    #[allow(unused)]
    pub(crate) async fn restart_actor(
        &mut self,
        actor_id: ActorUuid,
        router_namespace: RouterNamespace,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        tx_to_buffer: Sender<RoutedMessage>,
    ) -> Result<(), StateManagerError> {
        self.remove_actor(actor_id)?;
        self.new_actor(
            actor_id,
            router_namespace,
            device,
            default_model,
            tx_to_buffer,
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn shutdown_all_actors(&self) -> Result<(), StateManagerError> {
        // Send Shutdown message to every actor inbox; actors will flush and exit
        for entry in self.shared_router_state.actor_inboxes.iter() {
            let actor_id: ActorUuid = *entry.key();
            let tx: Sender<RoutedMessage> = entry.value().clone();
            let shutdown_msg = RoutedMessage {
                actor_id,
                protocol: RoutingProtocol::Shutdown,
                payload: RoutedPayload::Shutdown,
            };
            let _ = tx.send(shutdown_msg).await;

            let handle: Result<
                dashmap::mapref::one::Ref<'_, Uuid, Arc<JoinHandle<()>>>,
                StateManagerError,
            > = self.actor_handles.get(&actor_id).ok_or(
                StateManagerError::ActorHandleNotFoundError(
                    "[StateManager] Actor handle not found".to_string(),
                ),
            );

            while active_uuid_registry::interface::list_ids(
                self.client_namespace.as_ref(),
                crate::network::ACTOR_CONTEXT,
            )
            .contains(&actor_id)
            {
                tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tokio::time::sleep(std::time::Duration::from_secs(1)),
                )
                .await
                .map_err(|_| {
                    StateManagerError::ShutdownAllActorsError(format!(
                        "[StateManager] Shutdown all actors timeout: {}",
                        actor_id
                    ))
                })?;
            }

            if let Ok(handle) = handle {
                handle.abort();
            } else {
                continue;
            }
        }

        Ok(())
    }

    pub(crate) async fn clear_runtime_components(&mut self) -> Result<(), StateManagerError> {
        self.actor_handles.clear();
        self.shared_router_state.actor_inboxes.clear();
        self.actor_devices.clear();
        self.actor_model_handles.clear();
        self.shared_router_state.actor_router_addresses.clear();
        self.shared_local_models.clear();

        Ok(())
    }

    pub(crate) fn remove_actor(&mut self, id: Uuid) -> Result<(), StateManagerError> {
        if let Some((_, handle)) = self.actor_handles.remove(&id) {
            handle.abort();
        }

        self.shared_router_state.actor_inboxes.remove(&id);
        self.actor_envs.remove(&id);
        self.actor_devices.remove(&id);
        self.actor_model_handles.remove(&id);
        self.shared_router_state.actor_router_addresses.remove(&id);
        self.shared_actor_count.fetch_sub(1, Ordering::Relaxed);
        remove_id(
            self.client_namespace.as_ref(),
            crate::network::ACTOR_CONTEXT,
            id,
        )
        .map_err(StateManagerError::from)?;

        Ok(())
    }

    pub(crate) fn set_actor_id(
        &self,
        current_id: ActorUuid,
        new_id: ActorUuid,
    ) -> Result<(), StateManagerError> {
        let current_id_handle =
            match StateManager::<B, D_IN, D_OUT>::get_actor_handle(self, current_id) {
                Some(handle) => handle.clone(),
                None => {
                    return Err(StateManagerError::ActorHandleNotFoundError(format!(
                        "[StateManager] Actor ID {} not found",
                        current_id
                    )));
                }
            };
        let current_id_inbox =
            match StateManager::<B, D_IN, D_OUT>::get_actor_inbox(self, current_id) {
                Some(inbox) => inbox.clone(),
                None => {
                    return Err(StateManagerError::ActorInboxNotFoundError(format!(
                        "[StateManager] Actor ID {} not found",
                        current_id
                    )));
                }
            };

        if StateManager::<B, D_IN, D_OUT>::get_actor_handle(self, new_id).is_some()
            || StateManager::<B, D_IN, D_OUT>::get_actor_inbox(self, new_id).is_some()
        {
            return Err(StateManagerError::ActorAlreadyTakenError(format!(
                "[StateManager] Actor ID {} already taken",
                new_id
            )));
        }

        self.actor_handles.insert(new_id, current_id_handle);
        self.actor_handles.remove(&current_id);
        self.shared_router_state
            .actor_inboxes
            .insert(new_id, current_id_inbox);
        self.shared_router_state.actor_inboxes.remove(&current_id);
        if let Some((_, current_device)) = self.actor_devices.remove(&current_id) {
            self.actor_devices.insert(new_id, current_device);
        }
        if let Some((_, current_env)) = self.actor_envs.remove(&current_id) {
            self.actor_envs.insert(new_id, current_env);
        }
        if let Some((_, router_namespace)) = self
            .shared_router_state
            .actor_router_addresses
            .remove(&current_id)
        {
            self.shared_router_state
                .actor_router_addresses
                .insert(new_id, router_namespace);
        }

        replace_id(
            self.client_namespace.as_ref(),
            crate::network::ACTOR_CONTEXT,
            current_id,
            new_id,
        )
        .map_err(StateManagerError::from)?;

        Ok(())
    }

    pub(crate) fn distribute_actors(&self, router_namespaces: Vec<RouterNamespace>) {
        if router_namespaces.is_empty() {
            return;
        }

        let actor_ids: Vec<ActorUuid> = StateManager::<B, D_IN, D_OUT>::get_actor_id_list(self);

        for (i, actor_id) in actor_ids.iter().enumerate() {
            let router_namespace = router_namespaces[i % router_namespaces.len()].clone();
            self.shared_router_state
                .actor_router_addresses
                .insert(*actor_id, router_namespace);
        }
    }

    /// Replaces all actor-router mappings with the provided snapshot.
    ///
    /// Takes `&self` for the same reason as `distribute_actors`: `actor_router_addresses`
    /// is a `DashMap` and mutation is safe through a shared reference.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) fn restore_actor_router_mappings(
        &self,
        mappings: Vec<(ActorUuid, RouterNamespace)>,
    ) {
        self.shared_router_state.actor_router_addresses.clear();

        for (actor_id, router_namespace) in mappings {
            self.shared_router_state
                .actor_router_addresses
                .insert(actor_id, router_namespace);
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) fn get_actor_router_mappings(&self) -> Vec<(ActorUuid, RouterNamespace)> {
        self.shared_router_state
            .actor_router_addresses
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    pub(crate) fn get_actor_id_list(&self) -> Vec<ActorUuid> {
        self.actor_handles
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    fn sorted_actor_ids_for_model_updates(
        &self,
        actor_ids: Option<&[ActorUuid]>,
    ) -> Vec<ActorUuid> {
        let mut actor_ids = match actor_ids {
            Some(ids) => ids
                .iter()
                .copied()
                .filter(|actor_id| self.actor_handles.contains_key(actor_id))
                .collect(),
            None => self.get_actor_id_list(),
        };
        actor_ids.sort_by_key(|actor_id| actor_id.to_string());
        actor_ids.dedup();
        actor_ids
    }

    fn canonical_model_update_target_from_sorted_actor_ids(
        &self,
        actor_id: ActorUuid,
        sorted_actor_ids: &[ActorUuid],
    ) -> ActorUuid {
        match &self.shared_client_modes.actor_inference_mode {
            ActorInferenceMode::Local(ModelMode::Shared) => {
                let Some(actor_device) = self
                    .actor_devices
                    .get(&actor_id)
                    .map(|device_entry| device_entry.value().clone())
                else {
                    return actor_id;
                };

                sorted_actor_ids
                    .iter()
                    .copied()
                    .find(|candidate_actor_id| {
                        self.actor_devices
                            .get(candidate_actor_id)
                            .map(|device_entry| device_entry.value() == &actor_device)
                            .unwrap_or(false)
                    })
                    .unwrap_or(actor_id)
            }
            _ => actor_id,
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) fn canonical_model_update_target(&self, actor_id: ActorUuid) -> ActorUuid {
        let sorted_actor_ids = self.sorted_actor_ids_for_model_updates(None);
        self.canonical_model_update_target_from_sorted_actor_ids(actor_id, &sorted_actor_ids)
    }

    #[cfg(test)]
    pub(crate) fn model_update_dispatch_targets(&self) -> Vec<ActorUuid> {
        self.model_update_dispatch_targets_for_subset(None)
    }

    pub(crate) fn model_update_dispatch_targets_for_subset(
        &self,
        actor_ids: Option<&[ActorUuid]>,
    ) -> Vec<ActorUuid> {
        let sorted_actor_ids = self.sorted_actor_ids_for_model_updates(actor_ids);
        let mut dispatch_targets = Vec::new();

        for actor_id in sorted_actor_ids.iter().copied() {
            let canonical_target = self
                .canonical_model_update_target_from_sorted_actor_ids(actor_id, &sorted_actor_ids);
            if dispatch_targets.contains(&canonical_target) {
                continue;
            }

            dispatch_targets.push(canonical_target);
        }

        dispatch_targets
    }

    fn get_actor_handle(&self, id: Uuid) -> Option<Arc<JoinHandle<()>>> {
        self.actor_handles
            .get(&id)
            .map(|handle| Arc::clone(handle.value()))
    }

    fn get_actor_inbox(&self, id: Uuid) -> Option<Sender<RoutedMessage>> {
        self.shared_router_state
            .actor_inboxes
            .get(&id)
            .map(|tx| tx.value().clone())
    }

    #[allow(unexpected_cfgs)]
    fn tensor_data_to_any<const D: usize>(
        tensor_data: &TensorData,
        _device: &DeviceType,
    ) -> Result<AnyBurnTensor<B, D>, StateManagerError> {
        match &tensor_data.dtype {
            #[cfg(feature = "ndarray-backend")]
            relayrl_types::data::tensor::DType::NdArray(dtype) => match dtype {
                relayrl_types::data::tensor::NdArrayDType::F16
                | relayrl_types::data::tensor::NdArrayDType::F32
                | relayrl_types::data::tensor::NdArrayDType::F64 => tensor_data
                    .to_float_tensor::<B, D>(_device)
                    .map(AnyBurnTensor::Float)
                    .map_err(|e| StateManagerError::TensorConversionError(e.to_string())),
                relayrl_types::data::tensor::NdArrayDType::I8
                | relayrl_types::data::tensor::NdArrayDType::I16
                | relayrl_types::data::tensor::NdArrayDType::I32
                | relayrl_types::data::tensor::NdArrayDType::I64 => tensor_data
                    .to_int_tensor::<B, D>(_device)
                    .map(AnyBurnTensor::Int)
                    .map_err(|e| StateManagerError::TensorConversionError(e.to_string())),
                relayrl_types::data::tensor::NdArrayDType::Bool => tensor_data
                    .to_bool_tensor::<B, D>(_device)
                    .map(AnyBurnTensor::Bool)
                    .map_err(|e| StateManagerError::TensorConversionError(e.to_string())),
            },
            #[cfg(feature = "tch-backend")]
            relayrl_types::data::tensor::DType::Tch(dtype) => match dtype {
                relayrl_types::data::tensor::TchDType::F16
                | relayrl_types::data::tensor::TchDType::Bf16
                | relayrl_types::data::tensor::TchDType::F32
                | relayrl_types::data::tensor::TchDType::F64 => tensor_data
                    .to_float_tensor::<B, D>(_device)
                    .map(AnyBurnTensor::Float)
                    .map_err(|e| StateManagerError::TensorConversionError(e.to_string())),
                relayrl_types::data::tensor::TchDType::U8
                | relayrl_types::data::tensor::TchDType::I8
                | relayrl_types::data::tensor::TchDType::I16
                | relayrl_types::data::tensor::TchDType::I32
                | relayrl_types::data::tensor::TchDType::I64 => tensor_data
                    .to_int_tensor::<B, D>(_device)
                    .map(AnyBurnTensor::Int)
                    .map_err(|e| StateManagerError::TensorConversionError(e.to_string())),
                relayrl_types::data::tensor::TchDType::Bool => tensor_data
                    .to_bool_tensor::<B, D>(_device)
                    .map(AnyBurnTensor::Bool)
                    .map_err(|e| StateManagerError::TensorConversionError(e.to_string())),
            },
            _ => Err(StateManagerError::TensorConversionError(
                "[StateManager] Unsupported tensor dtype for action conversion".to_string(),
            )),
        }
    }

    async fn request_actions_direct(
        actor_id: ActorUuid,
        inbox: &Sender<RoutedMessage>,
        batch: Vec<(EnvironmentUuid, String, AnyBurnTensor<B, D_IN>, f32)>,
        device: &DeviceType,
    ) -> Result<Vec<(EnvironmentUuid, String, AnyBurnTensor<B, D_OUT>)>, StateManagerError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let env_ids: Vec<_> = batch.iter().map(|(env_id, _, _, _)| *env_id).collect();
        let env_labels: Vec<_> = batch
            .iter()
            .map(|(_, env_label, _, _)| env_label.clone())
            .collect();
        let observations: Vec<_> = batch
            .iter()
            .map(|(_, _, observation, _)| Arc::new(observation.clone()))
            .collect();
        let rewards: Vec<_> = batch.iter().map(|(_, _, _, reward)| *reward).collect();
        let msg = RoutedMessage {
            actor_id,
            protocol: RoutingProtocol::RequestInferenceBatch,
            payload: RoutedPayload::RequestInferenceBatch(Box::new(
                crate::network::client::runtime::router::BatchedInferenceRequest {
                    env_ids: env_ids.clone(),
                    env_labels: env_labels.clone(),
                    observations: Box::new(observations),
                    masks: Box::new(
                        std::iter::repeat_with(|| None::<Arc<AnyBurnTensor<B, D_OUT>>>)
                            .take(env_ids.len())
                            .collect::<Vec<_>>(),
                    ),
                    rewards,
                    reply_to: reply_tx,
                },
            )),
        };

        inbox
            .send(msg)
            .await
            .map_err(|e| StateManagerError::InferenceRequestError(e.to_string()))?;
        let actions = reply_rx
            .await
            .map_err(|e| StateManagerError::InferenceRequestError(e.to_string()))?;
        if actions.len() != env_ids.len() {
            return Err(StateManagerError::InferenceRequestError(format!(
                "[StateManager] Actor returned {} actions for {} envs",
                actions.len(),
                env_ids.len()
            )));
        }

        env_ids
            .into_iter()
            .zip(env_labels)
            .zip(actions)
            .map(|((env_id, env_label), action)| {
                let action_data = action.get_act().ok_or_else(|| {
                    StateManagerError::InferenceRequestError(
                        "[StateManager] Actor returned action without tensor payload".to_string(),
                    )
                })?;
                let action = Self::tensor_data_to_any::<D_OUT>(action_data, device)?;
                Ok((env_id, env_label, action))
            })
            .collect()
    }

    async fn flag_last_action_direct(
        actor_id: ActorUuid,
        inbox: &Sender<RoutedMessage>,
        reward: f32,
        env_id: Option<EnvironmentUuid>,
        env_label: Option<String>,
    ) -> Result<(), StateManagerError> {
        let msg = RoutedMessage {
            actor_id,
            protocol: RoutingProtocol::FlagLastInference,
            payload: RoutedPayload::FlagLastInference {
                reward,
                env_id,
                env_label,
            },
        };
        inbox
            .send(msg)
            .await
            .map_err(|e| StateManagerError::InferenceRequestError(e.to_string()))
    }

    pub(crate) fn run_env(
        &mut self,
        actor_id: ActorUuid,
        step_count: usize,
    ) -> Result<(), StateManagerError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let inbox = self.get_actor_inbox(actor_id).ok_or_else(|| {
                    StateManagerError::ActorInboxNotFoundError(format!(
                        "[StateManager] Actor inbox not found for {}",
                        actor_id
                    ))
                })?;
                let device = self
                    .actor_devices
                    .get(&actor_id)
                    .map(|device| device.clone())
                    .ok_or_else(|| {
                        StateManagerError::GetEnvInfoError(format!(
                            "[StateManager] Actor device not found for {}",
                            actor_id
                        ))
                    })?;

                let mut rewards: HashMap<EnvironmentUuid, f32> = HashMap::new();
                let mut env_labels: HashMap<EnvironmentUuid, String> = HashMap::new();
                let mut env_interface = self.actor_envs.get_mut(&actor_id).ok_or_else(|| {
                    StateManagerError::GetEnvInfoError(format!(
                        "[StateManager] Environment interface not found for {}",
                        actor_id
                    ))
                })?;

                for (env_id, _) in env_interface.ensure_ready()? {
                    rewards.entry(env_id).or_insert(0.0);
                }
                let mut known_env_ids: Vec<_> = rewards.keys().copied().collect();
                known_env_ids.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
                for (index, env_id) in known_env_ids.into_iter().enumerate() {
                    env_labels.insert(env_id, format!("env-{}", index + 1));
                }

                for _ in 0..step_count {
                    let mut observations = env_interface.current_observations()?;
                    observations.sort_unstable_by(|(left, _), (right, _)| {
                        left.as_bytes().cmp(right.as_bytes())
                    });
                    let batch: Vec<_> = observations
                        .into_iter()
                        .enumerate()
                        .map(|(index, (env_id, observation))| {
                            let reward = *rewards.get(&env_id).unwrap_or(&0.0);
                            let env_label = env_labels
                                .entry(env_id)
                                .or_insert_with(|| format!("env-{}", index + 1))
                                .clone();
                            (env_id, env_label, observation, reward)
                        })
                        .collect();
                    let batched_actions =
                        Self::request_actions_direct(actor_id, &inbox, batch, &device).await?;
                    let actions: Vec<_> = batched_actions
                        .into_iter()
                        .map(|(env_id, _, action)| (env_id, action))
                        .collect();

                    let steps = env_interface.step_once(&actions)?;
                    for step in steps {
                        rewards.insert(step.env_id, step.reward);
                        if step.terminated || step.truncated {
                            let env_label = env_labels
                                .get(&step.env_id)
                                .cloned()
                                .unwrap_or_else(|| step.env_id.to_string());
                            Self::flag_last_action_direct(
                                actor_id,
                                &inbox,
                                step.reward,
                                Some(step.env_id),
                                Some(env_label),
                            )
                            .await?;
                            rewards.insert(step.env_id, 0.0);
                        }
                    }
                }

                Ok(())
            })
        })
    }

    pub(crate) fn set_env<KindIn, KindOut>(
        &mut self,
        id: Uuid,
        env: Box<dyn Environment<B, D_IN, D_OUT, KindIn, KindOut>>,
        count: u32,
    ) -> Result<(), StateManagerError>
    where
        KindIn: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Send + Sync + 'static,
    {
        let device = self
            .actor_devices
            .get(&id)
            .map(|device| device.clone())
            .ok_or_else(|| {
                StateManagerError::SetEnvError(format!(
                    "[StateManager] Actor device not found for {}",
                    id
                ))
            })?;

        if let Some(mut env_interface) = self.actor_envs.get_mut(&id) {
            env_interface.set_env(Some(env), count as usize)?;
            Ok(())
        } else {
            let mut env_interface =
                EnvironmentInterface::new(self.client_namespace.clone(), device);
            env_interface.set_env(Some(env), count as usize)?;
            self.actor_envs.insert(id, env_interface);
            Ok(())
        }
    }

    pub(crate) fn get_env_count(&self, actor_id: ActorUuid) -> Result<u32, StateManagerError> {
        self.actor_envs
            .get(&actor_id)
            .ok_or_else(|| {
                StateManagerError::GetEnvCountError(format!(
                    "[StateManager] Environment interface not found for {}",
                    actor_id
                ))
            })?
            .get_env_count()
            .map_err(StateManagerError::from)
    }

    pub(crate) fn increase_env_count(
        &mut self,
        actor_id: ActorUuid,
        count: u32,
    ) -> Result<(), StateManagerError> {
        self.actor_envs
            .get_mut(&actor_id)
            .ok_or_else(|| {
                StateManagerError::IncreaseEnvCountError(format!(
                    "[StateManager] Environment interface not found for {}",
                    actor_id
                ))
            })?
            .increase_env_count(count)
            .map_err(StateManagerError::from)
    }

    pub(crate) fn decrease_env_count(
        &mut self,
        actor_id: ActorUuid,
        count: u32,
    ) -> Result<(), StateManagerError> {
        self.actor_envs
            .get_mut(&actor_id)
            .ok_or_else(|| {
                StateManagerError::DecreaseEnvCountError(format!(
                    "[StateManager] Environment interface not found for {}",
                    actor_id
                ))
            })?
            .decrease_env_count(count)
            .map_err(StateManagerError::from)
    }

    pub(crate) fn remove_env(&mut self, actor_id: ActorUuid) -> Result<(), StateManagerError> {
        self.actor_envs
            .get_mut(&actor_id)
            .ok_or_else(|| {
                StateManagerError::RemoveEnvError(format!(
                    "[StateManager] Environment interface not found for {}",
                    actor_id
                ))
            })?
            .remove_env()
            .map_err(StateManagerError::from)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::network::client::agent::{
        ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode,
    };
    use active_uuid_registry::interface::{reserve_id_with, reserve_namespace};
    use active_uuid_registry::registry_uuid::Uuid;
    use burn_ndarray::NdArray;
    use relayrl_types::data::tensor::DeviceType;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    type TestBackend = NdArray<f32>;
    const D_IN: usize = 4;
    const D_OUT: usize = 1;

    fn disabled_modes() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn shared_modes() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Shared),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn make_state_manager(
        modes: Arc<ClientModes>,
    ) -> (
        StateManager<TestBackend, D_IN, D_OUT>,
        tokio::sync::mpsc::Receiver<RoutedMessage>,
    ) {
        let namespace: Arc<str> = Arc::from(format!("test-sm-{}", Uuid::new_v4()));
        StateManager::<TestBackend, D_IN, D_OUT>::new(
            namespace,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            modes,
            Arc::new(RwLock::new(100)),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            Arc::new(RwLock::new(PathBuf::new())),
            None,
            #[cfg(feature = "metrics")]
            test_metrics(),
        )
    }

    #[cfg(feature = "metrics")]
    fn test_metrics() -> MetricsManager {
        MetricsManager::new(
            Arc::new(RwLock::new((
                "test-state-manager".to_string(),
                String::new(),
            ))),
            ("test-state-manager".to_string(), String::new()),
            None,
        )
    }

    fn deterministic_actor_id(last_byte: u8) -> Uuid {
        let mut bytes = [0_u8; 16];
        bytes[15] = last_byte;
        Uuid::from_bytes(bytes)
    }

    #[tokio::test]
    async fn distribute_actors_round_robin_2_routers() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let actor_ids: Vec<Uuid> = (0..4).map(|_| Uuid::new_v4()).collect();
        let ns1: RouterNamespace = Arc::from("r1");
        let ns2: RouterNamespace = Arc::from("r2");

        for id in &actor_ids {
            let handle = Arc::new(tokio::spawn(async {}));
            sm.actor_handles.insert(*id, handle);
        }

        sm.distribute_actors(vec![ns1.clone(), ns2.clone()]);

        assert_eq!(sm.shared_router_state.actor_router_addresses.len(), 4);
        for id in &actor_ids {
            let assigned = sm
                .shared_router_state
                .actor_router_addresses
                .get(id)
                .unwrap();
            assert!(
                *assigned == ns1 || *assigned == ns2,
                "Actor {} assigned to unexpected namespace",
                id
            );
        }
    }

    #[tokio::test]
    async fn distribute_actors_empty_namespaces_is_noop() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let actor_id = Uuid::new_v4();
        let original_ns: RouterNamespace = Arc::from("original");
        sm.actor_handles
            .insert(actor_id, Arc::new(tokio::spawn(async {})));
        sm.shared_router_state
            .actor_router_addresses
            .insert(actor_id, original_ns.clone());

        sm.distribute_actors(vec![]);

        let assigned = sm
            .shared_router_state
            .actor_router_addresses
            .get(&actor_id)
            .unwrap();
        assert_eq!(*assigned, original_ns, "Namespace should not change");
    }

    #[tokio::test]
    async fn distribute_actors_single_namespace() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let actor_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let ns: RouterNamespace = Arc::from("only");

        for id in &actor_ids {
            sm.actor_handles
                .insert(*id, Arc::new(tokio::spawn(async {})));
        }

        sm.distribute_actors(vec![ns.clone()]);

        for id in &actor_ids {
            let assigned = sm
                .shared_router_state
                .actor_router_addresses
                .get(id)
                .unwrap();
            assert_eq!(*assigned, ns);
        }
    }

    #[test]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    fn restore_replaces_all_mappings() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let old_id = Uuid::new_v4();
        let new_id = Uuid::new_v4();
        let old_ns: RouterNamespace = Arc::from("old");
        let new_ns: RouterNamespace = Arc::from("new");

        sm.shared_router_state
            .actor_router_addresses
            .insert(old_id, old_ns);
        sm.restore_actor_router_mappings(vec![(new_id, new_ns.clone())]);

        assert!(
            !sm.shared_router_state
                .actor_router_addresses
                .contains_key(&old_id),
            "Old mapping should be cleared"
        );
        let assigned = sm
            .shared_router_state
            .actor_router_addresses
            .get(&new_id)
            .unwrap();
        assert_eq!(*assigned, new_ns);
    }

    #[test]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    fn restore_with_empty_clears_all() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        sm.shared_router_state
            .actor_router_addresses
            .insert(Uuid::new_v4(), Arc::from("ns"));
        sm.restore_actor_router_mappings(vec![]);
        assert!(sm.shared_router_state.actor_router_addresses.is_empty());
    }

    #[test]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    fn get_returns_all_inserted_mappings() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let ns1: RouterNamespace = Arc::from("r1");
        let ns2: RouterNamespace = Arc::from("r2");
        let ns3: RouterNamespace = Arc::from("r3");

        sm.shared_router_state
            .actor_router_addresses
            .insert(id1, ns1.clone());
        sm.shared_router_state
            .actor_router_addresses
            .insert(id2, ns2.clone());
        sm.shared_router_state
            .actor_router_addresses
            .insert(id3, ns3.clone());

        let result = sm.get_actor_router_mappings();
        assert_eq!(result.len(), 3);
        assert!(result.iter().any(|(id, ns)| *id == id1 && *ns == ns1));
        assert!(result.iter().any(|(id, ns)| *id == id2 && *ns == ns2));
        assert!(result.iter().any(|(id, ns)| *id == id3 && *ns == ns3));
    }

    #[tokio::test]
    async fn get_actor_id_list_reflects_inserted_handles() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

        for id in &ids {
            sm.actor_handles
                .insert(*id, Arc::new(tokio::spawn(async {})));
        }

        let list = sm.get_actor_id_list();
        assert_eq!(list.len(), 3);
        for id in &ids {
            assert!(list.contains(id), "Actor {} not in list", id);
        }
    }

    #[tokio::test]
    async fn remove_actor_clears_device_and_router_metadata() {
        let (mut sm, _rx) = make_state_manager(disabled_modes());
        reserve_namespace(sm.client_namespace.as_ref());
        let actor_id = reserve_id_with(
            sm.client_namespace.as_ref(),
            crate::network::ACTOR_CONTEXT,
            117,
            100,
        )
        .unwrap();
        let (tx_to_actor, _actor_inbox_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

        sm.actor_handles
            .insert(actor_id, Arc::new(tokio::spawn(async {})));
        sm.shared_router_state
            .actor_inboxes
            .insert(actor_id, tx_to_actor);
        sm.actor_devices.insert(actor_id, DeviceType::Cpu);
        sm.shared_router_state
            .actor_router_addresses
            .insert(actor_id, Arc::from("router-a"));

        sm.remove_actor(actor_id).unwrap();

        assert!(sm.actor_handles.get(&actor_id).is_none());
        assert!(
            sm.shared_router_state
                .actor_inboxes
                .get(&actor_id)
                .is_none()
        );
        assert!(sm.actor_devices.get(&actor_id).is_none());
        assert!(
            sm.shared_router_state
                .actor_router_addresses
                .get(&actor_id)
                .is_none()
        );
    }

    #[tokio::test]
    async fn set_actor_id_moves_device_and_router_metadata() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        reserve_namespace(sm.client_namespace.as_ref());
        let current_id = reserve_id_with(
            sm.client_namespace.as_ref(),
            crate::network::ACTOR_CONTEXT,
            117,
            100,
        )
        .unwrap();
        let new_id = Uuid::new_v4();
        let (tx_to_actor, _actor_inbox_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

        sm.actor_handles
            .insert(current_id, Arc::new(tokio::spawn(async {})));
        sm.shared_router_state
            .actor_inboxes
            .insert(current_id, tx_to_actor);
        sm.actor_devices.insert(current_id, DeviceType::Cpu);
        sm.shared_router_state
            .actor_router_addresses
            .insert(current_id, Arc::from("router-a"));

        sm.set_actor_id(current_id, new_id).unwrap();

        assert!(sm.actor_handles.get(&current_id).is_none());
        assert!(
            sm.shared_router_state
                .actor_inboxes
                .get(&current_id)
                .is_none()
        );
        assert!(sm.actor_devices.get(&current_id).is_none());
        assert!(
            sm.shared_router_state
                .actor_router_addresses
                .get(&current_id)
                .is_none()
        );
        assert!(sm.actor_handles.get(&new_id).is_some());
        assert!(sm.shared_router_state.actor_inboxes.get(&new_id).is_some());
        assert!(matches!(
            sm.actor_devices.get(&new_id),
            Some(device) if *device == DeviceType::Cpu
        ));
        assert!(matches!(
            sm.shared_router_state.actor_router_addresses.get(&new_id),
            Some(namespace) if *namespace == Arc::<str>::from("router-a")
        ));
    }

    #[tokio::test]
    async fn model_update_dispatch_targets_returns_all_actor_ids_in_independent_mode() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

        for id in &ids {
            sm.actor_handles
                .insert(*id, Arc::new(tokio::spawn(async {})));
        }

        let mut expected = ids.clone();
        expected.sort_by_key(|actor_id| actor_id.to_string());

        assert_eq!(sm.model_update_dispatch_targets(), expected);
    }

    #[tokio::test]
    async fn model_update_dispatch_targets_deduplicates_shared_mode_by_device() {
        let (sm, _rx) = make_state_manager(shared_modes());
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

        for id in &ids {
            sm.actor_handles
                .insert(*id, Arc::new(tokio::spawn(async {})));
            sm.actor_devices.insert(*id, DeviceType::Cpu);
        }

        let expected_target = ids
            .iter()
            .min_by_key(|actor_id| actor_id.to_string())
            .copied()
            .unwrap();

        assert_eq!(sm.model_update_dispatch_targets(), vec![expected_target]);
    }

    #[tokio::test]
    async fn model_update_dispatch_targets_for_subset_returns_known_actor_ids_in_independent_mode()
    {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let id1 = deterministic_actor_id(1);
        let id2 = deterministic_actor_id(2);
        let id3 = deterministic_actor_id(3);
        let unknown_id = deterministic_actor_id(9);

        for actor_id in [id1, id2, id3] {
            sm.actor_handles
                .insert(actor_id, Arc::new(tokio::spawn(async {})));
        }

        let subset = vec![id3, unknown_id, id1, id3];
        assert_eq!(
            sm.model_update_dispatch_targets_for_subset(Some(&subset)),
            vec![id1, id3]
        );
    }

    #[tokio::test]
    async fn model_update_dispatch_targets_for_subset_ignores_unknown_actor_ids() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let known_id = deterministic_actor_id(1);
        let unknown_id = deterministic_actor_id(2);

        sm.actor_handles
            .insert(known_id, Arc::new(tokio::spawn(async {})));

        let subset = vec![unknown_id];
        assert!(
            sm.model_update_dispatch_targets_for_subset(Some(&subset))
                .is_empty()
        );

        let subset = vec![unknown_id, known_id];
        assert_eq!(
            sm.model_update_dispatch_targets_for_subset(Some(&subset)),
            vec![known_id]
        );
    }

    #[tokio::test]
    #[cfg(feature = "tch-backend")]
    async fn model_update_dispatch_targets_for_subset_deduplicates_selected_shared_devices() {
        let (sm, _rx) = make_state_manager(shared_modes());
        let cpu_small = deterministic_actor_id(1);
        let cpu_large = deterministic_actor_id(2);
        let cuda_small = deterministic_actor_id(3);
        let cuda_large = deterministic_actor_id(4);

        for (actor_id, device) in [
            (cpu_small, DeviceType::Cpu),
            (cpu_large, DeviceType::Cpu),
            (cuda_small, DeviceType::Cuda(0)),
            (cuda_large, DeviceType::Cuda(0)),
        ] {
            sm.actor_handles
                .insert(actor_id, Arc::new(tokio::spawn(async {})));
            sm.actor_devices.insert(actor_id, device);
        }

        let subset = vec![cuda_large, cpu_large, cpu_small];
        assert_eq!(
            sm.model_update_dispatch_targets_for_subset(Some(&subset)),
            vec![cpu_small, cuda_large]
        );
    }

    #[tokio::test]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn canonical_model_update_target_uses_shared_device_representative() {
        let (sm, _rx) = make_state_manager(shared_modes());
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

        for id in &ids {
            sm.actor_handles
                .insert(*id, Arc::new(tokio::spawn(async {})));
            sm.actor_devices.insert(*id, DeviceType::Cpu);
        }

        let expected_target = ids
            .iter()
            .min_by_key(|actor_id| actor_id.to_string())
            .copied()
            .unwrap();

        for id in &ids {
            assert_eq!(sm.canonical_model_update_target(*id), expected_target);
        }
    }

    #[tokio::test]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn canonical_model_update_target_preserves_independent_actor_ids() {
        let (sm, _rx) = make_state_manager(disabled_modes());
        let ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

        for id in &ids {
            sm.actor_handles
                .insert(*id, Arc::new(tokio::spawn(async {})));
        }

        for id in &ids {
            assert_eq!(sm.canonical_model_update_target(*id), *id);
        }
    }

    #[tokio::test]
    async fn shared_mode_second_actor_reuses_same_arc() {
        let (mut sm, _rx) = make_state_manager(shared_modes());

        let (h1, needs1) = sm
            .get_or_init_model_handle(None, DeviceType::Cpu)
            .await
            .unwrap();
        let (h2, needs2) = sm
            .get_or_init_model_handle(None, DeviceType::Cpu)
            .await
            .unwrap();

        assert!(
            Arc::ptr_eq(&h1, &h2),
            "Shared mode should reuse the same Arc"
        );
        assert!(needs1, "First call should need handshake (no model)");
        assert!(!needs2, "Second call should NOT need handshake");
    }

    #[tokio::test]
    async fn independent_mode_each_actor_gets_fresh_arc() {
        let (mut sm, _rx) = make_state_manager(disabled_modes());

        let (h1, _) = sm
            .get_or_init_model_handle(None, DeviceType::Cpu)
            .await
            .unwrap();
        let (h2, _) = sm
            .get_or_init_model_handle(None, DeviceType::Cpu)
            .await
            .unwrap();

        assert!(
            !Arc::ptr_eq(&h1, &h2),
            "Independent mode should create fresh Arc each time"
        );
    }

    #[tokio::test]
    async fn no_model_and_empty_path_sets_needs_handshake() {
        let (mut sm, _rx) = make_state_manager(disabled_modes());
        // shared_local_model_path is empty PathBuf, default_model is None
        let (_, needs_handshake) = sm
            .get_or_init_model_handle(None, DeviceType::Cpu)
            .await
            .unwrap();
        assert!(
            needs_handshake,
            "No model available → needs_handshake must be true"
        );
    }
}
