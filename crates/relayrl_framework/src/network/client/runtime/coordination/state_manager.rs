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
use relayrl_types::data::tensor::{BackendMatcher, DeviceType};
use relayrl_types::model::{HotReloadableModel, ModelModule};

use active_uuid_registry::registry_uuid::Uuid;

use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
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
    actor_handles: DashMap<ActorUuid, Arc<JoinHandle<()>>>,
    actor_devices: DashMap<ActorUuid, DeviceType>,
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
                actor_handles: DashMap::new(),
                actor_devices: DashMap::new(),
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
    /// - **Independent** mode: always creates a fresh `Arc<RwLock<Option<...>>>`.
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
                let handle: LocalModelHandle<B> = Arc::new(RwLock::new(reloadable));
                self.shared_local_models.push((device, handle.clone()));
                Ok((handle, needs_handshake))
            }

            // Independent or Server-inference: every actor gets its own fresh handle.
            _ => {
                let reloadable = self
                    .load_reloadable_model(default_model, device.clone())
                    .await?;
                let needs_handshake = reloadable.is_none();
                let handle: LocalModelHandle<B> = Arc::new(RwLock::new(reloadable));
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
        self.shared_router_state.actor_router_addresses.clear();
        self.shared_local_models.clear();

        Ok(())
    }

    pub(crate) fn remove_actor(&mut self, id: Uuid) -> Result<(), StateManagerError> {
        if let Some((_, handle)) = self.actor_handles.remove(&id) {
            handle.abort();
        }

        self.shared_router_state.actor_inboxes.remove(&id);
        self.actor_devices.remove(&id);
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
