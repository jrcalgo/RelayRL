use crate::network::client::agent::ClientCapabilities;
use crate::network::client::runtime::actor::{Actor, ActorEntity};
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::data::transport::TransportClient;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::utilities::configuration::ClientConfigLoader;

use std::path::PathBuf;
use thiserror::Error;

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::{remove, replace};
use relayrl_types::types::data::tensor::{AnyBurnTensor, BackendMatcher, DeviceType};
use relayrl_types::types::model::{HotReloadableModel, ModelModule};

use dashmap::DashMap;
use std::sync::Arc;

use burn_tensor::backend::Backend;
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Debug, Error)]
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
    SubscribeShutdownError(#[from] LifeCycleManagerError),
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

/// In-memory actor state management and global channel transport
pub(crate) struct StateManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    shared_client_capabilities: Arc<ClientCapabilities>,
    shared_max_traj_length: Arc<RwLock<u128>>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    shared_local_model_path: Arc<RwLock<PathBuf>>,
    default_model: Option<ModelModule<B>>,
    pub(crate) global_dispatcher_tx: Sender<RoutedMessage>,
    pub(crate) actor_inboxes: DashMap<ActorUuid, Sender<RoutedMessage>>,
    actor_handles: DashMap<ActorUuid, Arc<JoinHandle<()>>>,
    pub(crate) actor_router_addresses: DashMap<ActorUuid, RouterUuid>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    StateManager<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        shared_client_capabilities: Arc<ClientCapabilities>,
        shared_max_traj_length: Arc<RwLock<u128>>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
        shared_local_model_path: Arc<RwLock<PathBuf>>,
        default_model: Option<ModelModule<B>>,
    ) -> (Self, Receiver<RoutedMessage>) {
        let (global_dispatcher_tx, global_dispatcher_rx) =
            mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT * 2);
        (
            Self {
                shared_client_capabilities,
                shared_max_traj_length,
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                shared_server_addresses,
                shared_local_model_path,
                default_model,
                global_dispatcher_tx,
                actor_inboxes: DashMap::new(),
                actor_handles: DashMap::new(),
                actor_router_addresses: DashMap::new(),
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
        default_model: Option<ModelModule<B>>,
        device: DeviceType,
    ) -> Result<Option<HotReloadableModel<B>>, StateManagerError> {
        // Check fn param
        if let Some(model) = default_model {
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
        let local_model_path_str = self
            .shared_local_model_path
            .read()
            .await
            .to_str()
            .unwrap_or("")
            .to_string();

        if !local_model_path_str.is_empty() {
            return Ok(Some(
                HotReloadableModel::<B>::new_from_path(local_model_path_str.clone(), device)
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

    pub(crate) async fn __new_actor(
        &mut self,
        actor_id: ActorUuid,
        router_id: RouterUuid,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] shared_transport: Arc<
            TransportClient<B>,
        >,
        tx_to_sender: Sender<RoutedMessage>,
    ) -> Result<(), StateManagerError> {
        if self.actor_handles.contains_key(&actor_id) {
            println!(
                "[StateManager] Actor ID {} already exists, replacing existing actor...",
                actor_id
            );
            self.__remove_actor(actor_id)?
        }

        // Use helper function to load model from various sources
        let reloadable_model: Option<HotReloadableModel<B>> = self
            .load_reloadable_model(default_model, device.clone())
            .await?;

        // Create actor inbox for receiving messages from the filter
        let (tx_to_actor, actor_inbox_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        self.actor_inboxes.insert(actor_id, tx_to_actor.clone());

        let shared_local_model_path = self.shared_local_model_path.clone();

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_server_addresses = self.shared_server_addresses.clone();

        let shared_max_traj_length = self.shared_max_traj_length.clone();

        let shared_client_capabilities = self.shared_client_capabilities.clone();

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_transport_clone = shared_transport.clone();

        let handle: Arc<JoinHandle<()>> = Arc::new(tokio::spawn(async move {
            let (mut actor, handshake_flag): (Actor<B, D_IN, D_OUT>, bool) =
                Actor::<B, D_IN, D_OUT>::new(
                    actor_id,
                    device.clone(),
                    reloadable_model,
                    shared_local_model_path,
                    shared_max_traj_length,
                    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                    shared_server_addresses,
                    actor_inbox_rx,
                    tx_to_sender,
                    shared_client_capabilities,
                )
                .await;

            // Set transport after creation
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            actor.with_transport(shared_transport_clone).await;

            if handshake_flag {
                let model_handshake_ms = RoutedMessage {
                    actor_id,
                    protocol: RoutingProtocol::ModelHandshake,
                    payload: RoutedPayload::ModelHandshake,
                };
                let _ = tx_to_actor.send(model_handshake_ms).await;
            }

            if let Err(e) = actor.spawn_loop().await {
                eprintln!("[StateManager] Actor {:?} loop error: {}", actor_id, e);
            }
        }));

        self.actor_handles.insert(actor_id, handle);
        self.actor_router_addresses.insert(actor_id, router_id);
        Ok(())
    }

    pub(crate) async fn _restart_actor(
        &mut self,
        actor_id: ActorUuid,
        router_id: RouterUuid,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] shared_transport: Arc<
            TransportClient<B>,
        >,
        tx_to_sender: Sender<RoutedMessage>,
    ) -> Result<(), StateManagerError> {
        self.__remove_actor(actor_id)?;
        self.__new_actor(
            actor_id,
            router_id,
            device,
            default_model,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_transport,
            tx_to_sender,
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn __shutdown_all_actors(&self) -> Result<(), StateManagerError> {
        // Send Shutdown message to every actor inbox; actors will flush and exit
        for entry in self.actor_inboxes.iter() {
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
            if let Ok(handle) = handle {
                handle.abort();
            } else {
                continue;
            }
            remove("actor", actor_id).map_err(StateManagerError::from)?;
        }
        Ok(())
    }

    pub(crate) async fn clear_runtime_components(&mut self) -> Result<(), StateManagerError> {
        self.actor_handles.clear();
        self.actor_inboxes.clear();
        self.actor_router_addresses.clear();

        Ok(())
    }

    pub(crate) fn __remove_actor(&mut self, id: Uuid) -> Result<(), StateManagerError> {
        if let Some((_, handle)) = self.actor_handles.remove(&id) {
            handle.abort();
        }

        self.actor_inboxes.remove(&id);
        remove("actor", id).map_err(StateManagerError::from)?;
        Ok(())
    }

    pub(crate) fn __set_actor_id(
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
        self.actor_inboxes.insert(new_id, current_id_inbox);
        self.actor_inboxes.remove(&current_id);

        replace("actor", current_id, new_id).map_err(StateManagerError::from)?;

        Ok(())
    }

    pub(crate) fn distribute_actors(&self, router_ids: Vec<RouterUuid>) {
        if router_ids.is_empty() {
            return;
        }

        let actor_ids: Vec<ActorUuid> = StateManager::<B, D_IN, D_OUT>::get_actor_id_list(self);

        for (i, actor_id) in actor_ids.iter().enumerate() {
            let router_id = router_ids[i % router_ids.len()];
            self.actor_router_addresses.insert(*actor_id, router_id);
        }
    }

    pub(crate) fn restore_actor_router_mappings(&self, mappings: Vec<(ActorUuid, RouterUuid)>) {
        self.actor_router_addresses.clear();

        for (actor_id, router_id) in mappings {
            self.actor_router_addresses.insert(actor_id, router_id);
        }
    }

    pub(crate) fn get_actor_router_mappings(&self) -> Vec<(ActorUuid, RouterUuid)> {
        self.actor_router_addresses
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }

    pub(crate) fn get_actor_id_list(&self) -> Vec<ActorUuid> {
        self.actor_handles
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    fn get_actor_handle(&self, id: Uuid) -> Option<Arc<JoinHandle<()>>> {
        self.actor_handles
            .get(&id)
            .map(|handle| Arc::clone(handle.value()))
    }

    fn get_actor_inbox(&self, id: Uuid) -> Option<Sender<RoutedMessage>> {
        self.actor_inboxes.get(&id).map(|tx| tx.value().clone())
    }
}
