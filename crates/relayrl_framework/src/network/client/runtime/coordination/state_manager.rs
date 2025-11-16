use crate::network::client::runtime::actor::{Actor, ActorEntity};
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::transport::TransportClient;
use crate::utilities::configuration::ClientConfigLoader;

use thiserror::Error;

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
}

type ActorInstance<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> = Arc<dyn ActorEntity<B>>;
pub type ActorUuid = Uuid;

/// In-memory actor state management and global channel transport
pub(crate) struct StateManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    shared_config: Arc<RwLock<ClientConfigLoader>>,
    default_model: Option<ModelModule<B>>,
    pub(crate) global_bus_tx: Sender<RoutedMessage>,
    tx_to_sender: Sender<RoutedMessage>,
    pub(crate) actor_inboxes: DashMap<ActorUuid, Sender<RoutedMessage>>,
    actor_handles: DashMap<ActorUuid, Arc<JoinHandle<()>>>,
    actor_router_addresses: DashMap<ActorUuid, RouterUuid>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    StateManager<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        shared_config: Arc<RwLock<ClientConfigLoader>>,
        default_model: Option<ModelModule<B>>,
    ) -> (Self, Receiver<RoutedMessage>, Receiver<RoutedMessage>) {
        let (global_bus_tx, global_bus_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        let (tx_to_sender, rx_from_actor) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        (
            Self {
                shared_config,
                default_model,
                global_bus_tx,
                tx_to_sender,
                actor_inboxes: DashMap::new(),
                actor_handles: DashMap::new(),
                actor_router_addresses: DashMap::new(),
            },
            global_bus_rx,
            rx_from_actor,
        )
    }

    pub(crate) async fn __new_actor(
        &mut self,
        id: Uuid,
        device: DeviceType,
        default_model: Option<ModelModule<B>>,
        transport: Arc<TransportClient<B>>,
    ) -> Result<(), StateManagerError> {
        if self.actor_handles.contains_key(&id) {
            println!(
                "[StateManager] Actor ID {} already exists, replacing existing actor...",
                id
            );
            self.__remove_actor(id)?
        }

        // check fn param
        let reloadable_model: Option<HotReloadableModel<B>> = if let Some(model) = default_model {
            Some(
                HotReloadableModel::<B>::new_from_module(model, device.clone())
                    .await
                    .map_err(|_| {
                        StateManagerError::FailedToCreateReloadableModelError(format!("[StateManager] Failed to create reloadable default model from ModelModule argument..."))
                    })?,
            )
        // check if current struct default_model is Some
        } else if let Some(model) = { self.default_model.clone() } {
            Some(
                HotReloadableModel::<B>::new_from_module(model, device.clone())
                    .await
                    .map_err(|_| {
                        StateManagerError::FailedToCreateReloadableModelError(format!("[StateManager] Failed to create reloadable default model from cached default_model..."))
                    })?,
            )
        // try taking once from cached default_model
        } else {
            // lazily load from config if configured
            let loader: Arc<RwLock<ClientConfigLoader>> = self.shared_config.clone();
            let default_model_path_str: String = loader
                .read()
                .await
                .client_config
                .default_model_path
                .to_str()
                .unwrap_or_default()
                .to_string();

            if !default_model_path_str.is_empty() {
                Some(
                    HotReloadableModel::<B>::new_from_path(
                        loader.read().await.client_config.default_model_path.clone(),
                        device.clone(),
                    )
                    .await
                    .map_err(|_| {
                        StateManagerError::FailedToCreateReloadableModelError(format!(
                            "[StateManager] Failed to load model from path"
                        ))
                    })?,
                )
            } else {
                let local_model_path_str: String = loader
                    .read()
                    .await
                    .transport_config
                    .local_model_path
                    .to_str()
                    .unwrap_or_default()
                    .to_string();

                if !local_model_path_str.is_empty() {
                    Some(
                        HotReloadableModel::<B>::new_from_path(
                            loader
                                .read()
                                .await
                                .transport_config
                                .local_model_path
                                .clone(),
                            device.clone(),
                        )
                        .await
                        .map_err(|_| {
                            StateManagerError::FailedToCreateReloadableModelError(format!(
                                "[StateManager] Failed to load model from path"
                            ))
                        })?,
                    )
                } else {
                    None
                }
            }
        };

        let shared_config: Arc<RwLock<ClientConfigLoader>> = self.shared_config.clone();
        let (tx_to_actor, rx_from_global) = mpsc::channel(CHANNEL_THROUGHPUT);
        self.actor_inboxes.insert(id, tx_to_actor.clone());
        let model_path = shared_config
            .read()
            .await
            .transport_config
            .local_model_path
            .clone();

        let tx_to_sender = self.tx_to_sender.clone();
        let transport = transport.clone();
        let shared_config_cloned = shared_config.clone();

        let handle = Arc::new(tokio::spawn(async move {
            let (mut actor, handshake_flag) = Actor::<B, D_IN, D_OUT>::new(
                id,
                device.clone(),
                reloadable_model,
                model_path,
                shared_config_cloned,
                rx_from_global,
                tx_to_sender,
                transport,
            )
            .await;

            if handshake_flag {
                let model_handshake_ms = RoutedMessage {
                    actor_id: id,
                    protocol: RoutingProtocol::ModelHandshake,
                    payload: RoutedPayload::ModelHandshake,
                };
                let _ = tx_to_actor.send(model_handshake_ms).await;
            }

            actor.spawn_loop().await
        }));
        self.actor_handles.insert(id, handle);
        Ok(())
    }

    pub(crate) async fn __shutdown_all_actors(&self) -> Result<(), StateManagerError> {
        // Send Shutdown message to every actor inbox; actors will flush and exit
        for entry in self.actor_inboxes.iter() {
            let actor_id = *entry.key();
            let tx = entry.value().clone();
            let shutdown_msg = RoutedMessage {
                actor_id,
                protocol: RoutingProtocol::Shutdown,
                payload: RoutedPayload::Shutdown,
            };
            let _ = tx.send(shutdown_msg).await;

            let handle = self.actor_handles.get(&actor_id).ok_or(
                StateManagerError::ActorHandleNotFoundError(format!(
                    "[StateManager] Actor handle not found"
                )),
            );
            if let Ok(handle) = handle {
                handle.abort();
            } else {
                continue;
            }
        }
        Ok(())
    }

    pub(crate) fn spawn_shutdown_watcher(
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
        lifecycle: LifeCycleManager,
    ) {
        tokio::spawn(async move {
            match lifecycle.subscribe_shutdown() {
                Ok(mut rx) => {
                    let _ = rx.recv().await;
                    if let Err(e) = shared_state.read().await.__shutdown_all_actors().await {
                        eprintln!("[StateManager] Failed to shutdown all actors: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("[StateManager] Failed to subscribe to shutdown signal: {}", e);
                }
            }
        });
    }

    pub(crate) fn __remove_actor(&mut self, id: Uuid) -> Result<(), StateManagerError> {
        if let Some((_, handle)) = self.actor_handles.remove(&id) {
            handle.abort();
        }

        self.actor_inboxes.remove(&id);
        Ok(())
    }

    pub(crate) fn __get_actors(
        &self,
    ) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), StateManagerError> {
        let actor_ids: Vec<Uuid> = self.get_actor_id_list();
        let actor_handles: Vec<Arc<JoinHandle<()>>> = actor_ids
            .iter()
            .map(|id| {
                self.get_actor_handle(*id)
                    .ok_or(StateManagerError::ActorHandleNotFoundError(
                        "[StateManager] Actor handle not found".to_string(),
                    ))
            })
            .collect::<Result<Vec<Arc<JoinHandle<()>>>, StateManagerError>>()?;
        Ok((actor_ids, actor_handles))
    }

    pub(crate) fn __set_actor_id(
        &self,
        current_id: Uuid,
        new_id: Uuid,
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

        Ok(())
    }

    pub(crate) fn distribute_actors(&self, router_ids: Vec<RouterUuid>) {
        if router_ids.is_empty() {
            return;
        }

        let actor_ids: Vec<Uuid> = StateManager::<B, D_IN, D_OUT>::get_actor_id_list(self);

        for (i, actor_id) in actor_ids.iter().enumerate() {
            let router_id = router_ids[i % router_ids.len()];
            self.actor_router_addresses.insert(*actor_id, router_id);
        }
    }

    pub(crate) fn get_actor_router_mappings(&self) -> Vec<(ActorUuid, RouterUuid)> {
        self.actor_router_addresses
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect()
    }

    pub(crate) fn restore_actor_router_mappings(&self, mappings: Vec<(ActorUuid, RouterUuid)>) {
        self.actor_router_addresses.clear();

        for (actor_id, router_id) in mappings {
            self.actor_router_addresses.insert(actor_id, router_id);
        }
    }

    fn get_actor_id_list(&self) -> Vec<ActorUuid> {
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
