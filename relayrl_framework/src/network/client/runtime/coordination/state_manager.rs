use crate::network::HotReloadableModel;
use crate::network::client::runtime::actor::{Actor, ActorEntity};
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::transport::TransportClient;
use crate::utilities::configuration::ClientConfigLoader;
use dashmap::DashMap;
use std::sync::Arc;
use tch::Device;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;
use uuid::Uuid;

type ActorInstance = Arc<dyn ActorEntity>;
pub type ActorUuid = Uuid;

/// In-memory actor state management and global channel transport
pub(crate) struct StateManager {
    shared_config: Arc<ClientConfigLoader>,
    default_model: Option<HotReloadableModel>,
    pub(crate) global_bus_tx: Sender<RoutedMessage>,
    tx_to_sender: Sender<RoutedMessage>,
    pub(crate) actor_inboxes: DashMap<Uuid, Sender<RoutedMessage>>,
    actor_handles: DashMap<ActorUuid, Arc<JoinHandle<()>>>,
}

impl StateManager {
    pub(crate) fn new(
        shared_config: Arc<ClientConfigLoader>,
        shared_default_model: Option<HotReloadableModel>,
    ) -> (Self, Receiver<RoutedMessage>, Receiver<RoutedMessage>) {
        let (global_bus_tx, global_bus_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        let (tx_to_sender, rx_from_actor) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
        (
            Self {
                shared_config,
                default_model: shared_default_model,
                global_bus_tx,
                tx_to_sender,
                actor_inboxes: DashMap::new(),
                actor_handles: DashMap::new(),
            },
            global_bus_rx,
            rx_from_actor,
        )
    }

    pub(crate) async fn __new_actor(
        &mut self,
        id: Uuid,
        device: Device,
        default_model: Option<HotReloadableModel>,
        transport: Arc<TransportClient>,
    ) {
        if self.actor_handles.contains_key(&id) {
            eprintln!("[StateManager] Actor ID already exists, replacing existing actor...");
            self.__remove_actor(id);
        }

        let default_model = if let Some(model) = default_model {
            Some(model)
        } else if let Some(model) = { self.default_model.clone() } {
            Some(model)
        // try taking once from cached default_model
        } else {
            // lazily load from config if configured
            let loader =
                ClientConfigLoader::load_config(&self.shared_config.client_config.config_path);
            let default_model_path_str = loader
                .client_config
                .default_model_path
                .to_str()
                .expect("[StateManager] Failed to convert path to string")
                .to_string();

            if !default_model_path_str.is_empty() {
                Some(
                    HotReloadableModel::new_from_path(
                        loader.client_config.default_model_path,
                        device,
                    )
                    .await
                    .expect("[StateManager] Failed to load model from path"),
                )
            } else {
                let local_model_path_str = loader
                    .transport_config
                    .local_model_path
                    .to_str()
                    .expect("[StateManager] Failed to convert path to string")
                    .to_string();
                if !local_model_path_str.is_empty() {
                    Some(
                        HotReloadableModel::new_from_path(
                            loader.transport_config.local_model_path,
                            device,
                        )
                        .await
                        .expect("[StateManager] Failed to load model from path"),
                    )
                } else {
                    None
                }
            }
        };

        let shared_config: Arc<ClientConfigLoader> = self.shared_config.clone();
        let (tx_to_actor, rx_from_global) = mpsc::channel(CHANNEL_THROUGHPUT);
        self.actor_inboxes.insert(id, tx_to_actor.clone());
        let model_path = shared_config.transport_config.local_model_path.clone();

        let tx_to_sender = self.tx_to_sender.clone();
        let transport = transport.clone();
        let shared_config_cloned = shared_config.clone();

        let handle = Arc::new(tokio::spawn(async move {
            let (mut actor, handshake_flag) = Actor::new(
                id,
                device,
                default_model,
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
    }

    pub(crate) fn __remove_actor(&mut self, id: Uuid) {
        if let Some((_, handle)) = self.actor_handles.remove(&id) {
            handle.abort();
        }

        self.actor_inboxes.remove(&id);
    }

    pub(crate) fn __get_actors(
        &self,
    ) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), String> {
        let actor_ids = self.get_actor_id_list();
        let actor_handles = actor_ids
            .iter()
            .map(|id| self.get_actor_handle(*id).unwrap())
            .collect();
        Ok((actor_ids, actor_handles))
    }

    pub(crate) fn __set_actor_id(&self, current_id: Uuid, new_id: Uuid) -> Result<(), String> {
        let current_id_handle = match self.get_actor_handle(current_id) {
            Some(handle) => handle.clone(),
            None => return Err(format!("[StateManager] Actor ID {} not found", current_id)),
        };
        let current_id_inbox = match self.get_actor_inbox(current_id) {
            Some(inbox) => inbox.clone(),
            None => return Err(format!("[StateManager] Actor ID {} not found", current_id)),
        };

        if self.get_actor_handle(new_id).is_some() || self.get_actor_inbox(new_id).is_some() {
            return Err(format!("[StateManager] Actor ID {} already taken", new_id));
        }

        self.actor_handles.insert(new_id, current_id_handle);
        self.actor_handles.remove(&current_id);
        self.actor_inboxes.insert(new_id, current_id_inbox);
        self.actor_inboxes.remove(&current_id);

        Ok(())
    }

    fn get_actor_id_list(&self) -> Vec<ActorUuid> {
        self.actor_handles
            .iter()
            .map(|entry| entry.key().clone())
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
