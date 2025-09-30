use crate::network::client::runtime::actor::{Actor, ActorEntity};
use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::utilities::configuration::ClientConfigLoader;
use crate::network::client::runtime::transport::TransportClient;
use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::{CModule, Device};
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;
use uuid::Uuid;

type ActorInstance = Arc<dyn ActorEntity>;
pub type ActorUuid = Uuid;

/// In-memory actor state management and global channel transport
pub(crate) struct StateManager {
    shared_config: Arc<ClientConfigLoader>,
    default_model: Option<CModule>,
    pub(crate) global_bus_tx: Sender<RoutedMessage>,
    tx_to_sender: Sender<RoutedMessage>,
    pub(crate) actor_inboxes: DashMap<Uuid, Sender<RoutedMessage>>,
    actor_handles: DashMap<ActorUuid, Arc<JoinHandle<()>>>,
}

impl StateManager {
    pub(crate) fn new(
        shared_config: Arc<ClientConfigLoader>,
        default_model: Option<CModule>,
    ) -> (Self, Receiver<RoutedMessage>, Receiver<RoutedMessage>) {
        let (global_bus_tx, global_bus_rx) = mpsc::channel::<RoutedMessage>(
            CHANNEL_THROUGHPUT,
        );
        let (tx_to_sender, rx_from_actor) = mpsc::channel::<RoutedMessage>(
            CHANNEL_THROUGHPUT,
        );
        (
            Self {
                shared_config,
                default_model,
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
        model_path: Option<PathBuf>,
        transport: Arc<TransportClient>,
    ) {
        if self.actor_handles.contains_key(&id) {
            eprintln!("[StateManager] Actor ID already exists, replacing existing actor...");
            self.__remove_actor(id);
        }

        let default_model: Option<CModule> = Some(self.default_model.unwrap_or_else(|| {
            let loader = ClientConfigLoader::load_config(&self.shared_config.client_config.config_path);
            loader.client_config.default_model
        }));
        let shared_config: Arc<ClientConfigLoader> = self.shared_config.clone();

        let (tx_to_actor, rx_from_global) = mpsc::channel(
            CHANNEL_THROUGHPUT,
        );
        self.actor_inboxes.insert(id, tx_to_actor.clone());

        let model_path = match model_path {
            Some(path) => path,
            None => shared_config.transport_config.local_model_path.clone(),
        };

        let handle = Arc::new(tokio::spawn(async move {
            let (mut actor, handshake_flag) = Actor::new(
                id,
                device,
                default_model,
                model_path,
                shared_config,
                rx_from_global,
                self.tx_to_sender.clone(),
                transport.clone()
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

    pub(crate) fn __get_actors(&self) -> Result<(Vec<ActorUuid>, Vec<Arc<JoinHandle<()>>>), String> {
        let actor_ids = self.get_actor_id_list();
        let actor_handles = actor_ids.iter().map(|id| self.get_actor_handle(*id).unwrap()).collect();
        Ok((actor_ids, actor_handles))
    }

    pub(crate) fn __set_actor_id(&mut self, current_id: Uuid, new_id: Uuid) -> Result<(), String> {
        let current_id_handle = match self.get_actor_handle(current_id) {
            Some(handle) => handle.clone(),
            None => return Err(format!("[StateManager] Actor ID {} not found", current_id)),
        };
        let current_id_inbox = match self.get_actor_inbox(current_id) {
            Some(inbox) => inbox.clone(),
            None => return Err(format!("[StateManager] Actor ID {} not found", current_id)),
        };

        match self.get_actor_handle(new_id) {
            Some(handle) => return Err(format!("[StateManager] Actor ID {} already taken", new_id)),
            None => (),
        };
        match self.get_actor_inbox(new_id) {
            Some(inbox) => return Err(format!("[StateManager] Actor ID {} already taken", new_id)),
            None => (),
        };

        self.actor_handles.insert(new_id, current_id_handle);
        self.actor_handles.remove(&current_id);
        self.actor_inboxes.insert(new_id, current_id_inbox);
        self.actor_inboxes.remove(&current_id);

        Ok(())
    }

    fn get_actor_id_list(&self) -> Vec<ActorUuid> {
        self.actor_handles.iter().map(|entry| entry.key().clone()).collect()
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
