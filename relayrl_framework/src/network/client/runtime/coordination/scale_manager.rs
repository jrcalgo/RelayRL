use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::{
    ClientExternalReceiver, ClientExternalSender, ClientFilter, RoutedMessage,
};
use crate::network::client::runtime::transport::TransportClient;
use crate::network::random_uuid;
use crate::sys_utils::configuration::ClientConfigLoader;
use dashmap::DashMap;
use rand::Rng;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;
use uuid::Uuid;
use tokio::sync::mpsc::Receiver;

pub(crate) struct ServerAddresses {
    agent_listener_address: String,
    training_server_address: String,
}

pub(crate) struct RouterRuntimeParams {
    receiver_loop: JoinHandle<()>,
    filter_loop: JoinHandle<()>,
    sender_loop: JoinHandle<()>,
    tx_to_router: Sender<RoutedMessage>,
}

type RouterUuid = Uuid;

pub(crate) struct ScaleManager {
    shared_state: Arc<StateManager>,
    shared_config: Arc<ClientConfigLoader>,
    shared_transport: Arc<TransportClient>,
    runtime_params: Option<DashMap<RouterUuid, RouterRuntimeParams>>,
    server_addresses: ServerAddresses,
    rx_from_actor: Receiver<RoutedMessage>,
}

impl ScaleManager {
    pub(crate) fn new(shared_state: Arc<StateManager>, shared_config: Arc<ClientConfigLoader>, shared_transport: Arc<TransportClient>, rx_from_actor: Receiver<RoutedMessage>, agent_listener_address: String, training_server_address: String) -> Self {
        Self {
            shared_state,
            shared_config,
            shared_transport,
            runtime_params: None,
            server_addresses: ServerAddresses {
                agent_listener_address,
                training_server_address,
            },
            rx_from_actor,
        }
    }

    pub(crate) async fn __scale_up(&mut self, router_add: i32, global_bus_rx: Receiver<RoutedMessage>, shared_state: Arc<StateManager>) {
        if self.runtime_params.is_none() {
            self.runtime_params = Some(DashMap::new());
        }

        for i in 1..router_add {
            let receiver_state = self.shared_state.clone();
            let receiver = ClientExternalReceiver::new(receiver_state.global_bus_tx.clone(), self.server_addresses.agent_listener_address.clone()).with_transport(self.shared_transport.clone());

            let filter_state = self.shared_state.clone();
            let filter_rx = global_bus_rx;
            let filter = ClientFilter::new(filter_rx, filter_state);

            let sender_state = self.shared_state.clone();
            let sender = ClientExternalSender::new(self.rx_from_actor, self.server_addresses.training_server_address.clone()).with_transport(self.shared_transport.clone());

            let receiver_loop = Self::_spawn_external_receiver(receiver).await;
            let filter_loop = Self::_spawn_central_filter(filter).await;
            let sender_loop = Self::_spawn_external_sender(sender).await;

            let runtime_params = RouterRuntimeParams {
                receiver_loop,
                filter_loop,
                sender_loop,
                tx_to_router: sender_state.global_bus_tx.clone(),
            };

            let mut router_id = random_uuid(i as u32);
            while let Some(existing_id) = self
                .runtime_params
                .as_ref()
                .and_then(|map| map.get(&router_id))
            {
                eprintln!(
                    "Router ID {} already exists, generating a new one...",
                    existing_id.key()
                );
                router_id = random_uuid(i as u32);
            }

            self.runtime_params
                .as_ref()
                .and_then(|map| map.insert(router_id, runtime_params));
        }
    }

    pub(crate) async fn __scale_down(&mut self, router_remove: i32) {
        match self.runtime_params {
            Some(ref mut params) => {
                for _ in 0..router_remove {
                    if let Some(router) = params.iter().next() {
                        let router_id = router.key().clone();
                        router.value().receiver_loop.abort();
                        router.value().filter_loop.abort();
                        router.value().sender_loop.abort();
                        params.remove(&router_id);
                        eprintln!("Router with ID {} has been removed.", router_id);
                    } else {
                        eprintln!("No more routers to remove.");
                        break;
                    }
                }
            }
            None => {
                eprintln!("No routers to scale down.");
            }
        }
    }

    async fn _spawn_central_filter(mut router: ClientFilter) -> JoinHandle<()> {
        tokio::task::spawn(async move { router.spawn_loop().await })
    }

    async fn _spawn_external_receiver(receiver: ClientExternalReceiver) -> JoinHandle<()> {
        tokio::task::spawn(async move { receiver.spawn_loop().await })
    }

    async fn _spawn_external_sender(sender: ClientExternalSender) -> JoinHandle<()> {
        tokio::task::spawn(async move { sender.spawn_loop().await })
    }
}
