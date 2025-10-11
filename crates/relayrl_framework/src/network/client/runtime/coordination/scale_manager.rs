use crate::network::client::runtime::coordination::lifecycle_manager::LifeCycleManager;
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::{
    ClientExternalReceiver, ClientExternalSender, ClientFilter, RoutedMessage,
};
use crate::network::client::runtime::transport::TransportClient;
use crate::network::random_uuid;
use crate::utilities::configuration::ClientConfigLoader;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;
use uuid::Uuid;

pub(crate) struct ServerAddresses {
    agent_listener_address: String,
    training_server_address: String,
}

pub(crate) enum ScalingOperation {
    ScaleUp,
    ScaleDown,
}

pub(crate) struct RouterRuntimeParams {
    pub(crate) receiver_loop: JoinHandle<()>,
    pub(crate) filter_loop: JoinHandle<()>,
    pub(crate) sender_loop: JoinHandle<()>,
    pub(crate) tx_to_router: Sender<RoutedMessage>,
}

pub type RouterUuid = Uuid;

pub(crate) struct ScaleManager {
    shared_state: Arc<RwLock<StateManager>>,
    shared_config: Arc<RwLock<ClientConfigLoader>>,
    pub(crate) shared_global_bus_rx: Arc<RwLock<Receiver<RoutedMessage>>>,
    pub(crate) shared_transport: Arc<TransportClient>,
    pub(crate) runtime_params: Option<DashMap<RouterUuid, RouterRuntimeParams>>,
    server_addresses: ServerAddresses,
    rx_from_actor: Arc<RwLock<Receiver<RoutedMessage>>>,
    lifecycle: Option<LifeCycleManager>,
}

impl ScaleManager {
    pub(crate) fn new(
        shared_state: Arc<RwLock<StateManager>>,
        shared_config: Arc<RwLock<ClientConfigLoader>>,
        shared_global_bus_rx: Arc<RwLock<Receiver<RoutedMessage>>>,
        transport: TransportClient,
        rx_from_actor: Receiver<RoutedMessage>,
        agent_listener_address: String,
        training_server_address: String,
    ) -> Self {
        Self {
            shared_state,
            shared_config,
            shared_global_bus_rx,
            shared_transport: Arc::new(transport),
            runtime_params: None,
            server_addresses: ServerAddresses {
                agent_listener_address,
                training_server_address,
            },
            rx_from_actor: Arc::new(RwLock::new(rx_from_actor)),
            lifecycle: None,
        }
    }

    pub(crate) fn with_lifecycle(mut self, lifecycle: LifeCycleManager) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    pub(crate) async fn __scale_up(&mut self, router_add: u32) {
        if let Err(e) = self._send_scaling_warning(ScalingOperation::ScaleUp).await {
            eprintln!("Failed to send scaling warning via transport: {}", e);
            eprintln!("Aborting scale up operation...");
            return;
        }

        if self.runtime_params.is_none() {
            self.runtime_params = Some(DashMap::new());
        }

        let mut new_router_ids: Vec<RouterUuid> = Vec::new();
        let initial_router_count = self
            .runtime_params
            .as_ref()
            .map(|params| params.len())
            .unwrap_or(0);

        for i in 1..=router_add {
            let shared_receiver_state = self.shared_state.clone();
            let receiver = ClientExternalReceiver::new(
                shared_receiver_state.write().await.global_bus_tx.clone(),
                self.server_addresses.agent_listener_address.clone(),
            )
            .with_transport(self.shared_transport.clone());

            let receiver = if let Some(lc) = &self.lifecycle {
                receiver.with_shutdown(lc.subscribe_shutdown())
            } else {
                receiver
            };

            let shared_filter_state = self.shared_state.clone();
            let filter_rx = self.shared_global_bus_rx.clone();
            let filter = ClientFilter::new(filter_rx, shared_filter_state);
            let filter = if let Some(lc) = &self.lifecycle {
                filter.with_shutdown(lc.subscribe_shutdown())
            } else {
                filter
            };

            let shared_sender_state = self.shared_state.clone();
            let sender = ClientExternalSender::new(
                self.rx_from_actor.clone(),
                self.server_addresses.training_server_address.clone(),
            )
            .with_transport(self.shared_transport.clone());
            let sender = if let Some(lc) = &self.lifecycle {
                sender.with_shutdown(lc.subscribe_shutdown())
            } else {
                sender
            };

            let receiver_loop: JoinHandle<()> = Self::_spawn_external_receiver(receiver).await;
            let filter_loop: JoinHandle<()> = Self::_spawn_central_filter(filter).await;
            let sender_loop: JoinHandle<()> = Self::_spawn_external_sender(sender).await;

            let runtime_params = RouterRuntimeParams {
                receiver_loop,
                filter_loop,
                sender_loop,
                tx_to_router: shared_sender_state.write().await.global_bus_tx.clone(),
            };

            let mut router_id: RouterUuid = random_uuid(i);
            while let Some(existing_id) = self
                .runtime_params
                .as_ref()
                .and_then(|map| map.get(&router_id))
            {
                eprintln!(
                    "Router ID {} already exists, generating a new one...",
                    existing_id.key()
                );
                router_id = random_uuid(i);
            }

            new_router_ids.push(router_id);

            self.runtime_params
                .as_ref()
                .and_then(|map| map.insert(router_id, runtime_params));
        }

        let current_router_count = self
            .runtime_params
            .as_ref()
            .map(|params| params.len())
            .unwrap_or(0);

        if current_router_count != initial_router_count + (router_add as usize) {
            eprintln!(
                "Router creation failed: expected {} routers, but have {}",
                initial_router_count + (router_add as usize),
                current_router_count
            );
            eprintln!("Rolling back newly created routers...");
            self._rollback_routers(&new_router_ids).await;

            let _ = self._send_scaling_complete(ScalingOperation::ScaleUp).await;
            return;
        }

        let router_ids: Vec<RouterUuid> = self
            .runtime_params
            .as_ref()
            .expect("Runtime params should be initialized")
            .iter()
            .map(|router| *router.key())
            .collect();

        let old_actor_mappings = self
            .shared_state
            .blocking_read()
            .get_actor_router_mappings();

        self.shared_state
            .blocking_write()
            .distribute_actors(router_ids.clone());

        if let Err(e) = self._send_scaling_complete(ScalingOperation::ScaleUp).await {
            eprintln!("Failed to send scaling confirmation via transport: {}", e);
            eprintln!("Server was not notified of scaling completion.");
            eprintln!(
                "Rolling back: removing newly created routers and restoring actor mappings..."
            );

            self.shared_state
                .blocking_write()
                .restore_actor_router_mappings(old_actor_mappings);

            self._rollback_routers(&new_router_ids).await;

            eprintln!("Rollback complete. System restored to pre-scaling state.");
            return;
        }

        eprintln!(
            "Scale up successful: {} new router(s) added, total routers: {}",
            router_add, current_router_count
        );
    }

    pub(crate) async fn __scale_down(&mut self, router_remove: u32) {
        if let Err(e) = self
            ._send_scaling_warning(ScalingOperation::ScaleDown)
            .await
        {
            eprintln!("Failed to send scaling warning via transport: {}", e);
            eprintln!("Aborting scale down operation...");
            return;
        }

        match self.runtime_params {
            Some(ref mut params) => {
                let initial_router_count = params.len();

                if initial_router_count < router_remove as usize {
                    eprintln!(
                        "Cannot remove {} routers: only {} routers exist",
                        router_remove, initial_router_count
                    );

                    let _ = self
                        ._send_scaling_complete(ScalingOperation::ScaleDown)
                        .await;
                    return;
                }

                let old_actor_mappings = self
                    .shared_state
                    .blocking_read()
                    .get_actor_router_mappings();

                let (router_ids, removed_routers, current_router_count) =
                    tokio::task::block_in_place(|| {
                        let mut removed: Vec<(RouterUuid, RouterRuntimeParams)> = Vec::new();

                        for _ in 0..router_remove {
                            if let Some(router_entry) = params.iter().next() {
                                let router_id = *router_entry.key();

                                if let Some((id, params_val)) = params.remove(&router_id) {
                                    removed.push((id, params_val));
                                }
                            }
                        }

                        let remaining_router_ids: Vec<RouterUuid> =
                            params.iter().map(|router| *router.key()).collect();
                        let count = params.len();

                        (remaining_router_ids, removed, count)
                    });

                if current_router_count != initial_router_count - (router_remove as usize) {
                    eprintln!(
                        "Router removal verification failed: expected {} routers, but have {}",
                        initial_router_count - (router_remove as usize),
                        current_router_count
                    );
                    eprintln!("Rolling back: restoring removed routers...");

                    for (router_id, router_params) in removed_routers {
                        params.insert(router_id, router_params);
                    }

                    let _ = self
                        ._send_scaling_complete(ScalingOperation::ScaleDown)
                        .await;
                    return;
                }

                for (router_id, router_params) in &removed_routers {
                    router_params.receiver_loop.abort();
                    router_params.filter_loop.abort();
                    router_params.sender_loop.abort();
                    eprintln!("Router with ID {} has been removed.", router_id);
                }

                self.shared_state
                    .blocking_write()
                    .distribute_actors(router_ids.clone());

                if let Err(e) = self
                    ._send_scaling_complete(ScalingOperation::ScaleDown)
                    .await
                {
                    eprintln!("Failed to send scaling confirmation via transport: {}", e);
                    eprintln!("Server was not notified of scaling completion.");
                    eprintln!("WARNING: Routers have been removed and cannot be restored.");
                    eprintln!("Restoring actor mappings to best-effort state...");

                    self.shared_state
                        .blocking_write()
                        .restore_actor_router_mappings(old_actor_mappings);

                    eprintln!("Partial rollback complete. Manual intervention may be required.");
                    return;
                }

                eprintln!(
                    "Scale down successful: {} router(s) removed, total routers: {}",
                    router_remove, current_router_count
                );
            }
            None => {
                eprintln!("No routers to scale down.");

                let _ = self
                    ._send_scaling_complete(ScalingOperation::ScaleDown)
                    .await;
            }
        }
    }

    async fn _rollback_routers(&mut self, router_ids: &[RouterUuid]) {
        if let Some(ref params) = self.runtime_params {
            for router_id in router_ids {
                if let Some((_, router_params)) = params.remove(router_id) {
                    router_params.receiver_loop.abort();
                    router_params.filter_loop.abort();
                    router_params.sender_loop.abort();
                    eprintln!("Rolled back router with ID {}", router_id);
                }
            }
        }
    }

    async fn _send_scaling_warning(&self, operation: ScalingOperation) -> Result<(), String> {
        match &*self.shared_transport {
            #[cfg(feature = "grpc_network")]
            TransportClient::Async(async_transport) => {
                async_transport.send_scaling_warning(operation).await
            }
            #[cfg(feature = "zmq_network")]
            TransportClient::Sync(sync_transport) => {
                tokio::task::block_in_place(|| sync_transport.send_scaling_warning(operation))
            }
        }
    }

    async fn _send_scaling_complete(&self, operation: ScalingOperation) -> Result<(), String> {
        match &*self.shared_transport {
            #[cfg(feature = "grpc_network")]
            TransportClient::Async(async_transport) => {
                async_transport.send_scaling_complete(operation).await
            }
            #[cfg(feature = "zmq_network")]
            TransportClient::Sync(sync_transport) => {
                tokio::task::block_in_place(|| sync_transport.send_scaling_complete(operation))
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
