use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::{
    ClientExternalReceiver, ClientExternalSender, ClientFilter, ExternalReceiverError,
    ExternalSenderError, RoutedMessage,
};
use crate::network::client::runtime::router_dispatcher::RouterDispatcher;
use crate::network::client::runtime::transport::{TransportClient, TransportError};
use crate::network::random_uuid;
use crate::utilities::configuration::ClientConfigLoader;
use crate::utilities::misc_utils::ServerAddresses;

use thiserror::Error;

use burn_tensor::backend::Backend;
use relayrl_types::types::data::action::CodecConfig;
use relayrl_types::types::data::tensor::BackendMatcher;

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ScaleManagerError {
    #[error("Failed to send scaling warning: {0}")]
    SendScalingWarningError(#[from] TransportError),
    #[error("Failed to send scaling complete: {0}")]
    SendScalingCompleteError(TransportError),
    #[error("Failed to subscribe to shutdown: {0}")]
    SubscribeShutdownError(#[source] LifeCycleManagerError),
    #[error("Failed to spawn central filter: {0}")]
    SpawnCentralFilterError(String),
    #[error("Failed to spawn external receiver: {0}")]
    SpawnExternalReceiverError(#[source] ExternalReceiverError),
    #[error("Failed to spawn external sender: {0}")]
    SpawnExternalSenderError(#[source] ExternalSenderError),
    #[error("Router runtime params not found: {0}")]
    GetRouterRuntimeParamsError(String),
    #[error("Failed to send action request: {0}")]
    SendActionRequestError(String),
    #[error("Failed to receive action response: {0}")]
    ReceiveActionResponseError(String),
    #[error("Failed to send flag last action message: {0}")]
    SendFlagLastActionMessageError(String),
    #[error("Failed to send model version message: {0}")]
    SendModelVersionMessageError(String),
    #[error("Failed to receive model version response: {0}")]
    ReceiveModelVersionResponseError(String),
}

pub(crate) enum ScalingOperation {
    ScaleUp,
    ScaleDown,
}

pub(crate) struct RouterRuntimeParams {
    pub(crate) receiver_loop: JoinHandle<()>,
    pub(crate) filter_loop: JoinHandle<()>,
    pub(crate) sender_loop: JoinHandle<()>,
    pub(crate) filter_tx: Sender<RoutedMessage>,
    pub(crate) sender_tx: Sender<RoutedMessage>,
}

pub type RouterUuid = Uuid;

pub(crate) struct ScaleManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    pub(crate) shared_transport: Arc<TransportClient<B>>,
    pub(crate) router_dispatcher: Option<JoinHandle<()>>,
    pub(crate) router_filter_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
    pub(crate) runtime_params: Option<DashMap<RouterUuid, RouterRuntimeParams>>,
    lifecycle: Option<LifeCycleManager>,
    codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ScaleManager<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
        global_dispatcher_rx: Receiver<RoutedMessage>,
        transport: TransportClient<B>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
        codec: Option<CodecConfig>,
        lifecycle: Option<LifeCycleManager>,
    ) -> Self {
        // Spawn the RouterDispatcher
        let router_filter_channels: Arc<DashMap<Uuid, Sender<RoutedMessage>>> =
            Arc::new(DashMap::new());
        let dispatcher = RouterDispatcher::new(
            global_dispatcher_rx,
            router_filter_channels.clone(),
            shared_state.clone(),
        );

        let dispatcher: RouterDispatcher<B, D_IN, D_OUT> = if let Some(ref lc) = lifecycle {
            match lc.subscribe_shutdown() {
                Ok(rx) => dispatcher.with_shutdown(rx),
                Err(e) => {
                    eprintln!(
                        "[ScaleManager] Failed to subscribe dispatcher to shutdown: {}",
                        e
                    );
                    dispatcher
                }
            }
        } else {
            dispatcher
        };

        let router_dispatcher: Option<JoinHandle<()>> = Some(tokio::spawn(async move {
            if let Err(e) = dispatcher.spawn_loop().await {
                eprintln!("[ScaleManager] RouterDispatcher error: {}", e);
            }
        }));

        Self {
            shared_state,
            shared_transport: Arc::new(transport),
            router_dispatcher,
            router_filter_channels,
            runtime_params: None,
            shared_server_addresses,
            lifecycle,
            codec: codec.unwrap_or_default(),
        }
    }

    pub(crate) fn with_lifecycle(mut self, lifecycle: LifeCycleManager) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    pub(crate) async fn __scale_up(&mut self, router_add: u32) -> Result<(), ScaleManagerError> {
        self._send_scaling_warning(ScalingOperation::ScaleUp)
            .await?;

        if self.runtime_params.is_none() {
            self.runtime_params = Some(DashMap::new());
        }

        let mut new_router_ids: Vec<RouterUuid> = Vec::new();
        let initial_router_count: usize = self
            .runtime_params
            .as_ref()
            .map(|params| params.len())
            .unwrap_or(0);

        for i in 1..=router_add {
            let mut router_id: RouterUuid = random_uuid(i);
            let mut retry_count = 0;
            const MAX_UUID_RETRIES: usize = 100;

            while let Some(existing_id) = self
                .runtime_params
                .as_ref()
                .and_then(|map| map.get(&router_id))
            {
                retry_count += 1;
                if retry_count >= MAX_UUID_RETRIES {
                    eprintln!(
                        "[ScaleManager] Failed to generate unique router ID after {} attempts",
                        MAX_UUID_RETRIES
                    );
                    return Err(ScaleManagerError::GetRouterRuntimeParamsError(format!(
                        "UUID collision: failed to generate unique router ID after {} retries",
                        MAX_UUID_RETRIES
                    )));
                }

                eprintln!(
                    "Router ID {} already exists, generating a new one... (attempt {}/{})",
                    existing_id.key(),
                    retry_count,
                    MAX_UUID_RETRIES
                );
                router_id = random_uuid(i + retry_count as u32);
            }

            // Create per-router channels
            let (filter_tx, filter_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
            let (sender_tx, sender_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

            // Store filter_tx so dispatcher can route to it
            self.router_filter_channels
                .insert(router_id, filter_tx.clone());

            // Create ExternalReceiver - sends to global dispatcher
            let shared_receiver_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                self.shared_state.clone();
            let receiver = ClientExternalReceiver::new(
                router_id,
                shared_receiver_state
                    .write()
                    .await
                    .global_dispatcher_tx
                    .clone(),
                self.shared_server_addresses.clone(),
            )
            .with_transport(self.shared_transport.clone());

            let receiver: ClientExternalReceiver<B> = if let Some(lc) = &self.lifecycle {
                receiver.with_shutdown(
                    lc.subscribe_shutdown()
                        .map_err(|e| ScaleManagerError::SubscribeShutdownError(e))?,
                )
            } else {
                receiver
            };

            let shared_filter_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                self.shared_state.clone();
            let filter: ClientFilter<B, D_IN, D_OUT> =
                ClientFilter::new(router_id, filter_rx, shared_filter_state);
            let filter: ClientFilter<B, D_IN, D_OUT> = if let Some(lc) = &self.lifecycle {
                filter.with_shutdown(
                    lc.subscribe_shutdown()
                        .map_err(|e| ScaleManagerError::SubscribeShutdownError(e))?,
                )
            } else {
                filter
            };

            // Create Sender - receives from actors via sender_rx
            let sender: ClientExternalSender<B> = ClientExternalSender::new(
                router_id,
                sender_rx,
                self.shared_server_addresses.clone(),
                self.codec.clone(),
            )
            .with_transport(self.shared_transport.clone());
            let sender = if let Some(lc) = &self.lifecycle {
                sender.with_shutdown(
                    lc.subscribe_shutdown()
                        .map_err(|e| ScaleManagerError::SubscribeShutdownError(e))?,
                )
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
                filter_tx,
                sender_tx,
            };

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

            return self._send_scaling_complete(ScalingOperation::ScaleUp).await;
        }

        let router_ids: Vec<RouterUuid> = self
            .runtime_params
            .as_ref()
            .ok_or(ScaleManagerError::GetRouterRuntimeParamsError(
                "[ScaleManager] Runtime params should be initialized".to_string(),
            ))?
            .iter()
            .map(|router| *router.key())
            .collect();

        let old_actor_mappings: Vec<(Uuid, Uuid)> = {
            let state = self.shared_state.read().await;
            StateManager::<B, D_IN, D_OUT>::get_actor_router_mappings(&state)
        };

        {
            let state: tokio::sync::RwLockReadGuard<'_, StateManager<B, D_IN, D_OUT>> =
                self.shared_state.read().await;
            StateManager::<B, D_IN, D_OUT>::distribute_actors(&state, router_ids.clone());
        }

        if let Err(e) = self._send_scaling_complete(ScalingOperation::ScaleUp).await {
            eprintln!(
                "Rolling back: removing newly created routers and restoring actor mappings..."
            );

            {
                let state = self.shared_state.read().await;
                StateManager::<B, D_IN, D_OUT>::restore_actor_router_mappings(
                    &state,
                    old_actor_mappings,
                );
            }

            self._rollback_routers(&new_router_ids).await;

            eprintln!(
                "[ScaleManager] Failed to send scaling confirmation via transport: {}.
            \nServer was not notified of scaling completion.
            \n\n Rollback complete. System restored to pre-scaling router state.",
                e
            );

            return Err(e);
        }

        println!(
            "Scale up successful: {} new router(s) added, total routers: {}",
            router_add, current_router_count
        );

        Ok(())
    }

    pub(crate) async fn __scale_down(
        &mut self,
        router_remove: u32,
    ) -> Result<(), ScaleManagerError> {
        if let Err(e) = self
            ._send_scaling_warning(ScalingOperation::ScaleDown)
            .await
        {
            return Err(e);
        }

        match self.runtime_params {
            Some(ref mut params) => {
                let initial_router_count = params.len();

                if initial_router_count < router_remove as usize {
                    eprintln!(
                        "Cannot remove {} routers: only {} routers exist",
                        router_remove, initial_router_count
                    );

                    let result: Result<(), ScaleManagerError> = self
                        ._send_scaling_complete(ScalingOperation::ScaleDown)
                        .await;

                    return result;
                }

                let old_actor_mappings: Vec<(Uuid, Uuid)> = {
                    let state = self.shared_state.read().await;
                    StateManager::<B, D_IN, D_OUT>::get_actor_router_mappings(&state)
                };

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

                    let result: Result<(), ScaleManagerError> = self
                        ._send_scaling_complete(ScalingOperation::ScaleDown)
                        .await;

                    return result;
                }

                for (router_id, router_params) in &removed_routers {
                    router_params.receiver_loop.abort();
                    router_params.filter_loop.abort();
                    router_params.sender_loop.abort();
                    eprintln!("Router with ID {} has been removed.", router_id);
                }

                {
                    let state = self.shared_state.read().await;
                    state.distribute_actors(router_ids.clone());
                }

                if let Err(e) = self
                    ._send_scaling_complete(ScalingOperation::ScaleDown)
                    .await
                {
                    eprintln!("Restoring actor mappings to best-effort state...");
                    {
                        let state = self.shared_state.read().await;
                        state.restore_actor_router_mappings(old_actor_mappings);
                    }

                    eprintln!(
                        "[ScaleManager] Failed to send scaling confirmation via transport: {}.
                        \nServer was not notified of scaling completion.
                        \n\nWARNING: Routers have been removed and cannot be restored.
                        \n\nPartial rollback complete. Manual intervention may be required.",
                        e
                    );

                    return Err(e);
                }

                println!(
                    "Scale down successful: {} router(s) removed, total routers: {}",
                    router_remove, current_router_count
                );
                Ok(())
            }
            None => {
                println!("No routers to scale down.");

                let result = self
                    ._send_scaling_complete(ScalingOperation::ScaleDown)
                    .await;

                return result;
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

    async fn _send_scaling_warning(
        &self,
        operation: ScalingOperation,
    ) -> Result<(), ScaleManagerError> {
        match &*self.shared_transport {
            #[cfg(feature = "grpc_network")]
            TransportClient::Async(async_transport) => async_transport
                .send_scaling_warning(operation)
                .await
                .map_err(|e| ScaleManagerError::SendScalingWarningError(e)),
            #[cfg(feature = "zmq_network")]
            TransportClient::Sync(sync_transport) => {
                tokio::task::block_in_place(|| sync_transport.send_scaling_warning(operation))
                    .map_err(|e| ScaleManagerError::SendScalingWarningError(e))
            }
        }
    }

    async fn _send_scaling_complete(
        &self,
        operation: ScalingOperation,
    ) -> Result<(), ScaleManagerError> {
        match &*self.shared_transport {
            #[cfg(feature = "grpc_network")]
            TransportClient::Async(async_transport) => async_transport
                .send_scaling_complete(operation)
                .await
                .map_err(|e| ScaleManagerError::SendScalingCompleteError(e)),
            #[cfg(feature = "zmq_network")]
            TransportClient::Sync(sync_transport) => {
                tokio::task::block_in_place(|| sync_transport.send_scaling_complete(operation))
                    .map_err(|e| ScaleManagerError::SendScalingCompleteError(e))
            }
        }
    }

    async fn _spawn_central_filter(filter: ClientFilter<B, D_IN, D_OUT>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = filter.spawn_loop().await {
                eprintln!("[ScaleManager] Central filter error: {}", e);
            }
        })
    }

    async fn _spawn_external_receiver(receiver: ClientExternalReceiver<B>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = receiver.spawn_loop().await {
                eprintln!("[ScaleManager] External receiver error: {}", e);
            }
        })
    }

    async fn _spawn_external_sender(sender: ClientExternalSender<B>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = sender.spawn_loop().await {
                eprintln!("[ScaleManager] External sender error: {}", e);
            }
        })
    }
}
