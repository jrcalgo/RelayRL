use crate::network::HyperparameterArgs;
use crate::network::client::agent::ClientCapabilities;
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::network::client::runtime::coordination::lifecycle_manager::FormattedTrajectoryFileParams;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError, ServerAddresses,
};
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::buffer::{TrajectoryBufferTrait, TrajectorySinkError};
use crate::network::client::runtime::router::receiver::TransportReceiverError;
use crate::network::client::runtime::router::{
    RoutedMessage, buffer::ClientTrajectoryBuffer, filter::ClientCentralFilter,
    receiver::ClientTransportModelReceiver,
};
use crate::network::client::runtime::router_dispatcher::RouterDispatcher;
use crate::network::client::runtime::transport::{
    DispatcherError, ScalingDispatcher, TrainingDispatcher, TransportClient, TransportError,
};
use crate::utilities::configuration::Algorithm;
use crate::utilities::configuration::HyperparameterConfig;

use thiserror::Error;

use burn_tensor::backend::Backend;
use relayrl_types::types::data::action::CodecConfig;
use relayrl_types::types::data::tensor::BackendMatcher;
use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::{reserve_with, remove, add};

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ScaleManagerError {
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error(transparent)]
    TransportError(#[from] TransportError),
    #[error(transparent)]
    DispatcherError(#[from] DispatcherError),
    #[error("Failed to subscribe to shutdown: {0}")]
    SubscribeShutdownError(#[source] LifeCycleManagerError),
    #[error("Failed to spawn central filter: {0}")]
    SpawnCentralFilterError(String),
    #[error("Failed to spawn external receiver: {0}")]
    SpawnTransportReceiverError(#[source] TransportReceiverError),
    #[error("Failed to spawn external sender: {0}")]
    SpawnTransportSenderError(#[source] TrajectorySinkError),
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
    #[error("Failed to get config: {0}")]
    GetConfigError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalingOperation {
    ScaleOut,
    ScaleIn,
}

pub(crate) struct RouterRuntimeParams {
    pub(crate) receiver_loop: JoinHandle<()>,
    pub(crate) filter_loop: JoinHandle<()>,
    pub(crate) sender_loop: JoinHandle<()>,
    pub(crate) _filter_tx: Sender<RoutedMessage>,
    pub(crate) sender_tx: Sender<RoutedMessage>,
}

#[derive(Debug, Clone)]
pub(crate) struct AlgorithmArgs {
    pub(crate) algorithm: Algorithm,
    pub(crate) hyperparams: Option<HyperparameterArgs>,
}

pub type RouterUuid = Uuid;
pub type ScaleManagerUuid = Uuid;

pub(crate) struct ScaleManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    pub(crate) scaling_id: ScaleManagerUuid,
    shared_client_capabilities: Arc<ClientCapabilities>,
    shared_algorithm_args: Arc<AlgorithmArgs>,
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    shared_init_hyperparameters: Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>>,
    shared_trajectory_file_output: Arc<RwLock<FormattedTrajectoryFileParams>>,
    pub(crate) scaling_dispatcher: Arc<ScalingDispatcher<B>>,
    pub(crate) training_dispatcher: Arc<TrainingDispatcher<B>>,
    pub(crate) transport: Arc<TransportClient<B>>,
    pub(crate) router_dispatcher: Option<JoinHandle<()>>,
    pub(crate) router_filter_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
    pub(crate) runtime_params: Option<DashMap<RouterUuid, RouterRuntimeParams>>,
    codec: CodecConfig,
    cached_hyperparameters: HashMap<Algorithm, HyperparameterArgs>,
    lifecycle: Option<LifeCycleManager>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ScaleManager<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        shared_client_capabilities: Arc<ClientCapabilities>,
        shared_algorithm_args: Arc<AlgorithmArgs>,
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
        global_dispatcher_rx: Receiver<RoutedMessage>,
        transport: Arc<TransportClient<B>>,
        scaling_dispatcher: Arc<ScalingDispatcher<B>>,
        training_dispatcher: Arc<TrainingDispatcher<B>>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
        codec: Option<CodecConfig>,
        lifecycle: LifeCycleManager,
    ) -> Result<Self, ScaleManagerError> {
        let scaling_id: ScaleManagerUuid =
            reserve_with("scale_manager", 67, 100).map_err(ScaleManagerError::from)?;

        // Spawn the RouterDispatcher
        let router_filter_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>> =
            Arc::new(DashMap::new());
        let dispatcher = RouterDispatcher::new(
            global_dispatcher_rx,
            router_filter_channels.clone(),
            shared_state.clone(),
        );

        let dispatcher: RouterDispatcher<B, D_IN, D_OUT> = match lifecycle.subscribe_shutdown() {
            Ok(rx) => dispatcher.with_shutdown(rx),
            Err(e) => {
                eprintln!(
                    "[ScaleManager] Failed to subscribe dispatcher to shutdown: {}",
                    e
                );
                dispatcher
            }
        };

        let router_dispatcher: Option<JoinHandle<()>> = Some(tokio::spawn(async move {
            if let Err(e) = dispatcher.spawn_loop().await {
                eprintln!("[ScaleManager] RouterDispatcher error: {}", e);
            }
        }));

        let shared_init_hyperparameters = lifecycle.get_init_hyperparameters();
        let shared_trajectory_file_output = lifecycle.get_trajectory_file_output();

        Ok(Self {
            scaling_id,
            shared_client_capabilities,
            shared_algorithm_args,
            shared_state,
            transport,
            scaling_dispatcher,
            training_dispatcher,
            router_dispatcher,
            router_filter_channels,
            runtime_params: None,
            shared_server_addresses,
            shared_init_hyperparameters,
            shared_trajectory_file_output,
            codec: codec.unwrap_or_default(),
            cached_hyperparameters: HashMap::new(),
            lifecycle: Some(lifecycle),
        })
    }

    pub(crate) async fn clear_runtime_components(&mut self) -> Result<(), ScaleManagerError> {
        let router_count: u32 = self
            .runtime_params
            .as_ref()
            .map(|m| m.len() as u32)
            .unwrap_or(0);
        if router_count > 0 {
            self.__scale_in(router_count, false).await?;
        }
        self.router_dispatcher.take().map(|handle| handle.abort());
        self.router_filter_channels.clear();
        let _ = self.runtime_params.take();
        let _ = self.lifecycle.take();
        self.cached_hyperparameters.clear();
        Ok(())
    }

    pub(crate) async fn _send_client_ids_to_server(&self) -> Result<(), ScaleManagerError> {
        let scaling_server_address = self
            .shared_server_addresses
            .read()
            .await
            .scaling_server_address
            .clone();

        self.scaling_dispatcher
            .send_client_ids_to_server(&self.scaling_id, &scaling_server_address)
            .await
            .map_err(ScaleManagerError::from)
    }

    pub(crate) async fn _send_shutdown_signal_to_server(
        &mut self,
    ) -> Result<(), ScaleManagerError> {
        let scaling_server_address = self
            .shared_server_addresses
            .read()
            .await
            .scaling_server_address
            .clone();

        self.scaling_dispatcher
            .send_shutdown_signal_to_server(&self.scaling_id, &scaling_server_address)
            .await
            .map_err(ScaleManagerError::from)
    }

    pub(crate) async fn _send_algorithm_init_request(&mut self) -> Result<(), ScaleManagerError> {
        let scaling_server_address = self
            .shared_server_addresses
            .read()
            .await
            .scaling_server_address
            .clone();
        let algorithm_arg = self.shared_algorithm_args.algorithm.clone();

        let hyperparameter_args: HashMap<Algorithm, HyperparameterArgs> =
            if let Some(param_args) = &self.shared_algorithm_args.hyperparams {
                self.cached_hyperparameters
                    .insert(algorithm_arg.clone(), param_args.clone());

                self.cached_hyperparameters.clone()
            } else {
                // Use init_hyperparameters from lifecycle manager
                let hp_map = self.shared_init_hyperparameters.read().await.clone();
                for (k, v) in &hp_map {
                    self.cached_hyperparameters.insert(k.clone(), v.clone());
                }
                hp_map
            };

        self.scaling_dispatcher
            .send_algorithm_init_request(
                &self.scaling_id,
                algorithm_arg.to_owned(),
                hyperparameter_args,
                &scaling_server_address,
            )
            .await
            .map_err(ScaleManagerError::from)
    }

    pub(crate) async fn __scale_out(
        &mut self,
        router_add: u32,
        send_ids: bool,
    ) -> Result<(), ScaleManagerError> {
        let scaling_server_address = self
            .shared_server_addresses
            .read()
            .await
            .scaling_server_address
            .clone();
        self._send_scaling_warning(ScalingOperation::ScaleOut, &scaling_server_address)
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
            let router_id: RouterUuid =
                reserve_with("router", i * router_add, 100).map_err(ScaleManagerError::from)?;

            // Create per-router channels
            let (_filter_tx, filter_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
            let (sender_tx, sender_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

            // Store filter_tx so dispatcher can route to it
            self.router_filter_channels
                .insert(router_id, _filter_tx.clone());

            add("external_receiver", router_id).map_err(ScaleManagerError::from)?;
            // Create ExternalReceiver - sends to global dispatcher
            let shared_receiver_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                self.shared_state.clone();
            let receiver = ClientTransportModelReceiver::new(
                router_id,
                shared_receiver_state
                    .write()
                    .await
                    .global_dispatcher_tx
                    .clone(),
                self.shared_server_addresses.clone(),
            )
            .with_transport(self.transport.clone());

            let receiver: ClientTransportModelReceiver<B> = if let Some(lc) = &self.lifecycle {
                receiver.with_shutdown(
                    lc.subscribe_shutdown()
                        .map_err(ScaleManagerError::SubscribeShutdownError)?,
                )
            } else {
                receiver
            };

            let shared_filter_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                self.shared_state.clone();
            let filter: ClientCentralFilter<B, D_IN, D_OUT> =
                ClientCentralFilter::new(router_id, filter_rx, shared_filter_state);
            let filter: ClientCentralFilter<B, D_IN, D_OUT> = if let Some(lc) = &self.lifecycle {
                filter.with_shutdown(
                    lc.subscribe_shutdown()
                        .map_err(ScaleManagerError::SubscribeShutdownError)?,
                )
            } else {
                filter
            };

            add("external_sender", router_id).map_err(ScaleManagerError::from)?;
            // Create Sender - receives from actors via sender_rx
            let mut sender: ClientTrajectoryBuffer<B> =
                ClientTrajectoryBuffer::new(router_id, sender_rx, self.codec.clone());
            sender.with_transport(self.transport.clone(), self.shared_server_addresses.clone());
            sender.with_trajectory_writer(self.shared_trajectory_file_output.clone());
            if let Some(lc) = &self.lifecycle {
                sender.with_shutdown(
                    lc.subscribe_shutdown()
                        .map_err(ScaleManagerError::SubscribeShutdownError)?,
                );
            };

            let receiver_loop: JoinHandle<()> = Self::_spawn_external_receiver(receiver).await;
            let filter_loop: JoinHandle<()> = Self::_spawn_central_filter(filter).await;
            let sender_loop: JoinHandle<()> = Self::_spawn_external_sender(sender);

            let runtime_params = RouterRuntimeParams {
                receiver_loop,
                filter_loop,
                sender_loop,
                _filter_tx,
                sender_tx,
            };

            new_router_ids.push(router_id);

            self.runtime_params
                .as_ref()
                .and_then(|map| map.insert(router_id, runtime_params));
        }

        let current_router_count: usize = self
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

            return self
                ._send_scaling_complete(ScalingOperation::ScaleOut, &scaling_server_address)
                .await;
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

        if let Err(e) = self
            ._send_scaling_complete(ScalingOperation::ScaleOut, &scaling_server_address)
            .await
        {
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
        if send_ids {
            self._send_client_ids_to_server().await?;
        }

        println!(
            "Scale up successful: {} new router(s) added, total routers: {}",
            router_add, current_router_count
        );

        Ok(())
    }

    pub(crate) async fn __scale_in(
        &mut self,
        router_remove: u32,
        send_ids: bool,
    ) -> Result<(), ScaleManagerError> {
        let scaling_server_address = self
            .shared_server_addresses
            .read()
            .await
            .scaling_server_address
            .clone();
        self._send_scaling_warning(ScalingOperation::ScaleIn, &scaling_server_address)
            .await?;

        match self.runtime_params {
            Some(ref mut params) => {
                let initial_router_count = params.len();

                if initial_router_count < router_remove as usize {
                    eprintln!(
                        "Cannot remove {} routers: only {} routers exist",
                        router_remove, initial_router_count
                    );

                    let result: Result<(), ScaleManagerError> = self
                        ._send_scaling_complete(ScalingOperation::ScaleIn, &scaling_server_address)
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

                        for _ in 1..=router_remove {
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
                        ._send_scaling_complete(ScalingOperation::ScaleIn, &scaling_server_address)
                        .await;

                    return result;
                }

                for (router_id, router_params) in &removed_routers {
                    router_params.receiver_loop.abort();
                    router_params.filter_loop.abort();
                    router_params.sender_loop.abort();
                    remove("router", *router_id).map_err(ScaleManagerError::from)?;
                    remove("external_receiver", *router_id).map_err(ScaleManagerError::from)?;
                    remove("external_sender", *router_id).map_err(ScaleManagerError::from)?;
                    println!("Router with ID {} has been removed.", router_id);
                }

                {
                    let state = self.shared_state.read().await;
                    state.distribute_actors(router_ids.clone());
                }

                if let Err(e) = self
                    ._send_scaling_complete(ScalingOperation::ScaleIn, &scaling_server_address)
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

                if send_ids {
                    self._send_client_ids_to_server().await?;
                }

                println!(
                    "Scale down successful: {} router(s) removed, total routers: {}",
                    router_remove, current_router_count
                );
                Ok(())
            }
            None => {
                println!("No routers to scale down.");

                self._send_scaling_complete(ScalingOperation::ScaleIn, &scaling_server_address)
                    .await
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
        scaling_server_address: &str,
    ) -> Result<(), ScaleManagerError> {
        self.scaling_dispatcher
            .send_scaling_warning(&self.scaling_id, operation, scaling_server_address)
            .await
            .map_err(ScaleManagerError::from)
    }

    async fn _send_scaling_complete(
        &self,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), ScaleManagerError> {
        self.scaling_dispatcher
            .send_scaling_complete(&self.scaling_id, operation, scaling_server_address)
            .await
            .map_err(ScaleManagerError::from)
    }

    async fn _spawn_central_filter(filter: ClientCentralFilter<B, D_IN, D_OUT>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = filter.spawn_loop().await {
                eprintln!("[ScaleManager] Central filter error: {}", e);
            }
        })
    }

    async fn _spawn_external_receiver(receiver: ClientTransportModelReceiver<B>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = receiver.spawn_loop().await {
                eprintln!("[ScaleManager] External receiver error: {}", e);
            }
        })
    }

    fn _spawn_external_sender(mut sender: ClientTrajectoryBuffer<B>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = sender.spawn_loop() {
                eprintln!("[ScaleManager] External sender error: {}", e);
            }
        })
    }
}
