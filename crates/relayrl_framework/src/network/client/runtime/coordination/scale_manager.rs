use crate::network::HyperparameterArgs;
use crate::network::client::agent::LocalTrajectoryFileParams;
use crate::network::client::agent::{ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode};
use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::lifecycle_manager::{
    LifeCycleManager, LifeCycleManagerError,
};
use crate::network::client::runtime::coordination::state_manager::StateManager;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::TransportError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    ScalingDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::router::buffer::{TrajectoryBufferTrait, TrajectorySinkError};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::router::receiver::{
    ClientTransportModelReceiver, TransportReceiverError,
};
use crate::network::client::runtime::router::{
    RoutedMessage, buffer::ClientTrajectoryBuffer, filter::ClientCentralFilter,
};
use crate::network::client::runtime::router_dispatcher::RouterDispatcher;
use crate::utilities::configuration::Algorithm;
use crate::utilities::configuration::HyperparameterConfig;

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::{add_id, get_all_pairs, remove_id, reserve_id_with};
use burn_tensor::backend::Backend;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::tensor::BackendMatcher;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ScaleManagerError {
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error(transparent)]
    TransportError(#[from] TransportError),
    #[error("Scaling operation not supported: {0}")]
    ScalingOperationNotSupportedError(String),
    #[error("Failed to subscribe to shutdown: {0}")]
    SubscribeShutdownError(#[source] LifeCycleManagerError),
    #[error("Failed to spawn central filter: {0}")]
    SpawnCentralFilterError(String),
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error("Failed to spawn external receiver: {0}")]
    SpawnTransportReceiverError(#[source] TransportReceiverError),
    #[error("Failed to spawn external sender: {0}")]
    SpawnTrajectoryBufferError(#[source] TrajectorySinkError),
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
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) receiver_loop: Option<JoinHandle<()>>,
    pub(crate) filter_loop: JoinHandle<()>,
    pub(crate) trajectory_buffer_loop: Option<JoinHandle<()>>,
    pub(crate) _filter_tx: Sender<RoutedMessage>,
    pub(crate) trajectory_buffer_tx: Sender<RoutedMessage>,
}

#[derive(Debug, Clone)]
pub(crate) struct AlgorithmArgs {
    pub(crate) algorithm: Algorithm,
    pub(crate) hyperparams: Option<HyperparameterArgs>,
}

impl Default for AlgorithmArgs {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::ConfigInit,
            hyperparams: None,
        }
    }
}

pub type RouterUuid = Uuid;
pub type ScaleManagerUuid = Uuid;

pub(crate) struct ScaleManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    client_namespace: Arc<str>,
    pub(crate) scaling_id: ScaleManagerUuid,
    shared_client_modes: Arc<ClientModes>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_algorithm_args: Arc<AlgorithmArgs>,
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_server_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_init_hyperparameters: Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>>,
    shared_trajectory_file_output: Arc<RwLock<LocalTrajectoryFileParams>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) scaling_dispatcher: Option<Arc<ScalingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    pub(crate) router_dispatcher: Option<JoinHandle<()>>,
    pub(crate) router_filter_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
    pub(crate) runtime_params: Option<DashMap<RouterUuid, RouterRuntimeParams>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    codec: CodecConfig,
    cached_hyperparameters: HashMap<Algorithm, HyperparameterArgs>,
    lifecycle: Option<LifeCycleManager>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ScaleManager<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_algorithm_args: Arc<AlgorithmArgs>,
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
        global_dispatcher_rx: Receiver<RoutedMessage>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        scaling_dispatcher: Option<Arc<ScalingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_server_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: Option<
            CodecConfig,
        >,
        lifecycle: LifeCycleManager,
    ) -> Result<Self, ScaleManagerError> {
        let scaling_id: ScaleManagerUuid =
            reserve_id_with(client_namespace.as_ref(), "scale_manager", 67, 100)
                .map_err(ScaleManagerError::from)?;

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

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_init_hyperparameters = lifecycle.get_init_hyperparameters();

        let shared_trajectory_file_output = lifecycle.get_trajectory_file_output();

        Ok(Self {
            client_namespace,
            scaling_id,
            shared_client_modes,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_algorithm_args,
            shared_state,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            scaling_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            training_dispatcher,
            router_dispatcher,
            router_filter_channels,
            runtime_params: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_server_addresses,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_init_hyperparameters,
            shared_trajectory_file_output,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
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
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            self.__scale_in(router_count, false).await?;
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            self.__scale_in(router_count).await?;
        }
        self.router_dispatcher.take().map(|handle| handle.abort());
        self.router_filter_channels.clear();
        let _ = self.runtime_params.take();
        let _ = self.lifecycle.take();
        self.cached_hyperparameters.clear();
        Ok(())
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn _send_client_ids_to_server(
        &self,
        client_ids: Vec<(String, Uuid)>,
    ) -> Result<(), ScaleManagerError> {
        if let (Some(scaling_dispatcher), Some(server_addresses)) =
            (&self.scaling_dispatcher, &self.shared_server_addresses)
        {
            scaling_dispatcher
                .send_client_ids(&self.scaling_id, client_ids, server_addresses.clone())
                .await
                .map_err(ScaleManagerError::from)
        } else {
            Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Send client IDs to server failed; scaling dispatcher or server addresses not found".to_string(),
            ))
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn _send_shutdown_signal_to_server(
        &mut self,
    ) -> Result<(), ScaleManagerError> {
        if let (Some(scaling_dispatcher), Some(server_addresses)) =
            (&self.scaling_dispatcher, &self.shared_server_addresses)
        {
            scaling_dispatcher
                .send_shutdown_signal(&self.scaling_id, server_addresses.clone())
                .await
                .map_err(ScaleManagerError::from)
        } else {
            Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Shutdown signal failed; scaling dispatcher or server addresses not found"
                    .to_string(),
            ))
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn _send_algorithm_init_request(&mut self) -> Result<(), ScaleManagerError> {
        if let (Some(training_dispatcher), Some(server_addresses)) =
            (&self.training_dispatcher, &self.shared_server_addresses)
        {
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

            let algorithm_model_mode = match self.shared_client_modes.actor_training_data_mode.clone() {
                ActorTrainingDataMode::Online(params) => params.model_mode,
                ActorTrainingDataMode::Hybrid(params, _) => params.model_mode,
                _ => ModelMode::Independent, // realistically, this should never happen due to client mode gating
            };

            training_dispatcher
                .send_algorithm_init_request(
                    &self.scaling_id,
                    algorithm_model_mode,
                    algorithm_arg.to_owned(),
                    hyperparameter_args,
                    server_addresses.clone(),
                )
                .await
                .map_err(ScaleManagerError::from)
        } else {
            Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Algorithm init request failed; training dispatcher or server addresses not found"
                    .to_string(),
            ))
        }
    }

    pub(crate) async fn __scale_out(
        &mut self,
        router_add: u32,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), ScaleManagerError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if self.scaling_dispatcher.is_some() {
            if let Some(shared_server_addresses) = &self.shared_server_addresses {
                self._send_scaling_warning(
                    ScalingOperation::ScaleOut,
                    shared_server_addresses.clone(),
                )
                .await?;
            } else {
                return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                    "Scale out operation failed; server addresses not found".to_string(),
                ));
            };
        } else {
            ()
        };

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
            let router_id: RouterUuid = reserve_id_with(
                self.client_namespace.as_ref(),
                "router",
                i * router_add,
                100,
            )
            .map_err(ScaleManagerError::from)?;

            // Create per-router channels
            let (_filter_tx, filter_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
            let (trajectory_buffer_tx, trajectory_buffer_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

            // Store filter_tx so router dispatcher can route to it
            self.router_filter_channels
                .insert(router_id, _filter_tx.clone());

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let receiver = match (&self.training_dispatcher, &self.shared_server_addresses) {
                (Some(training_dispatcher), Some(server_addresses)) => {
                    add_id(
                        self.client_namespace.as_ref(),
                        "transport_receiver",
                        router_id,
                    )
                    .map_err(ScaleManagerError::from)?;

                    let shared_receiver_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                        self.shared_state.clone();
                    let global_dispatcher_tx = shared_receiver_state
                        .write()
                        .await
                        .global_dispatcher_tx
                        .clone();
                    let receiver_init = ClientTransportModelReceiver::new(
                        router_id,
                        global_dispatcher_tx,
                        server_addresses.clone(),
                        training_dispatcher.clone(),
                    );

                    if let Some(lc) = &self.lifecycle {
                        Some(
                            receiver_init.with_shutdown(
                                lc.subscribe_shutdown()
                                    .map_err(ScaleManagerError::SubscribeShutdownError)?,
                            ),
                        )
                    } else {
                        Some(receiver_init)
                    }
                }
                _ => None,
            };

            let filter = {
                let shared_filter_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                    self.shared_state.clone();
                let filter_init: ClientCentralFilter<B, D_IN, D_OUT> =
                    ClientCentralFilter::new(router_id, filter_rx, shared_filter_state);

                if let Some(lc) = &self.lifecycle {
                    filter_init.with_shutdown(
                        lc.subscribe_shutdown()
                            .map_err(ScaleManagerError::SubscribeShutdownError)?,
                    )
                } else {
                    filter_init
                }
            };

            let buffer: Option<ClientTrajectoryBuffer<B>> = {
                if self.shared_client_modes.actor_training_data_mode
                    != ActorTrainingDataMode::Disabled
                {
                    add_id(
                        self.client_namespace.as_ref(),
                        "trajectory_buffer",
                        router_id,
                    )
                    .map_err(ScaleManagerError::from)?;

                    let mut buffer_init: ClientTrajectoryBuffer<B> = ClientTrajectoryBuffer::new(
                        router_id,
                        trajectory_buffer_rx,
                        self.shared_client_modes.clone(),
                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        self.codec.clone(),
                    );

                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    if let (Some(training_dispatcher), Some(server_addresses)) =
                        (&self.training_dispatcher, &self.shared_server_addresses)
                    {
                        buffer_init
                            .with_transport(training_dispatcher.clone(), server_addresses.clone());
                    }

                    if let ActorTrainingDataMode::Offline(_) | ActorTrainingDataMode::Hybrid(_, _) =
                        &self.shared_client_modes.actor_training_data_mode
                    {
                        buffer_init
                            .with_trajectory_writer(self.shared_trajectory_file_output.clone());
                    }

                    if let Some(lc) = &self.lifecycle {
                        buffer_init.with_shutdown(
                            lc.subscribe_shutdown()
                                .map_err(ScaleManagerError::SubscribeShutdownError)?,
                        );
                    };

                    Some(buffer_init)
                } else {
                    None
                }
            };

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            let receiver_loop: Option<JoinHandle<()>> = if let Some(_receiver) = receiver {
                Some(Self::_spawn_transport_receiver(_receiver).await)
            } else {
                None
            };
            let filter_loop: JoinHandle<()> = Self::_spawn_central_filter(filter).await;
            let trajectory_buffer_loop: Option<JoinHandle<()>> = if let Some(_buffer) = buffer {
                Some(Self::_spawn_trajectory_buffer(_buffer))
            } else {
                None
            };

            let runtime_params = RouterRuntimeParams {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                receiver_loop,
                filter_loop,
                trajectory_buffer_loop,
                _filter_tx,
                trajectory_buffer_tx,
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

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if self.scaling_dispatcher.is_some() {
                if let Some(server_addresses) = &self.shared_server_addresses {
                    return self
                        ._send_scaling_complete(
                            ScalingOperation::ScaleOut,
                            server_addresses.clone(),
                        )
                        .await;
                } else {
                    return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                        "Scale out operation failed; server addresses not found".to_string(),
                    ));
                }
            }

            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Scale out operation failed; created routers were not properly initialized"
                    .to_string(),
            ));
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

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let old_actor_mappings: Vec<(Uuid, Uuid)> = {
            let state = self.shared_state.read().await;
            StateManager::<B, D_IN, D_OUT>::get_actor_router_mappings(&state)
        };

        {
            let state: tokio::sync::RwLockReadGuard<'_, StateManager<B, D_IN, D_OUT>> =
                self.shared_state.read().await;
            StateManager::<B, D_IN, D_OUT>::distribute_actors(&state, router_ids.clone());
        }

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if self.scaling_dispatcher.is_some() {
            if let Some(server_addresses) = &self.shared_server_addresses {
                if let Err(e) = self
                    ._send_scaling_complete(ScalingOperation::ScaleOut, server_addresses.clone())
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
                    let client_ids = get_all_pairs().map_err(ScaleManagerError::from)?;
                    self._send_client_ids_to_server(client_ids).await?;
                }
            }
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
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), ScaleManagerError> {
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if self.scaling_dispatcher.is_some() {
            if let Some(server_addresses) = &self.shared_server_addresses {
                self._send_scaling_warning(ScalingOperation::ScaleIn, server_addresses.clone())
                    .await?;
            } else {
                return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                    "Scale in operation not supported; server addresses not found".to_string(),
                ));
            }
        }

        match self.runtime_params {
            Some(ref mut params) => {
                let initial_router_count = params.len();

                if initial_router_count < router_remove as usize {
                    eprintln!(
                        "Cannot remove {} routers: only {} routers exist",
                        router_remove, initial_router_count
                    );
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    if self.scaling_dispatcher.is_some() {
                        if let Some(server_addresses) = &self.shared_server_addresses {
                            return self
                                ._send_scaling_complete(
                                    ScalingOperation::ScaleIn,
                                    server_addresses.clone(),
                                )
                                .await;
                        } else {
                            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                                "Scale in operation not supported; server addresses not found"
                                    .to_string(),
                            ));
                        }
                    }

                    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                    return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                        format!(
                            "Cannot remove routers: only {} routers exist",
                            initial_router_count
                        ),
                    ));
                }

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
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

                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    if self.scaling_dispatcher.is_some() {
                        if let Some(server_addresses) = &self.shared_server_addresses {
                            return self
                                ._send_scaling_complete(
                                    ScalingOperation::ScaleIn,
                                    server_addresses.clone(),
                                )
                                .await;
                        } else {
                            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                                "Scale in operation not supported; server addresses not found"
                                    .to_string(),
                            ));
                        }
                    }

                    return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                        "Scale in operation failed; removal of routers was not successful"
                            .to_string(),
                    ));
                }

                for (router_id, router_params) in &removed_routers {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    if let Some(receiver_loop) = &router_params.receiver_loop {
                        receiver_loop.abort();
                    }

                    router_params.filter_loop.abort();

                    if let Some(trajectory_buffer_loop) = &router_params.trajectory_buffer_loop {
                        trajectory_buffer_loop.abort();
                    }

                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    remove_id(
                        self.client_namespace.as_ref(),
                        "transport_receiver",
                        *router_id,
                    )
                    .map_err(ScaleManagerError::from)?;
                    remove_id(self.client_namespace.as_ref(), "router", *router_id)
                        .map_err(ScaleManagerError::from)?;
                    remove_id(
                        self.client_namespace.as_ref(),
                        "trajectory_buffer",
                        *router_id,
                    )
                    .map_err(ScaleManagerError::from)?;

                    println!("Router with ID {} has been removed.", router_id);
                }

                {
                    let state = self.shared_state.read().await;
                    state.distribute_actors(router_ids.clone());
                }

                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if self.scaling_dispatcher.is_some() {
                    if let Some(server_addresses) = &self.shared_server_addresses {
                        if let Err(e) = self
                            ._send_scaling_complete(
                                ScalingOperation::ScaleIn,
                                server_addresses.clone(),
                            )
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

                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        if send_ids {
                            let client_ids = get_all_pairs().map_err(ScaleManagerError::from)?;
                            self._send_client_ids_to_server(client_ids).await?;
                        }
                    }
                }

                println!(
                    "Scale down successful: {} router(s) removed, total routers: {}",
                    router_remove, current_router_count
                );
                Ok(())
            }
            None => {
                println!("No routers to scale down.");
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if let Some(server_addresses) = &self.shared_server_addresses {
                    return self
                        ._send_scaling_complete(ScalingOperation::ScaleIn, server_addresses.clone())
                        .await;
                } else {
                    return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                        "Scale in operation not supported; server addresses not found".to_string(),
                    ));
                }

                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                    "Scale in operation not supported".to_string(),
                ));
            }
        }
    }

    async fn _rollback_routers(&mut self, router_ids: &[RouterUuid]) {
        if let Some(ref params) = self.runtime_params {
            for router_id in router_ids {
                if let Some((_, router_params)) = params.remove(router_id) {
                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    if let Some(receiver_loop) = &router_params.receiver_loop {
                        receiver_loop.abort();
                    }

                    router_params.filter_loop.abort();

                    if let Some(trajectory_buffer_loop) = &router_params.trajectory_buffer_loop {
                        trajectory_buffer_loop.abort();
                    }

                    eprintln!("Rolled back router with ID {}", router_id);
                }
            }
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn _send_scaling_warning(
        &self,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), ScaleManagerError> {
        match &self.scaling_dispatcher {
            Some(scaling_dispatcher) => scaling_dispatcher
                .send_scaling_warning(&self.scaling_id, operation, shared_server_addresses)
                .await
                .map_err(ScaleManagerError::from),
            None => Ok(()),
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn _send_scaling_complete(
        &self,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), ScaleManagerError> {
        match &self.scaling_dispatcher {
            Some(scaling_dispatcher) => scaling_dispatcher
                .send_scaling_complete(&self.scaling_id, operation, shared_server_addresses)
                .await
                .map_err(ScaleManagerError::from),
            None => Ok(()),
        }
    }

    async fn _spawn_central_filter(filter: ClientCentralFilter<B, D_IN, D_OUT>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = filter.spawn_loop().await {
                eprintln!("[ScaleManager] Central filter error: {}", e);
            }
        })
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn _spawn_transport_receiver(
        mut receiver: ClientTransportModelReceiver<B>,
    ) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = receiver.spawn_loop().await {
                eprintln!("[ScaleManager] Transport receiver error: {}", e);
            }
        })
    }

    fn _spawn_trajectory_buffer(mut buffer: ClientTrajectoryBuffer<B>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = buffer.spawn_loop() {
                eprintln!("[ScaleManager] Trajectory buffer error: {}", e);
            }
        })
    }
}

#[cfg(test)]
mod tests {}
