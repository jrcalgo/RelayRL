use crate::network::HyperparameterArgs;
use crate::network::client::agent::LocalTrajectoryFileParams;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::agent::{ActorInferenceMode, AlgorithmArgs, ModelMode};
use crate::network::client::agent::{ActorTrainingDataMode, ClientModes, uses_local_file_writing};
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
    ProcessInitRequest, ScalingDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::router::buffer::{TrajectoryBufferTrait, TrajectorySinkError};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::router::receiver::{
    ClientTransportModelReceiver, TransportReceiverError,
};
use crate::network::client::runtime::router::router_dispatcher::RouterDispatcher;
use crate::network::client::runtime::router::{
    RoutedMessage, buffer::ClientTrajectoryBuffer, filter::ClientCentralFilter,
};
use crate::utilities::configuration::Algorithm;
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::MetricsManager;

use active_uuid_registry::interface::{
    get_namespace_entries, remove_id, remove_namespace, reserve_id_with, reserve_namespace,
};
use active_uuid_registry::{ContextString, NamespaceString, UuidPoolError, registry_uuid::Uuid};
use burn_tensor::backend::Backend;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::data::action::CodecConfig;
use relayrl_types::data::tensor::BackendMatcher;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::model::ModelModule;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
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
    #[error("Failed to send model update message: {0}")]
    SendModelUpdateMessageError(String),
    #[error("Failed to receive model version response: {0}")]
    ReceiveModelVersionResponseError(String),
    #[error("Failed to get config: {0}")]
    GetConfigError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) enum ScalingOperation {
    ScaleOut,
    ScaleIn,
}

#[derive(Clone)]
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) enum ProcessInitFlag<B: Backend + BackendMatcher<Backend = B>> {
    TrainingAlgorithmInit,
    InferenceModelInit(Option<ModelModule<B>>),
}

pub(crate) struct RouterRuntimeParams {
    pub(crate) filter_loop: JoinHandle<()>,
    pub(crate) trajectory_buffer_loop: Option<JoinHandle<()>>,
    #[allow(dead_code)]
    pub(crate) filter_tx: Sender<RoutedMessage>,
    pub(crate) trajectory_buffer_tx: Sender<RoutedMessage>,
}

pub type RouterNamespace = Arc<str>;
pub type ScaleManagerUuid = Uuid;

pub(crate) struct ScaleManager<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    client_namespace: Arc<str>,
    router_namespace_counter: u32,
    #[allow(unused)]
    pub(crate) scaling_id: ScaleManagerUuid,
    shared_client_modes: Arc<ClientModes>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_algorithm_args: Arc<AlgorithmArgs>,
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_init_hyperparameters: Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>>,
    shared_trajectory_file_output: Arc<RwLock<LocalTrajectoryFileParams>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) scaling_dispatcher: Option<Arc<ScalingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) router_receiver_loop: Option<JoinHandle<()>>,
    pub(crate) router_dispatcher: Option<JoinHandle<()>>,
    pub(crate) router_filter_channels: Arc<DashMap<RouterNamespace, Sender<RoutedMessage>>>,
    pub(crate) runtime_params: Option<DashMap<RouterNamespace, RouterRuntimeParams>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    codec: CodecConfig,
    cached_hyperparameters: HashMap<Algorithm, HyperparameterArgs>,
    lifecycle: Option<LifeCycleManager>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ScaleManager<B, D_IN, D_OUT>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new(
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
        shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: Option<
            CodecConfig,
        >,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
        lifecycle: LifeCycleManager,
    ) -> Result<Self, ScaleManagerError> {
        let scaling_id: ScaleManagerUuid = reserve_id_with(
            client_namespace.as_ref(),
            crate::network::SCALE_MANAGER_CONTEXT,
            67,
            100,
        )
        .map_err(ScaleManagerError::from)?;

        // Spawn the RouterDispatcher
        let router_filter_channels: Arc<DashMap<RouterNamespace, Sender<RoutedMessage>>> =
            Arc::new(DashMap::new());
        let dispatcher = RouterDispatcher::new(
            global_dispatcher_rx,
            router_filter_channels.clone(),
            shared_state.read().await.shared_router_state.clone(),
            #[cfg(feature = "metrics")]
            metrics,
        )
        .await;

        let dispatcher: RouterDispatcher = match lifecycle.subscribe_shutdown() {
            Ok(rx) => dispatcher.with_shutdown(rx),
            Err(e) => {
                log::error!(
                    "[ScaleManager] Failed to subscribe dispatcher to shutdown: {}",
                    e
                );
                dispatcher
            }
        };

        let router_dispatcher: Option<JoinHandle<()>> = Some(tokio::spawn(async move {
            if let Err(e) = dispatcher.spawn_loop().await {
                log::error!("[ScaleManager] RouterDispatcher error: {}", e);
            }
        }));

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_init_hyperparameters = lifecycle.get_init_hyperparameters();

        let shared_trajectory_file_output = lifecycle.get_trajectory_file_output();

        Ok(Self {
            client_namespace,
            router_namespace_counter: 0,
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
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            router_receiver_loop: None,
            router_filter_channels,
            runtime_params: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_transport_addresses,
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
            self.scale_in(router_count, false).await?;
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            self.scale_in(router_count).await?;
        }
        if let Some(handle) = self.router_dispatcher.take() {
            handle.abort()
        };
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let Some(handle) = self.router_receiver_loop.take() {
            handle.abort()
        };
        self.router_filter_channels.clear();
        let _ = self.runtime_params.take();
        let _ = self.lifecycle.take();
        self.cached_hyperparameters.clear();
        Ok(())
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn send_client_ids_to_server(
        &self,
        client_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        replace_context: bool,
    ) -> Result<(), ScaleManagerError> {
        if let (Some(scaling_dispatcher), Some(transport_addresses)) =
            (&self.scaling_dispatcher, &self.shared_transport_addresses)
        {
            let scaling_entry = (
                self.client_namespace.to_string(),
                crate::network::SCALE_MANAGER_CONTEXT.to_string(),
                self.scaling_id,
            );
            scaling_dispatcher
                .send_client_ids(
                    scaling_entry,
                    client_entries,
                    replace_context,
                    transport_addresses.clone(),
                )
                .await
                .map_err(ScaleManagerError::from)
        } else {
            Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Send client IDs to server failed; scaling dispatcher or server addresses not found".to_string(),
            ))
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn send_shutdown_signal_to_server(&mut self) -> Result<(), ScaleManagerError> {
        if let (Some(scaling_dispatcher), Some(transport_addresses)) =
            (&self.scaling_dispatcher, &self.shared_transport_addresses)
        {
            let scaling_entry = (
                self.client_namespace.to_string(),
                crate::network::SCALE_MANAGER_CONTEXT.to_string(),
                self.scaling_id,
            );
            scaling_dispatcher
                .send_shutdown_signal(scaling_entry, transport_addresses.clone())
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
    pub(crate) async fn send_process_init_request(
        &mut self,
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        process_init_flag: ProcessInitFlag<B>,
    ) -> Result<(), ScaleManagerError> {
        if let (Some(scaling_dispatcher), Some(transport_addresses)) =
            (&self.scaling_dispatcher, &self.shared_transport_addresses)
        {
            let scaling_entry = (
                self.client_namespace.to_string(),
                crate::network::SCALE_MANAGER_CONTEXT.to_string(),
                self.scaling_id,
            );

            let built_process_init_request = match process_init_flag {
                ProcessInitFlag::TrainingAlgorithmInit => {
                    let algorithm_args = self.shared_algorithm_args.algorithm.clone();

                    let hyperparamter_args =
                        if let Some(param_args) = &self.shared_algorithm_args.hyperparams {
                            self.cached_hyperparameters
                                .insert(algorithm_args.clone(), param_args.clone());
                            self.cached_hyperparameters.clone()
                        } else {
                            let hp_map = self.shared_init_hyperparameters.read().await.clone();
                            for (k, v) in &hp_map {
                                self.cached_hyperparameters.insert(k.clone(), v.clone());
                            }
                            hp_map
                        };

                    let algorithm_model_mode =
                        match self.shared_client_modes.actor_training_data_mode.clone() {
                            ActorTrainingDataMode::Online(params) => params.model_mode,
                            ActorTrainingDataMode::Hybrid(params, _) => params.model_mode,
                            _ => ModelMode::Independent, // realistically, this should never happen due to client mode gating
                        };

                    ProcessInitRequest::TrainingAlgorithmInit(
                        algorithm_model_mode,
                        algorithm_args,
                        hyperparamter_args,
                    )
                }
                ProcessInitFlag::InferenceModelInit(default_model) => {
                    let model_mode = match self.shared_client_modes.actor_inference_mode.clone() {
                        ActorInferenceMode::Server(params) => params.model_mode,
                        ActorInferenceMode::Local(params) => params,
                    };

                    ProcessInitRequest::InferenceModelInit(model_mode, default_model)
                }
            };

            scaling_dispatcher
                .send_process_init_request(
                    scaling_entry,
                    actor_entries,
                    built_process_init_request,
                    transport_addresses.clone(),
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

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn start_transport_receiver(&mut self) -> Result<(), ScaleManagerError> {
        if self.router_receiver_loop.is_some() {
            log::debug!("[ScaleManager] Transport receiver loop already started");
            return Ok(());
        }

        match (&self.training_dispatcher, &self.shared_transport_addresses) {
            (Some(training_dispatcher), Some(transport_addresses)) => {
                let _ = reserve_id_with(
                    self.client_namespace.as_ref(),
                    crate::network::RECEIVER_CONTEXT,
                    1,
                    100,
                )
                .map_err(ScaleManagerError::from)?;

                let global_dispatcher_tx =
                    self.shared_state.read().await.global_dispatcher_tx.clone();
                let receiver = ClientTransportModelReceiver::new(
                    self.client_namespace.clone(),
                    global_dispatcher_tx,
                    self.shared_state.clone(),
                    transport_addresses.clone(),
                    training_dispatcher.clone(),
                );

                let receiver = if let Some(lc) = &self.lifecycle {
                    match lc.subscribe_shutdown() {
                        Ok(rx) => receiver.with_shutdown(rx),
                        Err(e) => {
                            log::error!(
                                "[ScaleManager] Failed to subscribe transport receiver to shutdown: {}",
                                e
                            );
                            receiver
                        }
                    }
                } else {
                    receiver
                };

                log::info!(
                    "[ScaleManager] Spawning transport receiver loop for client namespace: {}",
                    self.client_namespace
                );
                let receiver_loop = Self::spawn_transport_receiver(receiver).await;
                self.router_receiver_loop = Some(receiver_loop);
            }
            _ => {
                log::debug!(
                    "[ScaleManager] Transport receiver loop not started; training dispatcher or server addresses not found"
                );
            }
        }

        Ok(())
    }

    pub(crate) async fn scale_out(
        &mut self,
        router_add: u32,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), ScaleManagerError> {
        let router_add = router_add as usize;

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        self.start_transport_receiver().await?;

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let Some(transport_addresses) = self.get_transport_addresses()? {
            self.send_scaling_warning(ScalingOperation::ScaleOut, transport_addresses)
                .await?;
        }

        if self.runtime_params.is_none() {
            self.runtime_params = Some(DashMap::new());
        }

        let initial_router_count: usize = self
            .runtime_params
            .as_ref()
            .map(|params| params.len())
            .unwrap_or(0);

        let mut new_router_namespaces: Vec<RouterNamespace> = Vec::new();

        for _ in 0..router_add {
            // For each router, there will be the following contexts:
            // - a receiver (if enabled)
            // - a filter
            // - a trajectory buffer
            // each context will contain a single UUID
            self.router_namespace_counter += 1;
            let counter = self.router_namespace_counter;
            let router_ns_str = format!(
                "{}/{}-{}",
                self.client_namespace.as_ref(),
                crate::network::ROUTER_NAMESPACE_PREFIX,
                counter
            );
            reserve_namespace(&router_ns_str);
            let router_namespace: RouterNamespace = Arc::from(router_ns_str.as_str());

            // Create per-router channels
            let (filter_tx, filter_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);
            let (trajectory_buffer_tx, trajectory_buffer_rx) =
                tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

            let filter = {
                let shared_filter_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>> =
                    self.shared_state.clone();
                let filter_init: ClientCentralFilter<B, D_IN, D_OUT> = ClientCentralFilter::new(
                    router_namespace.clone(),
                    filter_rx,
                    shared_filter_state,
                );

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
                    let _ = reserve_id_with(
                        router_namespace.as_ref(),
                        crate::network::BUFFER_CONTEXT,
                        1,
                        100,
                    )
                    .map_err(ScaleManagerError::from)?;

                    let mut buffer_init: ClientTrajectoryBuffer<B> = ClientTrajectoryBuffer::new(
                        router_namespace.clone(),
                        trajectory_buffer_rx,
                        self.shared_client_modes.clone(),
                        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                        self.codec.clone(),
                    );

                    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                    if let (Some(training_dispatcher), Some(transport_addresses)) =
                        (&self.training_dispatcher, &self.shared_transport_addresses)
                    {
                        buffer_init.with_transport(
                            training_dispatcher.clone(),
                            transport_addresses.clone(),
                        );
                    }

                    if uses_local_file_writing(&self.shared_client_modes.actor_training_data_mode) {
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

            let filter_loop: JoinHandle<()> = Self::spawn_central_filter(filter).await;
            let trajectory_buffer_loop: Option<JoinHandle<()>> =
                buffer.map(Self::spawn_trajectory_buffer);

            let runtime_params = RouterRuntimeParams {
                filter_loop,
                trajectory_buffer_loop,
                filter_tx: filter_tx.clone(),
                trajectory_buffer_tx,
            };

            if let Some(ref params) = self.runtime_params
                && let Some(old_params) = params.insert(router_namespace.clone(), runtime_params)
            {
                old_params.filter_loop.abort();
                if let Some(h) = old_params.trajectory_buffer_loop {
                    h.abort();
                }
            }

            self.router_filter_channels
                .insert(router_namespace.clone(), filter_tx);
            new_router_namespaces.push(router_namespace);
        }

        let current_router_count: usize = self
            .runtime_params
            .as_ref()
            .map(|params| params.len())
            .unwrap_or(0);

        if current_router_count != initial_router_count + router_add {
            log::error!(
                "Router creation failed: expected {} routers, but have {}",
                initial_router_count + router_add,
                current_router_count
            );
            log::warn!("Rolling back newly created routers...");
            self.rollback_routers(&new_router_namespaces).await;

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if let Some(transport_addresses) = self.get_transport_addresses()? {
                let _ = self
                    .send_scaling_complete(ScalingOperation::ScaleOut, transport_addresses)
                    .await;
            }

            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Scale out operation failed; created routers were not properly initialized"
                    .to_string(),
            ));
        }

        let router_namespaces: Vec<RouterNamespace> = self
            .runtime_params
            .as_ref()
            .ok_or(ScaleManagerError::GetRouterRuntimeParamsError(
                "[ScaleManager] Runtime params should be initialized".to_string(),
            ))?
            .iter()
            .map(|router| router.key().clone())
            .collect();

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let old_actor_mappings: Vec<(Uuid, RouterNamespace)> = {
            let state = self.shared_state.read().await;
            StateManager::<B, D_IN, D_OUT>::get_actor_router_mappings(&state)
        };

        {
            let state = self.shared_state.write().await;
            StateManager::<B, D_IN, D_OUT>::distribute_actors(&state, router_namespaces.clone());
        }

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let Some(transport_addresses) = self.get_transport_addresses()? {
            if let Err(e) = self
                .send_scaling_complete(ScalingOperation::ScaleOut, transport_addresses)
                .await
            {
                log::warn!(
                    "Rolling back: removing newly created routers and restoring actor mappings..."
                );

                {
                    let state = self.shared_state.write().await;
                    StateManager::<B, D_IN, D_OUT>::restore_actor_router_mappings(
                        &state,
                        old_actor_mappings,
                    );
                }

                self.rollback_routers(&new_router_namespaces).await;

                log::error!(
                    "[ScaleManager] Failed to send scaling confirmation via transport: {}.\n\
                    Server was not notified of scaling completion.\n\
                    Rollback complete. System restored to pre-scaling router state.",
                    e
                );

                return Err(e);
            }

            if send_ids {
                let client_ids = get_namespace_entries(self.client_namespace.as_ref())
                    .map_err(ScaleManagerError::from)?;
                self.send_client_ids_to_server(client_ids, true).await?;
            }
        }

        log::info!(
            "Scale up successful: {} new router(s) added, total routers: {}",
            router_add,
            current_router_count
        );

        Ok(())
    }

    pub(crate) async fn scale_in(
        &mut self,
        router_remove: u32,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] send_ids: bool,
    ) -> Result<(), ScaleManagerError> {
        let router_remove = router_remove as usize;

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let transport_addresses_opt = self.get_transport_addresses()?;

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let Some(transport_addresses) = transport_addresses_opt.clone() {
            self.send_scaling_warning(ScalingOperation::ScaleIn, transport_addresses)
                .await?;
        }

        if self.runtime_params.is_none() {
            log::warn!("No routers to scale down.");
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            {
                if let Some(transport_addresses) = transport_addresses_opt {
                    return self
                        .send_scaling_complete(ScalingOperation::ScaleIn, transport_addresses)
                        .await;
                }
                return Ok(());
            }
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Scale in operation not supported".to_string(),
            ));
        }

        let initial_router_count = self
            .runtime_params
            .as_ref()
            .ok_or_else(|| {
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[ScaleManager] runtime_params unexpectedly None after is_none() check"
                        .to_string(),
                )
            })?
            .len();

        if initial_router_count < router_remove {
            log::error!(
                "Cannot remove {} routers: only {} routers exist",
                router_remove,
                initial_router_count
            );
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if let Some(transport_addresses) = transport_addresses_opt {
                let _ = self
                    .send_scaling_complete(ScalingOperation::ScaleIn, transport_addresses)
                    .await;
            }
            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                format!(
                    "Cannot remove {} routers: only {} routers exist",
                    router_remove, initial_router_count
                ),
            ));
        }

        // Phase 1: Remove from runtime_params
        let (removed_routers, current_router_count, remaining_router_namespaces) = {
            let params = self.runtime_params.as_mut().ok_or_else(|| {
                ScaleManagerError::GetRouterRuntimeParamsError(
                    "[ScaleManager] runtime_params unexpectedly None in phase 1".to_string(),
                )
            })?;
            let keys_to_remove: Vec<RouterNamespace> = params
                .iter()
                .map(|e| e.key().clone())
                .take(router_remove)
                .collect();
            let mut removed: Vec<(RouterNamespace, RouterRuntimeParams)> =
                Vec::with_capacity(router_remove);
            for key in &keys_to_remove {
                if let Some((ns, rp)) = params.remove(key) {
                    removed.push((ns, rp));
                }
            }
            let count = params.len();
            let remaining: Vec<RouterNamespace> = params.iter().map(|e| e.key().clone()).collect();
            (removed, count, remaining)
        };

        if current_router_count != initial_router_count - router_remove {
            log::error!(
                "Router removal verification failed: expected {} routers, but have {}",
                initial_router_count - router_remove,
                current_router_count
            );

            {
                let params = self.runtime_params.as_mut().ok_or_else(|| {
                    ScaleManagerError::GetRouterRuntimeParamsError(
                        "[ScaleManager] runtime_params unexpectedly None during count-verify rollback"
                            .to_string(),
                    )
                })?;
                for (ns, rp) in removed_routers {
                    params.insert(ns, rp);
                }
            }

            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            if let Some(transport_addresses) = transport_addresses_opt {
                let _ = self
                    .send_scaling_complete(ScalingOperation::ScaleIn, transport_addresses)
                    .await;
            }

            return Err(ScaleManagerError::ScalingOperationNotSupportedError(
                "Scale in operation failed; removal of routers was not successful".to_string(),
            ));
        }

        // Phase 2: Redistribute actors to remaining routers (reversible).
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let old_actor_mappings: Vec<(Uuid, RouterNamespace)> = {
            let state = self.shared_state.read().await;
            StateManager::<B, D_IN, D_OUT>::get_actor_router_mappings(&state)
        };

        {
            let state = self.shared_state.write().await;
            state.distribute_actors(remaining_router_namespaces);
        }

        // Phase 3: Notify server. On failure, full rollback — tasks are still alive.
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        if let Some(transport_addresses) = transport_addresses_opt {
            if let Err(e) = self
                .send_scaling_complete(ScalingOperation::ScaleIn, transport_addresses)
                .await
            {
                {
                    let params = self.runtime_params.as_mut().ok_or_else(|| {
                        ScaleManagerError::GetRouterRuntimeParamsError(
                            "[ScaleManager] runtime_params unexpectedly None during transport-fail rollback"
                                .to_string(),
                        )
                    })?;
                    for (ns, rp) in removed_routers {
                        params.insert(ns, rp);
                    }
                }
                {
                    let state = self.shared_state.write().await;
                    state.restore_actor_router_mappings(old_actor_mappings);
                }
                log::error!(
                    "[ScaleManager] Failed to send scaling confirmation via transport: {}.\n\
                    Full rollback complete. All routers restored.",
                    e
                );
                return Err(e);
            }

            if send_ids {
                let client_ids = get_namespace_entries(self.client_namespace.as_ref())?;
                self.send_client_ids_to_server(client_ids, true).await?;
            }
        }

        // Phase 4: Server confirmed — safe to perform destructive teardown.
        for (router_namespace, router_params) in &removed_routers {
            router_params.filter_loop.abort();

            if let Some(trajectory_buffer_loop) = &router_params.trajectory_buffer_loop {
                trajectory_buffer_loop.abort();
            }

            let namespace_entries: Vec<(NamespaceString, ContextString, Uuid)> =
                get_namespace_entries(router_namespace.as_ref())?;

            for (_, context, id) in namespace_entries.iter() {
                let _ = remove_id(router_namespace, context, *id);
            }

            remove_namespace(router_namespace.as_ref());
            self.router_filter_channels.remove(router_namespace);

            log::info!(
                "Router namespace {} removed from registry.",
                router_namespace
            );
        }

        log::info!(
            "Scale down successful: {} router(s) removed, total routers: {}",
            router_remove,
            current_router_count
        );
        Ok(())
    }

    async fn rollback_routers(&mut self, router_namespaces: &[RouterNamespace]) {
        if let Some(ref params) = self.runtime_params {
            for router_namespace in router_namespaces {
                if let Some((_, router_params)) = params.remove(router_namespace) {
                    router_params.filter_loop.abort();

                    if let Some(trajectory_buffer_loop) = &router_params.trajectory_buffer_loop {
                        trajectory_buffer_loop.abort();
                    }

                    remove_namespace(router_namespace.as_ref());
                    self.router_filter_channels.remove(router_namespace);

                    log::warn!("Rolled back router with namespace tag {}", router_namespace);
                }
            }
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    fn get_transport_addresses(
        &self,
    ) -> Result<Option<Arc<RwLock<SharedTransportAddresses>>>, ScaleManagerError> {
        if self.scaling_dispatcher.is_some() {
            match &self.shared_transport_addresses {
                Some(addrs) => Ok(Some(addrs.clone())),
                None => Err(ScaleManagerError::ScalingOperationNotSupportedError(
                    "Scaling operation failed; server addresses not found".to_string(),
                )),
            }
        } else {
            Ok(None)
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn send_scaling_warning(
        &self,
        operation: ScalingOperation,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), ScaleManagerError> {
        match &self.scaling_dispatcher {
            Some(scaling_dispatcher) => {
                let scaling_entry = (
                    self.client_namespace.to_string(),
                    crate::network::SCALE_MANAGER_CONTEXT.to_string(),
                    self.scaling_id,
                );
                scaling_dispatcher
                    .send_scaling_warning(
                        scaling_entry,
                        operation,
                        shared_transport_addresses.clone(),
                    )
                    .await
                    .map_err(ScaleManagerError::from)
            }
            None => Ok(()),
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn send_scaling_complete(
        &self,
        operation: ScalingOperation,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), ScaleManagerError> {
        match &self.scaling_dispatcher {
            Some(scaling_dispatcher) => {
                let scaling_entry = (
                    self.client_namespace.to_string(),
                    crate::network::SCALE_MANAGER_CONTEXT.to_string(),
                    self.scaling_id,
                );
                scaling_dispatcher
                    .send_scaling_complete(
                        scaling_entry,
                        operation,
                        shared_transport_addresses.clone(),
                    )
                    .await
                    .map_err(ScaleManagerError::from)
            }
            None => Ok(()),
        }
    }

    async fn spawn_central_filter(filter: ClientCentralFilter<B, D_IN, D_OUT>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = filter.spawn_loop().await {
                log::error!("[ScaleManager] Central filter error: {}", e);
            }
        })
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn spawn_transport_receiver(
        mut receiver: ClientTransportModelReceiver<B, D_IN, D_OUT>,
    ) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = receiver.spawn_loop().await {
                log::error!("[ScaleManager] Transport receiver error: {}", e);
            }
        })
    }

    fn spawn_trajectory_buffer(mut buffer: ClientTrajectoryBuffer<B>) -> JoinHandle<()> {
        tokio::task::spawn(async move {
            if let Err(e) = buffer.spawn_loop() {
                log::error!("[ScaleManager] Trajectory buffer error: {}", e);
            }
        })
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn scaling_not_supported_error_display_contains_message() {
        let err = ScaleManagerError::ScalingOperationNotSupportedError("test message".into());
        let display = format!("{}", err);
        assert!(display.contains("test message"));
    }

    #[test]
    fn get_router_runtime_params_error_display_contains_message() {
        let err = ScaleManagerError::GetRouterRuntimeParamsError("x".into());
        let display = format!("{}", err);
        assert!(display.contains("x"));
    }
}
