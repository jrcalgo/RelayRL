use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::StateManager;
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::AsyncClientTransport;
use crate::network::client::runtime::transport::TransportClient;
use crate::network::client::runtime::transport::TransportError;
use crate::utilities::misc_utils::ServerAddresses;

use crate::utilities::orchestration::tonic_utils::relayrl_encoded_trajectory_to_grpc_encoded_trajectory;
use thiserror::Error;

use relayrl_types::types::data::action::CodecConfig;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::{AnyBurnTensor, BackendMatcher, TensorData};
use relayrl_types::types::data::trajectory::{RelayRLTrajectory, TrajectoryError};

use burn_tensor::Tensor;
use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::any::Any;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::sync::broadcast;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;

pub(crate) struct RoutedMessage {
    pub actor_id: Uuid,
    pub protocol: RoutingProtocol,
    pub payload: RoutedPayload,
}

pub(crate) enum RoutingProtocol {
    ModelHandshake,
    RequestInference,
    FlagLastInference,
    ModelVersion,
    ModelUpdate,
    ActorStatistics,
    SendTrajectory,
    Shutdown,

    ScalingWarning,
    ScalingConfirmation,
    ScalingAck,
}

pub(crate) enum RoutedPayload {
    ModelHandshake,
    RequestInference(Box<InferenceRequest>),
    FlagLastInference {
        reward: f32,
    },
    ModelVersion {
        reply_to: oneshot::Sender<i64>,
    },
    ModelUpdate {
        model_bytes: Vec<u8>,
        version: i64,
    },
    ActorStatistics {
        reply_to: oneshot::Sender<Vec<(Uuid, i64)>>,
    },
    SendTrajectory {
        timestamp: (u128, u128),
        trajectory: RelayRLTrajectory,
    },
    Shutdown,

    ScalingWarning {
        operation: ScalingOperation,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    ScalingConfirmation {
        operation: ScalingOperation,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    ScalingAck {
        success: bool,
        message: String,
    },
}

/// observation and mask are Arc<AnyBurnTensor<B, D_IN>> and Arc<Option<AnyBurnTensor<B, D_OUT>>> respectively
///
/// Using Box<dyn Any + Send> to avoid adding generic parameters to this struct.
/// This is (probably) safe because InferenceRequest is only sent to the actor from the coordinator layer
pub(crate) struct InferenceRequest {
    pub(crate) observation: Box<dyn Any + Send>,
    pub(crate) mask: Box<dyn Any + Send>,
    pub(crate) reward: f32,
    pub(crate) reply_to: oneshot::Sender<Arc<RelayRLAction>>,
}

/// Intermediary routing process/filter for routing received models to specified ActorEntity
pub(crate) struct ClientFilter<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    associated_router_id: RouterUuid,
    rx_from_receiver: Receiver<RoutedMessage>,
    shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientFilter<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        associated_router_id: RouterUuid,
        rx_from_receiver: Receiver<RoutedMessage>,
        shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Self {
        Self {
            associated_router_id,
            rx_from_receiver,
            shared_agent_state,
            shutdown: None,
        }
    }

    pub(crate) fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(mut self) -> Result<(), RouterError> {
        let mut shutdown: Option<broadcast::Receiver<()>> = self.shutdown.take();
        let mut rx: Receiver<RoutedMessage> = self.rx_from_receiver;
        let this_router_id: RouterUuid = self.associated_router_id;
        let shared_agent_state = self.shared_agent_state.clone();

        loop {
            tokio::select! {
                msg_opt = rx.recv() => {
                    match msg_opt {
                        Some(msg) => {
                            if let RoutingProtocol::Shutdown = msg.protocol {
                                Self::route_message(msg, &this_router_id, &shared_agent_state).await?;
                                break Ok(());
                            }
                            Self::route_message(msg, &this_router_id, &shared_agent_state).await?;
                        }
                        None => break Ok(()),
                    }
                }
                _ = async {
                    match &mut shutdown {
                        Some(rx) => { let _ = rx.recv().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    break Ok(());
                }
            }
        }
    }

    async fn route_message(
        msg: RoutedMessage,
        router_id: &RouterUuid,
        shared_agent_state: &Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Result<(), RouterError> {
        let actor_id: Uuid = msg.actor_id;
        let shared_state = shared_agent_state.read().await;

        match shared_state.actor_router_addresses.get(&actor_id) {
            Some(assigned_router_id) if *assigned_router_id == *router_id => {
                match shared_state.actor_inboxes.get(&actor_id) {
                    Some(tx) => {
                        if let Err(e) = tx.send(msg).await {
                            return Err(RouterError::FilterError(FilterError::RoutingError(
                                format!("Cannot send message to actor: {}", e),
                            )));
                        }
                        Ok(())
                    }
                    None => Err(RouterError::FilterError(FilterError::RoutingError(
                        format!("Actor inbox not found: {}", actor_id),
                    ))),
                }
            }
            Some(other_router_id) => Err(RouterError::FilterError(FilterError::RoutingError(
                format!(
                    "Actor {} is assigned to router {:?}, but message is for router {}",
                    actor_id, other_router_id, router_id
                ),
            ))),
            None => Err(RouterError::FilterError(FilterError::RoutingError(
                format!(
                    "Actor {} is not assigned to any router or does not exist",
                    actor_id
                ),
            ))),
        }
    }
}

//// End of Filtering (center of routing process)
//// Start of Packet Transportation/Receiver/Sender

#[derive(Debug, Error)]
pub enum FilterError {
    #[error("Filter routing error: {0}")]
    RoutingError(String),
}

#[derive(Debug, Error)]
pub enum RouterError {
    #[error(transparent)]
    FilterError(#[from] FilterError),
    #[error(transparent)]
    ExternalReceiverError(#[from] ExternalReceiverError),
    #[error(transparent)]
    ExternalSenderError(#[from] ExternalSenderError),
}

#[derive(Debug, Error)]
pub enum ExternalReceiverError {
    #[error("Transport error: {0}")]
    TransportError(#[from] TransportError),
}

/// Listens & receives model bytes from a training server
pub(crate) struct ClientExternalReceiver<B: Backend + BackendMatcher<Backend = B>> {
    associated_router_id: RouterUuid,
    active: AtomicBool,
    global_dispatcher_tx: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient<B>>>,
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientExternalReceiver<B> {
    pub fn new(
        associated_router_id: RouterUuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Self {
        Self {
            associated_router_id,
            active: AtomicBool::new(false),
            global_dispatcher_tx,
            transport: None,
            shared_server_addresses,
            shutdown: None,
        }
    }

    pub fn with_transport(mut self, transport: Arc<TransportClient<B>>) -> Self {
        self.transport = Some(Arc::from(transport));
        self
    }

    pub fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(&self) -> Result<(), RouterError> {
        self.active.store(true, Ordering::SeqCst);

        if let Some(transport) = &self.transport {
            match &**transport {
                #[cfg(feature = "zmq_network")]
                TransportClient::Sync(_) => {
                    while self.active.load(Ordering::SeqCst) {
                        let agent_listener_address = self
                            .shared_server_addresses
                            .read()
                            .await
                            .agent_listener_address
                            .clone();
                        let global_dispatcher_tx: Sender<RoutedMessage> =
                            self.global_dispatcher_tx.clone();
                        let transport_clone: Arc<TransportClient<B>> = transport.clone();
                        let identity: RouterUuid = self.associated_router_id;

                        let zmq_handle: tokio::task::JoinHandle<()> =
                            tokio::task::spawn_blocking(move || {
                                if let TransportClient::Sync(sync_tr) = &*transport_clone {
                                    match sync_tr.listen_for_model(
                                        &identity,
                                        agent_listener_address.as_str(),
                                        global_dispatcher_tx.clone(),
                                    ) {
                                        Ok(()) => {}
                                        Err(e) => {
                                            eprintln!(
                                                "[ClientExternalReceiver] ZMQ listen error: {}",
                                                e
                                            );
                                            std::thread::sleep(std::time::Duration::from_secs(5));
                                        }
                                    }
                                }
                            });

                        if let Some(mut shutdown_rx) =
                            self.shutdown.as_ref().and_then(|s| Some(s.resubscribe()))
                        {
                            let _ = shutdown_rx.recv().await;
                        } else {
                            std::future::pending::<()>().await;
                        }

                        self.active.store(false, Ordering::SeqCst);
                        zmq_handle.abort();
                    }
                }
                #[cfg(feature = "grpc_network")]
                TransportClient::Async(async_tr) => {
                    let mut shutdown_rx = self.shutdown.as_ref().map(|s| s.resubscribe());

                    while self.active.load(Ordering::SeqCst) {
                        let agent_listener_address: String = self
                            .shared_server_addresses
                            .read()
                            .await
                            .agent_listener_address
                            .clone();
                        let global_dispatcher_tx: Sender<RoutedMessage> =
                            self.global_dispatcher_tx.clone();
                        let identity: RouterUuid = self.associated_router_id;

                        tokio::select! {
                            result = async_tr.listen_for_model(&identity, agent_listener_address.as_str(), global_dispatcher_tx.clone()) => {
                                match result {
                                    Ok(()) => {
                                        // this should never happen, but if it does, we need to break the loop
                                        eprintln!("[ClientExternalReceiver] listen_for_model returned Ok");
                                        break;
                                    }
                                    Err(e) => {
                                        eprintln!("[ClientExternalReceiver] Failed to listen for model: {}", e);
                                        tokio::time::sleep(Duration::from_secs(1)).await;
                                    }
                                }
                            }
                            _ = async {
                                match &mut shutdown_rx {
                                    Some(rx) => rx.recv().await.map(|_| ()).map_err(|_| ()),
                                    None => {
                                        std::future::pending::<()>().await;
                                        Ok(())
                                    }
                                }
                            } => {
                                self.active.store(false, Ordering::SeqCst);
                                break;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

type PriorityRank = i64;

struct SenderQueueEntry {
    priority: PriorityRank, // lower = sooner, higher = later
    actor_id: Uuid,
    traj_to_send: RelayRLTrajectory,
}

impl Eq for SenderQueueEntry {}

impl PartialEq<Self> for SenderQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.actor_id == other.actor_id
    }
}

impl PartialOrd<Self> for SenderQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SenderQueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority)
    }
}

#[derive(Debug, Error)]
pub enum ExternalSenderError {
    #[error("Transport error: {0}")]
    TransportError(#[from] TransportError),
    #[error("Failed to encode trajectory: {0}")]
    EncodeTrajectoryError(#[from] TrajectoryError),
}

/// Receives trajectories from ActorEntity and creates send_traj tasks to send to a training server
pub(crate) struct ClientExternalSender<B: Backend + BackendMatcher<Backend = B>> {
    associated_router_id: RouterUuid,
    active: AtomicBool,
    rx_from_actor: Receiver<RoutedMessage>,
    actor_last_sent: DashMap<Uuid, i64>,
    traj_heap: Arc<Mutex<BinaryHeap<SenderQueueEntry>>>,
    transport: Option<Arc<TransportClient<B>>>,
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    shutdown: Option<broadcast::Receiver<()>>,
    codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientExternalSender<B> {
    pub fn new(
        associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
        codec: CodecConfig,
    ) -> Self {
        Self {
            associated_router_id,
            active: AtomicBool::new(false),
            rx_from_actor,
            actor_last_sent: DashMap::new(),
            traj_heap: Arc::new(Mutex::new(BinaryHeap::new())),
            transport: None,
            shared_server_addresses,
            shutdown: None,
            codec,
        }
    }

    pub fn with_transport(mut self, transport: Arc<TransportClient<B>>) -> Self {
        self.transport = Some(Arc::from(transport));
        self
    }

    pub fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(mut self) -> Result<(), RouterError> {
        self.active.store(true, Ordering::SeqCst);

        let mut rx: Receiver<RoutedMessage> = self.rx_from_actor;
        let mut shutdown: Option<broadcast::Receiver<()>> = self.shutdown.take();
        let mut tick: tokio::time::Interval = tokio::time::interval(Duration::from_millis(100));

        // Extract fields we need to avoid borrowing self
        let actor_last_sent: &DashMap<Uuid, i64> = &self.actor_last_sent;
        let traj_heap: Arc<Mutex<BinaryHeap<SenderQueueEntry>>> = self.traj_heap.clone();
        let transport: Option<Arc<TransportClient<B>>> = self.transport.clone();
        let model_server_address: String = self
            .shared_server_addresses
            .read()
            .await
            .model_server_address
            .clone();
        let trajectory_server_address: String = self
            .shared_server_addresses
            .read()
            .await
            .trajectory_server_address
            .clone();
        let codec: CodecConfig = self.codec.clone();
        let identity: RouterUuid = self.associated_router_id;

        while self.active.load(Ordering::SeqCst) {
            tokio::select! {
                msg_opt = rx.recv() => {
                    if let Some(msg) = msg_opt {
                        if let RoutedPayload::SendTrajectory { timestamp, trajectory } = msg.payload {
                            let priority: i64 = Self::_compute_priority(&msg.actor_id, actor_last_sent, timestamp);
                            let queue_entry = SenderQueueEntry { priority, actor_id: msg.actor_id, traj_to_send: trajectory };

                            let mut heap = traj_heap.lock().await;
                            heap.push(queue_entry);
                        }
                    } else {
                        break;
                    }
                }
                _ = tick.tick() => {
                    let job_option = { let mut heap = traj_heap.lock().await; heap.pop() };
                    if let Some(job) = job_option {
                        if let Err(e) = Self::send_trajectory(&identity, job, &model_server_address, &trajectory_server_address, &transport, &codec, actor_last_sent).await {
                            eprintln!("[ClientExternalSender] Failed to send trajectory: {}", e);
                            break;
                        }
                    }
                }
                _ = async {
                    match &mut shutdown {
                        Some(rx) => { let _ = rx.recv().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    self.active.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }

        self.active.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn send_trajectory(
        associated_router_id: &RouterUuid,
        entry: SenderQueueEntry,
        model_server_address: &str,
        trajectory_server_address: &str,
        transport: &Option<Arc<TransportClient<B>>>,
        codec: &CodecConfig,
        actor_last_sent: &DashMap<Uuid, i64>,
    ) -> Result<(), RouterError> {
        if let Some(transport) = transport {
            // Update last sent timestamp for this actor
            actor_last_sent.insert(entry.actor_id, entry.priority);

            // Send trajectory via transport
            match &**transport {
                #[cfg(feature = "zmq_network")]
                TransportClient::Sync(sync_client) => {
                    let encoded_traj = entry
                        .traj_to_send
                        .encode(codec)
                        .map_err(|e| ExternalSenderError::EncodeTrajectoryError(e))?;

                    sync_client
                        .send_traj_to_server(
                            associated_router_id,
                            encoded_traj,
                            model_server_address,
                            trajectory_server_address,
                        )
                        .map_err(|e| ExternalSenderError::TransportError(e))?;
                    Ok(())
                }
                #[cfg(feature = "grpc_network")]
                TransportClient::Async(async_client) => {
                    let encoded_traj = entry
                        .traj_to_send
                        .encode(codec)
                        .map_err(|e| ExternalSenderError::EncodeTrajectoryError(e))?;

                    async_client
                        .send_traj_to_server(
                            associated_router_id,
                            encoded_traj,
                            model_server_address,
                            trajectory_server_address,
                        )
                        .await
                        .map_err(|e| ExternalSenderError::TransportError(e))?;
                    Ok(())
                }
            }
        } else {
            return Err(RouterError::ExternalSenderError(
                ExternalSenderError::TransportError(TransportError::NoTransportConfiguredError(
                    "No transport configured for sending trajectories".to_string(),
                )),
            ));
        }
    }

    /// Enqueue a trajectory with a specific priority
    ///
    /// Note: This method should be called from an async context
    pub async fn enqueue_traj_set_priority(
        &self,
        actor_id: ActorUuid,
        priority_rank: PriorityRank,
        trajectory: RelayRLTrajectory,
    ) {
        let queue_entry = SenderQueueEntry {
            priority: priority_rank as PriorityRank,
            actor_id,
            traj_to_send: trajectory,
        };

        let mut traj_heap: tokio::sync::MutexGuard<'_, BinaryHeap<SenderQueueEntry>> =
            self.traj_heap.lock().await;
        traj_heap.push(queue_entry);
    }

    /// Round robin priority computation
    fn _compute_priority(
        actor_id: &ActorUuid,
        actor_last_sent: &DashMap<Uuid, i64>,
        timestamp: (u128, u128),
    ) -> PriorityRank {
        let (traj_millis, _) = timestamp;
        let now_millis: u128 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        const MAX_AGE_MILLIS: u128 = 300_000; // 5 mins

        let age_millis: u128 = now_millis.saturating_sub(traj_millis).min(MAX_AGE_MILLIS);

        let recent_sends: i64 = match actor_last_sent.get(actor_id) {
            Some(last_ref) => (*last_ref / 1000).max(0), // Decay factor
            None => 0,
        };

        let actor_burden: i64 = recent_sends * 10_000; // Weight actor balance
        let priority: i64 = actor_burden - (age_millis.min(i64::MAX as u128) as i64);

        priority as PriorityRank
    }
}
