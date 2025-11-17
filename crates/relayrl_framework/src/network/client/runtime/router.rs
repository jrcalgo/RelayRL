use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::coordination::state_manager::StateManager;
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::AsyncClientTransport;
use crate::network::client::runtime::transport::TransportClient;
use crate::network::client::runtime::transport::TransportError;

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
    rx_from_receiver: Arc<RwLock<Receiver<RoutedMessage>>>,
    shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientFilter<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        rx_from_receiver: Arc<RwLock<Receiver<RoutedMessage>>>,
        shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Self {
        Self {
            rx_from_receiver,
            shared_agent_state,
            shutdown: None,
        }
    }

    pub(crate) fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(&mut self) -> Result<(), RouterError> {
        let mut shutdown = self.shutdown.take();
        let mut rx_guard = self.rx_from_receiver.write().await;
        loop {
            tokio::select! {
                msg_opt = rx_guard.recv() => {
                    match msg_opt {
                        Some(msg) => {
                            if let RoutingProtocol::Shutdown = msg.protocol {
                                self.route(msg).await;
                                break Ok(());
                            }
                            self.route(msg).await;
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

    async fn route(&self, msg: RoutedMessage) {
        let actor_id = msg.actor_id;
        match self
            .shared_agent_state
            .read()
            .await
            .actor_inboxes
            .get(&actor_id)
        {
            Some(tx) => {
                if let Err(e) = tx.send(msg).await {
                    eprintln!("[ClientFilter] Cannot send message to actor: {}", e);
                }
            }
            None => {
                eprintln!("[ClientFilter] Actor not found: {}", actor_id);
            }
        }
    }
}

//// End of Filtering (center of routing process) Process
//// Start of Packet Transportation/Receiver/Sender

#[derive(Debug, Error)]
pub enum ClientFilterError {
    #[error("Filter routing error: {0}")]
    RoutingError(String),
}

#[derive(Debug, Error)]
pub enum RouterError {
    #[error(transparent)]
    CentralFilterError(#[from] ClientFilterError),
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
    tx_to_router: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient<B>>>,
    server_address: String,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientExternalReceiver<B> {
    pub fn new(tx_to_router: Sender<RoutedMessage>, server_address: String) -> Self {
        Self {
            tx_to_router,
            transport: None,
            server_address,
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
        if let Some(transport) = &self.transport {
            match &**transport {
                #[cfg(feature = "zmq_network")]
                TransportClient::Sync(sync_tr) => {
                    sync_tr
                        .listen_for_model(&self.server_address, self.tx_to_router.clone())
                        .map_err(|e| {
                            RouterError::ExternalReceiverError(
                                ExternalReceiverError::TransportError(e),
                            )
                        })?;
                }
                #[cfg(feature = "grpc_network")]
                TransportClient::Async(async_tr) => {
                    async_tr
                        .listen_for_model(&self.server_address)
                        .await
                        .map_err(|e| {
                            RouterError::ExternalReceiverError(
                                ExternalReceiverError::TransportError(e),
                            )
                        })?;
                }
            }
        }

        std::future::pending::<()>().await;
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
    active: AtomicBool,
    rx_from_actor: Arc<RwLock<Receiver<RoutedMessage>>>,
    actor_last_sent: DashMap<Uuid, i64>,
    traj_heap: Arc<Mutex<BinaryHeap<SenderQueueEntry>>>,
    transport: Option<Arc<TransportClient<B>>>,
    server_address: String,
    shutdown: Option<broadcast::Receiver<()>>,
    codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientExternalSender<B> {
    pub fn new(
        rx_from_actor: Arc<RwLock<Receiver<RoutedMessage>>>,
        server_address: String,
        codec: CodecConfig,
    ) -> Self {
        Self {
            active: AtomicBool::new(false),
            rx_from_actor,
            actor_last_sent: DashMap::new(),
            traj_heap: Arc::new(Mutex::new(BinaryHeap::new())),
            transport: None,
            server_address,
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

        let (_tx_dummy, rx_dummy) = tokio::sync::mpsc::channel(1);
        let rx = std::mem::replace(&mut self.rx_from_actor, Arc::new(RwLock::new(rx_dummy)));
        let mut rx_guard = rx.write().await;
        let mut shutdown = self.shutdown.take();
        let mut tick = tokio::time::interval(Duration::from_millis(100));

        while self.active.load(Ordering::SeqCst) {
            tokio::select! {
                msg_opt = rx_guard.recv() => {
                    if let Some(msg) = msg_opt {
                        if let RoutedPayload::SendTrajectory { timestamp, trajectory } = msg.payload {
                            let priority: i64 = Self::_compute_priority(&self.actor_last_sent, &msg.actor_id, timestamp);
                            let queue_entry = SenderQueueEntry { priority, actor_id: msg.actor_id, traj_to_send: trajectory };
                            let heap_arc2 = self.traj_heap.clone();
                            tokio::spawn(async move {
                                let mut traj_heap = heap_arc2.lock().await;
                                traj_heap.push(queue_entry);
                            });
                        }
                    } else {
                        break;
                    }
                }
                _ = tick.tick() => {
                    let job_option = { let mut heap = self.traj_heap.lock().await; heap.pop() };
                    if let Some(job) = job_option { if let Err(e) = self._send_trajectory(job, self.server_address.clone()).await { eprintln!("[ClientExternalSender] Failed to send trajectory: {}", e); break;} }
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

    pub fn enqueue_traj_set_priority(
        &self,
        id: Uuid,
        priority_rank: i64,
        trajectory: RelayRLTrajectory,
    ) {
        let queue_entry = SenderQueueEntry {
            priority: priority_rank as PriorityRank,
            actor_id: id,
            traj_to_send: trajectory,
        };

        let heap_arc: Arc<Mutex<BinaryHeap<SenderQueueEntry>>> = Arc::clone(&self.traj_heap);
        tokio::spawn(async move {
            let mut traj_heap = heap_arc.lock().await;
            traj_heap.push(queue_entry);
        });
    }

    /// Round robin priority computation
    fn _compute_priority(
        actor_last_sent: &DashMap<Uuid, i64>,
        id: &Uuid,
        timestamp: (u128, u128),
    ) -> PriorityRank {
        let (traj_millis, _) = timestamp;
        let now_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        const MAX_AGE_MILLIS: u128 = 300_000; // 5 mins

        let age_millis: u128 = now_millis.saturating_sub(traj_millis).min(MAX_AGE_MILLIS);

        let recent_sends: i64 = match actor_last_sent.get(&id) {
            Some(last_ref) => (*last_ref / 1000).max(0), // Decay factor
            None => 0,
        };

        let actor_burden: i64 = recent_sends * 10_000; // Weight actor balance
        let priority = actor_burden - (age_millis.min(i64::MAX as u128) as i64);

        priority as PriorityRank
    }

    async fn _send_trajectory(
        &self,
        entry: SenderQueueEntry,
        server_address: String,
    ) -> Result<(), RouterError> {
        if let Some(transport) = &self.transport {
            // Update last sent timestamp for this actor
            self.actor_last_sent.insert(entry.actor_id, entry.priority);

            // Send trajectory via transport
            match &**transport {
                #[cfg(feature = "zmq_network")]
                TransportClient::Sync(sync_client) => {
                    let encoded_traj = entry
                        .traj_to_send
                        .encode(&self.codec)
                        .map_err(|e| ExternalSenderError::EncodeTrajectoryError(e))?;

                    sync_client
                        .send_traj_to_server(encoded_traj, &server_address)
                        .map_err(|e| ExternalSenderError::TransportError(e))?;
                    Ok(())
                }
                #[cfg(feature = "grpc_network")]
                TransportClient::Async(async_client) => {
                    let encoded_traj = entry
                        .traj_to_send
                        .encode(&self.codec)
                        .map_err(|e| ExternalSenderError::EncodeTrajectoryError(e))?;

                    async_client
                        .send_traj_to_server(encoded_traj, &server_address)
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
}
