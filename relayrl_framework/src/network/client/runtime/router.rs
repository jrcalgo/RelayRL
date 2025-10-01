use crate::network::client::runtime::coordination::state_manager::StateManager;
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::AsyncClientTransport;
use crate::network::client::runtime::transport::TransportClient;
use crate::types::action::RL4SysAction;
use crate::types::trajectory::RL4SysTrajectory;
use dashmap::DashMap;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tch::Tensor;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, oneshot};
use uuid::{Timestamp, Uuid};

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
}

pub(crate) enum RoutedPayload {
    ModelHandshake,
    RequestInference {
        observation: Tensor,
        mask: Tensor,
        reward: f32,
        reply_to: oneshot::Sender<Arc<RL4SysAction>>,
    },
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
        timestamp: Timestamp,
        trajectory: RL4SysTrajectory,
    },
    Shutdown,
}

/// Intermediary routing process/filter for routing received models to specified ActorEntity
pub(crate) struct ClientFilter {
    rx_from_receiver: Receiver<RoutedMessage>,
    shared_agent_state: Arc<StateManager>,
}

impl ClientFilter {
    pub(crate) fn new(
        rx_from_receiver: Receiver<RoutedMessage>,
        shared_agent_state: Arc<StateManager>,
    ) -> Self {
        Self {
            rx_from_receiver,
            shared_agent_state,
        }
    }

    pub(crate) async fn spawn_loop(&mut self) {
        while let Some(msg) = self.rx_from_receiver.recv().await {
            if let RoutingProtocol::Shutdown = msg.protocol {
                self.route(msg).await;
                break;
            }
            self.route(msg).await;
        }
    }

    async fn route(&self, msg: RoutedMessage) {
        let actor_id = msg.actor_id;
        self.shared_agent_state
            .actor_inboxes
            .get(&actor_id)
            .expect("Actor not found")
            .send(msg)
            .await
            .expect("Cannot send message to actor");
    }
}

//// End of Routing Process
//// Start of Packet Transportation/Receiver/Sender

/// Listens & receives model bytes from a training server
pub(crate) struct ClientExternalReceiver {
    tx_to_router: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient>>,
    server_address: String,
}

impl ClientExternalReceiver {
    pub fn new(tx_to_router: Sender<RoutedMessage>, server_address: String) -> Self {
        Self {
            tx_to_router,
            transport: None,
            server_address,
        }
    }

    pub fn with_transport(mut self, transport: Arc<TransportClient>) -> Self {
        self.transport = Some(transport);
        self
    }

    pub(crate) async fn spawn_loop(&self) {
        if let Some(transport) = &self.transport {
            match &**transport {
                #[cfg(feature = "zmq_network")]
                TransportClient::Sync(sync_tr) => {
                    // Start model listening using sync transport
                    sync_tr.listen_for_model(&self.server_address, self.tx_to_router.clone());
                }
                #[cfg(feature = "grpc_network")]
                TransportClient::Async(async_tr) => {
                    async_tr.listen_for_model(&self.server_address).await;
                }
            }
        }
        // Receiver itself does nothing else; model updates come through tx_to_router
        // Waiting forever
        std::future::pending::<()>().await;
    }

    async fn _start_listen_task(&self) {}
}

type PriorityRank = i64;

struct SenderQueueEntry {
    priority: PriorityRank, // lower = sooner, higher = later
    actor_id: Uuid,
    traj_to_send: RL4SysTrajectory,
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

/// Receives trajectories from ActorEntity and creates send_traj tasks to send to a training server
pub(crate) struct ClientExternalSender {
    active: AtomicBool,
    rx_from_actor: Receiver<RoutedMessage>,
    actor_last_sent: DashMap<Uuid, i64>,
    traj_heap: Arc<Mutex<BinaryHeap<SenderQueueEntry>>>,
    transport: Option<Arc<TransportClient>>,
    server_address: String,
}

impl ClientExternalSender {
    pub fn new(rx_from_actor: Receiver<RoutedMessage>, server_address: String) -> Self {
        Self {
            active: AtomicBool::new(false),
            rx_from_actor,
            actor_last_sent: DashMap::new(),
            traj_heap: Arc::new(Mutex::new(BinaryHeap::new())),
            transport: None,
            server_address,
        }
    }

    pub fn with_transport(mut self, transport: TransportClient) -> Self {

        match transport {
            #[cfg(feature = "zmq_network")]
            TransportClient::Sync(sync_client) => {
                self.transport = Some(Arc::new(TransportClient::Sync(sync_client)));
            }
            #[cfg(feature = "grpc_network")]
            TransportClient::Async(async_client) => {
                self.transport = Some(Arc::new(TransportClient::Async(async_client)));
            }
        }
        self
    }

    pub(crate) async fn spawn_loop(mut self) {
        self.active.store(true, Ordering::SeqCst);

        let (_tx_dummy, rx_dummy) = tokio::sync::mpsc::channel(1);
        let mut rx = std::mem::replace(&mut self.rx_from_actor, rx_dummy);
        let heap_arc = self.traj_heap.clone();
        let actor_last_sent = self.actor_last_sent.clone();

        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let RoutedPayload::SendTrajectory {
                    timestamp,
                    trajectory,
                } = msg.payload
                {
                    let priority: i64 =
                        Self::_compute_priority(actor_last_sent.clone(), msg.actor_id, timestamp);

                    let queue_entry = SenderQueueEntry {
                        priority,
                        actor_id: msg.actor_id,
                        traj_to_send: trajectory,
                    };
                    let heap_arc2 = heap_arc.clone();
                    tokio::spawn(async move {
                        let mut traj_heap = heap_arc2.lock().await;
                        traj_heap.push(queue_entry);
                    });
                }
            }
        });

        self._process_heap().await;
        self.active.store(false, Ordering::SeqCst);
    }

    pub fn enqueue_traj_set_priority(
        &self,
        id: Uuid,
        priority_rank: i64,
        trajectory: RL4SysTrajectory,
    ) {
        let queue_entry = SenderQueueEntry {
            priority: priority_rank as PriorityRank,
            actor_id: id,
            traj_to_send: trajectory,
        };

        let heap_arc = Arc::clone(&self.traj_heap);
        tokio::spawn(async move {
            let mut traj_heap = heap_arc.lock().await;
            traj_heap.push(queue_entry);
        });
    }

    fn _compute_priority(
        _actor_last_sent: DashMap<Uuid, i64>,
        id: Uuid,
        timestamp: Timestamp,
    ) -> PriorityRank {
        let last = match _actor_last_sent.get(&id) {
            Some(last_ref) => *last_ref as u64,
            None => 0,
        };

        let (secs, nanos) = timestamp.to_unix();
        let ts_combined: u64 = (secs as usize + 1 / nanos as usize) as u64;

        let priority = ((last) << 32) | ((ts_combined) & 0xFFFF_FFFF);
        priority as PriorityRank
    }

    async fn _process_heap(&mut self) {
        // TODO: add polling here instead of loop
        while self.active.load(Ordering::SeqCst) {
            let job_option = {
                let mut heap = self.traj_heap.lock().await;
                heap.pop()
            };

            match job_option {
                Some(job) => {
                    self._send_trajectory(job, self.server_address.clone())
                        .await
                }
                None => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    async fn _send_trajectory(&self, entry: SenderQueueEntry, server_address: String) {
        if let Some(transport) = &self.transport {
            // Update last sent timestamp for this actor
            self.actor_last_sent.insert(entry.actor_id, entry.priority);

            // Send trajectory via transport
            match &**transport {
                #[cfg(feature = "zmq_network")]
                TransportClient::Sync(sync_client) => {
                    match sync_client.send_traj_to_server(entry.traj_to_send, &server_address) {
                        Ok(_) => {
                            println!(
                                "[ClientExternalSender] Successfully sent trajectory for actor {}",
                                entry.actor_id
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[ClientExternalSender] Failed to send trajectory for actor {}: {}",
                                entry.actor_id, e
                            );
                        }
                    }
                }
                #[cfg(feature = "grpc_network")]
                TransportClient::Async(async_client) => {
                    let grpc_traj =
                        async_client.convert_rl4sys_to_proto_trajectory(&entry.traj_to_send);
                    async_client
                        .send_traj_to_server(grpc_traj, &server_address)
                        .await;
                    println!(
                        "[ClientExternalSender] Successfully sent trajectory for actor {}",
                        entry.actor_id
                    );
                }
                _ => {
                    eprintln!(
                        "[ClientExternalSender] No transport configured for sending trajectories"
                    );
                }
            }
        } else {
            eprintln!("[ClientExternalSender] No transport configured for sending trajectories");
        }
    }
}
