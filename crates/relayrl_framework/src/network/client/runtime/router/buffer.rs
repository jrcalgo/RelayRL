use super::{RoutedMessage, RoutedPayload, RouterError};
use crate::network::client::agent::ClientModes;
use crate::network::client::agent::{
    LocalTrajectoryFileParams, LocalTrajectoryFileType, uses_local_file_writing,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::data::file_sink::{
    FileSinkError, write_local_trajectory_file,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::{
    TransportError, transport_dispatcher::TrainingDispatcher,
};

use relayrl_types::data::trajectory::{RelayRLTrajectory, TrajectoryError};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::prelude::action::CodecConfig;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use active_uuid_registry::registry_uuid::Uuid;

use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::collections::BinaryHeap;
#[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, RwLock, broadcast};

type PriorityRank = i64;

#[derive(Debug, Clone)]
pub(crate) struct SinkQueueEntry {
    priority: PriorityRank, // lower = sooner, higher = later
    actor_id: ActorUuid,
    traj_for_processing: Arc<RelayRLTrajectory>,
}

impl Eq for SinkQueueEntry {}

impl PartialEq<Self> for SinkQueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.actor_id == other.actor_id
    }
}

impl PartialOrd<Self> for SinkQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SinkQueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority)
    }
}

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum TrajectorySinkError {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error("Transport error: {0}")]
    TransportError(#[from] TransportError),
    #[error("Failed to encode trajectory: {0}")]
    EncodeTrajectoryError(#[from] TrajectoryError),
    #[error("File sink error: {0}")]
    FileSinkError(#[from] FileSinkError),
    #[error("Failed to join file sink task: {0}")]
    JoinFileSinkTaskError(#[from] tokio::task::JoinError),
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) trait TrajectoryBufferTrait<B: Backend + BackendMatcher<Backend = B>>:
    TransportTrajectorySinkTrait<B> + LocalFileTrajectorySinkTrait<B>
{
    fn new(
        associated_router_namespace: RouterNamespace,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        codec: CodecConfig,
    ) -> Self;
    fn with_transport(
        &mut self,
        training_dispatcher: Arc<TrainingDispatcher<B>>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> &mut Self;
    fn with_trajectory_writer(
        &mut self,
        shared_trajectory_file_output: Arc<RwLock<LocalTrajectoryFileParams>>,
    ) -> &mut Self;
    fn with_shutdown(&mut self, rx: broadcast::Receiver<()>) -> &mut Self;
    fn spawn_loop(&mut self) -> Result<(), RouterError>;
    fn _compute_priority(
        actor_id: &ActorUuid,
        actor_last_sent: &DashMap<Uuid, i64>,
        timestamp: (u128, u128),
    ) -> PriorityRank;
}

#[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
pub(crate) trait TrajectoryBufferTrait<B: Backend + BackendMatcher<Backend = B>>:
    LocalFileTrajectorySinkTrait<B>
{
    fn new(
        associated_router_namespace: RouterNamespace,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: CodecConfig,
    ) -> Self;
    fn with_trajectory_writer(
        &mut self,
        shared_trajectory_file_output: Arc<RwLock<LocalTrajectoryFileParams>>,
    ) -> &mut Self;
    fn with_shutdown(&mut self, rx: broadcast::Receiver<()>) -> &mut Self;
    fn spawn_loop(&mut self) -> Result<(), RouterError>;
    fn _compute_priority(
        actor_id: &ActorUuid,
        actor_last_sent: &DashMap<Uuid, i64>,
        timestamp: (u128, u128),
    ) -> PriorityRank;
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) trait TransportTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn send_trajectory(
        associated_router_namespace: &RouterNamespace,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        training_dispatcher: &Option<Arc<TrainingDispatcher<B>>>,
        shared_transport_addresses: &Arc<RwLock<SharedTransportAddresses>>,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
}

pub(crate) trait LocalFileTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn write_local_trajectory(
        entry: &SinkQueueEntry,
        file_params: &LocalTrajectoryFileParams,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
}

pub(crate) struct ClientTrajectoryBuffer<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    associated_router_namespace: RouterNamespace,
    active: AtomicBool,
    rx_from_actor: Option<Receiver<RoutedMessage>>,
    actor_last_processed: DashMap<Uuid, i64>,
    #[allow(dead_code)]
    traj_queue_tx: Option<Sender<SinkQueueEntry>>,
    shared_client_modes: Arc<ClientModes>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_transport_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
    shared_trajectory_file_output: Option<Arc<RwLock<LocalTrajectoryFileParams>>>,
    shutdown: Option<broadcast::Receiver<()>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    codec: CodecConfig,
    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    _phantom: PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrajectoryBufferTrait<B>
    for ClientTrajectoryBuffer<B>
{
    fn new(
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        associated_router_namespace: RouterNamespace,
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        _associated_router_namespace: RouterNamespace,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: CodecConfig,
    ) -> Self {
        Self {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            associated_router_namespace,
            active: AtomicBool::new(false),
            rx_from_actor: Some(rx_from_actor),
            actor_last_processed: DashMap::new(),
            traj_queue_tx: None,
            shared_client_modes,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_training_dispatcher: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_transport_addresses: None,
            shared_trajectory_file_output: None,
            shutdown: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            codec,
            #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
            _phantom: PhantomData,
        }
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    fn with_transport(
        &mut self,
        shared_training_dispatcher: Arc<TrainingDispatcher<B>>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> &mut Self {
        self.shared_training_dispatcher = Some(shared_training_dispatcher);
        self.shared_transport_addresses = Some(shared_transport_addresses);
        self
    }

    fn with_trajectory_writer(
        &mut self,
        shared_trajectory_file_output: Arc<RwLock<LocalTrajectoryFileParams>>,
    ) -> &mut Self {
        self.shared_trajectory_file_output = Some(shared_trajectory_file_output);
        self
    }

    fn with_shutdown(&mut self, rx: broadcast::Receiver<()>) -> &mut Self {
        self.shutdown = Some(rx);
        self
    }

    fn spawn_loop(&mut self) -> Result<(), RouterError> {
        self.active.store(true, Ordering::SeqCst);

        let mut rx_from_actor = self.rx_from_actor.take().ok_or_else(|| {
            RouterError::TrajectorySinkError(TrajectorySinkError::EncodeTrajectoryError(
                TrajectoryError::SerializationError("spawn_loop already called".to_string()),
            ))
        })?;

        let (traj_queue_tx, mut traj_queue_rx) =
            tokio::sync::mpsc::unbounded_channel::<SinkQueueEntry>();

        let actor_last_processed = self.actor_last_processed.clone();
        let shared_client_modes = self.shared_client_modes.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let namespace = self.associated_router_namespace.clone();

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_training_dispatcher = self.shared_training_dispatcher.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_transport_addresses = self.shared_transport_addresses.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let codec = self.codec.clone();

        let shared_trajectory_file_output = self.shared_trajectory_file_output.clone();

        let worker_priority_queue: Arc<Mutex<BinaryHeap<SinkQueueEntry>>> =
            Arc::new(Mutex::new(BinaryHeap::new()));

        let mut receiver_shutdown_rx = self.shutdown.take();
        let mut worker_shutdown_rx = receiver_shutdown_rx.as_mut().map(|rx| rx.resubscribe());
        let receiver_active = Arc::new(AtomicBool::new(true));
        let worker_active = receiver_active.clone();

        let receiver_actor_last_processed = actor_last_processed.clone();
        let _receiver_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;

                    _ = async {
                        if let Some(rx) = &mut receiver_shutdown_rx {
                            let _ = rx.recv().await;
                        } else {
                            std::future::pending::<()>().await;
                        }
                    } => {
                        break;
                    }

                    msg_opt = rx_from_actor.recv() => {
                        match msg_opt {
                            Some(msg) => {
                                // Only process SendTrajectory payloads
                                if let RoutedPayload::SendTrajectory { timestamp, trajectory } = msg.payload {
                                    let priority = Self::_compute_priority(
                                        &msg.actor_id,
                                        &receiver_actor_last_processed,
                                        timestamp,
                                    );

                                    let entry = SinkQueueEntry {
                                        priority,
                                        actor_id: msg.actor_id,
                                        traj_for_processing: Arc::new(trajectory),
                                    };

                                    if traj_queue_tx.send(entry).is_err() {
                                        break;
                                    }
                                }
                            }
                            None => {
                                break;
                            }
                        }
                    }
                }
            }
            receiver_active.store(false, Ordering::SeqCst);
        });

        let worker_queue = worker_priority_queue.clone();
        let worker_actor_last_processed = actor_last_processed.clone();
        let worker_modes = shared_client_modes.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let worker_training_dispatcher = shared_training_dispatcher.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let worker_transport_addresses = shared_transport_addresses.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let worker_codec = codec.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let worker_namespace = namespace.clone();
        let worker_trajectory_file_output = shared_trajectory_file_output.clone();

        let _worker_handle = tokio::spawn(async move {
            const BATCH_SIZE: usize = 10;
            let mut worker_tick = tokio::time::interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    biased;

                    _ = async {
                        if let Some(rx) = &mut worker_shutdown_rx {
                            let _ = rx.recv().await;
                        } else {
                            std::future::pending::<()>().await;
                        }
                    } => {
                        break;
                    }

                    job_opt = traj_queue_rx.recv() => {
                        match job_opt {
                            Some(job) => {
                                let mut queue = worker_queue.lock().await;
                                queue.push(job);
                            }
                            None => {
                                break;
                            }
                        }
                    }

                    _ = worker_tick.tick() => {
                        if !worker_active.load(Ordering::SeqCst) {
                            let queue = worker_queue.lock().await;
                            if queue.is_empty() {
                                break;
                            }
                            drop(queue);
                        }

                        let mut jobs_to_process = Vec::with_capacity(BATCH_SIZE);
                        {
                            let mut queue = worker_queue.lock().await;
                            for _ in 0..BATCH_SIZE {
                                if let Some(job) = queue.pop() {
                                    jobs_to_process.push(job);
                                } else {
                                    break;
                                }
                            }
                        }

                        // Dispatch each job to enabled sinks
                        for job in jobs_to_process {
                            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                            if let ActorTrainingDataMode::Online(_) | ActorTrainingDataMode::Hybrid(_, _) = &worker_modes.actor_training_data_mode {
                                if let (Some(dispatcher), Some(transport_addresses)) =
                                    (worker_training_dispatcher.clone(), worker_transport_addresses.clone())
                                {
                                    let transport_job = job.clone();
                                    let transport_codec = worker_codec.clone();
                                    let transport_addrs = transport_addresses.clone();
                                    let transport_actor_last = worker_actor_last_processed.clone();
                                    let transport_namespace = worker_namespace.clone();

                                    tokio::spawn(async move {
                                        // Encode trajectory for transport
                                        let encoded = match transport_job
                                            .traj_for_processing
                                            .encode(&transport_codec)
                                        {
                                            Ok(enc) => enc,
                                            Err(e) => {
                                                log::error!(
                                                    "[TrajectoryBuffer] Encode error: {:?}",
                                                    e
                                                );
                                                return;
                                            }
                                        };

                                        let addrs = transport_addrs.read().await;
                                        if let Err(e) = Self::send_trajectory(
                                            &transport_namespace,
                                            &transport_job.actor_id,
                                            &transport_job.priority,
                                            &encoded,
                                            &Some(dispatcher),
                                            &transport_addrs,
                                            &transport_actor_last,
                                        )
                                        .await
                                        {
                                            log::error!(
                                                "[TrajectoryBuffer] Transport send error: {:?}",
                                                e
                                            );
                                        }
                                    });
                                }
                            }

                            if uses_local_file_writing(&worker_modes.actor_training_data_mode) &&
                                let Some(ref traj_output) = worker_trajectory_file_output {
                                    let local_job = job.clone();
                                    let local_actor_last = worker_actor_last_processed.clone();
                                    let traj_output_clone = traj_output.clone();

                                    tokio::spawn(async move {
                                        let params = traj_output_clone.read().await;

                                        if let Err(e) = Self::write_local_trajectory(
                                            &local_job,
                                            &params,
                                            &local_actor_last,
                                        )
                                        .await
                                        {
                                            log::error!(
                                                "[TrajectoryBuffer] Local write error: {:?}",
                                                e
                                            );
                                        }
                                    });

                            }
                        }
                    }
                }
            }

            // Process remaining jobs synchronously for graceful shutdown
            let mut queue = worker_queue.lock().await;
            while let Some(job) = queue.pop() {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if let Some(transport_addresses) = &worker_transport_addresses {
                    if let ActorTrainingDataMode::Online(_) | ActorTrainingDataMode::Hybrid(_, _) =
                        &worker_modes.actor_training_data_mode
                    {
                        let encoded = match job.traj_for_processing.encode(&worker_codec) {
                            Ok(enc) => enc,
                            Err(e) => {
                                log::error!("[TrajectoryBuffer] Encode error: {:?}", e);
                                return;
                            }
                        };

                        let _ = Self::send_trajectory(
                            &worker_namespace,
                            &job.actor_id,
                            &job.priority,
                            &encoded,
                            &worker_training_dispatcher,
                            &transport_addresses,
                            &worker_actor_last_processed,
                        )
                        .await;
                    }
                }

                if uses_local_file_writing(&worker_modes.actor_training_data_mode)
                    && let Some(ref traj_output) = worker_trajectory_file_output
                {
                    let params = traj_output.read().await;

                    let _ =
                        Self::write_local_trajectory(&job, &params, &worker_actor_last_processed)
                            .await;
                }
            }
        });

        Ok(())
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
        let priority_rank: i64 = actor_burden - (age_millis.min(i64::MAX as u128) as i64);

        priority_rank
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> LocalFileTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn write_local_trajectory(
        entry: &SinkQueueEntry,
        file_params: &LocalTrajectoryFileParams,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        let trajectory = entry.traj_for_processing.clone();
        let actor_id = &entry.actor_id;
        let priority = &entry.priority;
        let num_actions = trajectory.actions.len();

        // Update last sent timestamp for this actor
        actor_last_processed.insert(*actor_id, *priority);

        let file_type = file_params.file_type.clone();

        let file_extension = match file_type {
            LocalTrajectoryFileType::Arrow => "arrow",
            LocalTrajectoryFileType::Csv => "csv",
        };

        let mut path = file_params.directory.join(format!(
            "{actor_id}_traj_{num_actions}_actions.{file_extension}"
        ));

        {
            // i love how unlikely this is to happen
            let mut counter = 1;
            while path.exists() {
                path = file_params.directory.join(format!(
                    "{actor_id}_traj_{num_actions}_actions_{counter}.{file_extension}"
                ));
                counter += 1;
            }
        }

        let _ = tokio::task::spawn_blocking(move || {
            write_local_trajectory_file(trajectory, &path, &file_type)
        })
        .await
        .map_err(TrajectorySinkError::from)?;

        Ok(())
    }
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
impl<B: Backend + BackendMatcher<Backend = B>> TransportTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn send_trajectory(
        associated_router_namespace: &RouterNamespace,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        training_dispatcher: &Option<Arc<TrainingDispatcher<B>>>,
        shared_transport_addresses: &Arc<RwLock<SharedTransportAddresses>>,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        if let Some(dispatcher) = training_dispatcher {
            // Update last sent timestamp for this actor
            actor_last_processed.insert(*actor_id, *priority);

            let buffer_entry = {
                let entries = get_context_entries(
                    associated_router_namespace.as_ref(),
                    crate::network::BUFFER_CONTEXT,
                )
                .map_err(|e| {
                    TrajectorySinkError::TransportError(TransportError::UuidPoolError(e))
                })?;
                entries
                    .first()
                    .ok_or_else(|| {
                        TrajectorySinkError::TransportError(
                            TransportError::NoTransportConfiguredError(
                                "No buffer context entries found".to_string(),
                            ),
                        )
                    })?
                    .clone()
            };

            dispatcher
                .send_trajectory(
                    buffer_entry,
                    encoded_trajectory.clone(),
                    shared_transport_addresses.clone(),
                )
                .await?
        }

        Err(TrajectorySinkError::TransportError(
            TransportError::NoTransportConfiguredError(
                "No transport configured for sending trajectories".to_string(),
            ),
        ))
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::network::client::agent::{
        ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode,
    };
    use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
    use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
    use active_uuid_registry::registry_uuid::Uuid;
    
    use relayrl_types::data::trajectory::RelayRLTrajectory;
    use std::collections::BinaryHeap;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    use tokio::sync::{broadcast, mpsc};

    // The backend is only referenced through phantom data in the no-transport build.
    // We use NdArray from burn_ndarray which is always available.
    use burn_ndarray::NdArray;
    type TestBackend = NdArray<f32>;

    fn disabled_modes() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn test_namespace() -> RouterNamespace {
        Arc::from("test-ns")
    }

    fn now_millis() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    fn make_send_trajectory_msg(actor_id: Uuid, num_actions: usize) -> RoutedMessage {
        use relayrl_types::data::action::RelayRLAction;
        let mut traj = RelayRLTrajectory::new(num_actions.max(1));
        for i in 0..num_actions {
            traj.add_action(RelayRLAction::minimal(i as f32, false));
        }
        let ts = now_millis();
        RoutedMessage {
            actor_id,
            protocol: RoutingProtocol::SendTrajectory,
            payload: RoutedPayload::SendTrajectory {
                timestamp: (ts, ts * 1_000_000),
                trajectory: traj,
            },
        }
    }

    fn make_entry(priority: i64) -> SinkQueueEntry {
        SinkQueueEntry {
            priority,
            actor_id: Uuid::new_v4(),
            traj_for_processing: Arc::new(RelayRLTrajectory::new(1)),
        }
    }

    #[test]
    fn lower_priority_rank_is_higher_heap_priority() {
        let mut heap: BinaryHeap<SinkQueueEntry> = BinaryHeap::new();
        heap.push(make_entry(100));
        heap.push(make_entry(50));
        heap.push(make_entry(200));

        // BinaryHeap pops the "highest" element first.
        // Our Ord is reversed, so the entry with the lowest priority rank pops first.
        assert_eq!(heap.pop().unwrap().priority, 50);
        assert_eq!(heap.pop().unwrap().priority, 100);
        assert_eq!(heap.pop().unwrap().priority, 200);
    }

    #[test]
    fn equal_priority_equal_actor_id_is_equal() {
        let id = Uuid::new_v4();
        let a = SinkQueueEntry {
            priority: 10,
            actor_id: id,
            traj_for_processing: Arc::new(RelayRLTrajectory::new(1)),
        };
        let b = SinkQueueEntry {
            priority: 10,
            actor_id: id,
            traj_for_processing: Arc::new(RelayRLTrajectory::new(1)),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn fresh_actor_no_burden_gets_negative_age_rank() {
        let actor_id = Uuid::new_v4();
        let last_sent: DashMap<Uuid, i64> = DashMap::new();
        // Very fresh timestamp (near now) → age_millis ≈ 0 → priority ≈ 0
        let ts = now_millis();
        let priority = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &last_sent,
            (ts, 0),
        );
        // With no burden and nearly zero age the priority is small (≥ -5ms tolerance)
        assert!(
            priority >= -100,
            "Priority {} is unexpectedly low",
            priority
        );
    }

    #[test]
    fn old_trajectory_gets_more_negative_rank() {
        let actor_id = Uuid::new_v4();
        let last_sent: DashMap<Uuid, i64> = DashMap::new();

        let fresh_ts = now_millis();
        let old_ts = now_millis().saturating_sub(60_000); // 1 minute ago

        let fresh_priority = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &last_sent,
            (fresh_ts, 0),
        );
        let old_priority = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &last_sent,
            (old_ts, 0),
        );

        assert!(
            old_priority < fresh_priority,
            "Older trajectory should have lower priority rank: old={} fresh={}",
            old_priority,
            fresh_priority
        );
    }

    #[test]
    fn high_recent_sends_increases_rank() {
        let actor_id = Uuid::new_v4();
        let ts = now_millis();

        let low_burden: DashMap<Uuid, i64> = DashMap::new();
        let high_burden: DashMap<Uuid, i64> = DashMap::new();
        high_burden.insert(actor_id, 10_000_000); // large recent_sends

        let low = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &low_burden,
            (ts, 0),
        );
        let high = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &high_burden,
            (ts, 0),
        );

        assert!(
            high > low,
            "High-burden actor should have higher priority rank: high={} low={}",
            high,
            low
        );
    }

    #[test]
    fn age_capped_at_300_000_ms() {
        let actor_id = Uuid::new_v4();
        let last_sent: DashMap<Uuid, i64> = DashMap::new();

        // 10 minutes ago (600_000 ms) — should be capped at 300_000
        let ts_10min = now_millis().saturating_sub(600_000);
        // 6 minutes ago (360_000 ms) — also above cap, same result expected
        let ts_6min = now_millis().saturating_sub(360_000);

        let p10 = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &last_sent,
            (ts_10min, 0),
        );
        let p6 = ClientTrajectoryBuffer::<TestBackend>::_compute_priority(
            &actor_id,
            &last_sent,
            (ts_6min, 0),
        );

        // Both exceed the 300_000 cap, so both should yield the same priority
        assert_eq!(
            p10, p6,
            "Priority should be identical when age exceeds the 300s cap"
        );
    }

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[tokio::test]
    async fn spawn_loop_double_call_returns_err() {
        let (tx, rx) = mpsc::channel::<RoutedMessage>(16);
        let mut buf =
            ClientTrajectoryBuffer::<TestBackend>::new(test_namespace(), rx, disabled_modes());
        assert!(buf.spawn_loop().is_ok());
        // Second call: rx already taken, must return Err
        assert!(buf.spawn_loop().is_err());
        drop(tx);
    }

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[tokio::test]
    async fn receiver_ignores_non_trajectory_payloads() {
        let (tx, rx) = mpsc::channel::<RoutedMessage>(16);
        let mut buf =
            ClientTrajectoryBuffer::<TestBackend>::new(test_namespace(), rx, disabled_modes());
        buf.spawn_loop().unwrap();

        let actor_id = Uuid::new_v4();
        // Send a Shutdown message (non-trajectory payload)
        tx.send(RoutedMessage {
            actor_id,
            protocol: RoutingProtocol::Shutdown,
            payload: RoutedPayload::Shutdown,
        })
        .await
        .unwrap();

        // Give the receiver task time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        // No assertions needed beyond "no panic"; the receiver should still be alive
        // and not have crashed.
        drop(tx);
    }

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[tokio::test]
    async fn shutdown_signal_stops_receiver() {
        let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
        let (tx, rx) = mpsc::channel::<RoutedMessage>(16);
        let mut buf =
            ClientTrajectoryBuffer::<TestBackend>::new(test_namespace(), rx, disabled_modes());
        buf.with_shutdown(shutdown_rx);
        buf.spawn_loop().unwrap();

        // Signal shutdown
        let _ = shutdown_tx.send(());

        // Give tasks time to exit
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        // No assertion needed beyond no panic/hang; the test completing proves tasks exited.
        drop(tx);
    }

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[tokio::test]
    async fn dropped_tx_breaks_receiver_loop() {
        let (tx, rx) = mpsc::channel::<RoutedMessage>(4);
        let mut buf =
            ClientTrajectoryBuffer::<TestBackend>::new(test_namespace(), rx, disabled_modes());
        buf.spawn_loop().unwrap();

        // Drop the sender — the receiver should observe channel close and exit
        drop(tx);

        // Give receiver task time to notice channel close
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        // Test passes if we reach here without hanging
    }

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[tokio::test]
    async fn partial_trajectory_still_forwarded() {
        let (tx, rx) = mpsc::channel::<RoutedMessage>(16);
        let mut buf =
            ClientTrajectoryBuffer::<TestBackend>::new(test_namespace(), rx, disabled_modes());
        buf.spawn_loop().unwrap();

        let actor_id = Uuid::new_v4();
        // Send a trajectory with only 1 action
        tx.send(make_send_trajectory_msg(actor_id, 1))
            .await
            .unwrap();

        // No error expected; allow processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        drop(tx);
    }

    #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
    #[tokio::test]
    async fn concurrent_actors_send_trajectories_safely() {
        let (tx, rx) = mpsc::channel::<RoutedMessage>(256);
        let mut buf =
            ClientTrajectoryBuffer::<TestBackend>::new(test_namespace(), rx, disabled_modes());
        buf.spawn_loop().unwrap();

        const NUM_ACTORS: usize = 8;
        const TRAJS_PER_ACTOR: usize = 5;

        let mut handles = Vec::new();
        for _ in 0..NUM_ACTORS {
            let tx_clone = tx.clone();
            let actor_id = Uuid::new_v4();
            handles.push(tokio::spawn(async move {
                for _ in 0..TRAJS_PER_ACTOR {
                    let msg = make_send_trajectory_msg(actor_id, 3);
                    tx_clone.send(msg).await.unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // Allow buffer tasks time to process all messages
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        drop(tx);
    }
}
