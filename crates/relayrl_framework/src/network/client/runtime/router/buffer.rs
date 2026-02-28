use super::{RoutedMessage, RoutedPayload, RouterError};
use crate::network::client::agent::LocalTrajectoryFileParams;
use crate::network::client::agent::LocalTrajectoryFileType;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::agent::ModelMode;
use crate::network::client::agent::{ActorTrainingDataMode, ClientModes};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::data::file_sink::{
    FileSinkError, write_local_trajectory_file,
};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::{
    TransportError, transport_dispatcher::TrainingDispatcher,
};

use relayrl_types::data::tensor::{DType, NdArrayDType, TchDType, TensorData};
use relayrl_types::data::trajectory::{EncodedTrajectory, RelayRLTrajectory, TrajectoryError};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_types::prelude::action::CodecConfig;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use arrow::array::BinaryBuilder;
use arrow::array::{ArrayRef, BooleanArray, Float32Array, StringArray, UInt64Array};
use arrow::array::{Float32Builder, Float64Builder, ListBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::collections::BinaryHeap;
use std::fs::File;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, RwLock, broadcast};
use uuid::Uuid;

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
        associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        codec: CodecConfig,
    ) -> Self;
    fn with_transport(
        &mut self,
        training_dispatcher: Arc<TrainingDispatcher<B>>,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
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
        associated_router_id: RouterUuid,
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
        associated_router_id: &RouterUuid,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        training_dispatcher: &Option<Arc<TrainingDispatcher<B>>>,
        shared_server_addresses: &Arc<RwLock<SharedTransportAddresses>>,
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
    associated_router_id: RouterUuid,
    active: AtomicBool,
    rx_from_actor: Option<Receiver<RoutedMessage>>,
    actor_last_processed: DashMap<Uuid, i64>,
    #[allow(dead_code)]
    traj_queue_tx: Option<Sender<SinkQueueEntry>>,
    shared_client_modes: Arc<ClientModes>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_server_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
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
        associated_router_id: RouterUuid,
        #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
        _associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))] codec: CodecConfig,
    ) -> Self {
        Self {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            associated_router_id,
            active: AtomicBool::new(false),
            rx_from_actor: Some(rx_from_actor),
            actor_last_processed: DashMap::new(),
            traj_queue_tx: None,
            shared_client_modes,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_training_dispatcher: None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_server_addresses: None,
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
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> &mut Self {
        self.shared_training_dispatcher = Some(shared_training_dispatcher);
        self.shared_server_addresses = Some(shared_server_addresses);
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
        let identity = self.associated_router_id;

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_training_dispatcher = self.shared_training_dispatcher.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let shared_server_addresses = self.shared_server_addresses.clone();
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
        let worker_server_addresses = shared_server_addresses.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let worker_codec = codec.clone();
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let worker_identity = identity;
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
                                if let (Some(dispatcher), Some(server_addresses)) =
                                    (worker_training_dispatcher.clone(), worker_server_addresses.clone())
                                {
                                    let transport_job = job.clone();
                                    let transport_codec = worker_codec.clone();
                                    let transport_addrs = server_addresses.clone();
                                    let transport_actor_last = worker_actor_last_processed.clone();
                                    let transport_identity = worker_identity;

                                    tokio::spawn(async move {
                                        // Encode trajectory for transport
                                        let encoded = match transport_job
                                            .traj_for_processing
                                            .encode(&transport_codec)
                                        {
                                            Ok(enc) => enc,
                                            Err(e) => {
                                                eprintln!(
                                                    "[TrajectoryBuffer] Encode error: {:?}",
                                                    e
                                                );
                                                return;
                                            }
                                        };

                                        let addrs = transport_addrs.read().await;
                                        if let Err(e) = Self::send_trajectory(
                                            &transport_identity,
                                            &transport_job.actor_id,
                                            &transport_job.priority,
                                            &encoded,
                                            &Some(dispatcher),
                                            &transport_addrs,
                                            &transport_actor_last,
                                        )
                                        .await
                                        {
                                            eprintln!(
                                                "[TrajectoryBuffer] Transport send error: {:?}",
                                                e
                                            );
                                        }
                                    });
                                }
                            }

                            if let ActorTrainingDataMode::Offline(_) | ActorTrainingDataMode::Hybrid(_, _) = &worker_modes.actor_training_data_mode {
                                if let Some(ref traj_output) = worker_trajectory_file_output {
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
                                            eprintln!(
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
            }

            // Process remaining jobs synchronously for graceful shutdown
            let mut queue = worker_queue.lock().await;
            while let Some(job) = queue.pop() {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if let Some(server_addresses) = &worker_server_addresses {
                    if let ActorTrainingDataMode::Online(_) | ActorTrainingDataMode::Hybrid(_, _) =
                        &worker_modes.actor_training_data_mode
                    {
                        let encoded = match job.traj_for_processing.encode(&worker_codec) {
                            Ok(enc) => enc,
                            Err(e) => {
                                eprintln!("[TrajectoryBuffer] Encode error: {:?}", e);
                                return;
                            }
                        };

                        let _ = Self::send_trajectory(
                            &worker_identity,
                            &job.actor_id,
                            &job.priority,
                            &encoded,
                            &worker_training_dispatcher,
                            &server_addresses,
                            &worker_actor_last_processed,
                        )
                        .await;
                    }
                }

                if let ActorTrainingDataMode::Offline(_) | ActorTrainingDataMode::Hybrid(_, _) =
                    &worker_modes.actor_training_data_mode
                {
                    if let Some(ref traj_output) = worker_trajectory_file_output {
                        let params = traj_output.read().await;

                        let _ = Self::write_local_trajectory(
                            &job,
                            &params,
                            &worker_actor_last_processed,
                        )
                        .await;
                    }
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

        tokio::task::spawn_blocking(move || {
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
        associated_router_id: &RouterUuid,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        training_dispatcher: &Option<Arc<TrainingDispatcher<B>>>,
        shared_server_addresses: &Arc<RwLock<SharedTransportAddresses>>,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        if let Some(dispatcher) = training_dispatcher {
            // Update last sent timestamp for this actor
            actor_last_processed.insert(*actor_id, *priority);

            dispatcher
                .send_trajectory(
                    associated_router_id,
                    encoded_trajectory.clone(),
                    shared_server_addresses.clone(),
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
mod tests {}
