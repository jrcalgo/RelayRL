use super::{RoutedMessage, RoutedPayload, RouterError};
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::agent::ActorServerModelMode;
use crate::network::client::agent::ClientCapabilities;
#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
use crate::network::client::agent::DatabaseTypeParams;
use crate::network::client::agent::TrajectoryFileParams;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::data::transport::{TransportClient, TransportError};

use relayrl_types::prelude::BackendMatcher;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use relayrl_types::prelude::CodecConfig;
use relayrl_types::types::data::tensor::{DType, NdArrayDType, TchDType, TensorData};
use relayrl_types::types::data::trajectory::{RelayRLTrajectory, TrajectoryError};

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
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[error("Transport error: {0}")]
    TransportError(#[from] TransportError),
    #[error("Failed to encode trajectory: {0}")]
    EncodeTrajectoryError(#[from] TrajectoryError),
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    #[error("Failed to write to database: {0}")]
    DatabaseWriteError(String),
}

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
pub(crate) trait TrajectoryBufferTrait<B: Backend + BackendMatcher<Backend = B>>:
    TransportTrajectorySinkTrait<B> + PersistentTrajectoryDataSinkTrait<B>
{
    fn new(
        associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_capabilities: Arc<ClientCapabilities>,
        codec: CodecConfig,
    ) -> Self;
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    fn with_database(&mut self, database_params: DatabaseTypeParams) -> &mut Self;
    fn with_transport(
        &mut self,
        transport: Arc<TransportClient<B>>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> &mut Self;
    fn with_trajectory_writer(
        &mut self,
        shared_trajectory_file_output: Arc<RwLock<FormattedTrajectoryFileParams>>,
    ) -> &mut Self;
    fn with_shutdown(&mut self, rx: broadcast::Receiver<()>) -> &mut Self;
    fn spawn_loop(&mut self) -> Result<(), RouterError>;
    fn _compute_priority(
        actor_id: &ActorUuid,
        actor_last_sent: &DashMap<Uuid, i64>,
        timestamp: (u128, u128),
    ) -> PriorityRank;
}

#[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
pub(crate) trait TrajectoryBufferTrait<B: Backend + BackendMatcher<Backend = B>>:
    PersistentTrajectoryDataSinkTrait<B>
{
    fn new(
        associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_capabilities: Arc<ClientCapabilities>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] codec: CodecConfig,
    ) -> Self;
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    fn with_database(&mut self, database_params: DatabaseTypeParams) -> &mut Self;
    fn with_trajectory_writer(
        &mut self,
        shared_trajectory_file_output: Arc<RwLock<TrajectoryFileParams>>,
    ) -> &mut Self;
    fn with_shutdown(&mut self, rx: broadcast::Receiver<()>) -> &mut Self;
    fn spawn_loop(&mut self) -> Result<(), RouterError>;
    fn _compute_priority(
        actor_id: &ActorUuid,
        actor_last_sent: &DashMap<Uuid, i64>,
        timestamp: (u128, u128),
    ) -> PriorityRank;
}

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
trait TransportTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn send_trajectory(
        associated_router_id: &RouterUuid,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
        transport: &Option<Arc<TransportClient<B>>>,
        actor_last_sent: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
    async fn retry_send(
        associated_router_id: &RouterUuid,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
        transport: &Option<Arc<TransportClient<B>>>,
        actor_last_sent: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
}

#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
pub(crate) trait PersistentTrajectoryDataSinkTrait<B: Backend + BackendMatcher<Backend = B>>:
    DatabaseTrajectorySinkTrait<B> + LocalTrajectorySinkTrait<B>
{
}

#[cfg(not(any(feature = "postgres_db", feature = "sqlite_db")))]
pub(crate) trait PersistentTrajectoryDataSinkTrait<B: Backend + BackendMatcher<Backend = B>>:
    LocalTrajectorySinkTrait<B>
{
}

#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
pub(crate) trait DatabaseTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn write_database_trajectory(
        entry: &SinkQueueEntry,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
    async fn retry_write_database(
        entry: &SinkQueueEntry,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
}

pub(crate) trait LocalTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn write_local_trajectory(
        entry: &SinkQueueEntry,
        actor_last_processed: &DashMap<Uuid, i64>,
        output_directory: &std::path::Path,
    ) -> Result<(), TrajectorySinkError>;
}

pub(crate) struct ClientTrajectoryBuffer<B: Backend + BackendMatcher<Backend = B>> {
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    associated_router_id: RouterUuid,
    active: AtomicBool,
    rx_from_actor: Option<Receiver<RoutedMessage>>,
    actor_last_processed: DashMap<Uuid, i64>,
    #[allow(dead_code)]
    traj_queue_tx: Option<Sender<SinkQueueEntry>>,
    shared_client_capabilities: Arc<ClientCapabilities>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    shared_transport: Option<Arc<TransportClient<B>>>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    shared_server_addresses: Option<Arc<RwLock<ServerAddresses>>>,
    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    database_params: Option<DatabaseTypeParams>,
    shared_trajectory_file_output: Option<Arc<RwLock<TrajectoryFileParams>>>,
    shutdown: Option<broadcast::Receiver<()>>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    codec: CodecConfig,
    #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
    _phantom: PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrajectoryBufferTrait<B>
    for ClientTrajectoryBuffer<B>
{
    fn new(
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        associated_router_id: RouterUuid,
        #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
        _associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        shared_client_capabilities: Arc<ClientCapabilities>,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))] codec: CodecConfig,
    ) -> Self {
        Self {
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            associated_router_id,
            active: AtomicBool::new(false),
            rx_from_actor: Some(rx_from_actor),
            actor_last_processed: DashMap::new(),
            traj_queue_tx: None,
            shared_client_capabilities,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_transport: None,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            shared_server_addresses: None,
            #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
            database_params: None,
            shared_trajectory_file_output: None,
            shutdown: None,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            codec,
            #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
            _phantom: PhantomData,
        }
    }

    #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
    fn with_database(&mut self, database_params: DatabaseTypeParams) -> &mut Self {
        self.database_params = Some(database_params);
        self
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    fn with_transport(
        &mut self,
        transport: Arc<TransportClient<B>>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> &mut Self {
        self.shared_transport = Some(transport);
        self.shared_server_addresses = Some(shared_server_addresses);
        self
    }

    fn with_trajectory_writer(
        &mut self,
        shared_trajectory_file_output: Arc<RwLock<TrajectoryFileParams>>,
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
        let shared_client_capabilities = self.shared_client_capabilities.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let identity = self.associated_router_id;

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_transport = self.shared_transport.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let shared_server_addresses = self.shared_server_addresses.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
        let worker_caps = shared_client_capabilities.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let worker_transport = shared_transport.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let worker_server_addresses = shared_server_addresses.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        let worker_codec = codec.clone();
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
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
                            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                            if worker_caps.training_server_mode != ActorServerModelMode::Disabled {
                                if let (Some(transport), Some(server_addrs)) =
                                    (&worker_transport, &worker_server_addresses)
                                {
                                    let transport_job = job.clone();
                                    let transport_codec = worker_codec.clone();
                                    let transport_client = transport.clone();
                                    let transport_addrs = server_addrs.clone();
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
                                            &addrs.model_server_address,
                                            &addrs.trajectory_server_address,
                                            &Some(transport_client),
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

                            #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
                            if worker_caps.database_trajectory_persistence {
                                let db_job = job.clone();
                                let db_actor_last = worker_actor_last_processed.clone();

                                tokio::spawn(async move {
                                    if let Err(e) =
                                        Self::write_database_trajectory(&db_job, &db_actor_last)
                                            .await
                                    {
                                        eprintln!(
                                            "[TrajectoryBuffer] Database write error: {:?}",
                                            e
                                        );
                                    }
                                });
                            }

                            if worker_caps.local_trajectory_persistence {
                                if let Some(ref traj_output) = worker_trajectory_file_output {
                                    let local_job = job.clone();
                                    let local_actor_last = worker_actor_last_processed.clone();
                                    let traj_output_clone = traj_output.clone();

                                    tokio::spawn(async move {
                                        let params = traj_output_clone.read().await;
                                        if params.enabled {
                                            // Ensure directory exists
                                            if let Err(e) = resolve_trajectory_directory(&params.directory) {
                                                eprintln!(
                                                    "[TrajectoryBuffer] Directory resolve error: {:?}",
                                                    e
                                                );
                                                return;
                                            }
                                            if let Err(e) = Self::write_local_trajectory(
                                                &local_job,
                                                &local_actor_last,
                                                &params.directory,
                                            )
                                            .await
                                            {
                                                eprintln!(
                                                    "[TrajectoryBuffer] Local write error: {:?}",
                                                    e
                                                );
                                            }
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
                #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
                if worker_caps.training_server_mode != ActorServerModelMode::Disabled {
                    let _ = Self::send_trajectory(
                        &worker_identity,
                        &job.actor_id,
                        &job.priority,
                        &job.traj_for_processing,
                        &worker_transport,
                        &worker_server_addresses,
                    )
                    .await;
                }

                #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
                if worker_caps.database_trajectory_persistence {
                    let _ =
                        Self::write_database_trajectory(&job, &worker_actor_last_processed).await;
                }

                if worker_caps.local_trajectory_persistence {
                    if let Some(ref traj_output) = worker_trajectory_file_output {
                        let params = traj_output.read().await;
                        if params.enabled {
                            if let Err(e) = resolve_trajectory_directory(&params.directory) {
                                eprintln!(
                                    "[TrajectoryBuffer] Directory resolve error during shutdown: {:?}",
                                    e
                                );
                                continue;
                            }
                            let _ = Self::write_local_trajectory(
                                &job,
                                &worker_actor_last_processed,
                                &params.directory,
                            )
                            .await;
                        }
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

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
impl<B: Backend + BackendMatcher<Backend = B>> TransportTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn send_trajectory(
        associated_router_id: &RouterUuid,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
        transport: &Option<Arc<TransportClient<B>>>,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        if let Some(transport) = transport {
            // Update last sent timestamp for this actor
            actor_last_processed.insert(*actor_id, *priority);

            // Send trajectory via transport
            match &**transport {
                #[cfg(feature = "sync_transport")]
                TransportClient::Sync(sync_client) => {
                    sync_client
                        .send_traj_to_server(
                            associated_router_id,
                            encoded_trajectory.clone(),
                            model_server_address,
                            trajectory_server_address,
                        )
                        .map_err(TrajectorySinkError::TransportError)?;
                    Ok(())
                }
                #[cfg(feature = "async_transport")]
                TransportClient::Async(async_client) => {
                    async_client
                        .send_traj_to_server(
                            associated_router_id,
                            encoded_trajectory.clone(),
                            model_server_address,
                            trajectory_server_address,
                        )
                        .await
                        .map_err(TrajectorySinkError::TransportError)?;
                    Ok(())
                }
            }
        } else {
            Err(TrajectorySinkError::TransportError(
                TransportError::NoTransportConfiguredError(
                    "No transport configured for sending trajectories".to_string(),
                ),
            ))
        }
    }

    async fn retry_send(
        associated_router_id: &RouterUuid,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        encoded_trajectory: &EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
        transport: &Option<Arc<TransportClient<B>>>,
        actor_last_sent: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        const MAX_RETRIES: u32 = 3;
        const BASE_DELAY_MS: u64 = 100;

        let Some(transport) = transport else {
            return Err(TrajectorySinkError::TransportError(
                TransportError::NoTransportConfiguredError(
                    "No transport configured for sending trajectories".to_string(),
                ),
            ));
        };

        for attempt in 0..MAX_RETRIES {
            // Exponential backoff: 100ms, 200ms, 400ms
            let delay = Duration::from_millis(BASE_DELAY_MS * (1 << attempt));
            tokio::time::sleep(delay).await;

            let result = match &**transport {
                #[cfg(feature = "sync_transport")]
                TransportClient::Sync(sync_client) => sync_client
                    .send_traj_to_server(
                        associated_router_id,
                        encoded_trajectory.clone(),
                        model_server_address,
                        trajectory_server_address,
                    )
                    .map_err(TrajectorySinkError::TransportError),
                #[cfg(feature = "async_transport")]
                TransportClient::Async(async_client) => async_client
                    .send_traj_to_server(
                        associated_router_id,
                        encoded_trajectory.clone(),
                        model_server_address,
                        trajectory_server_address,
                    )
                    .await
                    .map_err(TrajectorySinkError::TransportError),
            };

            match result {
                Ok(()) => {
                    // Update last sent timestamp on success
                    actor_last_sent.insert(*actor_id, *priority);
                    return Ok(());
                }
                Err(e) => {
                    if attempt == MAX_RETRIES - 1 {
                        eprintln!(
                            "[ClientTrajectoryBuffer] Retry send exhausted after {} attempts: {}",
                            MAX_RETRIES, e
                        );
                        return Err(e);
                    }
                    eprintln!(
                        "[ClientTrajectoryBuffer] Retry send attempt {}/{} failed: {}",
                        attempt + 1,
                        MAX_RETRIES,
                        e
                    );
                }
            }
        }

        // Unreachable, but satisfies the compiler
        Err(TrajectorySinkError::TransportError(
            TransportError::NoTransportConfiguredError("Retry exhausted".to_string()),
        ))
    }
}

#[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
impl<B: Backend + BackendMatcher<Backend = B>> DatabaseTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn write_database_trajectory(
        entry: &SinkQueueEntry,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        // TODO: Implement database trajectory writing
        Ok(())
    }

    async fn retry_write_database(
        entry: &SinkQueueEntry,
        actor_last_processed: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        // TODO: Implement database trajectory retry writing
        Ok(())
    }
}

struct TensorArrowData {
    dtype_str: String,
    shape: Vec<u64>,
    f32_data: Option<Vec<f32>>,
    f64_data: Option<Vec<f64>>,
    binary_data: Option<Vec<u8>>,
}

fn tensor_to_arrow_data(tensor: &TensorData) -> TensorArrowData {
    let dtype_str = tensor.dtype.to_string();
    let shape: Vec<u64> = tensor.shape.iter().map(|&s| s as u64).collect();

    match &tensor.dtype {
        DType::NdArray(NdArrayDType::F32) => {
            let floats: Vec<f32> = tensor
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            TensorArrowData {
                dtype_str,
                shape,
                f32_data: Some(floats),
                f64_data: None,
                binary_data: None,
            }
        }
        DType::NdArray(NdArrayDType::F64) => {
            let floats: Vec<f64> = tensor
                .data
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect();
            TensorArrowData {
                dtype_str,
                shape,
                f32_data: None,
                f64_data: Some(floats),
                binary_data: None,
            }
        }
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F32) => {
            let floats: Vec<f32> = tensor
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            TensorArrowData {
                dtype_str,
                shape,
                f32_data: Some(floats),
                f64_data: None,
                binary_data: None,
            }
        }
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F64) => {
            let floats: Vec<f64> = tensor
                .data
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect();
            TensorArrowData {
                dtype_str,
                shape,
                f32_data: None,
                f64_data: Some(floats),
                binary_data: None,
            }
        }
        _ => TensorArrowData {
            dtype_str,
            shape,
            f32_data: None,
            f64_data: None,
            binary_data: Some(tensor.data.clone()),
        },
    }
}

fn get_backend_str(trajectory: &RelayRLTrajectory) -> String {
    trajectory
        .actions
        .iter()
        .find_map(|a| a.get_obs().map(|t| format!("{:?}", t.supported_backend)))
        .unwrap_or_else(|| "None".to_string())
}

fn build_f32_list_array(data: Vec<Option<Vec<f32>>>) -> arrow::array::ArrayRef {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for item in data {
        match item {
            Some(values) => {
                for v in values {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_f64_list_array(data: Vec<Option<Vec<f64>>>) -> arrow::array::ArrayRef {
    let mut builder = ListBuilder::new(Float64Builder::new());
    for item in data {
        match item {
            Some(values) => {
                for v in values {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_shape_list_array(data: Vec<Option<Vec<u64>>>) -> arrow::array::ArrayRef {
    let mut builder = ListBuilder::new(UInt64Builder::new());
    for item in data {
        match item {
            Some(values) => {
                for v in values {
                    builder.values().append_value(v);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_binary_array(data: Vec<Option<Vec<u8>>>) -> arrow::array::ArrayRef {
    let mut builder = BinaryBuilder::new();
    for item in data {
        match item {
            Some(bytes) => builder.append_value(&bytes),
            None => builder.append_null(),
        }
    }
    Arc::new(builder.finish())
}

/// Resolves the trajectory output directory, creating it if it doesn't exist.
fn resolve_trajectory_directory(directory: &std::path::Path) -> Result<(), TrajectorySinkError> {
    if !directory.exists() {
        std::fs::create_dir_all(directory).map_err(|e| {
            TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                format!(
                    "Failed to create trajectory directory '{}': {}",
                    directory.display(),
                    e
                ),
            ))
        })?;
    }
    Ok(())
}

impl<B: Backend + BackendMatcher<Backend = B>> LocalTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn write_local_trajectory(
        entry: &SinkQueueEntry,
        actor_last_processed: &DashMap<Uuid, i64>,
        output_directory: &std::path::Path,
    ) -> Result<(), TrajectorySinkError> {
        let trajectory = entry.traj_for_processing.clone();
        let actor_id = entry.actor_id;
        let priority = entry.priority;
        let num_actions = trajectory.actions.len();

        if num_actions == 0 {
            return Ok(());
        }

        let backend_str = get_backend_str(&trajectory);
        let mut backends: Vec<String> = Vec::with_capacity(num_actions);
        let mut rewards: Vec<f32> = Vec::with_capacity(num_actions);
        let mut dones: Vec<bool> = Vec::with_capacity(num_actions);
        let mut timestamps: Vec<u64> = Vec::with_capacity(num_actions);
        let mut agent_ids: Vec<Option<String>> = Vec::with_capacity(num_actions);

        // Observation tensor columns
        let mut obs_dtypes: Vec<Option<String>> = Vec::with_capacity(num_actions);
        let mut obs_shapes: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_actions);
        let mut obs_f32: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_actions);
        let mut obs_f64: Vec<Option<Vec<f64>>> = Vec::with_capacity(num_actions);
        let mut obs_binary: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_actions);

        // Action tensor columns
        let mut act_dtypes: Vec<Option<String>> = Vec::with_capacity(num_actions);
        let mut act_shapes: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_actions);
        let mut act_f32: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_actions);
        let mut act_f64: Vec<Option<Vec<f64>>> = Vec::with_capacity(num_actions);
        let mut act_binary: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_actions);

        // Mask tensor columns
        let mut mask_dtypes: Vec<Option<String>> = Vec::with_capacity(num_actions);
        let mut mask_shapes: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_actions);
        let mut mask_f32: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_actions);
        let mut mask_f64: Vec<Option<Vec<f64>>> = Vec::with_capacity(num_actions);
        let mut mask_binary: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_actions);

        for action in trajectory.actions.iter() {
            backends.push(backend_str.clone());
            rewards.push(action.get_rew());
            dones.push(action.get_done());
            timestamps.push(action.get_timestamp());
            agent_ids.push(action.get_agent_id().map(|id| id.to_string()));

            if let Some(obs) = action.get_obs() {
                let arrow_data = tensor_to_arrow_data(obs);
                obs_dtypes.push(Some(arrow_data.dtype_str));
                obs_shapes.push(Some(arrow_data.shape));
                obs_f32.push(arrow_data.f32_data);
                obs_f64.push(arrow_data.f64_data);
                obs_binary.push(arrow_data.binary_data);
            } else {
                obs_dtypes.push(None);
                obs_shapes.push(None);
                obs_f32.push(None);
                obs_f64.push(None);
                obs_binary.push(None);
            }

            if let Some(act) = action.get_act() {
                let arrow_data = tensor_to_arrow_data(act);
                act_dtypes.push(Some(arrow_data.dtype_str));
                act_shapes.push(Some(arrow_data.shape));
                act_f32.push(arrow_data.f32_data);
                act_f64.push(arrow_data.f64_data);
                act_binary.push(arrow_data.binary_data);
            } else {
                act_dtypes.push(None);
                act_shapes.push(None);
                act_f32.push(None);
                act_f64.push(None);
                act_binary.push(None);
            }

            if let Some(mask) = action.get_mask() {
                let arrow_data = tensor_to_arrow_data(mask);
                mask_dtypes.push(Some(arrow_data.dtype_str));
                mask_shapes.push(Some(arrow_data.shape));
                mask_f32.push(arrow_data.f32_data);
                mask_f64.push(arrow_data.f64_data);
                mask_binary.push(arrow_data.binary_data);
            } else {
                mask_dtypes.push(None);
                mask_shapes.push(None);
                mask_f32.push(None);
                mask_f64.push(None);
                mask_binary.push(None);
            }
        }

        let file_path = output_directory.join(format!(
            "trajectory_{}_{}.arrow",
            actor_id, trajectory.timestamp
        ));

        tokio::task::spawn_blocking(move || -> Result<(), TrajectorySinkError> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("backend", DataType::Utf8, false),
                Field::new("reward", DataType::Float32, false),
                Field::new("done", DataType::Boolean, false),
                Field::new("timestamp", DataType::UInt64, false),
                Field::new("agent_id", DataType::Utf8, true),
                // Observation columns
                Field::new("obs_dtype", DataType::Utf8, true),
                Field::new(
                    "obs_shape",
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                    true,
                ),
                Field::new(
                    "obs_f32",
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                    true,
                ),
                Field::new(
                    "obs_f64",
                    DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                    true,
                ),
                Field::new("obs_binary", DataType::Binary, true),
                // Action columns
                Field::new("act_dtype", DataType::Utf8, true),
                Field::new(
                    "act_shape",
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                    true,
                ),
                Field::new(
                    "act_f32",
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                    true,
                ),
                Field::new(
                    "act_f64",
                    DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                    true,
                ),
                Field::new("act_binary", DataType::Binary, true),
                // Mask columns
                Field::new("mask_dtype", DataType::Utf8, true),
                Field::new(
                    "mask_shape",
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                    true,
                ),
                Field::new(
                    "mask_f32",
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                    true,
                ),
                Field::new(
                    "mask_f64",
                    DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                    true,
                ),
                Field::new("mask_binary", DataType::Binary, true),
            ]));

            let backend_array = StringArray::from(backends);
            let reward_array = Float32Array::from(rewards);
            let done_array = BooleanArray::from(dones);
            let timestamp_array = UInt64Array::from(timestamps);
            let agent_id_array = StringArray::from(agent_ids);

            let obs_dtype_array = StringArray::from(obs_dtypes);
            let obs_shape_array = build_shape_list_array(obs_shapes);
            let obs_f32_array = build_f32_list_array(obs_f32);
            let obs_f64_array = build_f64_list_array(obs_f64);
            let obs_binary_array = build_binary_array(obs_binary);

            let act_dtype_array = StringArray::from(act_dtypes);
            let act_shape_array = build_shape_list_array(act_shapes);
            let act_f32_array = build_f32_list_array(act_f32);
            let act_f64_array = build_f64_list_array(act_f64);
            let act_binary_array = build_binary_array(act_binary);

            let mask_dtype_array = StringArray::from(mask_dtypes);
            let mask_shape_array = build_shape_list_array(mask_shapes);
            let mask_f32_array = build_f32_list_array(mask_f32);
            let mask_f64_array = build_f64_list_array(mask_f64);
            let mask_binary_array = build_binary_array(mask_binary);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(backend_array) as ArrayRef,
                    Arc::new(reward_array) as ArrayRef,
                    Arc::new(done_array) as ArrayRef,
                    Arc::new(timestamp_array) as ArrayRef,
                    Arc::new(agent_id_array) as ArrayRef,
                    Arc::new(obs_dtype_array) as ArrayRef,
                    obs_shape_array,
                    obs_f32_array,
                    obs_f64_array,
                    obs_binary_array,
                    Arc::new(act_dtype_array) as ArrayRef,
                    act_shape_array,
                    act_f32_array,
                    act_f64_array,
                    act_binary_array,
                    Arc::new(mask_dtype_array) as ArrayRef,
                    mask_shape_array,
                    mask_f32_array,
                    mask_f64_array,
                    mask_binary_array,
                ],
            )
            .map_err(|e| {
                TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                    e.to_string(),
                ))
            })?;

            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }

            let file = File::create(&file_path).map_err(|e| {
                TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                    format!("Failed to create file: {}", e),
                ))
            })?;

            let mut writer = FileWriter::try_new(file, &schema).map_err(|e| {
                TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                    format!("Failed to create Arrow writer: {}", e),
                ))
            })?;

            writer.write(&batch).map_err(|e| {
                TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                    format!("Failed to write batch: {}", e),
                ))
            })?;

            writer.finish().map_err(|e| {
                TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                    format!("Failed to finish file: {}", e),
                ))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| {
            TrajectorySinkError::EncodeTrajectoryError(TrajectoryError::SerializationError(
                format!("Task join error: {}", e),
            ))
        })??;

        actor_last_processed.insert(actor_id, priority);

        Ok(())
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> PersistentTrajectoryDataSinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
}
