use super::{RoutedMessage, RoutedPayload, RouterError};
use crate::network::client::agent::DatabaseTypeParams;
use crate::network::client::runtime::coordination::lifecycle_manager::FormattedTrajectoryFileParams;
use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::transport::{TransportClient, TransportError};

use relayrl_types::prelude::{BackendMatcher, CodecConfig};
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::trajectory::{
    EncodedTrajectory, RelayRLTrajectory, TrajectoryError,
};

use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, RwLock, broadcast};
use uuid::Uuid;

type PriorityRank = i64;

#[derive(Debug, Clone)]
struct SinkQueueEntry {
    priority: PriorityRank, // lower = sooner, higher = later
    actor_id: ActorUuid,
    traj_for_processing: Arc<RelayRLTrajectory>,
}

enum TrajectoryToWrite {
    Encoded(EncodedTrajectory),
    Raw(Arc<RelayRLTrajectory>),
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
    #[error("Transport error: {0}")]
    TransportError(#[from] TransportError),
    #[error("Failed to encode trajectory: {0}")]
    EncodeTrajectoryError(#[from] TrajectoryError),
    #[error("Failed to write to database: {0}")]
    DatabaseWriteError(String),
}

pub(crate) trait TrajectoryBufferTrait<B: Backend + BackendMatcher<Backend = B>>:
    TransportTrajectorySinkTrait<B> + DatabaseTrajectorySinkTrait<B> + LocalTrajectorySinkTrait<B>
{
    fn new(
        associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        codec: CodecConfig,
    ) -> Self;
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

pub(crate) trait TransportTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
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

pub(crate) trait DatabaseTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn write_database_trajectory(
        database_params: &DatabaseTypeParams,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        trajectory: &TrajectoryToWrite,
    ) -> Result<(), TrajectorySinkError>;
    async fn retry_write_database(
        database_params: &DatabaseTypeParams,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        trajectory: TrajectoryToWrite,
    ) -> Result<(), TrajectorySinkError>;
}

pub(crate) trait LocalTrajectorySinkTrait<B: Backend + BackendMatcher<Backend = B>> {
    async fn write_local_trajectory(
        associated_router_id: &RouterUuid,
        entry: &SinkQueueEntry,
        codec: &CodecConfig,
        actor_last_sent: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError>;
    async fn retry_write_local(
        shared_writer_params: &FormattedTrajectoryFileParams,
        actor_id: &ActorUuid,
        priority: &PriorityRank,
        trajectory: &TrajectoryToWrite,
    ) -> Result<(), TrajectorySinkError>;
}

pub(crate) struct ClientTrajectoryBuffer<B: Backend + BackendMatcher<Backend = B>> {
    associated_router_id: RouterUuid,
    active: AtomicBool,
    rx_from_actor: Receiver<RoutedMessage>,
    actor_last_processed: DashMap<Uuid, i64>,
    traj_queue_tx: Option<Sender<SinkQueueEntry>>,
    shared_transport: Option<Arc<TransportClient<B>>>,
    shared_server_addresses: Option<Arc<RwLock<ServerAddresses>>>,
    shared_trajectory_file_output: Option<Arc<RwLock<FormattedTrajectoryFileParams>>>,
    shutdown: Option<broadcast::Receiver<()>>,
    codec: CodecConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrajectoryBufferTrait<B>
    for ClientTrajectoryBuffer<B>
{
    fn new(
        associated_router_id: RouterUuid,
        rx_from_actor: Receiver<RoutedMessage>,
        codec: CodecConfig,
    ) -> Self {
        Self {
            associated_router_id,
            active: AtomicBool::new(false),
            rx_from_actor,
            actor_last_processed: DashMap::new(),
            traj_queue_tx: None,
            shared_transport: None,
            shared_server_addresses: None,
            shared_trajectory_file_output: None,
            shutdown: None,
            codec,
        }
    }

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
        shared_trajectory_file_output: Arc<RwLock<FormattedTrajectoryFileParams>>,
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

        // Extract fields we need to avoid borrowing self
        let trajectory_writer_enabled: bool = self.shared_trajectory_file_output.is_some();
        let transport_enabled: bool = self.shared_transport.is_some();
        let server_addresses_enabled: bool = self.shared_server_addresses.is_some();
        let actor_last_processed: DashMap<Uuid, i64> = self.actor_last_processed.clone();

        let (_traj_queue_tx, mut traj_queue_rx) =
            tokio::sync::mpsc::unbounded_channel::<SinkQueueEntry>();

        let worker_priority_queue: Arc<Mutex<BinaryHeap<SinkQueueEntry>>> =
            Arc::new(Mutex::new(BinaryHeap::new()));

        let shared_transport = self.shared_transport.clone();
        let codec: CodecConfig = self.codec.clone();
        let identity: RouterUuid = self.associated_router_id;

        // Clone for worker async tasks
        let worker_queue: Arc<Mutex<BinaryHeap<SinkQueueEntry>>> = worker_priority_queue.clone();
        let worker_actor_last_processed: DashMap<Uuid, i64> = actor_last_processed.clone();
        let worker_transport: Option<Arc<TransportClient<B>>> = shared_transport.clone();
        let worker_codec: CodecConfig = codec.clone();
        let worker_identity: Uuid = identity;
        let worker_trajectory_writer_enabled: bool = trajectory_writer_enabled;
        let worker_transport_enabled: bool = transport_enabled;
        let worker_server_addresses_enabled: bool = server_addresses_enabled;
        let worker_active: Arc<AtomicBool> = Arc::new(AtomicBool::new(true));

        let _worker_handle = tokio::spawn(async move {
            const BATCH_SIZE: usize = 10;
            let mut worker_tick = tokio::time::interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    job_opt = traj_queue_rx.recv() => {
                        if let Some(job) = job_opt {
                            let mut queue = worker_queue.lock().await;
                            queue.push(job);
                        } else {
                            break;
                        }
                    }
                    _ = worker_tick.tick() => {
                        if !worker_active.load(Ordering::SeqCst) {
                            break;
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

                        for _job in jobs_to_process {
                            let _job_identity = worker_identity;
                            let _job_codec = worker_codec.clone();
                            let _job_actor_last_processed = worker_actor_last_processed.clone();
                            let _job_transport = worker_transport.clone();
                            let _job_trajectory_writer_enabled = worker_trajectory_writer_enabled;
                            let _job_transport_enabled = worker_transport_enabled;
                            let _job_server_addresses_enabled = worker_server_addresses_enabled;

                            // TODO: IO task implementation
                        }
                    }
                }
            }

            // Drain remaining jobs on shutdown
            let mut queue = worker_queue.lock().await;
            while let Some(_job) = queue.pop() {
                // TODO: Process remaining jobs
            }
        });

        // TODO: Implement message receive loop
        // For now, just mark as inactive
        self.active.store(false, Ordering::SeqCst);
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

impl<B: Backend + BackendMatcher<Backend = B>> DatabaseTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn write_database_trajectory(
        _database_params: &DatabaseTypeParams,
        _actor_id: &ActorUuid,
        _priority: &PriorityRank,
        _trajectory: &TrajectoryToWrite,
    ) -> Result<(), TrajectorySinkError> {
        // TODO: Implement database trajectory writing
        Ok(())
    }

    async fn retry_write_database(
        _database_params: &DatabaseTypeParams,
        _actor_id: &ActorUuid,
        _priority: &PriorityRank,
        _trajectory: TrajectoryToWrite,
    ) -> Result<(), TrajectorySinkError> {
        // TODO: Implement database trajectory retry writing
        Ok(())
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> LocalTrajectorySinkTrait<B>
    for ClientTrajectoryBuffer<B>
{
    async fn write_local_trajectory(
        _associated_router_id: &RouterUuid,
        _entry: &SinkQueueEntry,
        _codec: &CodecConfig,
        _actor_last_sent: &DashMap<Uuid, i64>,
    ) -> Result<(), TrajectorySinkError> {
        // TODO: Implement local trajectory writing
        Ok(())
    }

    async fn retry_write_local(
        _writer_params: &FormattedTrajectoryFileParams,
        _actor_id: &ActorUuid,
        _priority: &PriorityRank,
        _trajectory: &TrajectoryToWrite,
    ) -> Result<(), TrajectorySinkError> {
        // TODO: Implement local trajectory retry writing
        Ok(())
    }
}
