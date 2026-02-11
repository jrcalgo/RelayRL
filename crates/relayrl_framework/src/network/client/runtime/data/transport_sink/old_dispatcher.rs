//! Transport Dispatcher - Reliability Layer
//!
//! This module provides three specialized dispatchers that wrap raw transport implementations
//! (ZMQ/Tonic) with retry logic, circuit breaking, and backpressure control.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐
//! │   Actor     │  │ TrajectoryBuffer │  │  ScaleManager   │
//! └──────┬──────┘  └────────┬─────────┘  └────────┬────────┘
//!        │                  │                     │
//!        ▼                  ▼                     ▼
//! ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐
//! │  Inference   │  │   Training     │  │    Scaling      │
//! │  Dispatcher  │  │   Dispatcher   │  │   Dispatcher    │
//! └──────┬───────┘  └───────┬────────┘  └────────┬────────┘
//!        │                  │                    │
//!        └──────────────────┼────────────────────┘
//!                           ▼
//!                  ┌────────────────┐
//!                  │ TransportClient│
//!                  │  (ZMQ/Tonic)   │
//!                  └────────────────┘
//! ```

use crate::network::HyperparameterArgs;
use crate::network::client::runtime::coordination::scale_manager::ScalingOperation;
use crate::network::client::runtime::data::transport::{TransportClient, TransportError};
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::Algorithm;

use active_uuid_registry::interface::reserve_with;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::BackendMatcher;
use relayrl_types::types::data::trajectory::EncodedTrajectory;
use relayrl_types::types::model::ModelModule;

use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::sync::{RwLock, Semaphore, SemaphorePermit};
use uuid::Uuid;

// ============================================================================
// Error Types
// ============================================================================

/// Unified error type for dispatcher operations with retry context.
#[derive(Debug, Error)]
pub enum DispatcherError {
    /// The underlying transport operation failed.
    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    /// Maximum retry attempts exceeded.
    #[error("Max retries exceeded after {attempts} attempts: {cause}")]
    MaxRetriesExceeded {
        cause: TransportError,
        attempts: u32,
    },

    /// Circuit breaker is open, rejecting requests.
    #[error("Circuit breaker open - server appears unavailable")]
    CircuitOpen,

    /// Backpressure limit reached, too many in-flight requests.
    #[error("Backpressure limit exceeded - too many concurrent requests")]
    BackpressureExceeded,

    /// Operation timed out.
    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),

    /// Dispatcher not initialized or in invalid state.
    #[error("Dispatcher error: {0}")]
    InvalidState(String),

    /// Task join error from spawn_blocking.
    #[error("Task join error: {0}")]
    JoinError(String),
}

// ============================================================================
// Retry Policy
// ============================================================================

/// Configurable retry behavior with exponential backoff and jitter.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_attempts: u32,
    /// Initial delay before first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier for exponential backoff.
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delay.
    pub add_jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            add_jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Calculate delay for a given attempt number (1-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let base_delay = self.initial_delay.as_millis() as f64
            * self.backoff_multiplier.powi((attempt - 1) as i32);
        let mut delay_ms = base_delay.min(self.max_delay.as_millis() as f64);

        if self.add_jitter {
            // Add up to 25% jitter
            let jitter = delay_ms * 0.25 * rand::random::<f64>();
            delay_ms += jitter;
        }

        Duration::from_millis(delay_ms as u64)
    }

    /// No retries policy.
    pub fn no_retries() -> Self {
        Self {
            max_attempts: 0,
            ..Default::default()
        }
    }

    /// Aggressive retry policy for critical operations.
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.5,
            add_jitter: true,
        }
    }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally.
    Closed,
    /// Circuit is open, requests are rejected.
    Open,
    /// Circuit is half-open, allowing a single test request.
    HalfOpen,
}

/// Circuit breaker to prevent cascading failures.
///
/// Tracks consecutive failures and opens the circuit when threshold is exceeded.
/// After a cooldown period, allows a single test request (half-open state).
pub struct CircuitBreaker {
    /// Current circuit state.
    state: RwLock<CircuitState>,
    /// Consecutive failure count.
    failure_count: AtomicU64,
    /// Failure threshold to open circuit.
    failure_threshold: u64,
    /// Duration circuit stays open before half-open.
    open_duration: Duration,
    /// Timestamp when circuit was opened.
    opened_at: RwLock<Option<Instant>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u64, open_duration: Duration) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU64::new(0),
            failure_threshold,
            open_duration,
            opened_at: RwLock::new(None),
        }
    }

    /// Check if the circuit is currently open (rejecting requests).
    pub async fn is_open(&self) -> bool {
        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                // Check if we should transition to half-open
                if let Some(opened_at) = *self.opened_at.read().await {
                    if opened_at.elapsed() >= self.open_duration {
                        // Transition to half-open
                        *self.state.write().await = CircuitState::HalfOpen;
                        return false; // Allow the test request
                    }
                }
                true
            }
            CircuitState::HalfOpen => false, // Allow test request
        }
    }

    /// Record a successful operation.
    pub async fn record_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
        *self.state.write().await = CircuitState::Closed;
        *self.opened_at.write().await = None;
    }

    /// Record a failed operation.
    pub async fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

        if failures >= self.failure_threshold {
            let current_state = *self.state.read().await;
            if current_state != CircuitState::Open {
                *self.state.write().await = CircuitState::Open;
                *self.opened_at.write().await = Some(Instant::now());
            }
        }
    }

    /// Get current state for monitoring.
    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Get current failure count for monitoring.
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::SeqCst)
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

// ============================================================================
// Backpressure Controller
// ============================================================================

/// Semaphore-based concurrency limiter for backpressure control.
pub struct BackpressureController {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl BackpressureController {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// Acquire a permit before sending. Blocks (async) if at capacity.
    pub async fn acquire(&self) -> Result<SemaphorePermit<'_>, DispatcherError> {
        self.semaphore
            .acquire()
            .await
            .map_err(|_| DispatcherError::BackpressureExceeded)
    }

    /// Try to acquire without blocking - useful for non-critical operations.
    pub fn try_acquire(&self) -> Result<SemaphorePermit<'_>, DispatcherError> {
        self.semaphore
            .try_acquire()
            .map_err(|_| DispatcherError::BackpressureExceeded)
    }

    /// Get current available permits for monitoring.
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Get max concurrent limit.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

// ============================================================================
// Dispatcher Configuration
// ============================================================================

/// Configuration for dispatcher behavior.
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    /// Retry policy for failed operations.
    pub retry_policy: RetryPolicy,
    /// Failure threshold to open circuit breaker.
    pub circuit_breaker_threshold: u64,
    /// Duration circuit breaker stays open.
    pub circuit_breaker_duration: Duration,
    /// Maximum concurrent in-flight requests.
    pub max_concurrent_requests: usize,
    /// Operation timeout.
    pub timeout: Duration,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            retry_policy: RetryPolicy::default(),
            circuit_breaker_threshold: 5,
            circuit_breaker_duration: Duration::from_secs(30),
            max_concurrent_requests: 50,
            timeout: Duration::from_secs(30),
        }
    }
}

impl DispatcherConfig {
    /// Configuration optimized for inference (high throughput, low latency).
    pub fn for_inference() -> Self {
        Self {
            retry_policy: RetryPolicy {
                max_attempts: 2,
                initial_delay: Duration::from_millis(50),
                max_delay: Duration::from_secs(1),
                backoff_multiplier: 2.0,
                add_jitter: true,
            },
            circuit_breaker_threshold: 10,
            circuit_breaker_duration: Duration::from_secs(15),
            max_concurrent_requests: 100,
            timeout: Duration::from_secs(5),
        }
    }

    /// Configuration for training operations (can tolerate higher latency).
    pub fn for_training() -> Self {
        Self {
            retry_policy: RetryPolicy::default(),
            circuit_breaker_threshold: 5,
            circuit_breaker_duration: Duration::from_secs(30),
            max_concurrent_requests: 20,
            timeout: Duration::from_secs(60),
        }
    }

    /// Configuration for scaling operations (rare, should be reliable).
    pub fn for_scaling() -> Self {
        Self {
            retry_policy: RetryPolicy::aggressive(),
            circuit_breaker_threshold: 3,
            circuit_breaker_duration: Duration::from_secs(60),
            max_concurrent_requests: 5,
            timeout: Duration::from_secs(120),
        }
    }
}

// ============================================================================
// Inference Dispatcher
// ============================================================================

/// Dispatcher for inference-related operations.
///
/// Handles sending observations to inference servers and receiving actions.
/// Optimized for high throughput and low latency.
pub struct InferenceDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<TransportClient<B>>,
    circuit_breaker: CircuitBreaker,
    backpressure: BackpressureController,
    config: DispatcherConfig,
}

impl<B: Backend + BackendMatcher<Backend = B>> InferenceDispatcher<B> {
    pub fn new(transport: Arc<TransportClient<B>>, config: DispatcherConfig) -> Self {
        Self {
            transport,
            circuit_breaker: CircuitBreaker::new(
                config.circuit_breaker_threshold,
                config.circuit_breaker_duration,
            ),
            backpressure: BackpressureController::new(config.max_concurrent_requests),
            config,
        }
    }

    pub fn with_default_config(transport: Arc<TransportClient<B>>) -> Self {
        Self::new(transport, DispatcherConfig::for_inference())
    }

    /// Send an inference request with reliability guarantees.
    ///
    /// # Arguments
    /// * `actor_id` - The actor making the request
    /// * `obs_bytes` - Serialized observation data
    /// * `inference_server_address` - Target server address
    pub async fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, DispatcherError> {
        // Acquire backpressure permit
        let _permit = self.backpressure.acquire().await?;

        // Check circuit breaker
        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        // Execute with retry
        let mut attempts = 0;
        loop {
            let result = self
                .execute_inference(actor_id, obs_bytes, inference_server_address)
                .await;

            match result {
                Ok(action) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(action);
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_inference(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                let actor_id = *actor_id;
                let obs_bytes = obs_bytes.to_vec();
                let address = inference_server_address.to_string();

                // Clone the transport reference for the blocking task
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_action_request(&actor_id, &obs_bytes, &address)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::NoTransportConfiguredError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(_async_tr) => {
                // Tonic async implementation would go here
                Err(TransportError::NoTransportConfiguredError(
                    "Async inference not yet implemented".to_string(),
                ))
            }
        }
    }

    /// Get circuit breaker state for monitoring.
    pub async fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state().await
    }

    /// Get available permits for monitoring.
    pub fn available_permits(&self) -> usize {
        self.backpressure.available_permits()
    }
}

// ============================================================================
// Training Dispatcher
// ============================================================================

/// Dispatcher for training-related operations.
///
/// Handles trajectory sending, model handshake, and model update listening.
pub struct TrainingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<TransportClient<B>>,
    circuit_breaker: CircuitBreaker,
    backpressure: BackpressureController,
    config: DispatcherConfig,
    /// Current model version (atomic for lock-free reads).
    current_version: Arc<AtomicI64>,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrainingDispatcher<B> {
    pub fn new(transport: Arc<TransportClient<B>>, config: DispatcherConfig) -> Self {
        Self {
            transport,
            circuit_breaker: CircuitBreaker::new(
                config.circuit_breaker_threshold,
                config.circuit_breaker_duration,
            ),
            backpressure: BackpressureController::new(config.max_concurrent_requests),
            config,
            current_version: Arc::new(AtomicI64::new(0)),
        }
    }

    pub fn with_default_config(transport: Arc<TransportClient<B>>) -> Self {
        Self::new(transport, DispatcherConfig::for_training())
    }

    /// Send a trajectory to the training server.
    pub async fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_send_trajectory(
                    sender_id,
                    encoded_trajectory.clone(),
                    model_server_address,
                    trajectory_server_address,
                )
                .await;

            match result {
                Ok(()) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(());
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        model_server_address: &str,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                let sender_id = *sender_id;
                let model_addr = model_server_address.to_string();
                let traj_addr = trajectory_server_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_traj_to_server(
                            &sender_id,
                            encoded_trajectory,
                            &model_addr,
                            &traj_addr,
                        )
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::SendTrajError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_traj_to_server(
                        sender_id,
                        encoded_trajectory,
                        model_server_address,
                        trajectory_server_address,
                    )
                    .await
            }
        }
    }

    /// Perform initial model handshake with the server.
    pub async fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        model_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_model_handshake(actor_id, model_server_address, agent_listener_address)
                .await;

            match result {
                Ok(model) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(model);
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_model_handshake(
        &self,
        actor_id: &Uuid,
        model_server_address: &str,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let actor_id = *actor_id;
                let model_addr = model_server_address.to_string();
                let agent_addr = agent_listener_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.initial_model_handshake(&actor_id, &model_addr, &agent_addr)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::ModelHandshakeError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .initial_model_handshake(actor_id, model_server_address, agent_listener_address)
                    .await
            }
        }
    }

    /// Start listening for model updates.
    ///
    /// This is a long-running operation that doesn't use retry (the listener handles reconnection).
    pub async fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        agent_listener_address: &str,
        global_dispatcher_tx: Sender<RoutedMessage>,
    ) -> Result<(), DispatcherError> {
        // No backpressure or circuit breaker for long-running listener
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let receiver_id = *receiver_id;
                let agent_addr = agent_listener_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.listen_for_model(&receiver_id, &agent_addr, global_dispatcher_tx)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| DispatcherError::JoinError(e.to_string()))?
                .map_err(DispatcherError::Transport)
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => async_tr
                .listen_for_model(receiver_id, agent_listener_address, global_dispatcher_tx)
                .await
                .map_err(DispatcherError::Transport),
        }
    }

    /// Get current model version.
    pub fn current_version(&self) -> i64 {
        self.current_version.load(Ordering::SeqCst)
    }

    /// Update current model version.
    pub fn set_current_version(&self, version: i64) {
        self.current_version.store(version, Ordering::SeqCst);
    }

    /// Get circuit breaker state for monitoring.
    pub async fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state().await
    }

    /// Get available permits for monitoring.
    pub fn available_permits(&self) -> usize {
        self.backpressure.available_permits()
    }
}

// ============================================================================
// Scaling Dispatcher
// ============================================================================

/// Dispatcher for scaling and lifecycle operations.
///
/// Handles client registration, algorithm initialization, scaling signals, and shutdown.
pub struct ScalingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<TransportClient<B>>,
    circuit_breaker: CircuitBreaker,
    backpressure: BackpressureController,
    config: DispatcherConfig,
    /// Unique transport identifier.
    transport_id: Uuid,
    /// Whether algorithm has been initialized.
    algorithm_initialized: Arc<AtomicBool>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ScalingDispatcher<B> {
    pub fn new(
        transport: Arc<TransportClient<B>>,
        config: DispatcherConfig,
    ) -> Result<Self, DispatcherError> {
        let transport_id = reserve_with("scaling_dispatcher", 117, 100)
            .map_err(|e| DispatcherError::InvalidState(e.to_string()))?;

        Ok(Self {
            transport,
            circuit_breaker: CircuitBreaker::new(
                config.circuit_breaker_threshold,
                config.circuit_breaker_duration,
            ),
            backpressure: BackpressureController::new(config.max_concurrent_requests),
            config,
            transport_id,
            algorithm_initialized: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn with_default_config(
        transport: Arc<TransportClient<B>>,
    ) -> Result<Self, DispatcherError> {
        Self::new(transport, DispatcherConfig::for_scaling())
    }

    /// Send client IDs to server for registration.
    pub async fn send_client_ids_to_server(
        &self,
        scaling_id: &Uuid,
        client_ids: Vec<(String, Uuid)>,
        scaling_server_address: &str,
    ) -> Result<(), DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_send_client_ids(scaling_id, client_ids.to_owned(), scaling_server_address)
                .await;

            match result {
                Ok(()) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(());
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: Vec<(String, Uuid)>,
        scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let scaling_id = *scaling_id;
                let address = scaling_server_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_client_ids_to_server(&scaling_id, &client_ids, &address)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::SendClientIdsToServerError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_client_ids_to_server(scaling_id, &client_ids, scaling_server_address)
                    .await
            }
        }
    }

    /// Send algorithm initialization request.
    pub async fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_algorithm_init(
                    scaling_id,
                    algorithm.clone(),
                    hyperparams.clone(),
                    agent_listener_address,
                )
                .await;

            match result {
                Ok(()) => {
                    self.circuit_breaker.record_success().await;
                    self.algorithm_initialized.store(true, Ordering::SeqCst);
                    return Ok(());
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_algorithm_init(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let scaling_id = *scaling_id;
                let address = agent_listener_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_algorithm_init_request(
                            &scaling_id,
                            algorithm,
                            hyperparams,
                            &address,
                        )
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::SendAlgorithmInitRequestError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_algorithm_init_request(
                        scaling_id,
                        algorithm,
                        hyperparams,
                        agent_listener_address,
                    )
                    .await
            }
        }
    }

    /// Send scaling warning notification.
    pub async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_scaling_warning(scaling_id, operation.clone(), scaling_server_address)
                .await;

            match result {
                Ok(()) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(());
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let scaling_id = *scaling_id;
                let address = scaling_server_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_scaling_warning(&scaling_id, operation, &address)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::SendScalingWarningError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_scaling_warning(scaling_id, operation, scaling_server_address)
                    .await
            }
        }
    }

    /// Send scaling complete notification.
    pub async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_scaling_complete(scaling_id, operation.clone(), scaling_server_address)
                .await;

            match result {
                Ok(()) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(());
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let scaling_id = *scaling_id;
                let address = scaling_server_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_scaling_complete(&scaling_id, operation, &address)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::SendScalingCompleteError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_scaling_complete(scaling_id, operation, scaling_server_address)
                    .await
            }
        }
    }

    /// Send shutdown signal to server.
    pub async fn send_shutdown_signal_to_server(
        &self,
        scaling_id: &Uuid,
        scaling_server_address: &str,
    ) -> Result<(), DispatcherError> {
        let _permit = self.backpressure.acquire().await?;

        if self.circuit_breaker.is_open().await {
            return Err(DispatcherError::CircuitOpen);
        }

        let mut attempts = 0;
        loop {
            let result = self
                .execute_shutdown_signal(scaling_id, scaling_server_address)
                .await;

            match result {
                Ok(()) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(());
                }
                Err(e) if attempts < self.config.retry_policy.max_attempts => {
                    attempts += 1;
                    self.circuit_breaker.record_failure().await;
                    let delay = self.config.retry_policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    self.circuit_breaker.record_failure().await;
                    return Err(DispatcherError::MaxRetriesExceeded { cause: e, attempts });
                }
            }
        }
    }

    async fn execute_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let scaling_id = *scaling_id;
                let address = scaling_server_address.to_string();
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.send_shutdown_signal_to_server(&scaling_id, &address)
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| TransportError::SendShutdownSignalError(e.to_string()))?
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_shutdown_signal_to_server(scaling_id, scaling_server_address)
                    .await
            }
        }
    }

    /// Shutdown the transport.
    pub async fn shutdown(&self) -> Result<(), DispatcherError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(_) => {
                let transport = Arc::clone(&self.transport);

                tokio::task::spawn_blocking(move || {
                    if let TransportClient::Sync(sync_tr) = &*transport {
                        sync_tr.shutdown()
                    } else {
                        Err(TransportError::NoTransportConfiguredError(
                            "Expected sync transport".to_string(),
                        ))
                    }
                })
                .await
                .map_err(|e| DispatcherError::JoinError(e.to_string()))?
                .map_err(DispatcherError::Transport)
            }
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => async_tr
                .shutdown()
                .await
                .map_err(DispatcherError::Transport),
        }
    }

    /// Get transport ID.
    pub fn transport_id(&self) -> Uuid {
        self.transport_id
    }

    /// Check if algorithm has been initialized.
    pub fn is_algorithm_initialized(&self) -> bool {
        self.algorithm_initialized.load(Ordering::SeqCst)
    }

    /// Get circuit breaker state for monitoring.
    pub async fn circuit_state(&self) -> CircuitState {
        self.circuit_breaker.state().await
    }

    /// Get available permits for monitoring.
    pub fn available_permits(&self) -> usize {
        self.backpressure.available_permits()
    }
}
