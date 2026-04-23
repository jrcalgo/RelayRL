use super::ZmqClientError;
use crossbeam_utils::CachePadded;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, RwLock};
use std::time::Duration;
use std::time::Instant;

/// Configurable retry behavior with exponential backoff and jitter.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
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
    state: RwLock<CircuitState>,
    failure_count: CachePadded<AtomicU64>,
    failure_threshold: u64,
    open_duration: Duration,
    opened_at: RwLock<Option<Instant>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u64, open_duration: Duration) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: CachePadded::new(AtomicU64::new(0)),
            failure_threshold,
            open_duration,
            opened_at: RwLock::new(None),
        }
    }

    /// Check if the circuit is currently open (rejecting requests).
    pub fn is_open(&self) -> bool {
        let mut state = self
            .state
            .write()
            .expect("CircuitBreaker state lock poisoned");
        match *state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                let opened_at = self
                    .opened_at
                    .write()
                    .expect("CircuitBreaker opened_at lock poisoned");
                if let Some(opened_at) = *opened_at && opened_at.elapsed() >= self.open_duration {
                    *state = CircuitState::HalfOpen;
                    return false; // Allow the test request
                }

                true
            }
            CircuitState::HalfOpen => false, // Allow test request
        }
    }

    /// Record a successful operation.
    pub fn record_success(&self) {
        let mut state = self
            .state
            .write()
            .expect("CircuitBreaker state lock poisoned");
        let mut opened_at = self
            .opened_at
            .write()
            .expect("CircuitBreaker opened_at lock poisoned");
        self.failure_count.store(0, Ordering::SeqCst);
        *state = CircuitState::Closed;
        *opened_at = None;
    }

    /// Record a failed operation.
    pub fn record_failure(&self) {
        let mut state = self
            .state
            .write()
            .expect("CircuitBreaker state lock poisoned");
        let mut opened_at = self
            .opened_at
            .write()
            .expect("CircuitBreaker opened_at lock poisoned");
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

        if failures >= self.failure_threshold && *state != CircuitState::Open {
            *state = CircuitState::Open;
            *opened_at = Some(Instant::now());
        }
    }

    /// Get current state for monitoring.
    pub fn state(&self) -> CircuitState {
        *self
            .state
            .read()
            .expect("CircuitBreaker state lock poisoned")
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

/// Semaphore-based concurrency limiter for backpressure control.
pub struct BackpressureController {
    available: CachePadded<AtomicUsize>,
    condvar: Condvar,
    wait_mutex: Mutex<()>,
    max_concurrent: usize,
}

pub struct BackpressurePermit<'a> {
    controller: &'a BackpressureController,
}

impl<'a> Drop for BackpressurePermit<'a> {
    fn drop(&mut self) {
        let _wait_guard = self
            .controller
            .wait_mutex
            .lock()
            .expect("Backpressure wait mutex poisoned");
        self.controller.available.fetch_add(1, Ordering::Release);
        self.controller.condvar.notify_one();
    }
}

impl BackpressureController {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            available: CachePadded::new(AtomicUsize::new(max_concurrent)),
            condvar: Condvar::new(),
            wait_mutex: Mutex::new(()),
            max_concurrent,
        }
    }

    fn try_decrement_available(&self) -> bool {
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            if current == 0 {
                return false;
            }
            match self.available.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(next) => current = next,
            }
        }
    }

    /// Acquire a permit before sending. Blocks (sync) if at capacity.
    pub fn acquire(&self) -> Result<BackpressurePermit<'_>, ZmqClientError> {
        loop {
            if self.try_decrement_available() {
                return Ok(BackpressurePermit { controller: self });
            }

            let wait_guard = self
                .wait_mutex
                .lock()
                .expect("Backpressure wait mutex poisoned");

            if self.available.load(Ordering::Acquire) == 0 {
                let _unused = self
                    .condvar
                    .wait(wait_guard)
                    .expect("Backpressure wait mutex poisoned");
            }
        }
    }

    /// Try to acquire without blocking - useful for non-critical operations.
    pub fn try_acquire(&self) -> Result<BackpressurePermit<'_>, ZmqClientError> {
        if self.try_decrement_available() {
            Ok(BackpressurePermit { controller: self })
        } else {
            Err(ZmqClientError::BackpressureExceeded)
        }
    }

    /// Get current available permits for monitoring.
    pub fn available_permits(&self) -> usize {
        self.available.load(Ordering::Acquire)
    }

    /// Get max concurrent limit.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

/// Configuration for dispatcher behavior.
#[derive(Debug, Clone)]
pub struct ZmqPolicyConfig {
    pub retry_policy: RetryPolicy,
    pub circuit_breaker_threshold: u64,
    pub circuit_breaker_duration: Duration,
    pub max_concurrent_requests: usize,
    pub timeout: Duration,
}

impl Default for ZmqPolicyConfig {
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

impl ZmqPolicyConfig {
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

#[cfg(test)]
mod unit_tests {
    use super::{BackpressureController, CircuitBreaker, CircuitState};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;
    use std::{thread, time::Duration as StdDuration};

    #[test]
    fn record_failure_opens_circuit_at_threshold() {
        let breaker = CircuitBreaker::new(2, Duration::from_secs(30));

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn is_open_transitions_to_half_open_after_cooldown() {
        let breaker = CircuitBreaker::new(1, Duration::ZERO);
        breaker.record_failure();

        assert!(!breaker.is_open());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn record_success_clears_failures_and_closes_circuit() {
        let breaker = CircuitBreaker::new(1, Duration::from_secs(30));
        breaker.record_failure();

        breaker.record_success();

        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(breaker.failure_count(), 0);
        assert!(!breaker.is_open());
    }

    #[test]
    fn dropping_permit_wakes_blocked_waiter() {
        let controller = Arc::new(BackpressureController::new(1));
        let held_permit = controller.acquire().expect("first permit should succeed");
        let (tx, rx) = mpsc::channel();
        let waiting_controller = controller.clone();

        let waiter = thread::spawn(move || {
            let permit = waiting_controller
                .acquire()
                .expect("waiter should acquire after wakeup");
            tx.send(()).expect("waiter should notify main thread");
            drop(permit);
        });

        thread::sleep(StdDuration::from_millis(20));
        drop(held_permit);

        rx.recv_timeout(StdDuration::from_secs(1))
            .expect("waiter should wake once the permit is released");
        waiter.join().expect("waiter thread should exit cleanly");
    }
}
