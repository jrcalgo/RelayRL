use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use tokio::sync::{Semaphore, SemaphorePermit, OwnedSemaphorePermit};
use std::sync::Arc;

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
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let base_delay = self.initial_delay.as_millis() as f64
            * self.backoff_multiplier.powi((attempt - 1) as i32);
        let mut delay_ms = base_delay.min(self.max_delay.as_millis() as f64);

        if self.add_jitter {
            let jitter = delay_ms * 0.25 * rand::random::<f64>();
            delay_ms += jitter;
        }

        Duration::from_millis(delay_ms as u64)
    }

    pub fn no_retries() -> Self {
        Self {
            max_attempts: 0,
            ..Default::default()
        }
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU64,
    failure_threshold: u64,
    open_duration: Duration,
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

    pub fn is_open(&self) -> bool {
        let state = *self.state.read().expect("CircuitBreaker state lock poisoned");
        match state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                if let Some(opened_at) = *self
                    .opened_at
                    .read()
                    .expect("CircuitBreaker opened_at lock poisoned")
                {
                    if opened_at.elapsed() >= self.open_duration {
                        *self.state.write().expect("CircuitBreaker state lock poisoned") =
                            CircuitState::HalfOpen;
                        return false;
                    }
                }
                true
            }
            CircuitState::HalfOpen => false,
        }
    }

    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
        *self.state.write().expect("CircuitBreaker state lock poisoned") = CircuitState::Closed;
        *self.opened_at.write().expect("CircuitBreaker opened_at lock poisoned") = None;
    }

    pub fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        if failures >= self.failure_threshold {
            let current = *self.state.read().expect("CircuitBreaker state lock poisoned");
            if current != CircuitState::Open {
                *self.state.write().expect("CircuitBreaker state lock poisoned") =
                    CircuitState::Open;
                *self.opened_at.write().expect("CircuitBreaker opened_at lock poisoned") =
                    Some(Instant::now());
            }
        }
    }

    pub fn state(&self) -> CircuitState {
        *self.state.read().expect("CircuitBreaker state lock poisoned")
    }

    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::SeqCst)
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

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

    pub async fn acquire(&self) -> Result<OwnedSemaphorePermit, tokio::sync::AcquireError> {
        self.semaphore.clone().acquire_owned().await
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}


#[derive(Debug, Clone)]
pub struct NatsPolicyConfig {
    pub retry_policy: RetryPolicy,
    pub circuit_breaker_threshold: u64,
    pub circuit_breaker_duration: Duration,
    pub max_concurrent_requests: usize,
    pub timeout: Duration,
}

impl Default for NatsPolicyConfig {
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

impl NatsPolicyConfig {
    /// Optimised for inference: low latency, high throughput.
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

    /// Tolerant of higher latency for training operations.
    pub fn for_training() -> Self {
        Self {
            retry_policy: RetryPolicy::default(),
            circuit_breaker_threshold: 5,
            circuit_breaker_duration: Duration::from_secs(30),
            max_concurrent_requests: 20,
            timeout: Duration::from_secs(60),
        }
    }

    /// Rare but critical scaling operations — aggressive retry, low concurrency.
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

pub enum NatsAuthentication {
    Anonymous,
    Token(String),
    UserPassword { username: String, password: String },
    NKey { seed: String },
    CredentialsFile { path: PathBuf },
    CredentialsString(String),
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::time::Duration;

    // -------------------------------------------------------------------------
    // RetryPolicy tests
    // -------------------------------------------------------------------------

    #[test]
    fn attempt_0_returns_zero() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.delay_for_attempt(0), Duration::ZERO);
    }

    #[test]
    fn attempt_1_returns_initial_delay_without_jitter() {
        let policy = RetryPolicy {
            add_jitter: false,
            ..RetryPolicy::default()
        };
        assert_eq!(policy.delay_for_attempt(1), policy.initial_delay);
    }

    #[test]
    fn attempt_2_applies_backoff_without_jitter() {
        let policy = RetryPolicy {
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            add_jitter: false,
            ..RetryPolicy::default()
        };
        // attempt 2 => initial * backoff^1 = 100 * 2 = 200ms
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(200));
    }

    #[test]
    fn delay_capped_at_max() {
        let policy = RetryPolicy {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            backoff_multiplier: 10.0,
            add_jitter: false,
            ..RetryPolicy::default()
        };
        // Very high attempt should be capped at max_delay
        let delay = policy.delay_for_attempt(20);
        assert!(delay <= Duration::from_millis(500 + 1)); // small tolerance
    }

    #[test]
    fn no_retries_policy_max_attempts_zero() {
        let policy = RetryPolicy::no_retries();
        assert_eq!(policy.max_attempts, 0);
    }

    #[test]
    fn aggressive_policy_has_correct_fields() {
        let policy = RetryPolicy::aggressive();
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.initial_delay, Duration::from_millis(50));
        assert_eq!(policy.backoff_multiplier, 1.5);
    }

    #[test]
    fn jitter_is_within_25_percent() {
        let policy = RetryPolicy {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.0,
            add_jitter: true,
            max_attempts: 3,
        };
        for _ in 0..20 {
            let delay = policy.delay_for_attempt(1);
            // base = 100ms, jitter up to 25% => max 125ms
            assert!(delay >= Duration::from_millis(100));
            assert!(delay <= Duration::from_millis(126));
        }
    }

    // -------------------------------------------------------------------------
    // CircuitBreaker tests
    // -------------------------------------------------------------------------

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(!cb.is_open());
    }

    #[test]
    fn remains_closed_below_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(!cb.is_open());
    }

    #[test]
    fn opens_at_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.is_open());
    }

    #[test]
    fn circuit_breaker_stays_open_within_duration() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60));
        cb.record_failure();
        assert!(cb.is_open(), "Should be open before duration elapses");
    }

    #[test]
    fn transitions_to_half_open_after_duration() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(15));
        // is_open() transitions to HalfOpen when duration has passed
        assert!(!cb.is_open());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn record_success_resets_to_closed() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert!(!cb.is_open());
    }

    #[test]
    fn does_not_double_open() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(30));
        // Record more failures than threshold
        for _ in 0..10 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        // opened_at should be set once (we can't observe it directly, but state is Open)
        assert!(cb.is_open());
    }

    #[test]
    fn failure_count_increments() {
        let cb = CircuitBreaker::new(10, Duration::from_secs(30));
        for i in 1..=5 {
            cb.record_failure();
            assert_eq!(cb.failure_count(), i);
        }
    }

    // -------------------------------------------------------------------------
    // BackpressureController tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn available_permits_starts_at_max() {
        let ctrl = BackpressureController::new(5);
        assert_eq!(ctrl.available_permits(), 5);
        assert_eq!(ctrl.max_concurrent(), 5);
    }

    #[tokio::test]
    async fn acquiring_permit_reduces_count() {
        let ctrl = BackpressureController::new(5);
        let _permit = ctrl.acquire().await.unwrap();
        assert_eq!(ctrl.available_permits(), 4);
    }

    #[tokio::test]
    async fn dropping_permit_restores_count() {
        let ctrl = BackpressureController::new(5);
        {
            let _permit = ctrl.acquire().await.unwrap();
            assert_eq!(ctrl.available_permits(), 4);
        }
        // permit dropped
        assert_eq!(ctrl.available_permits(), 5);
    }

    #[tokio::test]
    async fn blocks_when_at_capacity() {
        use std::sync::Arc;
        let ctrl = Arc::new(BackpressureController::new(1));
        // Acquire the only permit
        let permit = ctrl.acquire().await.unwrap();

        let ctrl2 = ctrl.clone();
        let task = tokio::spawn(async move {
            ctrl2.acquire().await.unwrap()
        });

        // Yield and ensure the task is parked (not yet completed)
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!task.is_finished());

        // Release the permit, task should unblock
        drop(permit);
        let _p = task.await.unwrap();
        assert_eq!(ctrl.available_permits(), 0); // 1 permit re-acquired by task
    }
}

impl NatsAuthentication {
    /// Apply this authentication to a [`async_nats::ConnectOptions`] builder.
    ///
    /// `CredentialsFile` is the only variant that performs async I/O; all others
    /// are infallible synchronous operations.
    pub async fn apply(
        self,
        options: async_nats::ConnectOptions,
    ) -> Result<async_nats::ConnectOptions, async_nats::Error> {
        match self {
            Self::Anonymous => Ok(options),
            Self::Token(token_string) => Ok(options.token(token_string)),
            Self::UserPassword { username, password } => {
                Ok(options.user_and_password(username, password))
            }
            Self::NKey { seed } => Ok(options.nkey(seed)),
            Self::CredentialsFile { path } => {
                options.credentials_file(path).await.map_err(Into::into)
            }
            Self::CredentialsString(credentials_string) => {
                options.credentials(&credentials_string).map_err(Into::into)
            }
        }
    }
}
