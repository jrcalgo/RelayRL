use std::path::PathBuf;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, SemaphorePermit};

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
        let state = *self
            .state
            .read()
            .expect("CircuitBreaker state lock poisoned");
        match state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                if let Some(opened_at) = *self
                    .opened_at
                    .read()
                    .expect("CircuitBreaker opened_at lock poisoned")
                {
                    if opened_at.elapsed() >= self.open_duration {
                        *self
                            .state
                            .write()
                            .expect("CircuitBreaker state lock poisoned") = CircuitState::HalfOpen;
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
        *self
            .state
            .write()
            .expect("CircuitBreaker state lock poisoned") = CircuitState::Closed;
        *self
            .opened_at
            .write()
            .expect("CircuitBreaker opened_at lock poisoned") = None;
    }

    pub fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        if failures >= self.failure_threshold {
            let current = *self
                .state
                .read()
                .expect("CircuitBreaker state lock poisoned");
            if current != CircuitState::Open {
                *self
                    .state
                    .write()
                    .expect("CircuitBreaker state lock poisoned") = CircuitState::Open;
                *self
                    .opened_at
                    .write()
                    .expect("CircuitBreaker opened_at lock poisoned") = Some(Instant::now());
            }
        }
    }

    pub fn state(&self) -> CircuitState {
        *self
            .state
            .read()
            .expect("CircuitBreaker state lock poisoned")
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
