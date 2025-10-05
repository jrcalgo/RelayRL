//! RelayRL Metrics Module
//!
//! This module provides metrics and telemetry capabilities for the RelayRL framework,
//! enabling performance monitoring, profiling, and distributed tracing.

use std::sync::{Arc, Mutex, Once};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Expose submodules
pub mod definitions;
pub mod registry;
pub mod export;

// Re-export commonly used types
pub use definitions::{Counter, Gauge, Histogram};
pub use registry::MetricsRegistry;

// Global initialization guard
static INIT: Once = Once::new();

// Global metrics registry
static mut METRICS_REGISTRY: Option<Arc<Mutex<MetricsRegistry>>> = None;

/// Initialize the metrics system with default configuration
///
/// This sets up the global metrics registry with default exporters.
pub fn init_metrics() {
    INIT.call_once(|| {
        let registry = MetricsRegistry::new();
        
        unsafe {
            METRICS_REGISTRY = Some(Arc::new(Mutex::new(registry)));
        }
        
        #[cfg(feature = "prometheus")]
        {
            export::prometheus::init_prometheus_exporter();
        }
        
        #[cfg(feature = "opentelemetry")]
        {
            export::open_telemetry::init_opentelemetry();
        }
    });
}

/// Get a reference to the global metrics registry
///
/// # Returns
///
/// * `Arc<Mutex<MetricsRegistry>>` - Thread-safe reference to the metrics registry
pub fn global_registry() -> Arc<Mutex<MetricsRegistry>> {
    unsafe {
        match &METRICS_REGISTRY {
            Some(registry) => Arc::clone(registry),
            None => {
                init_metrics();
                Arc::clone(METRICS_REGISTRY.as_ref().unwrap())
            }
        }
    }
}

/// Register a counter metric
///
/// # Arguments
///
/// * `name` - The name of the counter
/// * `help` - Description of what the counter measures
/// * `labels` - Optional labels as key-value pairs
///
/// # Returns
///
/// * `Arc<Counter>` - A thread-safe reference to the counter
pub fn register_counter(name: &str, help: &str, labels: Option<HashMap<String, String>>) -> Arc<Counter> {
    let registry = global_registry();
    let mut registry = registry.lock().unwrap();
    registry.register_counter(name, help, labels)
}

/// Register a gauge metric
///
/// # Arguments
///
/// * `name` - The name of the gauge
/// * `help` - Description of what the gauge measures
/// * `labels` - Optional labels as key-value pairs
///
/// # Returns
///
/// * `Arc<Gauge>` - A thread-safe reference to the gauge
pub fn register_gauge(name: &str, help: &str, labels: Option<HashMap<String, String>>) -> Arc<Gauge> {
    let registry = global_registry();
    let mut registry = registry.lock().unwrap();
    registry.register_gauge(name, help, labels)
}

/// Register a histogram metric
///
/// # Arguments
///
/// * `name` - The name of the histogram
/// * `help` - Description of what the histogram measures
/// * `buckets` - Optional custom buckets; uses defaults if None
/// * `labels` - Optional labels as key-value pairs
///
/// # Returns
///
/// * `Arc<Histogram>` - A thread-safe reference to the histogram
pub fn register_histogram(
    name: &str,
    help: &str,
    buckets: Option<Vec<f64>>,
    labels: Option<HashMap<String, String>>
) -> Arc<Histogram> {
    let registry = global_registry();
    let mut registry = registry.lock().unwrap();
    registry.register_histogram(name, help, buckets, labels)
}

/// Timer utility for recording durations
pub struct Timer {
    start: Instant,
    histogram: Arc<Histogram>,
}

impl Timer {
    /// Start a new timer associated with a histogram
    ///
    /// # Arguments
    ///
    /// * `histogram` - The histogram to record the duration in
    ///
    /// # Returns
    ///
    /// * `Self` - A new Timer instance
    pub fn start(histogram: Arc<Histogram>) -> Self {
        Self {
            start: Instant::now(),
            histogram,
        }
    }
    
    /// Observe the elapsed time and record it in the histogram
    pub fn observe(&self) {
        let duration = self.start.elapsed();
        let seconds = duration.as_secs_f64();
        self.histogram.observe(seconds);
    }
    
    /// Observe the elapsed time and reset the timer
    pub fn observe_and_reset(&mut self) {
        self.observe();
        self.start = Instant::now();
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.observe();
    }
} 