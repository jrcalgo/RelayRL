//! RelayRL Metrics Module
//!
//! This module provides metrics and telemetry capabilities for the RelayRL framework,
//! enabling performance monitoring, profiling, and distributed tracing.

use opentelemetry::global;
use std::sync::OnceLock;

// Expose submodules
pub mod export;
pub mod manager;

// Re-export commonly used types
pub use manager::MetricsManager;

// Global metrics manager (initialized once)
static METRICS_MANAGER: OnceLock<MetricsManager> = OnceLock::new();

/// Initialize the metrics system with default configuration
///
/// This sets up the global metrics registry with default exporters.
pub fn init_metrics() -> MetricsManager {
    #[cfg(feature = "prometheus")]
    let prometheus_registry = Some(export::prometheus::create_prometheus_registry());

    #[cfg(not(feature = "prometheus"))]
    let prometheus_registry = None;

    #[cfg(feature = "opentelemetry")]
    export::open_telemetry::init_opentelemetry_with_otlp("http://localhost:4317");

    let otel_meter = global::meter("relay-rl");

    let mgr_ref =
        METRICS_MANAGER.get_or_init(|| MetricsManager::new(prometheus_registry, otel_meter));

    mgr_ref.clone()
}
