//! RelayRL Metrics Module
//!
//! This module provides metrics and telemetry capabilities for the RelayRL framework,
//! enabling performance monitoring, profiling, and distributed tracing.

use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::RwLock;

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
pub async fn init_metrics(metrics_args: Arc<RwLock<(String, String)>>) -> MetricsManager {
    #[cfg(feature = "prometheus")]
    let prometheus_registry = Some(export::prometheus::create_prometheus_registry());

    #[cfg(not(feature = "prometheus"))]
    let prometheus_registry = None;

    let initial_metrics_args = metrics_args.read().await.clone();

    let mgr_ref = METRICS_MANAGER.get_or_init(move || {
        MetricsManager::new(
            metrics_args.clone(),
            initial_metrics_args,
            prometheus_registry,
        )
    });

    mgr_ref.clone()
}
