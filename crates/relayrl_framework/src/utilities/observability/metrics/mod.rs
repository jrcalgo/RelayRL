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

async fn init_metrics_with_slot(
    metrics_manager: &OnceLock<MetricsManager>,
    metrics_args: Arc<RwLock<(String, String)>>,
) -> MetricsManager {
    let initial_metrics_args = metrics_args.read().await.clone();
    let init_metrics_args = metrics_args.clone();

    let mgr_ref = metrics_manager.get_or_init(move || {
        #[cfg(feature = "prometheus")]
        let prometheus_registry = Some(export::prometheus::create_prometheus_registry());

        #[cfg(not(feature = "prometheus"))]
        let prometheus_registry = None;

        MetricsManager::new(init_metrics_args, initial_metrics_args, prometheus_registry)
    });

    mgr_ref.bind_metrics_args(metrics_args);
    mgr_ref.sync_config().await;
    mgr_ref.clone()
}

/// Initialize the metrics system with default configuration
///
/// This sets up the global metrics registry with default exporters.
pub async fn init_metrics(metrics_args: Arc<RwLock<(String, String)>>) -> MetricsManager {
    init_metrics_with_slot(&METRICS_MANAGER, metrics_args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn init_metrics_rebinds_singleton_to_latest_source() {
        let metrics_manager = OnceLock::new();
        let first_args = Arc::new(RwLock::new(("meter-one".to_string(), String::new())));
        let first_manager = init_metrics_with_slot(&metrics_manager, first_args).await;
        assert_eq!(
            first_manager.current_config_snapshot().await,
            ("meter-one".to_string(), String::new())
        );

        let second_args = Arc::new(RwLock::new(("meter-two".to_string(), String::new())));
        let second_manager = init_metrics_with_slot(&metrics_manager, second_args.clone()).await;
        assert_eq!(
            second_manager.current_config_snapshot().await,
            ("meter-two".to_string(), String::new())
        );

        *second_args.write().await = ("meter-three".to_string(), String::new());
        assert_eq!(
            first_manager.current_config_snapshot().await,
            ("meter-three".to_string(), String::new())
        );
    }
}
