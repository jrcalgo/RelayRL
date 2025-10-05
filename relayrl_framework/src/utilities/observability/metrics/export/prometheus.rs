//! RelayRL Prometheus Metrics Exporter
//!
//! This module provides Prometheus integration for the RelayRL metrics system,
//! enabling export of metrics to Prometheus monitoring infrastructure.

use prometheus::{Encoder, Registry, TextEncoder};

/// Initialize the Prometheus exporter with default settings
///
/// This function starts a web server on the configured host and port
/// to expose metrics in Prometheus format.
#[cfg(feature = "prometheus")]
pub fn create_prometheus_registry() -> Registry {
    Registry::new()
}

/// Return the current metrics in Prometheus text format
///
/// # Returns
///
/// * `String` - The metrics in Prometheus text format
#[cfg(feature = "prometheus")]
pub fn get_metrics_as_string(registry: &Registry) -> String {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

// No-op implementations for when the feature is disabled
#[cfg(not(feature = "prometheus"))]
pub fn create_prometheus_registry() -> () {}

#[cfg(not(feature = "prometheus"))]
pub fn get_metrics_as_string(_registry: &()) -> String {
    "Prometheus metrics export is disabled (feature not enabled)".to_string()
}
