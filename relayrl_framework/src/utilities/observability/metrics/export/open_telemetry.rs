//! RelayRL OpenTelemetry Metrics Exporter
//!
//! This module provides OpenTelemetry integration for the RelayRL metrics system,
//! enabling distributed tracing and metrics collection.

use opentelemetry::{global, KeyValue};
use std::collections::HashMap;

/// Initialize OpenTelemetry with OTLP exporter
///
/// # Arguments
///
/// * `otlp_endpoint` - The OTLP endpoint URL
#[cfg(feature = "opentelemetry")]
pub fn init_opentelemetry_with_otlp(_otlp_endpoint: &str) {
    // For the current dependency versions, the OTLP metrics pipeline setup is not used.
    // We fallback to the default global meter provider.
    log::warn!("OpenTelemetry OTLP metrics export not configured for current versions; using default meter provider");
}

// No-op implementations for when the feature is disabled
#[cfg(not(feature = "opentelemetry"))]
pub fn init_opentelemetry_with_otlp(_otlp_endpoint: &str) {
    log::warn!("OpenTelemetry OTLP metrics export is disabled (feature not enabled)");
}

/// Track an RelayRL counter with OpenTelemetry
///
/// # Arguments
///
/// * `name` - The name of the counter
/// * `value` - The value to increment by
/// * `labels` - Labels to attach to the counter
#[cfg(feature = "opentelemetry")]
pub fn track_counter(name: &str, value: u64, labels: &HashMap<String, String>) {
    let meter = global::meter("relay-rl");
    let counter = meter.u64_counter(name.to_string()).build();
    let attributes: Vec<KeyValue> = labels
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();
    counter.add(value, &attributes);
}

/// Track an RelayRL histogram with OpenTelemetry
///
/// # Arguments
///
/// * `name` - The name of the histogram
/// * `value` - The value to record
/// * `labels` - Labels to attach to the histogram
#[cfg(feature = "opentelemetry")]
pub fn track_histogram(name: &str, value: f64, labels: &HashMap<String, String>) {
    let meter = global::meter("relay-rl");
    let histogram = meter.f64_histogram(name.to_string()).build();
    let attributes: Vec<KeyValue> = labels
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();
    histogram.record(value, &attributes);
}

/// Create an RelayRL span for tracing
///
/// # Arguments
///
/// * `name` - The name of the span
/// * `labels` - Labels to attach to the span
///
/// # Returns
///
/// * `Option<opentelemetry::trace::Span>` - The created span, if OpenTelemetry is enabled
#[cfg(feature = "opentelemetry")]
pub fn create_span(_name: &str, _labels: &HashMap<String, String>) -> Option<()> {
    // Tracing spans are not configured with the current dependency set.
    None
}

// No-op implementations for when the feature is disabled
#[cfg(not(feature = "opentelemetry"))]
pub fn track_counter(_name: &str, _value: u64, _labels: &HashMap<String, String>) {
    // No-op when feature is disabled
}

#[cfg(not(feature = "opentelemetry"))]
pub fn track_histogram(_name: &str, _value: f64, _labels: &HashMap<String, String>) {
    // No-op when feature is disabled
}

#[cfg(not(feature = "opentelemetry"))]
pub fn create_span(_name: &str, _labels: &HashMap<String, String>) -> Option<()> {
    None
}
