//! RelayRL OpenTelemetry Metrics Exporter
//!
//! This module provides OpenTelemetry integration for the RelayRL metrics system,
//! enabling distributed tracing and metrics collection.

use opentelemetry::{
    KeyValue, global,
    global::BoxedSpan,
    trace::{Span, Tracer},
};
use opentelemetry_otlp::{MetricExporter, Protocol, WithExportConfig};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use std::collections::HashMap;

/// Initialize OpenTelemetry with OTLP exporter
///
/// # Arguments
///
/// * `otlp_endpoint` - The OTLP endpoint URL
#[cfg(feature = "opentelemetry")]
pub fn init_opentelemetry_with_otlp(otlp_endpoint: &str) {
    let exporter = match MetricExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .with_protocol(Protocol::Grpc)
        .build()
    {
        Ok(exporter) => exporter,
        Err(err) => {
            log::error!(
                "Failed to configure OpenTelemetry OTLP gRPC metrics exporter for endpoint `{}`: {}; leaving the current global meter provider unchanged",
                otlp_endpoint,
                err
            );
            return;
        }
    };

    let meter_provider = SdkMeterProvider::builder()
        .with_periodic_exporter(exporter)
        .build();

    global::set_meter_provider(meter_provider);
    log::info!(
        "Configured OpenTelemetry OTLP gRPC metrics exporter for endpoint `{}`",
        otlp_endpoint
    );
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
/// * `BoxedSpan` - The created span
#[cfg(feature = "opentelemetry")]
pub fn create_span(name: &str, labels: &HashMap<String, String>) -> BoxedSpan {
    // Tracing spans are not configured with the current dependency set.
    let tracer = global::tracer("relay-rl");
    let mut span = tracer.start(name.to_string());
    let attributes: Vec<KeyValue> = labels
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();
    span.set_attributes(attributes);
    span
}
