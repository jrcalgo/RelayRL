//! RL4Sys OpenTelemetry Metrics Exporter
//!
//! This module provides OpenTelemetry integration for the RL4Sys metrics system,
//! enabling distributed tracing and metrics collection.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use opentelemetry::global::BoxedSpan;
#[cfg(feature = "opentelemetry")]
use opentelemetry::{
    KeyValue,
    global,
    metrics::{self, Counter as OtelCounter, Histogram as OtelHistogram, MeterProvider},
    // sdk::metrics::{controllers, processors, selectors}, // TODO: Update for newer OpenTelemetry version
};

// Global initialization guard
static INIT: Once = Once::new();

// Static configuration
static mut OTEL_SERVICE_NAME: &str = "rl4sys";
static mut OTEL_EXPORT_INTERVAL_MILLIS: u64 = 10000; // 10 seconds

/// Set the service name for OpenTelemetry
///
/// # Arguments
///
/// * `service_name` - The name of the service
pub fn set_service_name(service_name: &'static str) {
    unsafe {
        OTEL_SERVICE_NAME = service_name;
    }
}

/// Set the export interval for OpenTelemetry metrics
///
/// # Arguments
///
/// * `interval_millis` - The interval in milliseconds
pub fn set_export_interval(interval_millis: u64) {
    unsafe {
        OTEL_EXPORT_INTERVAL_MILLIS = interval_millis;
    }
}

/// Initialize OpenTelemetry with default settings
#[cfg(feature = "opentelemetry")]
pub fn init_opentelemetry() {
    INIT.call_once(|| {
        let service_name = unsafe { OTEL_SERVICE_NAME };
        let export_interval = unsafe { OTEL_EXPORT_INTERVAL_MILLIS };

        // Create a meter provider with stdout exporter
        let controller = controllers::basic(
            processors::factory(
                selectors::simple::histogram([0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
                selectors::simple::sum(),
            )
            .with_memory(true)
            .build(),
        )
        .with_collect_period(std::time::Duration::from_millis(export_interval))
        .build();

        global::set_meter_provider(controller);

        let meter = global::meter_provider().meter(service_name);

        log::info!(
            "OpenTelemetry metrics initialized for service '{}'",
            service_name
        );
    });
}

/// Initialize OpenTelemetry with OTLP exporter
///
/// # Arguments
///
/// * `otlp_endpoint` - The OTLP endpoint URL
#[cfg(feature = "opentelemetry")]
pub fn init_opentelemetry_with_otlp(otlp_endpoint: &str) {
    INIT.call_once(|| {
        let service_name = unsafe { OTEL_SERVICE_NAME };
        let export_interval = unsafe { OTEL_EXPORT_INTERVAL_MILLIS };
        // This is a placeholder for OTLP exporter setup
        // The actual implementation would depend on the specific OTLP exporter crate
        log::info!("OpenTelemetry OTLP metrics enabled with endpoint: {}", otlp_endpoint);
        log::warn!("OTLP exporter is placeholder only - add opentelemetry-otlp dependency and implement this function");
        // Fall back to standard init for now
        init_opentelemetry_impl();
    });
}

#[cfg(feature = "opentelemetry")]
fn init_opentelemetry_impl() {
    let service_name = unsafe { OTEL_SERVICE_NAME };
    let export_interval = unsafe { OTEL_EXPORT_INTERVAL_MILLIS };

    // Create a meter provider with stdout exporter
    let controller = controllers::basic(
        processors::factory(
            selectors::simple::histogram([0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
            selectors::simple::sum(),
        )
        .with_memory(true)
        .build(),
    )
    .with_collect_period(std::time::Duration::from_millis(export_interval))
    .build();

    global::set_meter_provider(controller);

    log::info!(
        "OpenTelemetry metrics initialized for service '{}'",
        service_name
    );
}

/// Track an RL4Sys counter with OpenTelemetry
///
/// # Arguments
///
/// * `name` - The name of the counter
/// * `value` - The value to increment by
/// * `labels` - Labels to attach to the counter
#[cfg(feature = "opentelemetry")]
pub fn track_counter(name: &str, value: u64, labels: &HashMap<String, String>) {
    let meter = global::meter_provider().meter(unsafe { OTEL_SERVICE_NAME });

    let counter = meter.u64_counter(name.to_string()).build();

    let attributes: Vec<KeyValue> = labels
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();

    counter.add(value, &attributes);
}

/// Track an RL4Sys histogram with OpenTelemetry
///
/// # Arguments
///
/// * `name` - The name of the histogram
/// * `value` - The value to record
/// * `labels` - Labels to attach to the histogram
#[cfg(feature = "opentelemetry")]
pub fn track_histogram(name: &str, value: f64, labels: &HashMap<String, String>) {
    let meter = global::meter_provider().meter(unsafe { OTEL_SERVICE_NAME });

    let histogram = meter.f64_histogram(name.to_string()).build();

    let attributes: Vec<KeyValue> = labels
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();

    histogram.record(value, &attributes);
}

/// Create an RL4Sys span for tracing
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
pub fn create_span(name: &str, labels: &HashMap<String, String>) -> Option<BoxedSpan> {
    use opentelemetry::trace::{Tracer, TracerProvider};

    let tracer = global::tracer_provider().tracer(unsafe { OTEL_SERVICE_NAME });

    let attributes: Vec<KeyValue> = labels
        .iter()
        .map(|(k, v)| KeyValue::new(k.clone(), v.clone()))
        .collect();
    let span = tracer
        .span_builder(name.to_string())
        .with_attributes(attributes);

    Some(span.start(&tracer))
}

// No-op implementations for when the feature is disabled
#[cfg(not(feature = "opentelemetry"))]
pub fn init_opentelemetry() {
    log::warn!("OpenTelemetry metrics export is disabled (feature not enabled)");
}

#[cfg(not(feature = "opentelemetry"))]
pub fn init_opentelemetry_with_otlp(_otlp_endpoint: &str) {
    log::warn!("OpenTelemetry OTLP metrics export is disabled (feature not enabled)");
}

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
