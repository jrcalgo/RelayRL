//! RelayRL Metrics Export Module
//!
//! This module provides exporters for RelayRL metrics, allowing metrics
//! to be exposed to external monitoring systems like Prometheus and OpenTelemetry.

pub mod prometheus;
pub mod open_telemetry;

// Re-export commonly used functions
pub use prometheus::{
    init_prometheus_exporter,
    init_prometheus_exporter_with_settings,
    set_prometheus_host,
    set_prometheus_port,
    set_prometheus_endpoint,
};

pub use open_telemetry::{
    init_opentelemetry,
    init_opentelemetry_with_otlp,
    set_service_name,
    set_export_interval,
}; 