//! RelayRL Metrics Export Module
//!
//! This module provides exporters for RelayRL metrics, allowing metrics
//! to be exposed to external monitoring systems like Prometheus and OpenTelemetry.

pub mod prometheus;
pub mod open_telemetry;

// Re-export commonly used functions
pub use prometheus::{
    create_prometheus_registry,
    get_metrics_as_string,
};

pub use open_telemetry::{
    init_opentelemetry_with_otlp,
}; 