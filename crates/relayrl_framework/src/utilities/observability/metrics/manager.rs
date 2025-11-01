//! RelayRL Metrics Manager
//!
//! This module provides a unified interface for metrics management, abstracting
//! away the underlying Prometheus and OpenTelemetry implementations.

use opentelemetry::{
    KeyValue,
    metrics::{Counter, Histogram, Meter},
};
use prometheus::{Counter as PrometheusCounter, Histogram as PrometheusHistogram, Registry};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Manages metrics for both Prometheus and OpenTelemetry backends.
#[derive(Clone)]
pub struct MetricsManager {
    prometheus_registry: Option<Arc<Mutex<Registry>>>,
    otel_meter: Meter,
}

impl MetricsManager {
    /// Creates a new `MetricsManager`.
    pub fn new(prometheus_registry: Option<Registry>, otel_meter: Meter) -> Self {
        Self {
            prometheus_registry: prometheus_registry.map(|r| Arc::new(Mutex::new(r))),
            otel_meter,
        }
    }

    /// Records a value for a counter.
    pub fn record_counter(&self, name: &str, value: u64, labels: &[KeyValue]) {
        // OpenTelemetry
        let counter = self.otel_meter.u64_counter(name.to_string()).build();
        counter.add(value, labels);

        // Prometheus
        if let Some(registry_arc) = &self.prometheus_registry {
            let mut prom_labels = HashMap::new();
            for label in labels {
                prom_labels.insert(label.key.as_str(), label.value.as_str());
            }

            let prom_counter = PrometheusCounter::with_opts(
                prometheus::Opts::new(name, name).const_labels(
                    labels
                        .iter()
                        .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                        .collect(),
                ),
            )
            .unwrap();

            if let Ok(registry) = registry_arc.lock() {
                if registry.register(Box::new(prom_counter.clone())).is_ok() {
                    prom_counter.inc_by(value as f64);
                }
            }
        }
    }

    /// Records a value for a histogram.
    pub fn record_histogram(&self, name: &str, value: f64, labels: &[KeyValue]) {
        // OpenTelemetry
        let histogram = self.otel_meter.f64_histogram(name.to_string()).build();
        histogram.record(value, labels);

        // Prometheus
        if let Some(registry_arc) = &self.prometheus_registry {
            let mut prom_labels = HashMap::new();
            for label in labels {
                prom_labels.insert(label.key.as_str(), label.value.as_str());
            }

            let prom_histogram = PrometheusHistogram::with_opts(
                prometheus::HistogramOpts::new(name, name).const_labels(
                    labels
                        .iter()
                        .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                        .collect(),
                ),
            )
            .unwrap();

            if let Ok(registry) = registry_arc.lock() {
                if registry.register(Box::new(prom_histogram.clone())).is_ok() {
                    prom_histogram.observe(value);
                }
            }
        }
    }

    /// Returns the Prometheus registry.
    pub fn prometheus_registry(&self) -> Option<Arc<Mutex<Registry>>> {
        self.prometheus_registry.clone()
    }
}
