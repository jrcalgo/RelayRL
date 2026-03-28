//! RelayRL Metrics Manager
//!
//! This module provides a unified interface for metrics management, abstracting
//! away the underlying Prometheus and OpenTelemetry implementations.

use super::export;

use opentelemetry::{
    InstrumentationScope, KeyValue, global,
    metrics::{Meter, MeterProvider},
};
use prometheus::{Counter as PrometheusCounter, Histogram as PrometheusHistogram, Registry};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

/// Manages metrics for both Prometheus and OpenTelemetry backends.
#[derive(Clone)]
pub struct MetricsManager {
    prometheus_registry: Option<Arc<Mutex<Registry>>>,
    metrics_args: Arc<RwLock<(String, String)>>,
    cached_metrics_args: Arc<Mutex<(String, String)>>,
}

impl MetricsManager {
    /// Creates a new `MetricsManager`.
    pub fn new(
        metrics_args: Arc<RwLock<(String, String)>>,
        initial_metrics_args: (String, String),
        prometheus_registry: Option<Registry>,
    ) -> Self {
        Self::init_meter_provider(initial_metrics_args.1.as_str());

        Self {
            prometheus_registry: prometheus_registry.map(|r| Arc::new(Mutex::new(r))),
            metrics_args,
            cached_metrics_args: Arc::new(Mutex::new(initial_metrics_args)),
        }
    }

    fn init_meter_provider(metrics_otlp_endpoint: &str) {
        #[cfg(feature = "opentelemetry")]
        if !metrics_otlp_endpoint.is_empty() {
            export::open_telemetry::init_opentelemetry_with_otlp(metrics_otlp_endpoint);
        }
    }

    async fn current_meter(&self) -> Meter {
        let current_metrics_args = self.metrics_args.read().await.clone();

        if let Ok(mut cached_metrics_args) = self.cached_metrics_args.lock() {
            if *cached_metrics_args != current_metrics_args {
                Self::init_meter_provider(current_metrics_args.1.as_str());
                *cached_metrics_args = current_metrics_args.clone();
            }
        }

        let scope = InstrumentationScope::builder(current_metrics_args.0.clone()).build();
        global::meter_provider().meter_with_scope(scope)
    }

    /// Records a value for a counter.
    pub async fn record_counter(&self, name: &str, value: u64, labels: &[KeyValue]) {
        let otel_meter = self.current_meter().await;

        // OpenTelemetry
        let counter = otel_meter.u64_counter(name.to_string()).build();
        counter.add(value, labels);

        // Prometheus
        if let Some(registry_arc) = &self.prometheus_registry {
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
    pub async fn record_histogram(&self, name: &str, value: f64, labels: &[KeyValue]) {
        let otel_meter = self.current_meter().await;

        // OpenTelemetry
        let histogram = otel_meter.f64_histogram(name.to_string()).build();
        histogram.record(value, labels);

        // Prometheus
        if let Some(registry_arc) = &self.prometheus_registry {
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
