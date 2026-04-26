//! RelayRL Metrics Manager
//!
//! This module provides a unified interface for metrics management, abstracting
//! away the underlying Prometheus and OpenTelemetry implementations.

use super::export;

use opentelemetry::{InstrumentationScope, KeyValue, global, metrics::Meter};
use prometheus::{Counter as PrometheusCounter, Histogram as PrometheusHistogram, Registry};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, MutexGuard},
};
use tokio::sync::RwLock;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetricsConfig {
    meter_name: String,
    otlp_endpoint: String,
}

impl From<(String, String)> for MetricsConfig {
    fn from((meter_name, otlp_endpoint): (String, String)) -> Self {
        Self {
            meter_name,
            otlp_endpoint,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MetricKey {
    name: String,
    labels: Vec<(String, String)>,
}

impl MetricKey {
    fn new(name: &str, labels: &[KeyValue]) -> Self {
        let mut normalized_labels = BTreeMap::new();
        for label in labels {
            normalized_labels.insert(label.key.to_string(), label.value.to_string());
        }

        Self {
            name: name.to_string(),
            labels: normalized_labels.into_iter().collect(),
        }
    }

    fn const_labels(&self) -> HashMap<String, String> {
        self.labels.iter().cloned().collect()
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
static PROVIDER_SYNC_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Manages metrics for both Prometheus and OpenTelemetry backends.
#[derive(Clone)]
pub struct MetricsManager {
    prometheus_registry: Option<Arc<Mutex<Registry>>>,
    prometheus_counters: Arc<Mutex<HashMap<MetricKey, PrometheusCounter>>>,
    prometheus_histograms: Arc<Mutex<HashMap<MetricKey, PrometheusHistogram>>>,
    #[allow(clippy::type_complexity)]
    metrics_args: Arc<Mutex<Arc<RwLock<(String, String)>>>>,
    cached_metrics_args: Arc<Mutex<MetricsConfig>>,
}

impl MetricsManager {
    /// Creates a new `MetricsManager`.
    pub fn new(
        metrics_args: Arc<RwLock<(String, String)>>,
        initial_metrics_args: (String, String),
        prometheus_registry: Option<Registry>,
    ) -> Self {
        let initial_config = MetricsConfig::from(initial_metrics_args);
        Self::sync_meter_provider("", initial_config.otlp_endpoint.as_str());

        Self {
            prometheus_registry: prometheus_registry.map(|r| Arc::new(Mutex::new(r))),
            prometheus_counters: Arc::new(Mutex::new(HashMap::new())),
            prometheus_histograms: Arc::new(Mutex::new(HashMap::new())),
            metrics_args: Arc::new(Mutex::new(metrics_args)),
            cached_metrics_args: Arc::new(Mutex::new(initial_config)),
        }
    }

    pub(crate) fn bind_metrics_args(&self, metrics_args: Arc<RwLock<(String, String)>>) {
        *lock_unpoisoned(&self.metrics_args) = metrics_args;
    }

    pub(crate) async fn sync_config(&self) {
        let _ = self.sync_config_inner().await;
    }

    fn sync_meter_provider(previous_otlp_endpoint: &str, current_otlp_endpoint: &str) {
        if previous_otlp_endpoint == current_otlp_endpoint {
            return;
        }

        #[cfg(test)]
        PROVIDER_SYNC_CALLS.fetch_add(1, Ordering::SeqCst);

        #[cfg(feature = "opentelemetry")]
        if current_otlp_endpoint.is_empty() {
            export::open_telemetry::shutdown_opentelemetry_meter_provider();
        } else {
            export::open_telemetry::init_opentelemetry_with_otlp(current_otlp_endpoint);
        }
    }

    async fn read_metrics_config(&self) -> MetricsConfig {
        let metrics_args = { lock_unpoisoned(&self.metrics_args).clone() };
        MetricsConfig::from(metrics_args.read().await.clone())
    }

    async fn sync_config_inner(&self) -> MetricsConfig {
        let current_metrics_args = self.read_metrics_config().await;
        let previous_otlp_endpoint = {
            let mut cached_metrics_args = lock_unpoisoned(&self.cached_metrics_args);
            let previous_otlp_endpoint = cached_metrics_args.otlp_endpoint.clone();
            *cached_metrics_args = current_metrics_args.clone();
            previous_otlp_endpoint
        };

        Self::sync_meter_provider(
            previous_otlp_endpoint.as_str(),
            current_metrics_args.otlp_endpoint.as_str(),
        );

        current_metrics_args
    }

    fn get_or_register_counter(
        &self,
        name: &str,
        labels: &[KeyValue],
    ) -> Option<PrometheusCounter> {
        let registry_arc = self.prometheus_registry.as_ref()?.clone();
        let metric_key = MetricKey::new(name, labels);

        if let Some(counter) = lock_unpoisoned(&self.prometheus_counters)
            .get(&metric_key)
            .cloned()
        {
            return Some(counter);
        }

        let prom_counter = PrometheusCounter::with_opts(
            prometheus::Opts::new(name, name).const_labels(metric_key.const_labels()),
        )
        .expect("Prometheus counter options should be valid");

        let mut counters = lock_unpoisoned(&self.prometheus_counters);
        if let Some(counter) = counters.get(&metric_key).cloned() {
            return Some(counter);
        }

        let register_result = {
            let registry = lock_unpoisoned(&registry_arc);
            registry.register(Box::new(prom_counter.clone()))
        };

        match register_result {
            Ok(()) => {
                counters.insert(metric_key, prom_counter.clone());
                Some(prom_counter)
            }
            Err(err) => {
                log::error!("Failed to register Prometheus counter `{}`: {}", name, err);
                None
            }
        }
    }

    fn get_or_register_histogram(
        &self,
        name: &str,
        labels: &[KeyValue],
    ) -> Option<PrometheusHistogram> {
        let registry_arc = self.prometheus_registry.as_ref()?.clone();
        let metric_key = MetricKey::new(name, labels);

        if let Some(histogram) = lock_unpoisoned(&self.prometheus_histograms)
            .get(&metric_key)
            .cloned()
        {
            return Some(histogram);
        }

        let prom_histogram = PrometheusHistogram::with_opts(
            prometheus::HistogramOpts::new(name, name).const_labels(metric_key.const_labels()),
        )
        .expect("Prometheus histogram options should be valid");

        let mut histograms = lock_unpoisoned(&self.prometheus_histograms);
        if let Some(histogram) = histograms.get(&metric_key).cloned() {
            return Some(histogram);
        }

        let register_result = {
            let registry = lock_unpoisoned(&registry_arc);
            registry.register(Box::new(prom_histogram.clone()))
        };

        match register_result {
            Ok(()) => {
                histograms.insert(metric_key, prom_histogram.clone());
                Some(prom_histogram)
            }
            Err(err) => {
                log::error!(
                    "Failed to register Prometheus histogram `{}`: {}",
                    name,
                    err
                );
                None
            }
        }
    }

    async fn current_meter(&self) -> Meter {
        let current_metrics_args = self.sync_config_inner().await;
        let scope = InstrumentationScope::builder(current_metrics_args.meter_name).build();
        global::meter_provider().meter_with_scope(scope)
    }

    /// Records a value for a counter.
    pub async fn record_counter(&self, name: &str, value: u64, labels: &[KeyValue]) {
        let otel_meter = self.current_meter().await;

        let counter = otel_meter.u64_counter(name.to_string()).build();
        counter.add(value, labels);

        if let Some(prom_counter) = self.get_or_register_counter(name, labels) {
            prom_counter.inc_by(value as f64);
        }
    }

    /// Records a value for a histogram.
    pub async fn record_histogram(&self, name: &str, value: f64, labels: &[KeyValue]) {
        let otel_meter = self.current_meter().await;

        let histogram = otel_meter.f64_histogram(name.to_string()).build();
        histogram.record(value, labels);

        if let Some(prom_histogram) = self.get_or_register_histogram(name, labels) {
            prom_histogram.observe(value);
        }
    }

    /// Returns the Prometheus registry.
    #[allow(unused)]
    pub fn prometheus_registry(&self) -> Option<Arc<Mutex<Registry>>> {
        self.prometheus_registry.clone()
    }

    #[cfg(test)]
    pub(crate) async fn current_config_snapshot(&self) -> (String, String) {
        let current_metrics_args = self.sync_config_inner().await;
        (
            current_metrics_args.meter_name,
            current_metrics_args.otlp_endpoint,
        )
    }

    #[cfg(test)]
    fn provider_sync_call_count() -> usize {
        PROVIDER_SYNC_CALLS.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    fn reset_provider_sync_call_count() {
        PROVIDER_SYNC_CALLS.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager(prometheus_registry: Option<Registry>) -> MetricsManager {
        let metrics_args = ("test-meter".to_string(), String::new());
        MetricsManager::new(
            Arc::new(RwLock::new(metrics_args.clone())),
            metrics_args,
            prometheus_registry,
        )
    }

    fn prometheus_output(manager: &MetricsManager) -> String {
        let registry = manager
            .prometheus_registry()
            .expect("expected prometheus registry");
        let registry = lock_unpoisoned(&registry);
        export::prometheus::get_metrics_as_string(&registry)
    }

    fn parse_metric_value(metrics: &str, metric_name: &str) -> Option<f64> {
        metrics.lines().find_map(|line| {
            if line.starts_with('#') {
                return None;
            }

            let mut parts = line.split_whitespace();
            let name = parts.next()?;
            let value = parts.next()?;
            if name == metric_name {
                return value.parse().ok();
            }

            None
        })
    }

    #[tokio::test]
    async fn record_counter_accumulates_prometheus_value() {
        let manager = test_manager(Some(Registry::new()));

        manager.record_counter("test_counter", 1, &[]).await;
        manager.record_counter("test_counter", 2, &[]).await;

        let metrics = prometheus_output(&manager);
        assert_eq!(parse_metric_value(&metrics, "test_counter"), Some(3.0));
    }

    #[tokio::test]
    async fn record_histogram_accumulates_prometheus_count() {
        let manager = test_manager(Some(Registry::new()));

        manager.record_histogram("test_histogram", 1.5, &[]).await;
        manager.record_histogram("test_histogram", 2.5, &[]).await;

        let metrics = prometheus_output(&manager);
        assert_eq!(
            parse_metric_value(&metrics, "test_histogram_count"),
            Some(2.0)
        );
    }

    #[tokio::test]
    async fn meter_name_changes_do_not_resync_provider_but_endpoint_changes_do() {
        let metrics_args = Arc::new(RwLock::new(("meter-one".to_string(), String::new())));
        let manager = MetricsManager::new(
            metrics_args.clone(),
            ("meter-one".to_string(), String::new()),
            None,
        );

        MetricsManager::reset_provider_sync_call_count();

        *metrics_args.write().await = ("meter-two".to_string(), String::new());
        assert_eq!(
            manager.current_config_snapshot().await,
            ("meter-two".to_string(), String::new())
        );
        assert_eq!(MetricsManager::provider_sync_call_count(), 0);

        *metrics_args.write().await =
            ("meter-two".to_string(), "http://127.0.0.1:4317".to_string());
        assert_eq!(
            manager.current_config_snapshot().await,
            ("meter-two".to_string(), "http://127.0.0.1:4317".to_string())
        );
        assert_eq!(MetricsManager::provider_sync_call_count(), 1);

        *metrics_args.write().await = ("meter-two".to_string(), String::new());
        assert_eq!(
            manager.current_config_snapshot().await,
            ("meter-two".to_string(), String::new())
        );
        assert_eq!(MetricsManager::provider_sync_call_count(), 2);
    }
}
