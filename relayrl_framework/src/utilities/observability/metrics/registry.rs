//! RelayRL Metrics Registry
//!
//! This module provides a central registry for all metrics defined in the RelayRL framework,
//! ensuring proper organization and access to metrics across components.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::definitions::{Counter, Gauge, Histogram};

#[cfg(feature = "prometheus")]
use prometheus::{self, Registry};

/// A registry that manages all metrics
pub struct MetricsRegistry {
    counters: HashMap<String, Arc<Counter>>,
    gauges: HashMap<String, Arc<Gauge>>,
    histograms: HashMap<String, Arc<Histogram>>,
    #[cfg(feature = "prometheus")]
    prometheus_registry: Option<Registry>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    ///
    /// # Returns
    ///
    /// * `Self` - A new MetricsRegistry instance
    pub fn new() -> Self {
        #[cfg(feature = "prometheus")]
        let prometheus_registry = Some(Registry::new());

        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            #[cfg(feature = "prometheus")]
            prometheus_registry,
        }
    }

    /// Register a counter metric
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the counter
    /// * `help` - Description of what the counter measures
    /// * `labels` - Optional labels as key-value pairs
    ///
    /// # Returns
    ///
    /// * `Arc<Counter>` - A thread-safe reference to the counter
    pub fn register_counter(
        &mut self,
        name: &str,
        help: &str,
        labels: Option<HashMap<String, String>>,
    ) -> Arc<Counter> {
        if let Some(counter) = self.counters.get(name) {
            return Arc::clone(counter);
        }

        let mut counter = Counter::new(name, help, labels);

        #[cfg(feature = "prometheus")]
        if let Some(ref registry) = self.prometheus_registry {
            if let Some(ref labels) = counter.labels().iter().next() {
                // Create a counter vec with labels
                let label_names: Vec<&str> = counter.labels().keys().map(|k| k.as_str()).collect();
                let label_values: Vec<&str> =
                    counter.labels().values().map(|v| v.as_str()).collect();

                let opts = prometheus::Opts::new(counter.name(), counter.help());
                let counter_vec = prometheus::CounterVec::new(opts, &label_names).unwrap();
                registry.register(Box::new(counter_vec.clone())).unwrap();

                let prometheus_counter = counter_vec
                    .get_metric_with_label_values(&label_values)
                    .unwrap();
                counter.set_prometheus_counter(prometheus_counter);
            } else if counter.prometheus_counter.is_none() {
                // Create a simple counter
                let opts = prometheus::Opts::new(counter.name(), counter.help());
                let prometheus_counter = prometheus::Counter::with_opts(opts).unwrap();
                registry
                    .register(Box::new(prometheus_counter.clone()))
                    .unwrap();
                counter.set_prometheus_counter(prometheus_counter);
            }
        }

        let counter = Arc::new(counter);
        self.counters.insert(name.to_string(), Arc::clone(&counter));
        counter
    }

    /// Register a gauge metric
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the gauge
    /// * `help` - Description of what the gauge measures
    /// * `labels` - Optional labels as key-value pairs
    ///
    /// # Returns
    ///
    /// * `Arc<Gauge>` - A thread-safe reference to the gauge
    pub fn register_gauge(
        &mut self,
        name: &str,
        help: &str,
        labels: Option<HashMap<String, String>>,
    ) -> Arc<Gauge> {
        if let Some(gauge) = self.gauges.get(name) {
            return Arc::clone(gauge);
        }

        let mut gauge = Gauge::new(name, help, labels);

        #[cfg(feature = "prometheus")]
        if let Some(ref registry) = self.prometheus_registry {
            if let Some(ref labels) = gauge.labels().iter().next() {
                // Create a gauge vec with labels
                let label_names: Vec<&str> = gauge.labels().keys().map(|k| k.as_str()).collect();
                let label_values: Vec<&str> = gauge.labels().values().map(|v| v.as_str()).collect();

                let opts = prometheus::Opts::new(gauge.name(), gauge.help());
                let gauge_vec = prometheus::GaugeVec::new(opts, &label_names).unwrap();
                registry.register(Box::new(gauge_vec.clone())).unwrap();

                let prometheus_gauge = gauge_vec
                    .get_metric_with_label_values(&label_values)
                    .unwrap();
                gauge.set_prometheus_gauge(prometheus_gauge);
            } else if gauge.prometheus_gauge.is_none() {
                // Create a simple gauge
                let opts = prometheus::Opts::new(gauge.name(), gauge.help());
                let prometheus_gauge = prometheus::Gauge::with_opts(opts).unwrap();
                registry
                    .register(Box::new(prometheus_gauge.clone()))
                    .unwrap();
                gauge.set_prometheus_gauge(prometheus_gauge);
            }
        }

        let gauge = Arc::new(gauge);
        self.gauges.insert(name.to_string(), Arc::clone(&gauge));
        gauge
    }

    /// Register a histogram metric
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the histogram
    /// * `help` - Description of what the histogram measures
    /// * `buckets` - Optional custom buckets; uses defaults if None
    /// * `labels` - Optional labels as key-value pairs
    ///
    /// # Returns
    ///
    /// * `Arc<Histogram>` - A thread-safe reference to the histogram
    pub fn register_histogram(
        &mut self,
        name: &str,
        help: &str,
        buckets: Option<Vec<f64>>,
        labels: Option<HashMap<String, String>>,
    ) -> Arc<Histogram> {
        if let Some(histogram) = self.histograms.get(name) {
            return Arc::clone(histogram);
        }

        let mut histogram = Histogram::new(name, help, buckets, labels);

        #[cfg(feature = "prometheus")]
        if let Some(ref registry) = self.prometheus_registry {
            if let Some(ref labels) = histogram.labels().iter().next() {
                // Create a histogram vec with labels
                let label_names: Vec<&str> =
                    histogram.labels().keys().map(|k| k.as_str()).collect();
                let label_values: Vec<&str> =
                    histogram.labels().values().map(|v| v.as_str()).collect();

                let opts = prometheus::HistogramOpts::new(histogram.name(), histogram.help())
                    .buckets(histogram.buckets().to_vec());

                let histogram_vec = prometheus::HistogramVec::new(opts, &label_names).unwrap();
                registry.register(Box::new(histogram_vec.clone())).unwrap();

                let prometheus_histogram = histogram_vec
                    .get_metric_with_label_values(&label_values)
                    .unwrap();
                histogram.set_prometheus_histogram(prometheus_histogram);
            } else if histogram.prometheus_histogram.is_none() {
                // Create a simple histogram
                let opts = prometheus::HistogramOpts::new(histogram.name(), histogram.help())
                    .buckets(histogram.buckets().to_vec());

                let prometheus_histogram = prometheus::Histogram::with_opts(opts).unwrap();
                registry
                    .register(Box::new(prometheus_histogram.clone()))
                    .unwrap();
                histogram.set_prometheus_histogram(prometheus_histogram);
            }
        }

        let histogram = Arc::new(histogram);
        self.histograms
            .insert(name.to_string(), Arc::clone(&histogram));
        histogram
    }

    /// Get a counter by name
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the counter
    ///
    /// # Returns
    ///
    /// * `Option<Arc<Counter>>` - The counter, if it exists
    pub fn get_counter(&self, name: &str) -> Option<Arc<Counter>> {
        self.counters.get(name).cloned()
    }

    /// Get a gauge by name
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the gauge
    ///
    /// # Returns
    ///
    /// * `Option<Arc<Gauge>>` - The gauge, if it exists
    pub fn get_gauge(&self, name: &str) -> Option<Arc<Gauge>> {
        self.gauges.get(name).cloned()
    }

    /// Get a histogram by name
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the histogram
    ///
    /// # Returns
    ///
    /// * `Option<Arc<Histogram>>` - The histogram, if it exists
    pub fn get_histogram(&self, name: &str) -> Option<Arc<Histogram>> {
        self.histograms.get(name).cloned()
    }

    /// Get all counter names
    ///
    /// # Returns
    ///
    /// * `HashSet<String>` - A set of counter names
    pub fn counter_names(&self) -> HashSet<String> {
        self.counters.keys().cloned().collect()
    }

    /// Get all gauge names
    ///
    /// # Returns
    ///
    /// * `HashSet<String>` - A set of gauge names
    pub fn gauge_names(&self) -> HashSet<String> {
        self.gauges.keys().cloned().collect()
    }

    /// Get all histogram names
    ///
    /// # Returns
    ///
    /// * `HashSet<String>` - A set of histogram names
    pub fn histogram_names(&self) -> HashSet<String> {
        self.histograms.keys().cloned().collect()
    }

    #[cfg(feature = "prometheus")]
    /// Get a reference to the Prometheus registry
    ///
    /// # Returns
    ///
    /// * `Option<&Registry>` - The Prometheus registry, if available
    pub(crate) fn prometheus_registry(&self) -> Option<&Registry> {
        self.prometheus_registry.as_ref()
    }
}
