//! RL4Sys Metrics Definitions
//!
//! This module defines the core metric types used throughout the RL4Sys framework,
//! including counters, gauges, and histograms.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "prometheus")]
use prometheus::{
    self, Counter as PrometheusCounter, Gauge as PrometheusGauge, Histogram as PrometheusHistogram,
    core::{AtomicF64, AtomicU64, GenericCounter, GenericGauge},
};

/// A metric that represents a single numerical value that only ever goes up
pub struct Counter {
    name: String,
    help: String,
    labels: HashMap<String, String>,
    #[cfg(feature = "prometheus")]
    pub(crate) prometheus_counter: Option<PrometheusCounter>,
    value: Arc<Mutex<u64>>,
}

impl Counter {
    /// Create a new counter
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the counter
    /// * `help` - Description of what the counter measures
    /// * `labels` - Optional labels as key-value pairs
    ///
    /// # Returns
    ///
    /// * `Self` - A new Counter instance
    pub fn new(name: &str, help: &str, labels: Option<HashMap<String, String>>) -> Self {
        let labels = labels.unwrap_or_default();

        #[cfg(feature = "prometheus")]
        let prometheus_counter = {
            let opts = prometheus::Opts::new(name, help);
            if labels.is_empty() {
                Some(prometheus::Counter::with_opts(opts).unwrap())
            } else {
                None // Will be initialized with registry
            }
        };

        Self {
            name: name.to_string(),
            help: help.to_string(),
            #[cfg(feature = "prometheus")]
            prometheus_counter,
            labels,
            value: Arc::new(Mutex::new(0)),
        }
    }

    /// Increment the counter by 1
    pub fn inc(&self) {
        let mut value = self.value.lock().unwrap();
        *value += 1;

        #[cfg(feature = "prometheus")]
        if let Some(ref counter) = self.prometheus_counter {
            counter.inc();
        }
    }

    /// Increment the counter by the given amount
    ///
    /// # Arguments
    ///
    /// * `v` - The amount to increment by
    pub fn inc_by(&self, v: u64) {
        let mut value = self.value.lock().unwrap();
        *value += v;

        #[cfg(feature = "prometheus")]
        if let Some(ref counter) = self.prometheus_counter {
            counter.inc_by(v as f64);
        }
    }

    /// Get the current value of the counter
    ///
    /// # Returns
    ///
    /// * `u64` - The current counter value
    pub fn get(&self) -> u64 {
        *self.value.lock().unwrap()
    }

    /// Get the name of the counter
    ///
    /// # Returns
    ///
    /// * `&str` - The counter name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the help text for the counter
    ///
    /// # Returns
    ///
    /// * `&str` - The help text
    pub fn help(&self) -> &str {
        &self.help
    }

    /// Get the labels associated with this counter
    ///
    /// # Returns
    ///
    /// * `&HashMap<String, String>` - The labels
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    #[cfg(feature = "prometheus")]
    /// Set the Prometheus counter
    ///
    /// # Arguments
    ///
    /// * `counter` - The Prometheus counter
    pub(crate) fn set_prometheus_counter(&mut self, counter: PrometheusCounter) {
        self.prometheus_counter = Some(counter);
    }
}

/// A metric that represents a single numerical value that can go up and down
pub struct Gauge {
    name: String,
    help: String,
    labels: HashMap<String, String>,
    #[cfg(feature = "prometheus")]
    pub(crate) prometheus_gauge: Option<PrometheusGauge>,
    value: Arc<Mutex<f64>>,
}

impl Gauge {
    /// Create a new gauge
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the gauge
    /// * `help` - Description of what the gauge measures
    /// * `labels` - Optional labels as key-value pairs
    ///
    /// # Returns
    ///
    /// * `Self` - A new Gauge instance
    pub fn new(name: &str, help: &str, labels: Option<HashMap<String, String>>) -> Self {
        let labels = labels.unwrap_or_default();

        #[cfg(feature = "prometheus")]
        let prometheus_gauge = {
            let opts = prometheus::Opts::new(name, help);
            if labels.is_empty() {
                Some(prometheus::Gauge::with_opts(opts).unwrap())
            } else {
                None // Will be initialized with registry
            }
        };

        Self {
            name: name.to_string(),
            help: help.to_string(),
            #[cfg(feature = "prometheus")]
            prometheus_gauge,
            labels,
            value: Arc::new(Mutex::new(0.0)),
        }
    }

    /// Set the gauge to the given value
    ///
    /// # Arguments
    ///
    /// * `v` - The value to set
    pub fn set(&self, v: f64) {
        let mut value = self.value.lock().unwrap();
        *value = v;

        #[cfg(feature = "prometheus")]
        if let Some(ref gauge) = self.prometheus_gauge {
            gauge.set(v);
        }
    }

    /// Increment the gauge by 1
    pub fn inc(&self) {
        let mut value = self.value.lock().unwrap();
        *value += 1.0;

        #[cfg(feature = "prometheus")]
        if let Some(ref gauge) = self.prometheus_gauge {
            gauge.inc();
        }
    }

    /// Increment the gauge by the given amount
    ///
    /// # Arguments
    ///
    /// * `v` - The amount to increment by
    pub fn inc_by(&self, v: f64) {
        let mut value = self.value.lock().unwrap();
        *value += v;

        #[cfg(feature = "prometheus")]
        if let Some(ref gauge) = self.prometheus_gauge {
            gauge.add(v);
        }
    }

    /// Decrement the gauge by 1
    pub fn dec(&self) {
        let mut value = self.value.lock().unwrap();
        *value -= 1.0;

        #[cfg(feature = "prometheus")]
        if let Some(ref gauge) = self.prometheus_gauge {
            gauge.dec();
        }
    }

    /// Decrement the gauge by the given amount
    ///
    /// # Arguments
    ///
    /// * `v` - The amount to decrement by
    pub fn dec_by(&self, v: f64) {
        let mut value = self.value.lock().unwrap();
        *value -= v;

        #[cfg(feature = "prometheus")]
        if let Some(ref gauge) = self.prometheus_gauge {
            gauge.sub(v);
        }
    }

    /// Get the current value of the gauge
    ///
    /// # Returns
    ///
    /// * `f64` - The current gauge value
    pub fn get(&self) -> f64 {
        *self.value.lock().unwrap()
    }

    /// Get the name of the gauge
    ///
    /// # Returns
    ///
    /// * `&str` - The gauge name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the help text for the gauge
    ///
    /// # Returns
    ///
    /// * `&str` - The help text
    pub fn help(&self) -> &str {
        &self.help
    }

    /// Get the labels associated with this gauge
    ///
    /// # Returns
    ///
    /// * `&HashMap<String, String>` - The labels
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    #[cfg(feature = "prometheus")]
    /// Set the Prometheus gauge
    ///
    /// # Arguments
    ///
    /// * `gauge` - The Prometheus gauge
    pub(crate) fn set_prometheus_gauge(&mut self, gauge: PrometheusGauge) {
        self.prometheus_gauge = Some(gauge);
    }
}

/// A metric that tracks the distribution of a set of values
pub struct Histogram {
    name: String,
    help: String,
    labels: HashMap<String, String>,
    buckets: Vec<f64>,
    #[cfg(feature = "prometheus")]
    pub(crate) prometheus_histogram: Option<PrometheusHistogram>,
    values: Arc<Mutex<Vec<f64>>>,
}

impl Histogram {
    /// Create a new histogram
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
    /// * `Self` - A new Histogram instance
    pub fn new(
        name: &str,
        help: &str,
        buckets: Option<Vec<f64>>,
        labels: Option<HashMap<String, String>>,
    ) -> Self {
        let labels = labels.unwrap_or_default();
        let buckets = buckets.unwrap_or_else(|| {
            vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]
        });

        #[cfg(feature = "prometheus")]
        let prometheus_histogram = {
            let opts = prometheus::HistogramOpts::new(name, help).buckets(buckets.clone());

            if labels.is_empty() {
                Some(prometheus::Histogram::with_opts(opts).unwrap())
            } else {
                None // Will be initialized with registry
            }
        };

        Self {
            name: name.to_string(),
            help: help.to_string(),
            buckets,
            #[cfg(feature = "prometheus")]
            prometheus_histogram,
            labels,
            values: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Observe a value
    ///
    /// # Arguments
    ///
    /// * `v` - The value to observe
    pub fn observe(&self, v: f64) {
        let mut values = self.values.lock().unwrap();
        values.push(v);

        #[cfg(feature = "prometheus")]
        if let Some(ref histogram) = self.prometheus_histogram {
            histogram.observe(v);
        }
    }

    /// Get the name of the histogram
    ///
    /// # Returns
    ///
    /// * `&str` - The histogram name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the help text for the histogram
    ///
    /// # Returns
    ///
    /// * `&str` - The help text
    pub fn help(&self) -> &str {
        &self.help
    }

    /// Get the buckets for this histogram
    ///
    /// # Returns
    ///
    /// * `&[f64]` - The histogram buckets
    pub fn buckets(&self) -> &[f64] {
        &self.buckets
    }

    /// Get the labels associated with this histogram
    ///
    /// # Returns
    ///
    /// * `&HashMap<String, String>` - The labels
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    /// Get all observed values
    ///
    /// # Returns
    ///
    /// * `Vec<f64>` - A copy of all values observed
    pub fn values(&self) -> Vec<f64> {
        self.values.lock().unwrap().clone()
    }

    #[cfg(feature = "prometheus")]
    /// Set the Prometheus histogram
    ///
    /// # Arguments
    ///
    /// * `histogram` - The Prometheus histogram
    pub(crate) fn set_prometheus_histogram(&mut self, histogram: PrometheusHistogram) {
        self.prometheus_histogram = Some(histogram);
    }
}
