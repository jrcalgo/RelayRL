/// Wraps a Prometheus (or OpenTelemetry) registry, offering async counters/histogram.
pub(crate) struct MetricsManager;

impl MetricsManager {
    pub fn new() -> Self {
        MetricsManager
    }

    fn record_counter(&self, name: &str, value: u64) {
        todo!()
    }

    fn record_histogram(&self, name: &str, value: f64) {
        todo!()
    }
}
