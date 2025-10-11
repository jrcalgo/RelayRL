// RelayRL Observability Module
//
// This module provides observability features for the RelayRL framework,
// including logging and metrics functionality for distributed reinforcement
// learning applications.

// Re-export logging submodules
#[cfg(feature = "logging")]
pub mod logging;

// Re-export metrics submodules
#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "metrics")]
use metrics::MetricsManager;

/// Initialize all observability components based on current configuration
#[cfg(feature = "metrics")]
pub fn init_observability() -> MetricsManager {
    #[cfg(feature = "logging")]
    logging::init_logging();

    metrics::init_metrics()
}

#[cfg(not(feature = "metrics"))]
pub fn init_observability() {
    #[cfg(feature = "logging")]
    logging::init_logging();
}
