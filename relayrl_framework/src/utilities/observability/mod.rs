// RL4Sys Observability Module
//
// This module provides observability features for the RL4Sys framework,
// including logging and metrics functionality for distributed reinforcement
// learning applications.

// Re-export logging submodules
#[cfg(feature = "logging")]
pub mod logging;

// Re-export metrics submodules
#[cfg(feature = "metrics")]
pub mod metrics;

/// Initialize all observability components based on current configuration
pub fn init_observability() {
    #[cfg(feature = "logging")]
    logging::init_logging();
    
    #[cfg(feature = "metrics")]
    metrics::init_metrics();
} 