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
