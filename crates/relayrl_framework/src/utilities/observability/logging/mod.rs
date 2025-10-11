//! RelayRL Logging Module
//!
//! This module provides logging functionality for the RelayRL framework,
//! enabling client and server components to emit structured logs for debugging,
//! monitoring, and auditing purposes.

use log::LevelFilter;
use log4rs::{
    append::console::{ConsoleAppender, Target},
    config::{Appender, Config, Root},
    filter::threshold::ThresholdFilter,
};
use std::sync::{Mutex, Once};

// Re-export log to provide users with a consistent interface
pub use log::{Level, debug, error, info, trace, warn};

// Module exports
pub mod builder;
pub mod filters;
pub mod sinks;

// Global initialization guard
static INIT: Once = Once::new();

/// Initializes the logging subsystem with default configuration
///
/// This function is called automatically when the observability module
/// is initialized. It configures log4rs with sensible defaults.
pub fn init_logging() {
    INIT.call_once(|| {
        // Set up default console logger
        let stdout = ConsoleAppender::builder().target(Target::Stdout).build();

        // Create a basic configuration
        let config = Config::builder()
            .appender(
                Appender::builder()
                    .filter(Box::new(ThresholdFilter::new(LevelFilter::Info)))
                    .build("stdout", Box::new(stdout)),
            )
            .build(Root::builder().appender("stdout").build(LevelFilter::Info))
            .unwrap();

        // Initialize the logger with this configuration
        match log4rs::init_config(config) {
            Ok(_) => log::info!("RelayRL logging initialized with default configuration"),
            Err(e) => eprintln!("Failed to initialize RelayRL logging: {}", e),
        }
    });
}

/// Initializes logging with a custom configuration file
///
/// # Arguments
///
/// * `config_path` - Path to the log4rs YAML configuration file
///
/// # Returns
///
/// * `Result<(), String>` - Success or error message
pub fn init_logging_from_file(config_path: &str) -> Result<(), String> {
    let result = Mutex::new(Ok(()));
    INIT.call_once(
        || match log4rs::init_file(config_path, Default::default()) {
            Ok(_) => log::info!(
                "RelayRL logging initialized from config file: {}",
                config_path
            ),
            Err(e) => {
                let mut result_guard = result.lock().unwrap();
                *result_guard = Err(format!("Failed to initialize logging: {}", e));
            }
        },
    );

    result.into_inner().unwrap()
}

/// Resets and reconfigures logging with builder-created configuration
///
/// # Arguments
///
/// * `config` - log4rs::Config to use for reconfiguration
///
/// # Returns
///
/// * `Result<(), String>` - Success or error message
pub fn reconfigure_logging(config: Config) -> Result<(), String> {
    match log4rs::init_config(config) {
        Ok(_) => {
            log::info!("RelayRL logging reconfigured successfully");
            Ok(())
        }
        Err(e) => Err(format!("Failed to reconfigure logging: {}", e)),
    }
}
