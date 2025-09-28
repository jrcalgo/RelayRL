//! RL4Sys Logging Builder
//!
//! This module provides a fluent builder API for constructing
//! custom logging configurations for the RL4Sys framework.

use super::sinks::{console::create_console_appender, file::create_file_appender};
use log::LevelFilter;
use log4rs::config::runtime::{ConfigBuilder, RootBuilder};
use log4rs::{
    config::{Appender, Config, Logger, Root},
    filter::threshold::ThresholdFilter,
};
use std::path::Path;

/// LoggingBuilder provides a fluent API for configuring the logging subsystem
///
/// This builder simplifies the process of creating a custom log4rs configuration,
/// allowing users to specify log levels, appenders, and module-specific settings.
pub struct LoggingBuilder {
    config_builder: ConfigBuilder,
    root_builder: RootBuilder,
    appenders: Vec<String>,
}

impl LoggingBuilder {
    /// Create a new logging builder with a default configuration
    pub fn new() -> Self {
        Self {
            config_builder: Config::builder(),
            root_builder: Root::builder(),
            appenders: Vec::new(),
        }
    }

    /// Add a console appender to the configuration
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the appender
    /// * `level` - Minimum log level to display
    ///
    /// # Returns
    ///
    /// * `Self` - The builder instance for method chaining
    pub fn with_console(mut self, name: &str, level: LevelFilter) -> Self {
        let console = create_console_appender();
        let appender = Appender::builder()
            .filter(Box::new(ThresholdFilter::new(level)))
            .build(name, Box::new(console));

        self.config_builder = self.config_builder.appender(appender);
        self.appenders.push(name.to_string());
        self.root_builder = self.root_builder.appender(name);

        self
    }

    /// Add a file appender to the configuration
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the appender
    /// * `path` - Path to the log file
    /// * `level` - Minimum log level to record
    ///
    /// # Returns
    ///
    /// * `Self` - The builder instance for method chaining
    pub fn with_file(mut self, name: &str, path: &str, level: LevelFilter) -> Self {
        let path = Path::new(path);
        if let Some(parent_dir) = path.parent() {
            std::fs::create_dir_all(parent_dir).unwrap_or_else(|_| {
                eprintln!(
                    "Failed to create directories for log file: {:?}",
                    parent_dir
                );
            });
        }

        let file = create_file_appender(path);
        let appender = Appender::builder()
            .filter(Box::new(ThresholdFilter::new(level)))
            .build(name, Box::new(file));

        self.config_builder = self.config_builder.appender(appender);
        self.appenders.push(name.to_string());
        self.root_builder = self.root_builder.appender(name);

        self
    }

    /// Set the default log level for the root logger
    ///
    /// # Arguments
    ///
    /// * `level` - The log level to set
    ///
    /// # Returns
    ///
    /// * `Self` - The builder instance for method chaining
    pub fn with_level(mut self, level: LevelFilter) -> Self {
        self.root_builder = self.root_builder.build(level);
        self
    }

    /// Configure a specific module's log level
    ///
    /// # Arguments
    ///
    /// * `module` - The module path (e.g., "rl4sys_framework::network")
    /// * `level` - The log level for this module
    ///
    /// # Returns
    ///
    /// * `Self` - The builder instance for method chaining
    pub fn with_module_level(mut self, module: &str, level: LevelFilter) -> Self {
        let logger = Logger::builder().build(module, level);

        self.config_builder = self.config_builder.logger(logger);
        self
    }

    /// Build the final logging configuration
    ///
    /// # Returns
    ///
    /// * `Result<Config, String>` - The built configuration or an error
    pub fn build(self) -> Result<Config, String> {
        if self.appenders.is_empty() {
            return Err("At least one appender must be configured".to_string());
        }

        match self.config_builder.build(self.root_builder) {
            Ok(config) => Ok(config),
            Err(e) => Err(format!("Failed to build log configuration: {}", e)),
        }
    }
}
