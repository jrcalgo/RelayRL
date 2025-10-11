//! RelayRL Logging Sinks
//!
//! This module contains various logging sinks (outputs) used by the RelayRL
//! framework for directing logs to different destinations.

pub mod console;
pub mod file;

// Re-export commonly used types and functions
pub use console::{
    create_colored_console_appender, create_console_appender, create_json_console_appender,
};

pub use file::{create_file_appender, create_json_file_appender, create_size_rotating_appender};
