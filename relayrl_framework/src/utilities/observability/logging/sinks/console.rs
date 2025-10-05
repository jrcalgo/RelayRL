//! RelayRL Console Logging Sink
//!
//! This module provides console output capabilities for the RelayRL logging system,
//! allowing logs to be displayed to stdout and stderr with customizable formatting.

use log4rs::{
    append::console::{ConsoleAppender, Target},
    encode::pattern::PatternEncoder,
};

/// Creates a console appender with default settings (colored output to stdout)
///
/// # Returns
///
/// * `ConsoleAppender` - A configured console appender
pub fn create_console_appender() -> ConsoleAppender {
    ConsoleAppender::builder()
        .target(Target::Stdout)
        .encoder(Box::new(PatternEncoder::new(
            "[{d(%Y-%m-%d %H:%M:%S%.3f)} {h({l})} {M}] {m}{n}",
        )))
        .build()
}

/// Creates a console appender with custom settings
///
/// # Arguments
///
/// * `target` - Stdout or Stderr
/// * `pattern` - Log formatting pattern
///
/// # Returns
///
/// * `ConsoleAppender` - A customized console appender
pub fn create_custom_console_appender(target: Target, pattern: &str) -> ConsoleAppender {
    ConsoleAppender::builder()
        .target(target)
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build()
}

/// Creates an ANSI color-enabled console appender for enhanced readability
///
/// # Returns
///
/// * `ConsoleAppender` - A configured console appender with color support
pub fn create_colored_console_appender() -> ConsoleAppender {
    let pattern = r#"{d(%Y-%m-%d %H:%M:%S%.3f)} {highlight({l})} \
                    {magenta}[{T}]{end} {cyan}{M}{end} - {message}{n}"#;

    ConsoleAppender::builder()
        .target(Target::Stdout)
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build()
}

/// Creates a minimalistic console appender for performance-critical logging
///
/// # Returns
///
/// * `ConsoleAppender` - A configured console appender with minimal formatting
pub fn create_minimal_console_appender() -> ConsoleAppender {
    ConsoleAppender::builder()
        .target(Target::Stdout)
        .encoder(Box::new(PatternEncoder::new("{l} {m}{n}")))
        .build()
}

/// Creates a JSON-formatting console appender for machine-readable logs
///
/// # Returns
///
/// * `ConsoleAppender` - A configured console appender with JSON output
pub fn create_json_console_appender() -> ConsoleAppender {
    let pattern = r#"{"timestamp":"{d(%Y-%m-%dT%H:%M:%S%.3fZ)}","level":"{l}","target":"{T}","message":"{m}"}"#;

    ConsoleAppender::builder()
        .target(Target::Stdout)
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build()
}
