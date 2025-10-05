//! RelayRL File Logging Sink
//!
//! This module provides file output capabilities for the RelayRL logging system,
//! allowing logs to be persisted to disk with configurable rotation policies.

use log4rs::{
    append::file::FileAppender,
    append::rolling_file::{
        RollingFileAppender, policy::compound::CompoundPolicy,
        policy::compound::roll::fixed_window::FixedWindowRoller,
        policy::compound::trigger::size::SizeTrigger,
    },
    encode::pattern::PatternEncoder,
};
use std::path::Path;

/// Creates a simple file appender with no rotation
///
/// # Arguments
///
/// * `path` - Path to the log file
///
/// # Returns
///
/// * `FileAppender` - A configured file appender
pub fn create_file_appender(path: &Path) -> FileAppender {
    FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "[{d(%Y-%m-%d %H:%M:%S%.3f)} {l} {M}] {m}{n}",
        )))
        .build(path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to create file appender at {:?}: {}", path, e);
            panic!("Failed to create file appender");
        })
}

/// Creates a rotating file appender based on file size
///
/// # Arguments
///
/// * `base_path` - Base path for the log file
/// * `file_size_mb` - Size in megabytes to trigger rotation
/// * `rotation_count` - Number of archived log files to keep
///
/// # Returns
///
/// * `RollingFileAppender` - A configured rolling file appender
pub fn create_size_rotating_appender(
    base_path: &Path,
    file_size_mb: u64,
    rotation_count: u32,
) -> RollingFileAppender {
    let path_str = base_path.to_str().unwrap_or("logs/relayrl.log");
    let base_filename = path_str.to_string();

    // Configure the rotation policy
    let window_size = rotation_count;
    let fixed_window_roller = FixedWindowRoller::builder()
        .build(&format!("{}.{{}}", base_filename), window_size)
        .unwrap_or_else(|e| {
            eprintln!("Failed to create roller for {}: {}", base_filename, e);
            panic!("Failed to create log file roller");
        });

    // Set up the size trigger (in bytes)
    let size_mb = file_size_mb * 1024 * 1024;
    let size_trigger = SizeTrigger::new(size_mb);

    // Combine policies
    let compound_policy =
        CompoundPolicy::new(Box::new(size_trigger), Box::new(fixed_window_roller));

    // Create the rolling file appender
    RollingFileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S%.3f)} {h({l})} [{t}] {m}{n}",
        )))
        .build(base_path, Box::new(compound_policy))
        .unwrap_or_else(|e| {
            eprintln!(
                "Failed to create rolling file appender at {:?}: {}",
                base_path, e
            );
            panic!("Failed to create rolling file appender");
        })
}

/// Creates a daily rotating file appender
///
/// # Arguments
///
/// * `base_path` - Base path for the log file
/// * `rotation_count` - Number of archived log files to keep
///
/// # Returns
///
/// * `RollingFileAppender` - A configured rolling file appender
pub fn create_daily_rotating_appender(
    base_path: &Path,
    rotation_count: u32,
) -> RollingFileAppender {
    let path_str = base_path.to_str().unwrap_or("logs/relayrl.log");
    let base_filename = path_str.to_string();

    // Configure the daily rotation policy
    let window_size = rotation_count;
    let fixed_window_roller = FixedWindowRoller::builder()
        .build(&format!("{}.{{}}", base_filename), window_size)
        .unwrap_or_else(|e| {
            eprintln!("Failed to create roller for {}: {}", base_filename, e);
            panic!("Failed to create log file roller");
        });

    // Set up time-based rotation (you might want to change this to use a TimeTrigger)
    // For now, using size trigger as a placeholder
    let size_trigger = SizeTrigger::new(10 * 1024 * 1024); // 10MB as placeholder

    // Combine policies
    let compound_policy =
        CompoundPolicy::new(Box::new(size_trigger), Box::new(fixed_window_roller));

    // Create the rolling file appender
    RollingFileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S%.3f)} {h({l})} [{t}] {m}{n}",
        )))
        .build(base_path, Box::new(compound_policy))
        .unwrap_or_else(|e| {
            eprintln!(
                "Failed to create rolling file appender at {:?}: {}",
                base_path, e
            );
            panic!("Failed to create rolling file appender");
        })
}

/// Creates a JSON-formatted file appender for machine parsing
///
/// # Arguments
///
/// * `path` - Path to the log file
///
/// # Returns
///
/// * `FileAppender` - A configured file appender with JSON formatting
pub fn create_json_file_appender(path: &Path) -> FileAppender {
    let pattern = r#"{"time":"{d(%Y-%m-%dT%H:%M:%S%.3fZ)}","level":"{l}","target":"{t}","thread":"{T}","message":"{m}"}{n}"#;

    FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build(path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to create JSON file appender at {:?}: {}", path, e);
            panic!("Failed to create JSON file appender");
        })
}
