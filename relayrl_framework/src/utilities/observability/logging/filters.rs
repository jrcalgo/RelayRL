//! RL4Sys Logging Filters
//!
//! This module provides custom logging filters to control log output
//! based on the needs of the RL4Sys framework.

use log::Level;
use log4rs::filter::Filter;
use std::fmt;

/// A filter that only allows logs from specific components or modules
pub struct ComponentFilter {
    component_name: String,
}

impl ComponentFilter {
    /// Creates a new component filter
    ///
    /// # Arguments
    ///
    /// * `component_name` - The name of the component to allow logs from
    ///
    /// # Returns
    ///
    /// * `Self` - A new ComponentFilter instance
    pub fn new(component_name: &str) -> Self {
        Self {
            component_name: component_name.to_string(),
        }
    }
}

impl Filter for ComponentFilter {
    fn filter(&self, record: &log::Record) -> log4rs::filter::Response {
        if let Some(target) = record.module_path() {
            if target.contains(&self.component_name) {
                log4rs::filter::Response::Accept
            } else {
                log4rs::filter::Response::Neutral
            }
        } else {
            log4rs::filter::Response::Neutral
        }
    }
}

impl fmt::Debug for ComponentFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ComponentFilter")
            .field("component_name", &self.component_name)
            .finish()
    }
}

/// A filter that includes performance-related logs
pub struct PerformanceFilter;

impl PerformanceFilter {
    /// Creates a new performance filter
    pub fn new() -> Self {
        Self
    }
}

impl Filter for PerformanceFilter {
    fn filter(&self, record: &log::Record) -> log4rs::filter::Response {
        if record
            .key_values()
            .get(log::kv::Key::from_str("type"))
            .map(|v| v.to_string())
            == Some("performance".to_string())
        {
            log4rs::filter::Response::Accept
        } else {
            log4rs::filter::Response::Neutral
        }
    }
}

impl fmt::Debug for PerformanceFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("PerformanceFilter").finish()
    }
}

/// A filter that only allows logs with certain metadata tags
pub struct MetadataFilter {
    key: String,
    value: String,
}

impl MetadataFilter {
    /// Creates a new metadata filter
    ///
    /// # Arguments
    ///
    /// * `key` - The metadata key to filter on
    /// * `value` - The value that the metadata key should have
    ///
    /// # Returns
    ///
    /// * `Self` - A new MetadataFilter instance
    pub fn new(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
        }
    }
}

impl Filter for MetadataFilter {
    fn filter(&self, record: &log::Record) -> log4rs::filter::Response {
        if record
            .key_values()
            .get(log::kv::Key::from_str(&self.key))
            .map(|v| v.to_string())
            == Some(self.value.clone())
        {
            log4rs::filter::Response::Accept
        } else {
            log4rs::filter::Response::Neutral
        }
    }
}

impl fmt::Debug for MetadataFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MetadataFilter")
            .field("key", &self.key)
            .field("value", &self.value)
            .finish()
    }
}

/// A filter that only includes critical logs regardless of level
pub struct CriticalFilter;

impl CriticalFilter {
    /// Creates a new critical filter
    pub fn new() -> Self {
        Self
    }
}

impl Filter for CriticalFilter {
    fn filter(&self, record: &log::Record) -> log4rs::filter::Response {
        if record.level() >= Level::Error
            || record
                .key_values()
                .get(log::kv::Key::from_str("critical"))
                .is_some()
        {
            log4rs::filter::Response::Accept
        } else {
            log4rs::filter::Response::Neutral
        }
    }
}

impl fmt::Debug for CriticalFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("CriticalFilter").finish()
    }
}
