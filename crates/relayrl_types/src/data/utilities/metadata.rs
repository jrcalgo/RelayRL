//! Metadata and provenance tracking for RL telemetry data

use bincode::config;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub created_at: u64, // Unix timestamp
    pub model_version: i64,
    pub training_step: u64,
    /// Episode number (for trajectory data)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episode: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Tensor statistics (useful for monitoring)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistics: Option<TensorStatistics>,
    /// Network transport info
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<TransportMetadata>,
    /// Custom key-value metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub enum MetadataError {
    SerializationError(String),
    DeserializationError(String),
}

impl TensorMetadata {
    pub fn new(model_version: i64, training_step: u64) -> Self {
        Self {
            created_at: current_timestamp(),
            model_version,
            training_step,
            episode: None,
            agent_id: None,
            statistics: None,
            transport: None,
            custom: None,
        }
    }

    pub fn with_episode(mut self, episode: u64) -> Self {
        self.episode = Some(episode);
        self
    }

    pub fn with_agent_id(mut self, agent_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_statistics(mut self, stats: TensorStatistics) -> Self {
        self.statistics = Some(stats);
        self
    }

    pub fn with_transport(mut self, transport: TransportMetadata) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Age of this data in seconds
    pub fn age_seconds(&self) -> u64 {
        current_timestamp().saturating_sub(self.created_at)
    }

    /// Serialize to compact binary format
    #[cfg(feature = "metadata")]
    pub fn to_binary(&self) -> Result<Vec<u8>, MetadataError> {
        bincode::serde::encode_to_vec(self, config::standard())
            .map_err(|e| MetadataError::SerializationError(e.to_string()))
    }

    /// Deserialize from binary format
    #[cfg(feature = "metadata")]
    pub fn from_binary(data: &[u8]) -> Result<(Self, usize), MetadataError> {
        bincode::serde::decode_from_slice(data, config::standard())
            .map_err(|e| MetadataError::DeserializationError(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportMetadata {
    /// Was data compressed?
    pub compressed: bool,
    /// Was data encrypted?
    pub encrypted: bool,
    /// Original size in bytes
    pub original_size: usize,
    pub transmitted_size: usize,
    /// Compression ratio (if compressed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_ratio: Option<f32>,
    /// Checksum/hash (if verified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<[u8; 32]>,
}

impl TransportMetadata {
    pub fn new(original_size: usize, transmitted_size: usize) -> Self {
        Self {
            compressed: original_size != transmitted_size,
            encrypted: false,
            original_size,
            transmitted_size,
            compression_ratio: if original_size != transmitted_size {
                Some(original_size as f32 / transmitted_size as f32)
            } else {
                None
            },
            checksum: None,
        }
    }
}

pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
