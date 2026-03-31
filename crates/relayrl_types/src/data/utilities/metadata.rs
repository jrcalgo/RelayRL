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

#[cfg(test)]
mod unit_tests {
    use super::*;
    fn sample_statistics() -> TensorStatistics {
        TensorStatistics {
            mean: 1.5,
            std: 0.5,
            min: 1.0,
            max: 2.0,
            shape: vec![2, 2],
            dtype: "NdArray(F32)".to_string(),
        }
    }

    #[test]
    fn metadata_builder_helpers_populate_optional_fields() {
        let transport = TransportMetadata::new(100, 50);
        let metadata = TensorMetadata::new(3, 5)
            .with_episode(7)
            .with_agent_id("agent-1".to_string())
            .with_statistics(sample_statistics())
            .with_transport(transport.clone());

        assert_eq!(metadata.model_version, 3);
        assert_eq!(metadata.training_step, 5);
        assert_eq!(metadata.episode, Some(7));
        assert_eq!(metadata.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(metadata.statistics.as_ref().unwrap().shape, vec![2, 2]);
        assert_eq!(metadata.transport.as_ref().unwrap().original_size, 100);
    }

    #[test]
    fn metadata_age_uses_created_at_timestamp() {
        let mut metadata = TensorMetadata::new(1, 1);
        metadata.created_at = metadata.created_at.saturating_sub(2);

        assert!(metadata.age_seconds() >= 2);
    }

    #[test]
    fn transport_metadata_detects_compression() {
        let compressed = TransportMetadata::new(200, 50);
        let passthrough = TransportMetadata::new(128, 128);

        assert!(compressed.compressed);
        assert_eq!(compressed.compression_ratio, Some(4.0));
        assert!(!passthrough.compressed);
        assert!(passthrough.compression_ratio.is_none());
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn metadata_binary_helpers_encode_and_reject_truncated_payloads() {
        let metadata = TensorMetadata::new(2, 10)
            .with_episode(3)
            .with_agent_id("agent-2".to_string());

        let bytes = metadata.to_binary().unwrap();
        let err = TensorMetadata::from_binary(&bytes[..bytes.len() - 1])
            .expect_err("truncated metadata payloads should fail to deserialize");

        assert!(!bytes.is_empty());
        assert!(matches!(err, MetadataError::DeserializationError(_)));
    }
}
