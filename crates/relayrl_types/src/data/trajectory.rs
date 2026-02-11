//! RelayRL Trajectory types for collecting sequences of actions
//!
//! Trajectories are sequences of actions that form episodes in reinforcement learning.
//! Supports batching, compression, and network transmission optimizations.

use crate::types::data::action::{ActionError, RelayRLAction};
use bincode::config;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(feature = "metadata")]
use crate::types::data::action::CodecConfig;

#[cfg(feature = "integrity")]
use crate::types::data::utilities::chunking::{ChunkedTensor, TensorChunk};
#[cfg(feature = "compression")]
use crate::types::data::utilities::compress::{CompressedData, CompressionScheme};
#[cfg(feature = "encryption")]
use crate::types::data::utilities::encrypt::{EncryptedData, EncryptionKey};
#[cfg(feature = "integrity")]
use crate::types::data::utilities::integrity::{compute_checksum, Checksum};
#[cfg(feature = "metadata")]
use crate::types::data::utilities::metadata::TensorMetadata;

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Failed to get current timestamp")
        .as_secs()
}

/// Core trajectory structure for RelayRL
///
/// A trajectory is a sequence of actions representing an episode or partial episode.
/// Includes metadata for tracking provenance and enabling distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayRLTrajectory {
    pub actions: Vec<RelayRLAction>,
    pub max_length: usize,
    pub agent_id: Option<Uuid>,
    pub timestamp: u64,
    pub episode: Option<u64>,
    pub training_step: Option<u64>,
}

impl Default for RelayRLTrajectory {
    fn default() -> Self {
        let default_length: usize = 1000;
        Self {
            actions: Vec::with_capacity(default_length),
            max_length: default_length,
            agent_id: None,
            timestamp: current_timestamp(),
            episode: None,
            training_step: None,
        }
    }
}

impl RelayRLTrajectory {
    pub fn new(max_length: usize) -> Self {
        Self {
            actions: Vec::with_capacity(max_length),
            max_length,
            agent_id: None,
            timestamp: current_timestamp(),
            episode: None,
            training_step: None,
        }
    }

    pub fn with_agent_id(max_length: usize, agent_id: Uuid) -> Self {
        Self {
            actions: Vec::with_capacity(max_length),
            max_length,
            agent_id: Some(agent_id),
            timestamp: current_timestamp(),
            episode: None,
            training_step: None,
        }
    }

    pub fn with_metadata(
        max_length: usize,
        agent_id: Option<Uuid>,
        episode: Option<u64>,
        training_step: Option<u64>,
    ) -> Self {
        Self {
            actions: Vec::with_capacity(max_length),
            max_length,
            agent_id,
            timestamp: current_timestamp(),
            episode,
            training_step,
        }
    }

    /// Returns true if trajectory should be flushed (reached max length or episode ended)
    pub fn add_action(&mut self, action: RelayRLAction) -> bool {
        let is_done = action.get_done();
        self.actions.push(action);

        is_done || self.actions.len() >= self.max_length
    }

    pub fn add_action_ref(&mut self, action: &RelayRLAction) -> bool {
        self.add_action(action.clone())
    }

    pub fn clear(&mut self) {
        self.actions.clear();
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn is_complete(&self) -> bool {
        self.actions.last().is_some_and(|a| a.get_done())
    }

    pub fn is_full(&self) -> bool {
        self.actions.len() >= self.max_length
    }

    pub fn total_reward(&self) -> f32 {
        self.actions.iter().map(|a| a.get_rew()).sum()
    }

    pub fn avg_reward(&self) -> f32 {
        if self.actions.is_empty() {
            0.0
        } else {
            self.total_reward() / self.actions.len() as f32
        }
    }

    pub fn age_seconds(&self) -> u64 {
        current_timestamp().saturating_sub(self.timestamp)
    }

    pub fn get_actions(&self) -> &[RelayRLAction] {
        &self.actions
    }

    pub fn get_agent_id(&self) -> Option<&Uuid> {
        self.agent_id.as_ref()
    }

    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }

    pub fn get_episode(&self) -> Option<u64> {
        self.episode
    }

    pub fn get_training_step(&self) -> Option<u64> {
        self.training_step
    }

    pub fn set_agent_id(&mut self, agent_id: Uuid) {
        self.agent_id = Some(agent_id);
    }

    pub fn set_episode(&mut self, episode: u64) {
        self.episode = Some(episode);
    }

    pub fn set_training_step(&mut self, step: u64) {
        self.training_step = Some(step);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedTrajectory {
    pub data: Vec<u8>,

    #[cfg(feature = "metadata")]
    pub metadata: Option<TensorMetadata>,

    #[cfg(feature = "compression")]
    pub compressed: bool,

    #[cfg(feature = "encryption")]
    pub encrypted: bool,

    #[cfg(feature = "integrity")]
    pub checksum: Option<Checksum>,

    pub num_actions: usize,
    pub original_size: usize,
}

#[derive(Debug, Clone)]
pub enum TrajectoryError {
    SerializationError(String),
    DeserializationError(String),

    #[cfg(feature = "compression")]
    CompressionError(String),
    #[cfg(feature = "encryption")]
    EncryptionError(String),
    #[cfg(feature = "integrity")]
    IntegrityError(String),
    ChunkingError(String),

    ActionError(ActionError),
}

impl std::fmt::Display for TrajectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationError(e) => {
                write!(f, "[TrajectoryError] Serialization error: {}", e)
            }
            Self::DeserializationError(e) => {
                write!(f, "[TrajectoryError] Deserialization error: {}", e)
            }
            #[cfg(feature = "compression")]
            Self::CompressionError(e) => write!(f, "[TrajectoryError] Compression error: {}", e),
            #[cfg(feature = "encryption")]
            Self::EncryptionError(e) => write!(f, "[TrajectoryError] Encryption error: {}", e),
            #[cfg(feature = "integrity")]
            Self::IntegrityError(e) => write!(f, "[TrajectoryError] Integrity error: {}", e),
            Self::ChunkingError(e) => write!(f, "[TrajectoryError] Chunking error: {}", e),
            Self::ActionError(e) => write!(f, "[TrajectoryError] Action error: {}", e),
        }
    }
}

impl std::error::Error for TrajectoryError {}

impl From<ActionError> for TrajectoryError {
    fn from(err: ActionError) -> Self {
        TrajectoryError::ActionError(err)
    }
}

impl RelayRLTrajectory {
    /// Processing pipeline:
    /// 1. Serialize to bincode
    /// 2. Compress (if enabled)
    /// 3. Encrypt (if enabled)
    /// 4. Add integrity check (if enabled)
    #[cfg(feature = "metadata")]
    pub fn encode(&self, config: &CodecConfig) -> Result<EncodedTrajectory, TrajectoryError> {
        let original_data = bincode::serde::encode_to_vec(self, config::standard())
            .map_err(|e| TrajectoryError::SerializationError(e.to_string()))?;

        let original_size = original_data.len();
        let mut data = original_data;

        #[cfg(feature = "compression")]
        let compressed = if let Some(scheme) = config.compression {
            let compressed_data = CompressedData::compress(&data, scheme)
                .map_err(|e| TrajectoryError::CompressionError(e.to_string()))?;
            data = bincode::serde::encode_to_vec(&compressed_data, config::standard())
                .map_err(|e| TrajectoryError::SerializationError(e.to_string()))?;
            true
        } else {
            false
        };

        #[cfg(feature = "encryption")]
        let encrypted = if let Some(key) = &config.encryption_key {
            let encrypted_data = EncryptedData::encrypt(&data, key)
                .map_err(|e| TrajectoryError::EncryptionError(e.to_string()))?;
            data = bincode::serde::encode_to_vec(&encrypted_data, config::standard())
                .map_err(|e| TrajectoryError::SerializationError(e.to_string()))?;
            true
        } else {
            false
        };

        #[cfg(feature = "integrity")]
        let checksum = if config.verify_integrity {
            Some(compute_checksum(&data))
        } else {
            None
        };

        Ok(EncodedTrajectory {
            data,
            #[cfg(feature = "metadata")]
            metadata: None,
            #[cfg(feature = "compression")]
            compressed,
            #[cfg(feature = "encryption")]
            encrypted,
            #[cfg(feature = "integrity")]
            checksum,
            num_actions: self.actions.len(),
            original_size,
        })
    }

    /// Reverses the encoding pipeline:
    /// 1. Verify integrity (if enabled)
    /// 2. Decrypt (if encrypted)
    /// 3. Decompress (if compressed)
    /// 4. Deserialize from bincode
    #[cfg(feature = "metadata")]
    pub fn decode(
        encoded: &EncodedTrajectory,
        config: &CodecConfig,
    ) -> Result<(Self, usize), TrajectoryError> {
        let mut data = encoded.data.clone();

        #[cfg(feature = "integrity")]
        if config.verify_integrity && encoded.checksum.is_some() {
            let computed = compute_checksum(&data);
            if computed != encoded.checksum.unwrap() {
                return Err(TrajectoryError::IntegrityError(
                    "Checksum mismatch".to_string(),
                ));
            }
        }

        #[cfg(feature = "encryption")]
        if encoded.encrypted {
            if let Some(key) = &config.encryption_key {
                let (encrypted_data, _): (EncryptedData, usize) =
                    bincode::serde::decode_from_slice(&data, config::standard())
                        .map_err(|e| TrajectoryError::DeserializationError(e.to_string()))?;
                data = encrypted_data
                    .decrypt(key)
                    .map_err(|e| TrajectoryError::EncryptionError(e.to_string()))?;
            } else {
                return Err(TrajectoryError::EncryptionError(
                    "Encryption key required but not provided".to_string(),
                ));
            }
        }

        #[cfg(feature = "compression")]
        if encoded.compressed {
            let (compressed_data, _): (CompressedData, usize) =
                bincode::serde::decode_from_slice(&data, config::standard())
                    .map_err(|e| TrajectoryError::DeserializationError(e.to_string()))?;
            data = compressed_data
                .decompress()
                .map_err(|e| TrajectoryError::CompressionError(e.to_string()))?;
        }

        let (trajectory, bytes_read): (RelayRLTrajectory, usize) =
            bincode::serde::decode_from_slice(&data, config::standard())
                .map_err(|e| TrajectoryError::DeserializationError(e.to_string()))?;

        Ok((trajectory, bytes_read))
    }

    /// Serialize to bytes
    #[cfg(feature = "metadata")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, TrajectoryError> {
        bincode::serde::encode_to_vec(self, config::standard())
            .map_err(|e| TrajectoryError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    #[cfg(feature = "metadata")]
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), TrajectoryError> {
        bincode::serde::decode_from_slice(data, config::standard())
            .map_err(|e| TrajectoryError::DeserializationError(e.to_string()))
    }

    /// Encode with chunking for large trajectories
    #[cfg(all(feature = "metadata", feature = "integrity"))]
    pub fn encode_chunked(
        &self,
        config: &CodecConfig,
        chunk_size: usize,
    ) -> Result<Vec<TensorChunk>, TrajectoryError> {
        let encoded = self.encode(config)?;
        let encoded_bytes = bincode::serde::encode_to_vec(&encoded, config::standard())
            .map_err(|e| TrajectoryError::SerializationError(e.to_string()))?;

        let chunked = ChunkedTensor::from_data(&encoded_bytes, chunk_size);
        Ok(chunked.chunks().to_vec())
    }

    /// Reassemble from chunks
    #[cfg(all(feature = "metadata", feature = "integrity"))]
    pub fn decode_chunked(
        chunks: &[TensorChunk],
        config: &CodecConfig,
    ) -> Result<Self, TrajectoryError> {
        let reassembled = ChunkedTensor::reassemble(chunks)
            .map_err(|e| TrajectoryError::ChunkingError(e.to_string()))?;

        let (encoded, _): (EncodedTrajectory, usize) =
            bincode::serde::decode_from_slice(&reassembled, config::standard())
                .map_err(|e| TrajectoryError::DeserializationError(e.to_string()))?;

        Self::decode(&encoded, config).map(|(trajectory, _)| trajectory)
    }
}

pub trait RelayRLTrajectoryTrait {
    type Action;

    fn add_action(&mut self, action: &Self::Action);
}

impl RelayRLTrajectoryTrait for RelayRLTrajectory {
    type Action = RelayRLAction;

    fn add_action(&mut self, action: &Self::Action) {
        self.add_action_ref(action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let traj = RelayRLTrajectory::new(100);
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
        assert!(!traj.is_complete());
    }

    #[test]
    fn test_add_actions() {
        let mut traj = RelayRLTrajectory::new(10);
        let action = RelayRLAction::minimal(1.0, false);

        let should_flush = traj.add_action(action);
        assert!(!should_flush);
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_trajectory_completion() {
        let mut traj = RelayRLTrajectory::new(10);

        for _ in 0..5 {
            traj.add_action(RelayRLAction::minimal(1.0, false));
        }
        assert!(!traj.is_complete());

        traj.add_action(RelayRLAction::minimal(1.0, true));
        assert!(traj.is_complete());
    }

    #[test]
    fn test_trajectory_rewards() {
        let mut traj = RelayRLTrajectory::new(10);

        for i in 1..=5 {
            traj.add_action(RelayRLAction::minimal(i as f32, false));
        }

        assert_eq!(traj.total_reward(), 15.0);
        assert_eq!(traj.avg_reward(), 3.0);
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn test_trajectory_serialization() {
        let mut traj = RelayRLTrajectory::new(10);
        traj.add_action(RelayRLAction::minimal(1.5, true));

        let bytes = traj.to_bytes().unwrap();
        let (decoded, decoded_bytes_read) = RelayRLTrajectory::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded.get_actions()[0].get_rew(), 1.5);
        assert_eq!(decoded_bytes_read, bytes.len());
    }
}
