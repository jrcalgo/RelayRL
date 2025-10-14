//! RelayRL Action types with burn tensor backend support
//! 
//! Provides flexible action representation supporting multiple backends (ndarray, tch)
//! with integrated serialization, compression, encryption, and integrity checking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(feature = "compression")]
use crate::utilities::compress::{CompressedData, CompressionScheme};
#[cfg(feature = "encryption")]
use crate::utilities::encrypt::{EncryptedData, EncryptionKey};
#[cfg(feature = "integrity")]
use crate::utilities::integrity::Checksum;
#[cfg(feature = "metadata")]
use crate::utilities::metadata::TensorMetadata;
#[cfg(feature = "integrity")]
use crate::utilities::chunking::{ChunkedTensor, TensorChunk};

/// Tensor backend enumeration for runtime backend selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorBackend {
    /// CPU-based NdArray backend
    #[cfg(feature = "ndarray-backend")]
    NdArray,
    /// LibTorch backend (GPU/CPU)
    #[cfg(feature = "tch-backend")]
    Tch,
}

impl Default for TensorBackend {
    fn default() -> Self {
        #[cfg(feature = "ndarray-backend")]
        return TensorBackend::NdArray;
        
        #[cfg(all(feature = "tch-backend", not(feature = "ndarray-backend")))]
        return TensorBackend::Tch;
    }
}

/// Data type enumeration for tensor serialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    U8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Bool,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::U8 => write!(f, "U8"),
            DType::I16 => write!(f, "I16"),
            DType::I32 => write!(f, "I32"),
            DType::I64 => write!(f, "I64"),
            DType::F32 => write!(f, "F32"),
            DType::F64 => write!(f, "F64"),
            DType::Bool => write!(f, "Bool"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) data: Vec<u8>,
    pub(crate) backend: TensorBackend,
}

impl TensorData {
    pub fn new(shape: Vec<usize>, dtype: DType, data: Vec<u8>, backend: TensorBackend) -> Self {
        Self {
            shape,
            dtype,
            data,
            backend,
        }
    }

    pub fn num_el(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Additional data types that can be attached to actions via the `data` parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelayRLData {
    Tensor(TensorData),
    U8(u8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    Bool(bool),
}

/// Represents a single timestep in an RL environment, containing:
/// - Observation tensor
/// - Action tensor
/// - Action mask
/// - Reward value
/// - Terminal flag
/// - Auxiliary data
/// - Agent and timing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayRLAction {
    pub(crate) obs: Option<TensorData>,
    
    pub(crate) act: Option<TensorData>,
    
    pub(crate) mask: Option<TensorData>,
    
    pub(crate) rew: f32,
    
    pub(crate) done: bool,
    
    pub(crate) data: Option<HashMap<String, RelayRLData>>,
    
    pub(crate) agent_id: Option<Uuid>,
    
    pub(crate) timestamp: u64,
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl RelayRLAction {
    /// Create a new RelayRLAction
    pub fn new(
        obs: Option<TensorData>,
        act: Option<TensorData>,
        mask: Option<TensorData>,
        rew: f32,
        done: bool,
        data: Option<HashMap<String, RelayRLData>>,
        agent_id: Option<Uuid>,
    ) -> Self {
        Self {
            obs,
            act,
            mask,
            rew,
            done,
            data,
            agent_id,
            timestamp: current_timestamp(),
        }
    }

    pub fn minimal(rew: f32, done: bool) -> Self {
        Self {
            obs: None,
            act: None,
            mask: None,
            rew,
            done,
            data: None,
            agent_id: None,
            timestamp: current_timestamp(),
        }
    }

    // Getters
    pub fn get_obs(&self) -> Option<&TensorData> {
        self.obs.as_ref()
    }

    pub fn get_act(&self) -> Option<&TensorData> {
        self.act.as_ref()
    }

    pub fn get_mask(&self) -> Option<&TensorData> {
        self.mask.as_ref()
    }

    pub fn get_rew(&self) -> f32 {
        self.rew
    }

    pub fn get_done(&self) -> bool {
        self.done
    }

    pub fn get_data(&self) -> Option<&HashMap<String, RelayRLData>> {
        self.data.as_ref()
    }

    pub fn get_agent_id(&self) -> Option<&Uuid> {
        self.agent_id.as_ref()
    }

    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }

    pub fn update_reward(&mut self, reward: f32) {
        self.rew = reward;
    }

    pub fn set_done(&mut self, done: bool) {
        self.done = done;
    }

    pub fn set_agent_id(&mut self, agent_id: Uuid) {
        self.agent_id = Some(agent_id);
    }

    /// Age of this action in seconds
    pub fn age_seconds(&self) -> u64 {
        current_timestamp().saturating_sub(self.timestamp)
    }
}

/// Codec configuration for encoding/decoding actions
#[derive(Debug, Clone)]
pub struct CodecConfig {
    #[cfg(feature = "compression")]
    pub compression: Option<CompressionScheme>,
    
    #[cfg(feature = "encryption")]
    pub encryption_key: Option<EncryptionKey>,
    
    #[cfg(feature = "integrity")]
    pub verify_integrity: bool,
    
    #[cfg(feature = "metadata")]
    pub include_metadata: bool,
}

impl Default for CodecConfig {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            #[cfg(feature = "compression")]
            compression: Some(CompressionScheme::Lz4),
            
            #[cfg(feature = "encryption")]
            encryption_key: None,
            
            #[cfg(feature = "integrity")]
            verify_integrity: true,
            
            #[cfg(feature = "metadata")]
            include_metadata: true,
        }
    }
}

impl CodecConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub enum ActionError {
    SerializationError(String),
    DeserializationError(String),
    
    #[cfg(feature = "compression")]
    CompressionError(String),
    
    #[cfg(feature = "encryption")]
    EncryptionError(String),
    
    #[cfg(feature = "integrity")]
    IntegrityError(String),
    
    ChunkingError(String),
}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationError(e) => write!(f, "[ActionError] Serialization error: {}", e),
            Self::DeserializationError(e) => write!(f, "[ActionError] Deserialization error: {}", e),
            #[cfg(feature = "compression")]
            Self::CompressionError(e) => write!(f, "[ActionError] Compression error: {}", e),
            #[cfg(feature = "encryption")]
            Self::EncryptionError(e) => write!(f, "[ActionError] Encryption error: {}", e),
            #[cfg(feature = "integrity")]
            Self::IntegrityError(e) => write!(f, "[ActionError] Integrity error: {}", e),
            Self::ChunkingError(e) => write!(f, "[ActionError] Chunking error: {}", e),
        }
    }
}

impl std::error::Error for ActionError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedAction {
    pub data: Vec<u8>,
    #[cfg(feature = "metadata")]
    pub metadata: Option<TensorMetadata>,
    
    /// Was compression applied?
    #[cfg(feature = "compression")]
    pub compressed: bool,
    /// Was encryption applied?
    #[cfg(feature = "encryption")]
    pub encrypted: bool,
    /// Integrity checksum
    #[cfg(feature = "integrity")]
    pub checksum: Option<Checksum>,
    
    /// Original size in bytes before encoding
    pub original_size: usize,
}

impl RelayRLAction {
    /// Processing pipeline:
    /// 1. Serialize to bincode
    /// 2. Compress (if enabled)
    /// 3. Encrypt (if enabled)
    /// 4. Add integrity check (if enabled)
    #[cfg(feature = "metadata")]
    pub fn encode(&self, config: &CodecConfig) -> Result<EncodedAction, ActionError> {
        let original_data = bincode::serialize(self)
            .map_err(|e| ActionError::SerializationError(e.to_string()))?;
        
        let original_size = original_data.len();
        let mut data = original_data;

        #[cfg(feature = "compression")]
        let compressed = if let Some(scheme) = config.compression {
            let compressed_data = CompressedData::compress(&data, scheme)
                .map_err(|e| ActionError::CompressionError(e.to_string()))?;
            data = bincode::serialize(&compressed_data)
                .map_err(|e| ActionError::SerializationError(e.to_string()))?;
            true
        } else {
            false
        };

        #[cfg(feature = "encryption")]
        let encrypted = if let Some(key) = &config.encryption_key {
            let encrypted_data = EncryptedData::encrypt(&data, key)
                .map_err(|e| ActionError::EncryptionError(e.to_string()))?;
            data = bincode::serialize(&encrypted_data)
                .map_err(|e| ActionError::SerializationError(e.to_string()))?;
            true
        } else {
            false
        };

        #[cfg(feature = "integrity")]
        let checksum = if config.verify_integrity {
            Some(crate::utilities::integrity::compute_checksum(&data))
        } else {
            None
        };

        Ok(EncodedAction {
            data,
            #[cfg(feature = "metadata")]
            metadata: None,
            #[cfg(feature = "compression")]
            compressed,
            #[cfg(feature = "encryption")]
            encrypted,
            #[cfg(feature = "integrity")]
            checksum,
            original_size,
        })
    }

    /// Reverses the encoding pipeline:
    /// 1. Verify integrity (if enabled)
    /// 2. Decrypt (if encrypted)
    /// 3. Decompress (if compressed)
    /// 4. Deserialize from bincode
    #[cfg(feature = "metadata")]
    pub fn decode(encoded: &EncodedAction, config: &CodecConfig) -> Result<Self, ActionError> {
        let mut data = encoded.data.clone();

        #[cfg(feature = "integrity")]
        if config.verify_integrity {
            if let Some(expected_checksum) = encoded.checksum {
                let computed = crate::utilities::integrity::compute_checksum(&data);
                if computed != expected_checksum {
                    return Err(ActionError::IntegrityError(
                        "Checksum mismatch".to_string()
                    ));
                }
            }
        }

        #[cfg(feature = "encryption")]
        if encoded.encrypted {
            if let Some(key) = &config.encryption_key {
                let encrypted_data: EncryptedData = bincode::deserialize(&data)
                    .map_err(|e| ActionError::DeserializationError(e.to_string()))?;
                data = encrypted_data.decrypt(key)
                    .map_err(|e| ActionError::EncryptionError(e.to_string()))?;
            } else {
                return Err(ActionError::EncryptionError(
                    "Encryption key required but not provided".to_string()
                ));
            }
        }

        #[cfg(feature = "compression")]
        if encoded.compressed {
            let compressed_data: CompressedData = bincode::deserialize(&data)
                .map_err(|e| ActionError::DeserializationError(e.to_string()))?;
            data = compressed_data.decompress()
                .map_err(|e| ActionError::CompressionError(e.to_string()))?;
        }

        let action: RelayRLAction = bincode::deserialize(&data)
            .map_err(|e| ActionError::DeserializationError(e.to_string()))?;

        Ok(action)
    }

    /// Serialize to bytes
    #[cfg(feature = "metadata")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, ActionError> {
        bincode::serialize(self)
            .map_err(|e| ActionError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    #[cfg(feature = "metadata")]
    pub fn from_bytes(data: &[u8]) -> Result<Self, ActionError> {
        bincode::deserialize(data)
            .map_err(|e| ActionError::DeserializationError(e.to_string()))
    }

    /// Encode with chunking for large actions
    #[cfg(all(feature = "metadata", feature = "integrity"))]
    pub fn encode_chunked(
        &self,
        config: &CodecConfig,
        chunk_size: usize,
    ) -> Result<Vec<TensorChunk>, ActionError> {
        let encoded = self.encode(config)?;
        // Serialize the entire EncodedAction structure
        let encoded_bytes = bincode::serialize(&encoded)
            .map_err(|e| ActionError::SerializationError(e.to_string()))?;
        let chunked = ChunkedTensor::from_data(&encoded_bytes, chunk_size);
        Ok(chunked.chunks().to_vec())
    }

    /// Reassemble from chunks
    #[cfg(all(feature = "metadata", feature = "integrity"))]
    pub fn decode_chunked(
        chunks: &[TensorChunk],
        config: &CodecConfig,
    ) -> Result<Self, ActionError> {
        let reassembled = ChunkedTensor::reassemble(chunks)
            .map_err(|e| ActionError::ChunkingError(e.to_string()))?;
        
        // Deserialize the EncodedAction structure from reassembled data
        let encoded: EncodedAction = bincode::deserialize(&reassembled)
            .map_err(|e| ActionError::DeserializationError(e.to_string()))?;
        
        Self::decode(&encoded, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_action() {
        let action = RelayRLAction::minimal(1.0, false);
        assert_eq!(action.get_rew(), 1.0);
        assert!(!action.get_done());
        assert!(action.get_obs().is_none());
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn test_action_serialization() {
        let action = RelayRLAction::minimal(1.5, true);
        let bytes = action.to_bytes().unwrap();
        let decoded = RelayRLAction::from_bytes(&bytes).unwrap();
        
        assert_eq!(decoded.get_rew(), 1.5);
        assert_eq!(decoded.get_done(), true);
    }

    #[test]
    fn test_action_age() {
        let action = RelayRLAction::minimal(0.0, false);
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(action.age_seconds() >= 1);
    }
}
