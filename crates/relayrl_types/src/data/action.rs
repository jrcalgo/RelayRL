//! RelayRL Action types with burn tensor backend support
//!
//! Provides flexible action representation supporting multiple backends (ndarray, tch)
//! with integrated serialization, compression, encryption, and integrity checking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use bincode::config;
use uuid::Uuid;

#[cfg(feature = "ndarray-backend")]
use burn_ndarray::NdArray;
#[cfg(feature = "tch-backend")]
use burn_tch::LibTorch as Tch;
use burn_tensor::{backend::Backend, Bool, Float, Int, Tensor};

#[cfg(feature = "integrity")]
use crate::data::utilities::chunking::{ChunkedTensor, TensorChunk};
#[cfg(feature = "compression")]
use crate::data::utilities::compress::{CompressedData, CompressionScheme};
#[cfg(feature = "encryption")]
use crate::data::utilities::encrypt::{EncryptedData, EncryptionKey};
#[cfg(feature = "integrity")]
use crate::data::utilities::integrity::{compute_checksum, Checksum};
#[cfg(feature = "metadata")]
use crate::data::utilities::metadata::TensorMetadata;

use super::tensor::{
    BackendMatcher, DType, DeviceType, NdArrayDType, SupportedTensorBackend, TchDType, TensorData,
    TensorError,
};

/// Additional data types that can be attached to actions via the `data` parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelayRLData {
    DType(DType),
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

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl RelayRLAction {
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

    pub fn to_tensor<B: Backend + BackendMatcher + 'static>(
        tensor_data: &TensorData,
        device: &DeviceType,
    ) -> Result<Box<dyn std::any::Any>, TensorError> {
        if !B::matches_backend(&tensor_data.supported_backend) {
            return Err(TensorError::BackendError(format!(
                "Backend mismatch: expected {:?}, got {:?}",
                tensor_data.supported_backend,
                std::any::type_name::<B>()
            )));
        }

        match tensor_data.supported_backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => match &tensor_data.dtype {
                DType::NdArray(dtype) => match dtype {
                    NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => tensor_data
                        .to_float_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    NdArrayDType::I8
                    | NdArrayDType::I16
                    | NdArrayDType::I32
                    | NdArrayDType::I64 => tensor_data
                        .to_int_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    NdArrayDType::Bool => tensor_data
                        .to_bool_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    _ => Err(TensorError::DTypeError(format!(
                        "NdArray dtype not supported: {:?}",
                        tensor_data.dtype
                    ))),
                },
                _ => Err(TensorError::DTypeError(format!(
                    "Unsupported dtype for NdArray backend: {}",
                    tensor_data.dtype
                ))),
            },
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => match &tensor_data.dtype {
                DType::Tch(dtype) => match dtype {
                    TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => tensor_data
                        .to_float_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 => tensor_data
                        .to_int_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    TchDType::Bool => tensor_data
                        .to_bool_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    _ => Err(TensorError::DTypeError(format!(
                        "Tch dtype not supported: {:?}",
                        tensor_data.dtype
                    ))),
                },
                _ => Err(TensorError::DTypeError(format!(
                    "Unsupported dtype for Tch backend: {}",
                    tensor_data.dtype
                ))),
            },
            SupportedTensorBackend::None => {
                Err(TensorError::BackendError("No backend selected".to_string()))
            }
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

    pub fn get_obs(&self) -> Option<&TensorData> {
        self.obs.as_ref()
    }

    pub fn get_obs_tensor<B: Backend + BackendMatcher + 'static>(
        &self,
        device: &DeviceType,
    ) -> Option<Box<dyn std::any::Any>> {
        self.obs
            .as_ref()
            .and_then(|tensor_data| Self::to_tensor::<B>(tensor_data, device).ok())
    }

    pub fn get_act(&self) -> Option<&TensorData> {
        self.act.as_ref()
    }

    pub fn get_act_tensor<B: Backend + BackendMatcher + 'static>(
        &self,
        device: &DeviceType,
    ) -> Option<Box<dyn std::any::Any>> {
        self.act
            .as_ref()
            .and_then(|tensor_data| Self::to_tensor::<B>(tensor_data, device).ok())
    }

    pub fn get_mask(&self) -> Option<&TensorData> {
        self.mask.as_ref()
    }

    pub fn get_mask_tensor<B: Backend + BackendMatcher + 'static>(
        &self,
        device: &DeviceType,
    ) -> Option<Box<dyn std::any::Any>> {
        self.mask
            .as_ref()
            .and_then(|tensor_data| Self::to_tensor::<B>(tensor_data, device).ok())
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
    TensorError(TensorError),
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
            Self::TensorError(e) => write!(f, "[ActionError] Tensor error: {}", e),
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
    #[cfg(feature = "compression")]
    pub compressed: bool,
    #[cfg(feature = "encryption")]
    pub encrypted: bool,
    #[cfg(feature = "integrity")]
    pub checksum: Option<Checksum>,
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
        let original_data =
            bincode::serde::encode_to_vec(self, config::standard()).map_err(|e| {
                ActionError::TensorError(TensorError::SerializationError(e.to_string()))
            })?;

        let original_size = original_data.len();
        let mut data = original_data;

        #[cfg(feature = "compression")]
        let compressed = if let Some(scheme) = config.compression {
            let compressed_data = CompressedData::compress(&data, scheme)
                .map_err(|e| ActionError::CompressionError(e.to_string()))?;
            data = bincode::serde::encode_to_vec(&compressed_data, config::standard()).map_err(
                |e| ActionError::TensorError(TensorError::SerializationError(e.to_string())),
            )?;
            true
        } else {
            false
        };

        #[cfg(feature = "encryption")]
        let encrypted = if let Some(key) = &config.encryption_key {
            let encrypted_data = EncryptedData::encrypt(&data, key)
                .map_err(|e| ActionError::EncryptionError(e.to_string()))?;
            data = bincode::serde::encode_to_vec(&encrypted_data, config::standard()).map_err(
                |e| ActionError::TensorError(TensorError::SerializationError(e.to_string())),
            )?;
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
        if config.verify_integrity && encoded.checksum.is_some() {
            let computed = compute_checksum(&data);
            if computed != encoded.checksum.unwrap() {
                return Err(ActionError::IntegrityError("Checksum mismatch".to_string()));
            }
        }

        #[cfg(feature = "encryption")]
        if encoded.encrypted {
            if let Some(key) = &config.encryption_key {
                let (encrypted_data, _): (EncryptedData, usize) =
                    bincode::serde::decode_from_slice(&data, config::standard()).map_err(|e| {
                        ActionError::TensorError(TensorError::DeserializationError(e.to_string()))
                    })?;
                data = encrypted_data
                    .decrypt(key)
                    .map_err(|e| ActionError::EncryptionError(e.to_string()))?;
            } else {
                return Err(ActionError::EncryptionError(
                    "Encryption key required but not provided".to_string(),
                ));
            }
        }

        #[cfg(feature = "compression")]
        if encoded.compressed {
            let (compressed_data, _): (CompressedData, usize) =
                bincode::serde::decode_from_slice(&data, config::standard()).map_err(|e| {
                    ActionError::TensorError(TensorError::DeserializationError(e.to_string()))
                })?;
            data = compressed_data
                .decompress()
                .map_err(|e| ActionError::CompressionError(e.to_string()))?;
        }

        let (action, _): (RelayRLAction, usize) =
            bincode::serde::decode_from_slice(&data, config::standard()).map_err(|e| {
                ActionError::TensorError(TensorError::DeserializationError(e.to_string()))
            })?;

        Ok(action)
    }

    /// Serialize to bytes
    #[cfg(feature = "metadata")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, TensorError> {
        bincode::serde::encode_to_vec(self, config::standard())
            .map_err(|e| TensorError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    #[cfg(feature = "metadata")]
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize), TensorError> {
        bincode::serde::decode_from_slice(data, config::standard())
            .map_err(|e| TensorError::DeserializationError(e.to_string()))
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
        let encoded_bytes =
            bincode::serde::encode_to_vec(&encoded, config::standard()).map_err(|e| {
                ActionError::TensorError(TensorError::SerializationError(e.to_string()))
            })?;
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

        let (encoded, _): (EncodedAction, usize) =
            bincode::serde::decode_from_slice(&reassembled, config::standard()).map_err(|e| {
                ActionError::TensorError(TensorError::DeserializationError(e.to_string()))
            })?;

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
        let (decoded, decoded_bytes_read) = RelayRLAction::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.get_rew(), 1.5);
        assert!(decoded.get_done());
        assert_eq!(decoded_bytes_read, bytes.len());
    }

    #[test]
    fn test_action_age() {
        let action = RelayRLAction::minimal(0.0, false);
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(action.age_seconds() >= 1);
    }
}
