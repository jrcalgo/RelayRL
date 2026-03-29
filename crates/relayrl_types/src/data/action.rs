//! RelayRL Action types with burn tensor backend support
//!
//! Provides flexible action representation supporting multiple backends (ndarray, tch)
//! with integrated serialization, compression, encryption, and integrity checking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use bincode::config;
use uuid::Uuid;

use burn_tensor::backend::Backend;

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

#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
#[cfg(feature = "ndarray-backend")]
use crate::data::tensor::NdArrayDType;

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
use super::tensor::{BackendMatcher, DeviceType, SupportedTensorBackend };

use super::tensor::{DType, TensorData, TensorError};

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
                },
                #[cfg(feature = "tch-backend")]
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
                    TchDType::U8 | TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 => tensor_data
                        .to_int_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
                    TchDType::Bool => tensor_data
                        .to_bool_tensor::<B, 1>(device)
                        .map(|tensor| Box::new(tensor) as Box<dyn std::any::Any>),
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
        if config.verify_integrity && let Some(checksum) = encoded.checksum {
            let computed = compute_checksum(&data);
            if computed != checksum {
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
mod unit_tests {
    use super::*;
    #[cfg(feature = "ndarray-backend")]
    use burn_ndarray::NdArray;
    use uuid::Uuid;

    use crate::data::tensor::{
        DType, DeviceType, NdArrayDType, SupportedTensorBackend, TensorData, TensorError,
    };
    #[cfg(feature = "encryption")]
    use crate::data::utilities::encrypt::generate_key;

    fn f32_tensor(values: &[f32]) -> TensorData {
        TensorData::new(
            vec![values.len()],
            DType::NdArray(NdArrayDType::F32),
            values.iter().flat_map(|value| value.to_le_bytes()).collect(),
            SupportedTensorBackend::NdArray,
        )
    }

    fn bool_tensor(values: &[bool]) -> TensorData {
        TensorData::new(
            vec![values.len()],
            DType::NdArray(NdArrayDType::Bool),
            values.iter().map(|value| u8::from(*value)).collect(),
            SupportedTensorBackend::NdArray,
        )
    }

    fn rich_action() -> RelayRLAction {
        let mut aux = HashMap::new();
        aux.insert("score".to_string(), RelayRLData::F32(7.5));
        aux.insert("label".to_string(), RelayRLData::String("policy".to_string()));

        RelayRLAction::new(
            Some(f32_tensor(&[1.0, 2.0])),
            Some(f32_tensor(&[3.0, 4.0])),
            Some(bool_tensor(&[true, false])),
            1.5,
            true,
            Some(aux),
            Some(Uuid::from_u128(7)),
        )
    }

    #[test]
    fn minimal_action_has_expected_defaults() {
        let action = RelayRLAction::minimal(1.0, false);
        assert_eq!(action.get_rew(), 1.0);
        assert!(!action.get_done());
        assert!(action.get_obs().is_none());
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn action_serialization_round_trip() {
        let action = RelayRLAction::minimal(1.5, true);
        let bytes = action.to_bytes().unwrap();
        let (decoded, decoded_bytes_read) = RelayRLAction::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.get_rew(), 1.5);
        assert!(decoded.get_done());
        assert_eq!(decoded_bytes_read, bytes.len());
    }

    #[test]
    fn setters_update_action_metadata() {
        let mut action = RelayRLAction::minimal(0.0, false);
        let agent_id = Uuid::from_u128(11);

        action.update_reward(2.5);
        action.set_done(true);
        action.set_agent_id(agent_id);

        assert_eq!(action.get_rew(), 2.5);
        assert!(action.get_done());
        assert_eq!(action.get_agent_id(), Some(&agent_id));
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn tensor_accessors_return_tensors_for_matching_backend() {
        let action = rich_action();

        assert!(action.get_obs_tensor::<NdArray>(&DeviceType::Cpu).is_some());
        assert!(action.get_act_tensor::<NdArray>(&DeviceType::Cpu).is_some());
        assert!(action.get_mask_tensor::<NdArray>(&DeviceType::Cpu).is_some());
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn to_tensor_rejects_missing_backend() {
        let tensor = TensorData::new(
            vec![2],
            DType::NdArray(NdArrayDType::F32),
            [1.0f32, 2.0]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            SupportedTensorBackend::None,
        );

        let err = RelayRLAction::to_tensor::<NdArray>(&tensor, &DeviceType::Cpu)
            .expect_err("tensor conversion should reject the missing backend");

        assert!(matches!(err, TensorError::BackendError(message) if message.contains("Backend mismatch")));
    }

    #[test]
    fn codec_config_defaults_match_enabled_features() {
        let config = CodecConfig::default();

        #[cfg(feature = "compression")]
        assert!(config.compression.is_some());

        #[cfg(feature = "encryption")]
        assert!(config.encryption_key.is_none());

        #[cfg(feature = "integrity")]
        assert!(config.verify_integrity);

        #[cfg(feature = "metadata")]
        assert!(config.include_metadata);
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn encode_decode_round_trip_preserves_action_payloads() {
        let action = rich_action();
        let config = CodecConfig::default();

        let encoded = action.encode(&config).unwrap();
        let decoded = RelayRLAction::decode(&encoded, &config).unwrap();

        assert_eq!(decoded.get_rew(), action.get_rew());
        assert_eq!(decoded.get_done(), action.get_done());
        assert_eq!(decoded.get_agent_id(), action.get_agent_id());
        assert_eq!(decoded.get_obs().unwrap().data, action.get_obs().unwrap().data);
        assert_eq!(decoded.get_act().unwrap().data, action.get_act().unwrap().data);
        assert_eq!(decoded.get_mask().unwrap().data, action.get_mask().unwrap().data);
        assert!(matches!(
            decoded.get_data().unwrap().get("score"),
            Some(RelayRLData::F32(value)) if (*value - 7.5).abs() < f32::EPSILON
        ));

        #[cfg(feature = "compression")]
        assert!(encoded.compressed);

        #[cfg(feature = "integrity")]
        assert!(encoded.checksum.is_some());
    }

    #[test]
    #[cfg(all(feature = "metadata", feature = "integrity"))]
    fn decode_rejects_checksum_mismatch() {
        let action = rich_action();
        let config = CodecConfig::default();
        let mut encoded = action.encode(&config).unwrap();

        encoded.data[0] ^= 0xFF;

        let err = RelayRLAction::decode(&encoded, &config)
            .expect_err("tampering should invalidate the checksum");

        assert!(matches!(err, ActionError::IntegrityError(message) if message.contains("Checksum mismatch")));
    }

    #[test]
    #[cfg(all(feature = "metadata", feature = "encryption"))]
    fn decode_requires_encryption_key_when_payload_is_encrypted() {
        let action = rich_action();
        let mut encode_config = CodecConfig::default();
        encode_config.encryption_key = Some(generate_key());
        let encoded = action.encode(&encode_config).unwrap();

        let mut decode_config = CodecConfig::default();
        decode_config.encryption_key = None;

        let err = RelayRLAction::decode(&encoded, &decode_config)
            .expect_err("decoding encrypted payloads should require the key");

        assert!(matches!(
            err,
            ActionError::EncryptionError(message) if message.contains("Encryption key required")
        ));
    }

    #[test]
    #[cfg(all(feature = "metadata", feature = "integrity"))]
    fn chunked_encode_decode_round_trip_preserves_action() {
        let action = rich_action();
        let config = CodecConfig::default();

        let chunks = action.encode_chunked(&config, 8).unwrap();
        assert!(chunks.len() > 1);

        let decoded = RelayRLAction::decode_chunked(&chunks, &config).unwrap();
        assert_eq!(decoded.get_rew(), action.get_rew());
        assert_eq!(decoded.get_obs().unwrap().data, action.get_obs().unwrap().data);
    }

    #[test]
    fn age_seconds_uses_action_timestamp() {
        let mut action = RelayRLAction::minimal(0.0, false);
        action.timestamp = action.timestamp.saturating_sub(2);

        assert!(action.age_seconds() >= 2);
    }
}
