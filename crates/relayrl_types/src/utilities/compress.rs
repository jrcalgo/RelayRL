//! High-performance compression for tensor data transport
//!
//! Supports multiple compression schemes optimized for different scenarios:
//! - LZ4: Ultra-fast, real-time inference (3-4 GB/s)
//! - Zstd: Best compression ratio for training data (5-10x reduction)

use serde::{Deserialize, Serialize};

#[cfg(feature = "compression")]
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

#[cfg(feature = "compression")]
use zstd::bulk::{compress as zstd_compress, decompress as zstd_decompress};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionScheme {
    /// No compression (passthrough)
    None,
    Lz4,
    Zstd(i32),
}

impl Default for CompressionScheme {
    fn default() -> Self {
        Self::Lz4
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    pub data: Vec<u8>,
    pub original_size: usize,
    pub scheme: CompressionScheme,
}

impl CompressedData {
    #[cfg(feature = "compression")]
    pub fn compress(data: &[u8], scheme: CompressionScheme) -> Result<Self, CompressionError> {
        let original_size = data.len();
        let compressed = match scheme {
            CompressionScheme::None => data.to_vec(),
            CompressionScheme::Lz4 => compress_prepend_size(data),
            CompressionScheme::Zstd(level) => zstd_compress(data, level)
                .map_err(|e| CompressionError::ZstdError(e.to_string()))?,
        };
        Ok(Self {
            data: compressed,
            original_size,
            scheme,
        })
    }

    #[cfg(feature = "compression")]
    pub fn decompress(&self) -> Result<Vec<u8>, CompressionError> {
        match self.scheme {
            CompressionScheme::None => Ok(self.data.clone()),
            CompressionScheme::Lz4 => decompress_size_prepended(&self.data)
                .map_err(|e| CompressionError::Lz4Error(e.to_string())),
            CompressionScheme::Zstd(_) => zstd_decompress(&self.data, self.original_size)
                .map_err(|e| CompressionError::ZstdError(e.to_string())),
        }
    }

    pub fn compression_ratio(&self) -> f32 {
        self.original_size as f32 / self.data.len() as f32
    }

    /// Space saved in bytes
    pub fn space_saved(&self) -> isize {
        self.original_size as isize - self.data.len() as isize
    }
}

#[derive(Debug, Clone)]
pub enum CompressionError {
    Lz4Error(String),
    ZstdError(String),
    InvalidData,
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lz4Error(e) => write!(f, "LZ4 error: {}", e),
            Self::ZstdError(e) => write!(f, "Zstd error: {}", e),
            Self::InvalidData => write!(f, "Invalid compressed data"),
        }
    }
}

impl std::error::Error for CompressionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "compression")]
    fn test_lz4_compression() {
        let data = vec![42u8; 1000];
        let compressed = CompressedData::compress(&data, CompressionScheme::Lz4).unwrap();
        assert!(compressed.data.len() < data.len());
        let decompressed = compressed.decompress().unwrap();
        assert_eq!(data, decompressed);
    }
}
