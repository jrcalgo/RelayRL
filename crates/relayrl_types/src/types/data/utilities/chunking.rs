//! Chunking utilities for streaming large tensors over network

use serde::{Deserialize, Serialize};

#[cfg(feature = "integrity")]
use crate::types::data::utilities::integrity::{Checksum, compute_checksum};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorChunk {
    pub chunk_id: u32,
    pub total_chunks: u32,
    pub data: Vec<u8>,
    #[cfg(feature = "integrity")]
    pub checksum: Checksum,
    pub offset: usize,
}

/// Collection of chunks representing a full tensor
#[derive(Debug, Clone)]
pub struct ChunkedTensor {
    chunks: Vec<TensorChunk>,
    chunk_size: usize,
    total_size: usize,
}

impl ChunkedTensor {
    pub fn from_data(data: &[u8], chunk_size: usize) -> Self {
        let total_chunks = (data.len() + chunk_size - 1) / chunk_size;
        let mut chunks = Vec::with_capacity(total_chunks);
        for (i, chunk_data) in data.chunks(chunk_size).enumerate() {
            let chunk = TensorChunk {
                chunk_id: i as u32,
                total_chunks: total_chunks as u32,
                data: chunk_data.to_vec(),
                #[cfg(feature = "integrity")]
                checksum: compute_checksum(chunk_data),
                offset: i * chunk_size,
            };
            chunks.push(chunk);
        }
        Self {
            chunks,
            chunk_size,
            total_size: data.len(),
        }
    }

    pub fn chunks(&self) -> &[TensorChunk] {
        &self.chunks
    }

    pub fn reassemble(chunks: &[TensorChunk]) -> Result<Vec<u8>, ChunkError> {
        if chunks.is_empty() {
            return Err(ChunkError::NoChunks);
        }
        let mut sorted_chunks = chunks.to_vec();
        sorted_chunks.sort_by_key(|c| c.chunk_id);
        let total_chunks = sorted_chunks[0].total_chunks;
        if sorted_chunks.len() != total_chunks as usize {
            return Err(ChunkError::MissingChunks {
                expected: total_chunks,
                received: sorted_chunks.len() as u32,
            });
        }
        let mut result = Vec::new();
        for (i, chunk) in sorted_chunks.iter().enumerate() {
            if chunk.chunk_id != i as u32 {
                return Err(ChunkError::OutOfOrder);
            }
            #[cfg(feature = "integrity")]
            {
                let computed = compute_checksum(&chunk.data);
                if computed != chunk.checksum {
                    return Err(ChunkError::CorruptedChunk(chunk.chunk_id));
                }
            }
            result.extend_from_slice(&chunk.data);
        }
        Ok(result)
    }

    ///  network transmission chunk size (1MB)
    pub const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;
}

#[derive(Debug, Clone)]
pub enum ChunkError {
    NoChunks,
    MissingChunks { expected: u32, received: u32 },
    OutOfOrder,
    CorruptedChunk(u32),
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoChunks => write!(f, "No chunks provided"),
            Self::MissingChunks { expected, received } => {
                write!(
                    f,
                    "Missing chunks: expected {}, received {}",
                    expected, received
                )
            }
            Self::OutOfOrder => write!(f, "Chunks out of order"),
            Self::CorruptedChunk(id) => write!(f, "Chunk {} is corrupted", id),
        }
    }
}

impl std::error::Error for ChunkError {}
