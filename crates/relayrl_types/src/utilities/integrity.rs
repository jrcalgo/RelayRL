//! Cryptographic integrity verification for tensor data
//! 
//! Uses BLAKE3 for fast, parallel hashing with cryptographic security.

use serde::{Deserialize, Serialize};

#[cfg(feature = "integrity")]
use blake3::{Hash, Hasher};

pub type Checksum = [u8; 32]; // 256-bit BLAKE3 hash

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedData {
    pub data: Vec<u8>,
    pub checksum: Checksum,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

impl VerifiedData {
    #[cfg(feature = "integrity")]
    pub fn new(data: Vec<u8>) -> Self {
        let checksum = compute_checksum(&data);
        Self {
            data,
            checksum,
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()),
        }
    }

    #[cfg(feature = "integrity")]
    pub fn new_without_timestamp(data: Vec<u8>) -> Self {
        let checksum = compute_checksum(&data);
        Self {
            data,
            checksum,
            timestamp: None,
        }
    }

    #[cfg(feature = "integrity")]
    pub fn verify(&self) -> Result<(), IntegrityError> {
        let computed = compute_checksum(&self.data);
        if computed == self.checksum {
            Ok(())
        } else {
            Err(IntegrityError::ChecksumMismatch {
                expected: self.checksum,
                computed,
            })
        }
    }

    #[cfg(feature = "integrity")]
    pub fn verify_with_age(&self, max_age_secs: u64) -> Result<(), IntegrityError> {
        self.verify()?;
        if let Some(timestamp) = self.timestamp {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let age = now.saturating_sub(timestamp);
            if age > max_age_secs {
                return Err(IntegrityError::DataTooOld { age, max_age: max_age_secs });
            }
        }
        Ok(())
    }

    /// Consume and return data if valid
    #[cfg(feature = "integrity")]
    pub fn into_verified(self) -> Result<Vec<u8>, IntegrityError> {
        self.verify()?;
        Ok(self.data)
    }
}

/// Compute BLAKE3 checksum (fast, parallel, cryptographically secure)
#[cfg(feature = "integrity")]
pub fn compute_checksum(data: &[u8]) -> Checksum {
    blake3::hash(data).into()
}

/// Compute keyed hash (for HMAC-like authentication)
#[cfg(feature = "integrity")]
pub fn compute_keyed_hash(data: &[u8], key: &[u8; 32]) -> Checksum {
    blake3::keyed_hash(key, data).into()
}

#[derive(Debug, Clone)]
pub enum IntegrityError {
    ChecksumMismatch {
        expected: Checksum,
        computed: Checksum,
    },
    DataTooOld {
        age: u64,
        max_age: u64,
    },
}

impl std::fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChecksumMismatch { expected, computed } => {
                write!(f, "Checksum mismatch: expected {:?}, got {:?}", expected, computed)
            }
            Self::DataTooOld { age, max_age } => {
                write!(f, "Data too old: {} seconds (max {})", age, max_age)
            }
        }
    }
}

impl std::error::Error for IntegrityError {}
