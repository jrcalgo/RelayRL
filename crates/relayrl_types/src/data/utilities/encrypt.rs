//! Authenticated encryption for secure tensor transport
//!
//! Uses ChaCha20-Poly1305 AEAD cipher - fast, secure, pure Rust.

use serde::{Deserialize, Serialize};

#[cfg(feature = "encryption")]
use chacha20poly1305::{
    ChaCha20Poly1305, Key, Nonce,
    aead::{Aead, AeadCore, KeyInit, OsRng},
};

/// 256-bit encryption key
pub type EncryptionKey = [u8; 32];

/// 96-bit nonce
pub type NonceBytes = [u8; 12];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub ciphertext: Vec<u8>,
    /// Nonce msut be unique
    pub nonce: NonceBytes,
    /// Optional additional authenticated data (not encrypted)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aad: Option<Vec<u8>>,
}

impl EncryptedData {
    #[cfg(feature = "encryption")]
    pub fn encrypt(plaintext: &[u8], key: &EncryptionKey) -> Result<Self, EncryptionError> {
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| EncryptionError::EncryptionFailed(e.to_string()))?;
        Ok(Self {
            ciphertext,
            nonce: nonce.into(),
            aad: None,
        })
    }

    #[cfg(feature = "encryption")]
    pub fn encrypt_with_aad(
        plaintext: &[u8],
        key: &EncryptionKey,
        aad: &[u8],
    ) -> Result<Self, EncryptionError> {
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        use chacha20poly1305::aead::Payload;
        let payload = Payload {
            msg: plaintext,
            aad,
        };
        let ciphertext = cipher
            .encrypt(&nonce, payload)
            .map_err(|e| EncryptionError::EncryptionFailed(e.to_string()))?;
        Ok(Self {
            ciphertext,
            nonce: nonce.into(),
            aad: Some(aad.to_vec()),
        })
    }

    #[cfg(feature = "encryption")]
    pub fn decrypt(&self, key: &EncryptionKey) -> Result<Vec<u8>, EncryptionError> {
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let nonce = Nonce::from_slice(&self.nonce);
        let plaintext = if let Some(aad) = &self.aad {
            use chacha20poly1305::aead::Payload;
            let payload = Payload {
                msg: &self.ciphertext,
                aad,
            };
            cipher
                .decrypt(nonce, payload)
                .map_err(|e| EncryptionError::DecryptionFailed(e.to_string()))?
        } else {
            cipher
                .decrypt(nonce, self.ciphertext.as_ref())
                .map_err(|e| EncryptionError::DecryptionFailed(e.to_string()))?
        };
        Ok(plaintext)
    }

    pub const OVERHEAD_BYTES: usize = 16;
}

#[cfg(feature = "encryption")]
pub fn generate_key() -> EncryptionKey {
    let key = ChaCha20Poly1305::generate_key(&mut OsRng);
    key.into()
}

#[derive(Debug, Clone)]
pub enum EncryptionError {
    EncryptionFailed(String),
    DecryptionFailed(String),
    InvalidKey,
}

impl std::fmt::Display for EncryptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EncryptionFailed(e) => write!(f, "Encryption failed: {}", e),
            Self::DecryptionFailed(e) => write!(f, "Decryption failed: {}", e),
            Self::InvalidKey => write!(f, "Invalid encryption key"),
        }
    }
}

impl std::error::Error for EncryptionError {}

#[cfg(all(test, feature = "encryption"))]
mod unit_tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_round_trip_preserves_plaintext() {
        let key = generate_key();
        let plaintext = b"relayrl-encryption";

        let encrypted = EncryptedData::encrypt(plaintext, &key).unwrap();

        assert_eq!(
            encrypted.ciphertext.len(),
            plaintext.len() + EncryptedData::OVERHEAD_BYTES
        );
        assert_eq!(encrypted.decrypt(&key).unwrap(), plaintext);
    }

    #[test]
    fn encrypt_with_aad_preserves_associated_data() {
        let key = generate_key();
        let aad = b"relayrl";
        let encrypted = EncryptedData::encrypt_with_aad(b"payload", &key, aad).unwrap();

        assert_eq!(encrypted.aad.as_deref(), Some(aad.as_slice()));
        assert_eq!(encrypted.decrypt(&key).unwrap(), b"payload");
    }

    #[test]
    fn decrypt_with_the_wrong_key_fails() {
        let plaintext = b"secret";
        let key = generate_key();
        let encrypted = EncryptedData::encrypt(plaintext, &key).unwrap();
        let mut wrong_key = key;
        wrong_key[0] ^= 0xFF;

        let err = encrypted
            .decrypt(&wrong_key)
            .expect_err("ciphertext should not decrypt with the wrong key");

        assert!(matches!(err, EncryptionError::DecryptionFailed(_)));
    }

    #[test]
    fn generated_keys_have_expected_length() {
        let key = generate_key();
        assert_eq!(key.len(), 32);
    }
}
