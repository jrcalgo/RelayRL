# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-12-19

### Added
- Multi-backend support: `burn-ndarray` (CPU) and `burn-tch` (GPU)
- Runtime backend selection via `TensorBackend` enum
- Feature-based backend selection (`ndarray-backend`, `tch-backend`)
- LZ4 and Zstd compression schemes
- ChaCha20-Poly1305 AEAD encryption
- BLAKE3 cryptographic integrity verification
- Automatic chunking for large payloads
- Comprehensive metadata tracking
- Agent ID tracking (`agent_id: Option<Uuid>`)
- Timestamp support (`timestamp: u64`)
- Episode and training step metadata
- Enhanced `RelayRLData` enum with more primitive types
- `CodecConfig` for centralized encoding/decoding configuration
- `EncodedAction` and `EncodedTrajectory` structures
- `CompressedData`, `EncryptedData`, `VerifiedData` utility types
- `ChunkedTensor` for streaming large payloads
- `TensorMetadata` for provenance tracking
- `QuantizedData` for size optimization
- `encode()`, `decode()`, `encode_chunked()`, `decode_chunked()` methods
- `to_bytes()`, `from_bytes()` serialization methods
- `age_seconds()`, `total_reward()`, `avg_reward()` utility methods
- `is_complete()`, `is_full()` trajectory status methods
- `with_agent_id()`, `with_metadata()` constructor variants
- `minimal()` constructor for simple cases
- Comprehensive getter/setter methods
- Feature flags for optional dependencies
- Convenience feature bundles (`network-basic`, `network-secure`, `network-full`)
- Enhanced test coverage with feature-gated tests
- Updated documentation with examples and migration guide

### Changed
- Replaced `tch` dependency with `burn-tensor`, `burn-ndarray`, `burn-tch`
- All struct fields changed from `pub` to `pub(crate)`
- `RelayRLAction::new()` now requires `agent_id` parameter
- `RelayRLTrajectory::new()` changed `max_length` from `u128` to `usize`
- `DType` variants renamed for clarity (`Byte` → `U8`, `Short` → `I16`, etc.)
- Added `backend: TensorBackend` field to `TensorData`
- Replaced `SafeTensorError` with `ActionError` and `TrajectoryError`
- Enhanced error types with more specific variants
- Improved safetensors integration
- Replaced serde_pickle with bincode for better performance

### Removed
- Direct `tch::Tensor` integration
- Python integration (pyo3 dependencies)
- ZMQ network transport integration
- serde_pickle serialization
- Debug print statements from constructors
- `tch` dependency

### Fixed
- Memory efficiency with zero-copy operations
- Parallel hashing performance with BLAKE3
- Secure key generation for encryption
- Data provenance tracking
- Streaming support for large payloads

### Security
- Authenticated encryption with ChaCha20-Poly1305 AEAD
- Cryptographic integrity with BLAKE3
- Secure random key generation
- Comprehensive metadata for audit trails

---

## [0.1.x] - Legacy Version

The previous version used `tch` as the primary tensor backend with basic safetensors serialization. This version is now deprecated in favor of the more flexible and feature-rich 0.2.0 architecture.
