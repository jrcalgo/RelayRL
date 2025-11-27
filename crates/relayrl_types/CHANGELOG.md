# Changelog

All notable changes to this project will be documented in this file.

## [0.3.2] - 2025-11-27

### Changed
- **Memory Optimization** - Tensor wrappers and models now consume shared references (`Arc`) instead of cloning values
  - `FloatBurnTensor`, `IntBurnTensor`, and `BoolBurnTensor` now store tensors as `Arc<Tensor<...>>`
  - `AnyBurnTensor` conversion methods (`into_f16_data`, `into_f32_data`, etc.) now accept `Arc<Self>` instead of `Self`
  - Significantly reduces memory allocations and improves performance for tensor operations
  - Model inference paths updated to work with shared tensor references
- **ONNX Runtime Update** - Updated `ort` dependency from `1.16.3` to `2.0.0-rc.10`
  - Provides access to latest ONNX Runtime features and improvements
  - Better compatibility with newer ONNX models
- **Default Features** - Added `inference-models` to default features
  - `inference-models` feature bundle includes both `tch-model` and `onnx-model`
  - Enables model inference capabilities by default for better out-of-the-box experience
- **Code Simplification** - Reduced code complexity in model module
  - Simplified tensor extraction and conversion logic
  - Removed redundant type conversion paths
  - Improved code maintainability with cleaner helper functions

### Fixed
- **Tensor Conversion Methods** - Fixed tensor type extraction to use pattern matching on `Arc` references
  - Improved type safety for tensor conversions
  - Better error handling for unsupported tensor type conversions

## [0.3.12] - 2025-11-17

### Changed
- **ModelError Implementation** - Enhanced `ModelError` with `thiserror` derive for better error handling
  - Added `thiserror` dependency (v2.0.17) to Cargo.toml
  - Updated `ModelError` enum to derive from `thiserror::Error`
  - Added `#[error(...)]` attributes to all error variants for improved error messages
  - Provides better integration with error handling libraries and more consistent error formatting

## [0.3.11] - 2025-11-15

### Fixed
- **Model Step Mask Handling** - Fixed mask tensor conversion in `ModelModule::step()`
  - Corrected match statement syntax for `AnyBurnTensor` pattern matching
  - Fixed mask reference to use `ref` to avoid moving the mask value
  - Added proper error handling with `.expect()` for mask tensor conversion failures
- **Model Validation** - Updated `validate_model_shapes()` to use the new 3-tuple return signature from `step()`
  - Now correctly destructures `(TensorData, Option<TensorData>, HashMap)` return value

### Changed
- **Default Features** - Restored `codec-full` to default features for better out-of-the-box functionality
- **Package Metadata** - Updated author email address

## [0.3.1] - 2025-11-15

### Added
- **HotReloadableModel Getters** - Added convenience getter methods for better API ergonomics
  - `default_device()` - Access the default device configuration
  - `version()` - Get the current model version atomically
  - `input_dim()` - Get the input dimension
  - `output_dim()` - Get the output dimension
- **ModelError Display** - Implemented `std::fmt::Display` for `ModelError` for better error messages and logging

### Changed
- **Default Features** - Removed `tch-model` and `onnx-model` from default features to reduce default dependency footprint
  - Model inference features are now opt-in via `tch-model` or `onnx-model` feature flags
- **Model Module Feature Gating** - Model module is now conditionally compiled based on feature flags
  - Only available when `tch-model` or `onnx-model` features are enabled
  - Model types in prelude are also feature-gated
- **Step Method Simplification** - Simplified `ModelModule::step()` return signature
  - Now returns `(TensorData, Option<TensorData>, HashMap<String, RelayRLData>)`
  - Mask tensor is now returned directly as `Option<TensorData>` instead of complex runtime conversion logic
  - Simplified mask handling in `HotReloadableModel::forward()`
- **README Updates** - Clarified feature flag documentation and organization

### Fixed
- **LibTorch Bool Tensor Handling** - Fixed bool tensor conversion in LibTorch inference path
  - Corrected bool tensor serialization to use `u8` instead of direct bool casting
  - Fixed tensor shape handling for bool observations
- **Tensor Conversion Stability** - Improved tensor conversion reliability in ONNX inference paths
  - Fixed dtype cloning issues in `match_obs_to_act()` calls
  - Better error handling for tensor type mismatches
- **Tch Backend Fixes** - Various fixes for tch-backend tensor operations
  - Improved memory handling for zero-initialized tensors
  - Fixed tensor data lifetime issues

## [0.3.0] - 2025-11-10

### Added
- **Model Module** - Complete `ModelModule<B>` implementation with full ONNX and LibTorch inference support
  - `step()` method for running inference with optional masking
  - `zeros_action()` method for creating zero-initialized action tensors
  - `run_libtorch_step()` for PyTorch/LibTorch model inference
  - `run_onnx_step()` for ONNX Runtime model inference
  - Support for all tensor dtypes (F16, F32, F64, I8, I16, I32, I64, U8, Bool, BF16)
- **Hot-Reloadable Models** - `HotReloadableModel` for dynamic model reloading without service interruption
- **ONNX Runtime Integration** - Full ONNX model support with type-safe tensor conversions
  - `convert_obs_to_act()` helper for observation to action conversion
  - `match_obs_to_act()` helper for runtime dtype dispatching
  - Proper lifetime management for `OrtValue` and array handling
- **LibTorch Integration** - Complete PyTorch/LibTorch model support
  - Seamless conversion between Burn tensors and Tch tensors
  - Support for all numeric types and half-precision floats
- **AnyBurnTensor Enhancements** - Improved generic tensor wrapper
  - `into_f16_data()`, `into_bf16_data()`, `into_f32_data()`, `into_f64_data()` conversion methods
  - `into_i8_data()`, `into_i16_data()`, `into_i32_data()`, `into_i64_data()` conversion methods
  - `into_u8_data()`, `into_bool_data()` conversion methods
  - Better error handling for type conversions
- **Model Utilities** - Enhanced helper functions in `model/utils.rs`
- **Public API Reorganization** - Unified prelude for easier imports
  - Renamed `data_prelude` to `prelude`
  - Added model types to prelude: `ModelModule`, `ModelError`, `HotReloadableModel`
  - Exported `AnyBurnTensor`, `BoolBurnTensor`, `FloatBurnTensor`, `IntBurnTensor`

### Changed
- **Prelude Module** - Consolidated and renamed for better discoverability
  - `data_prelude` → `prelude`
  - Removed empty `model_prelude` module
  - Added comprehensive model type exports
- **Tensor Type Exports** - Added burn tensor wrapper types to public API
- **Model Inference** - Improved error handling and type safety across inference paths
- **Tensor Conversions** - Enhanced type-safe conversions between different tensor representations

### Fixed
- **ONNX Lifetime Issues** - Resolved lifetime management for `OrtValue::from_array()` calls
- **Type Conversion Stability** - Fixed type casting between different tensor backends
- **Generic Constraints** - Improved trait bounds for tensor element types
- **F16/BF16 Handling** - Proper conversion to F32 for ONNX models (ONNX doesn't support half-precision)

### Breaking
- Model inference API expanded - if using models directly, review new `ModelModule` API
- Tensor wrapper types now part of public API - may affect type resolution in some contexts

## [0.2.11] - 2025-10-26

### Changed
- Made `TensorData` fields public (was `pub(crate)`) for better external access to backend information
- Improved code formatting and readability in `ConversionTensor` implementation
- Better error handling formatting for quantization feature requirements

### Fixed
- Import ordering in tensor module for better consistency

## [0.2.1] - 2025-10-19

### Added
- Generic tensor conversion: `ConversionTensor<B, D, K>` → `TensorData` using a target `conversion_dtype` (supports K = Float, Int, Bool)
- Runtime device selection via `DeviceType` (Cpu, Cuda(idx), Mps) and backend resolution via `BackendMatcher`
- Full support for `bf16` (when `quantization`/`half` feature is enabled) and `u8` int tensors on the `tch` backend

### Changed
- Refactored tensor/backends API into `types/tensor.rs` (centralized dtype, backend, device, and conversion utilities)
- Replaced `TensorBackend` with `SupportedTensorBackend` for clearer runtime/backend intent
- `RelayRLAction::to_tensor` and getters now accept `&DeviceType` for user-selected CPU/GPU
- Adopted `bincode::serde::{encode_to_vec, decode_from_slice}` across encoding paths for consistency
- Updated `burn-tch` import to `burn_tch::LibTorch as Tch` to align with 0.18 APIs
- Crate metadata updated (repository, documentation URLs)

### Fixed
- Correct handling of `TchDType::Bf16` (distinct from `f16`) by converting through `bf16` to `f32`
- Stable bool serialization by packing `Vec<bool>` to `Vec<u8>`
- Resolved device associated-type mismatches by routing devices through `BackendMatcher`

### Breaking
- `RelayRLAction::to_tensor` signature changed to require `&DeviceType`; corresponding `get_*_tensor` helpers updated
- `DType` streamlined under backends (`NdArray(...)` / `Tch(...)`); removed legacy `None` variant
- Renamed/standardized backend enums to `SupportedTensorBackend`

## [0.2.0] - 2025-10-15

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
