# Changelog

All notable changes to this project will be documented in this file.

## [0.7.1] - 2026-04-26

### Added
- **Flattened batched model inference helpers** - `ModelModule::flat_batch_inference()` exposes direct `TensorData`-to-`TensorData` inference via the existing flattened tensor-data path, and `ModelModule::flat_batch_zeros()` exposes flattened zero-action batch creation for callers that already operate on batched tensor payloads

### Changed
- **Package metadata** - `relayrl_types` crate version is now `0.7.1`
- **README changelog badge** - Updated the README changelog badge from `0.5.4` to `0.7.0`

## [0.7.0] - 2026-04-23

### Added
- **Batched model inference** - `ModelModule::step_batch()` runs a single forward pass over a batch of observations with per-row optional masks, returning action tensors, mask tensors, and auxiliary maps per row
- **Batched hot-reload forward** - `HotReloadableModel::forward_batch()` maps batched observations and rewards into a `Vec<RelayRLAction>` using the active module

### Changed
- **Trajectory environment metadata** - `RelayRLTrajectory` now carries optional `env_id` and `env_label` with `get_env_id`, `get_env_label`, `set_env_id`, and `set_env_label`; defaults and constructors initialize them to `None`
- **Records and trajectory builders** - CSV and Arrow trajectory reconstruction paths construct trajectories with the new fields set to `None` so existing serialized data continues to load without env columns

### Breaking
- **`RelayRLTrajectory::with_metadata` signature** - The constructor now takes `env_id: Option<Uuid>` and `env_label: Option<String>` after `agent_id`; call sites must pass the two new arguments (or `None`) before `episode` and `training_step`

## [0.6.0] - 2026-04-13

### Added
- **In-memory ONNX model construction** - Added APIs for building ONNX-backed models directly from raw bytes without writing model files first
  - `Model::from_onnx_bytes()` and `ModelModule::from_onnx_bytes()` now support byte-backed initialization for metadata-driven model loading paths

### Changed
- **Model bundle serialization** - Serialized model modules now carry `metadata.json` alongside the model bytes so deserialization can reconstruct a complete module from one payload
  - `serialize_model_module()` now packages metadata and model bytes together, and `deserialize_model_module()` restores both before loading the module
- **Default feature set and feature wiring** - Default builds now target `ndarray-backend` with `onnx-model` instead of enabling `codec-full` and both inference backends by default
  - `ndarray-backend` and `tch-backend` now pull in `half`
  - `compression` and `encryption` now pull in `bincode`

### Fixed
- **Reduced-feature builds** - Corrected optional codec imports so `action` and `trajectory` compile cleanly when `metadata`, `compression`, and `encryption` are not enabled

### Breaking
- **Hot-reload model handle API** - `HotReloadableModel` now uses atomic swaps internally and exposes `current_module()` for direct access to the active module
  - The public `Clone` implementation was removed, so code that cloned `HotReloadableModel` directly must switch to external shared ownership

## [0.5.4] - 2026-03-29

### Fixed
- **tch-backend-only feature sets** - Correct compilation and warnings when using `tch-backend` without `ndarray-backend`
  - Tch `to_tensor` match arms and records/tensor NdArray paths are feature-gated so `tch-backend`-only builds do not hit unreachable or inconsistent branches
- **inference-models and prelude cfg** - Resolved warnings related to `inference-models` and tightened prelude `cfg` for model exports

### Added
- **Comprehensive unit testing** - Expanded `#[cfg(test)]` coverage across core data, records, utilities, and model code
  - Data: `action`, `tensor`, `trajectory`; records (`csv`, `arrow`, shared helpers); utilities (`chunking`, `compress`, `encrypt`, `integrity`, `metadata`, `quantize`)
  - Model: `mod`, `utils`, `hot_reloadable`
  - Tests are feature-aware (`ndarray-backend`, `metadata`, `compression`, `encryption`, `integrity`, etc.) so optional stacks stay verified without breaking minimal feature sets

### Changed
- **Feature matrix for modules and prelude** - Stricter alignment between optional backends, inference features, and public surface
  - `model` is built only when both an inference feature (`tch-model` / `onnx-model`) and a tensor backend (`ndarray-backend` / `tch-backend`) are enabled
  - `action`, `tensor`, `trajectory`, and `records` (and their prelude namespaces) require at least one of `ndarray-backend` or `tch-backend`
  - Prelude `model` exports require inference features plus a tensor backend
  - NdArray-specific conversion and record paths compile only with `ndarray-backend`

## [0.5.3] - 2026-03-08

### Changed
- **Dependency and workspace alignment** - Dependency updates and workspace inheritance
  - Dependencies now use workspace inheritance where applicable (serde, serde_json, bytemuck, dashmap, uuid, tokio, tempfile, bincode)
  - Exact version pins for `tch` (=0.22.0) and `ort` (=2.0.0-rc.11)
  - Version bumps: blake3 1.8.2 → 1.8.3, bytes 1.10.1 → 1.11.1, lz4_flex 0.11.5 → 0.12.0, half 2.7.0 → 2.7.1

## [0.5.2] - 2026-02-19

### Added
- **TensorError variant added** - New variant added
  - `ShapeError(String)` added

## [0.5.1] - 2026-02-14

### Fixed
- **README formatting issues resolved** - Removed merge conflict text lol

## [0.5.0] - 2026-02-14

### Added
- **CSV Trajectory Serialization** - Complete CSV file support for trajectory data
  - `CsvTrajectory` struct with `to_csv()` and `from_csv()` methods
  - JSON-encoded tensor data with dtype, shape, and payload columns
  - Smart writer caching with append support and duplicate prevention
  - Validation cache for ensuring data consistency (backend, actor_id, timestamp)
  - Support for all tensor backends (NdArray, Tch) and data types
  - Auxiliary data serialization via JSON
- **Arrow Trajectory Serialization** - Apache Arrow IPC format support for trajectory data
  - `ArrowTrajectory` struct with `to_arrow()` and `from_arrow()` methods
  - Columnar binary format for efficient storage and querying
  - Schema-validated data with typed columns for all fields
  - Nested list support for tensor shapes and float arrays
  - Binary column support for integer and bool tensor data
  - Better compression and performance than CSV for large datasets
- **Data Records Module** - New `data::records` module structure
  - Shared helper functions for tensor data frame conversion
  - Backend string extraction utilities
  - Exported in prelude as `prelude::records`

### Changed
- **Error Types** - Added comprehensive error handling for data serialization
  - `CsvDataError` with variants for CSV failures, validation, and cache errors
  - `ArrowDataError` with variants for Arrow schema, batch, and IO errors
  - Type aliases (`CsvTrajectoryError`, `ArrowTrajectoryError`) for backwards compatibility

### Breaking
- **Namespace Reorganization** - Internal module paths were changed from `crate::types::data` to split module roots:
  - Data modules now live under `crate::data`
  - Model modules now live under `crate::model`
  - Update imports that referenced old `crate::types::data::*` paths to the new module locations
  - Recommended migration path is via stable prelude exports under `crate::prelude::*`

## [0.4.1] - 2026-01-22

### Changed
- **Burn dependency updates** - Version change from 0.18.0 to 0.20.0 for the following Burn-related dependencies:
  - `burn-tensor`
  - `burn-ndarray`
  - `burn-tch`
- **Prelude namespace exports** - Nested segments of lib into categorized prelude namespaces for:
  - `action`
  - `tensor`
  - `trajectory`
  - `model`
  - `codec`

## [0.4.0] - 2026-01-12

### Added
- **Acquire Loaded Backend Fn** - `BackendMatcher` trait now provides a function for acquiring current `SupportedTensorBackend`
  - `get_supported_backend()` returns internal `Backend` value as a `SupportedTensorBackend` enum value

### Changed
- **Hyperparams Enum to HyperparameterArgs Enum** - Changed name and altered accepted enum values
  - Removal of DashMap `Map` value, replaced with HashMap
  - `Args(Vec<String>)` changed to `List(Vec<String>)`
  - Enum now uses `serde::Serialize` and `serde::Deserialize` attributes

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
