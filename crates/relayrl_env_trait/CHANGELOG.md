# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-04-26

### Added
- **Flattened byte-oriented environment APIs** - `Environment` now exposes `observation_dim()`, `action_dim()`, `flat_observation_bytes()`, and `action_is_discrete()` so runtimes can consume observations and action metadata without depending on Burn tensor generics
- **Dynamic scalar environment helpers** - `DynScalarEnvironment` now provides `dyn_flat_obs()`, `dyn_step()`, and `dyn_act_dim()` forwarding helpers for object-safe flattened observation, step, and action-dimension access
- **Vector environment runtime helpers** - `VectorEnvironment` now exposes `n_envs()` and `step_bytes()` for byte-buffer batched stepping that returns flattened observations, rewards, and completion flags

### Changed
- **Package metadata** - `relayrl_env_trait` crate version is now `1.2.0`
- **Environment reset payloads** - `ScalarEnvReset` and `VectorEnvReset` now carry observations as `Vec<u8>` instead of `burn_tensor::Tensor` values
- **Trait object and handle types** - `DynVectorEnv`, `DynScalarEnvironment`, `EnvironmentHandle`, `Environment`, `ScalarEnvironment`, and `VectorEnvironment` no longer require backend, dimension, or tensor-kind generic parameters
- **Scalar stepping API** - `ScalarEnvironment` now uses `step_bytes(&[u8]) -> Option<(Vec<u8>, f32, bool)>` instead of accepting a typed action tensor and returning `ScalarEnvStep`
- **Vector stepping API** - `VectorEnvironment` now uses `step_bytes(&[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)>` instead of accepting `(EnvironmentUuid, Tensor)` action pairs and returning `VectorEnvStep` values

### Removed
- **Burn tensor dependency** - Removed the `burn-tensor` dependency and the public re-export of `Backend`, `Tensor`, and `TensorKind`
- **Step result structs** - Removed `ScalarEnvStep` and `VectorEnvStep`; step methods now return flattened observation bytes, rewards, and completion flags directly

### Breaking
- **Environment trait implementation signatures** - Implementors must remove Burn tensor generic parameters from environment traits and implement the new flattened byte API methods
- **Observation and action representation** - Callers that used typed Burn tensors in reset or step payloads must convert to/from `Vec<u8>` plus explicit dtype and dimension metadata
- **Truncation and per-step info payloads** - The new `step_bytes()` return tuples expose completion flags but no longer include separate `truncated` booleans or optional `EnvInfo` values from the removed step result structs

## [1.1.0] - 2026-04-23

### Added
- **Vector environment support** - Added `VectorEnvironment`, `VectorEnvReset`, `VectorEnvStep`, and `EnvironmentUuid` so batched and multi-environment runtimes can address logical environments explicitly
- **Tensor-backed environment typing** - Added `EnvDType`, `EnvNdArrayDType`, and `EnvTchDType` along with `burn-tensor` and `uuid` dependencies for backend-aware environment APIs
- **Environment dispatch helpers** - Added `EnvironmentKind`, `kind()`, and `into_handle()` support so callers can route scalar and vector environments through a common handle

### Changed
- **Trait exports and crate surface** - The crate now exposes its public API from `traits` and re-exports it at the crate root with `pub use traits::*`
- **Structured environment info** - Reset and step payloads now use optional `EnvInfo` collections instead of requiring environment metadata on every response

### Breaking
- **Generic environment trait model** - `Environment` is now generic over backend, tensor dimensions, and tensor kinds, and implementors must provide `observation_dtype()`, `action_dtype()`, `kind()`, and `into_handle()`
  - Code using the older `environment_traits` module path must migrate to the crate root or `traits`
  - Implementations should now align with `ScalarEnvironment` or `VectorEnvironment` depending on whether they execute one environment at a time or batched environment steps
