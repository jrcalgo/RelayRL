# Changelog

All notable changes to this project will be documented in this file.

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
