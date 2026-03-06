# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0-alpha.1] - 2026-03-07

### Added
- **NATS Transport Scaffold** - `TransportType::NATS` variant with `nats-transport` feature and `async-nats` 0.46.0 dependency; scaffold modules for `NatsInterface` (not yet implemented)
- **Trait-Based Transport Abstraction** - `SyncClientTransportInterface`, `AsyncClientTransportInterface` base traits; separate operation traits: `SyncClientInferenceTransportOps`, `SyncClientTrainingTransportOps`, `SyncClientScalingTransportOps` (and async variants)
- **Training/Inference Dispatcher Split** - `InferenceDispatcher`, `TrainingDispatcher`, `ScalingDispatcher` replacing monolithic transport dispatcher; `ProcessInitRequest` enum for algorithm/inference init
- **Actor Model Modes** - `ModelMode::Shared` (per-device model pool, reused across actors) and `ModelMode::Independent` (per-actor model handle)
- **ClientModes System** - `ClientModes` struct with `ActorInferenceMode` (`Local(ModelMode)` / `Server`) and `ActorTrainingDataMode` (`Online` / `Offline` / `Hybrid` / `Disabled`) with invariant validation
- **CSV Trajectory File Sink** - `LocalTrajectoryFileType` enum (`Arrow`, `Csv`); `write_local_trajectory_file()` supporting both formats via `relayrl_types` `ArrowTrajectory` and `CsvTrajectory`
- **Transport Resilience Policies** - `RetryPolicy`, `CircuitBreaker`, `BackpressureController` in `zmq/policies` module with configurable backoff and concurrency limits
- **Network Feature Presets** - `full-zmq-network`, `zmq-training-network`, `zmq-inference-network`, `full-nats-network`, `nats-training-network`, `nats-inference-network`
- **tch-backend Feature** - Optional `tch-backend` feature flag (ndarray is now the default backend)
- **Prelude Submodules** - `tensor::burn`, `tensor::relayrl`, `action`, `trajectory`, `model`, `config::network_codec` submodules in prelude

### Changed
- **Transport Layer Rewrite** - Monolithic `transport/` module replaced with modular `transport_sink/` architecture; ZMQ split into `interface`, `ops`, `policies` submodules
- **Feature Flags Redesigned** - Old flags (`network`, `transport_layer`, `async_transport`, `sync_transport`, `database_layer`) replaced with transport-specific flags (`zmq-transport`, `nats-transport`) and server composition flags (`zmq-training-server`, `zmq-inference-server`, etc.)
- **Default Features** - Changed from `["client"]` to `["client", "zmq-transport"]`
- **Scaling System Rewrite** - `scale_in`/`scale_out` major rewrite; bare UUID args replaced with pool entries `(namespace, context, uuid)`; scaling protocol permits with backpressure; parallel scaling operations
- **Router Namespaces** - `router_ids` replaced with `RouterNamespace` (`Arc<str>`) for namespace-based routing and actor distribution
- **Server Addresses** - `server_addresses` renamed to `transport_addresses` / `SharedTransportAddresses`; split into `SharedInferenceAddresses` and `SharedTrainingAddresses`; address prefix system removed
- **Dependencies to Workspace** - `relayrl_types`, `tokio`, `serde`, `dashmap`, `thiserror`, `async-trait`, `burn-tensor`, `arrow`, `arrow-schema`, `arrow-array` now use workspace inheritance
- **Default Tensor Backend** - `ndarray-backend` is now the default compilation target; `tch-backend` made optional via feature flag
- **active-uuid-registry** - Bumped 0.3.0 to 0.7.0; namespace/context-based pool entry model
- **Actor Construction** - `new_actor(s)` / `remove_actor(s)` improved with `ClientModes` propagation through coordinator, scale manager, state manager, actor chain
- **Trajectory Buffer** - `PersistentTrajectoryDataSinkTrait` renamed to `LocalFileTrajectorySinkTrait`; uses `TrainingDispatcher` instead of raw `TransportClient`; `TrajectoryFileParams` renamed to `LocalTrajectoryFileParams`
- **Server Config Paths** - Distinct `training_server_config.json` and `inference_server_config.json` with dedicated macros (`resolve_training_server_config_json_path!`, `resolve_inference_server_config_json_path!`)
- **Environment Traits** - `EnvironmentTrainingTrait` and `EnvironmentTestingTrait` methods now return `Result<_, EnvironmentError>` with `thiserror`-based error type
- **Server Legacy Directory** - `server/old/` renamed to `server/legacy/`

### Removed
- **Database Layer** - `database_layer`, `postgres_db`, `sqlite_db` features removed; `postgres` and `sqlite` dependencies removed
- **Old Transport Module** - Client-side `transport/` directory and monolithic `transport_dispatcher.rs` replaced by `transport_sink/`
- **serde-pickle** - Dependency removed
- **Profile Sections** - `[profile.dev]` and `[profile.release]` removed from crate `Cargo.toml` (moved to workspace)

### Fixed
- **Scaling Initialization** - Coordinator incorrectly called `scale_in` instead of `scale_out` when transport was disabled, preventing routers from being created and leaving actors unable to receive data payloads
- **Prelude Struct Exports** - Stale `ServerConfigBuilder` / `ServerConfigLoader` / `ServerConfigParams` exports updated to match renamed `TrainingServerConfig*` types
- **Tensor Re-exports** - `prelude::tensor::burn` corrected to re-export from `relayrl_types::prelude::tensor::burn` instead of raw `burn_tensor`; `prelude::tensor::relayrl` corrected to `relayrl_types::prelude::tensor::relayrl`
- **Documentation URL** - Fixed `docs.rs` URL in `Cargo.toml` (`docs.rs/crates/...` to `docs.rs/crate/...`)

### Breaking
- Feature flags renamed: `transport_layer` / `async_transport` / `sync_transport` / `zmq_transport` to `zmq-transport` / `nats-transport` / `zmq-*-server` / `nats-*-server`
- Default features changed from `["client"]` to `["client", "zmq-transport"]`
- `ServerAddresses` renamed to `SharedTransportAddresses` with inference/training split
- `RouterUuid` / `router_ids` replaced by `RouterNamespace`
- `TrajectoryFileParams` renamed to `LocalTrajectoryFileParams`
- `PersistentTrajectoryDataSinkTrait` renamed to `LocalFileTrajectorySinkTrait`
- Database features and dependencies removed
- `tch-backend` no longer included by default; must opt in via `tch-backend` feature

---

## [0.5.0-alpha] - 2026-01-10

### Added
- **Multi-Actor Runtime** - Native support for concurrent actor execution with dynamic actor management
  - `new_actor()`, `new_actors()`, `remove_actor()` for runtime actor control
  - Per-actor model management with round-robin router assignment
  - `get_actor_ids()`, `set_actor_id()` for actor identification
- **Builder Pattern API** - Ergonomic agent construction with `AgentBuilder<B, D_IN, D_OUT, KindIn, KindOut>`
  - Fluent interface for configuration with type-safe parameter validation
  - Supports `actor_count()`, `router_scale()`, `default_device()`, `default_model()`, `config_path()`
- **Throughput Scaling** - Dynamic router worker scaling via `scale_throughput(n)` to add/remove routing workers
- **Action Flagging** - Mark actions as terminal with `flag_last_action(ids, reward)` for episode termination
- **Model Versioning** - Track model versions per actor with `get_model_version(ids)`
- **Backend-Agnostic Tensors** - `AnyBurnTensor<B, D>` enum with `FloatBurnTensor`, `IntBurnTensor`, `BoolBurnTensor` variants
- **Device Type Support** - `DeviceType` enum for hardware selection (`Cpu`, `Cuda(device_id)`, `Mps`)
- **Arrow File Sink** - Local trajectory data storage in Apache Arrow format for offline training
- **Observability Infrastructure** - Feature-gated logging and metrics systems
  - `LoggingBuilder` with console/file sinks (`logging` feature)
  - `MetricsManager` with Prometheus/OpenTelemetry export (`metrics` feature)
- **Database Layer** - PostgreSQL (`postgres_db`) and SQLite (`sqlite_db`) support (under development)
- **Environment Traits** - `EnvironmentTrainingTrait` and `EnvironmentTestingTrait` for custom environments
- **Algorithm Hyperparameters** - Expanded support for DDPG, PPO, REINFORCE, TD3, and custom algorithms
- **Transport Configuration** - Separate addresses for model server, trajectory server, agent listener, scaling server, and inference server
- **Prelude Module** - Convenient imports via `relayrl_framework::prelude::*`
- **Coordination Layer** - New runtime managers: `ClientCoordinator`, `ScaleManager`, `StateManager`, `LifecycleManager`
- **Routing Layer** - `RouterDispatcher` with scalable router workers and `TrajectoryBuffer` for message dispatching

### Changed
- **Architecture Redesign** - Complete rewrite from monolithic to layered architecture
  - New coordination, routing, actor, and data layers with separation of concerns
  - Multi-actor native design replacing single-agent focused approach
- **Agent API** - Now requires generic type parameters `RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>`
  - Old: `RelayRLAgent::new(model, config_path, server_type, ...).await`
  - New: `AgentBuilder::builder().actor_count(4).build().await?`
- **Action Request** - Returns `Vec<(Uuid, Arc<RelayRLAction>)>` instead of single action for multi-actor support
- **Configuration System** - Separated into client/server configurations
  - `ClientConfigLoader` with `client_config.json`, `ServerConfigLoader` with `server_config.json`
  - New JSON structure with nested `client_config` and `transport_config` sections
- **Type System** - Core types moved to external `relayrl_types` crate (`RelayRLAction`, `TensorData`, `RelayRLData`, `RelayRLTrajectory`, `ModelModule`, `HotReloadableModel`)
- **Tensor Backend** - Switched from `tch` to `burn-tensor` with `NdArray` (CPU) and `Tch` (CPU/CUDA/MPS) backend support
- **Error Handling** - Replaced panics with proper `Result` types using `thiserror`
  - New error types: `ClientError`, `CoordinatorError`, `ScaleManagerError`, `StateManagerError`, `LifecycleManagerError`
- **Feature Flags** - Complete reorganization
  - Old: `full`, `networks`, `grpc_network`, `zmq_network`, `data_types`, `python_bindings`
  - New: `client`, `network`, `inference_server`, `training_server`, `transport_layer`, `database_layer`, `logging`, `metrics`
- **Default Feature** - Changed from `full` to `client`
- **Crate Type** - Changed from `["rlib", "cdylib"]` to `["rlib"]`
- **Dependencies Updated** - `tokio` 1.44.2 → 1.48.0, `rand` 0.8.5 → 0.9.2
- **Dependencies Added** - `relayrl_types`, `active-uuid-registry`, `burn-tensor`, `arrow`, `dashmap`, `thiserror`, `async-trait`, `uuid`, `log`, `bincode`

### Removed
- **Python Bindings** - All PyO3-based bindings removed from core framework
  - `PyRelayRLAgent`, `PyTrainingServer`, `PyConfigLoader`, `PyRelayRLAction`, `PyRelayRLTrajectory`
  - Functionality will be available in separate `relayrl_python` crate
- **gRPC Transport** - All Tonic/Protobuf code removed
  - `agent_grpc.rs`, `training_grpc.rs`, `grpc_utils.rs`, `proto/relayrl_grpc.proto`
  - `grpc_network` and related feature flags
- **Python Algorithm Runtime** - Python subprocess management for algorithms removed
  - `python_subprocesses/` module, `native/python/` algorithm implementations
  - Functionality will be available in separate `relayrl_algorithms` crate
- **Direct TorchScript Support** - `tch` crate dependency removed; `CModule` replaced with `ModelModule<B>` abstraction
- **Dependencies Removed** - `tch`, `tonic`, `tonic-build`, `prost`, `pyo3`, `pyo3-build-config`, `safetensors`

### Fixed
- **Error Propagation** - Near complete removal of panics with proper upstream error propagation
- **Memory Management** - Improved Arc-based sharing for tensor data and actions

### Breaking
- Agent construction API changed to builder pattern with generic type parameters
- Configuration file format changed from `relayrl_config.json` to separate `client_config.json` / `server_config.json`
- Action request returns `Vec<(Uuid, Arc<RelayRLAction>)>` instead of single action
- All core types moved to `relayrl_types` crate - requires adding dependency
- Python bindings no longer available in this crate
- gRPC transport no longer supported

---

## [0.4.52] - Previous Release

Final release of the prototype version with Python-first design.

### Features
- gRPC and ZMQ transport support
- PyO3-based Python bindings
- TorchScript model inference via `tch`
- REINFORCE algorithm implementation (Python)
- Single-agent focused API
- Unified configuration system

*For detailed v0.4.52 documentation, see the prototype README in [RelayRL-prototype/relayrl_framework/](https://github.com/jrcalgo/RelayRL-prototype)*
