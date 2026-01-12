# Changelog

All notable changes to this project will be documented in this file.

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

*For detailed v0.4.52 documentation, see the prototype README in `RelayRL-prototype/relayrl_framework/`*
