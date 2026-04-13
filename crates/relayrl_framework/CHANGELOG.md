# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0-beta.1] - 2026-04-13

### Added
- **In-memory trajectory retrieval** - `RelayRLAgent::get_trajectory_memory()` added for draining accumulated per-actor trajectory memory from the runtime coordinator
  - Completed trajectories can now be retained in bounded in-memory buffers, and `flag_last_action` now stamps each emitted trajectory with an episode id before dispatch
- **Overflow inference mode** - `ActorInferenceMode::ServerOverflow(ModelMode, InferenceParams)` added as an experimental transport-gated mode for mixing local model ownership with remote inference fallback

### Changed
- **Local model/runtime concurrency** - Local actor model handles now use `ArcSwapOption` instead of lock-based storage so inference and model swaps can proceed through snapshot-style loads rather than blocking reload paths

### Fixed
- **Action request handling** - `request_action()` follow-up fixes improved actor reply coordination and corrected id/reference handling in the runtime action path
- **Trajectory-length defaults** - Generated config JSON and builder defaults now cap `max_traj_length` at `1000` instead of `100000000` to avoid runaway memory use in default configurations

### Breaking
- **Feature and codec surface** - Default features dropped `tch-backend`, `relayrl_framework` no longer forces `relayrl_types` `ndarray-backend` / `onnx-model` features in its dependency declaration, and `prelude::config::network_codec` now exists only when `nats-transport` or `zmq-transport` is enabled
- **Training-data mode API redesign** - `ActorTrainingDataMode` was expanded from the older `Offline` / `Hybrid` shape into explicit file and memory variants such as `OfflineWithFiles`, `OfflineWithMemory`, `OnlineWithFiles`, and `OnlineWithMemory`; the non-transport default is now `OfflineWithMemory`

## [0.5.0-beta] - 2026-04-06

### Added
- **Local model update API** - `RelayRLAgent::update_model(model, actor_ids: Option<Vec<ActorUuid>>)` added for refreshing all actors or targeted actors; validates `ModelModule` metadata before dispatch and rejects local updates when `ActorTrainingDataMode` is `Online` or `Hybrid`
- **Observability configuration** - `ClientConfigParams` and generated client config JSON now include `metrics_meter_name` and `metrics_otlp_endpoint`; metrics initialization is bound to live config state and Prometheus/OTLP exporter plumbing is threaded through the client runtime
- **Algorithm and environment prelude re-exports** - Framework prelude now exposes `relayrl_algorithms` and `relayrl_env_trait` via `prelude::algorithms`, `prelude::templates::algorithms`, and `prelude::templates::environment`
- **Expanded client test coverage** - Added `tests/local_client_smoke.rs` for a TorchScript-backed build/start/request/shutdown path and broadened unit coverage across agent, configuration, file sink, metrics, router, and lifecycle modules
- **Transport model-listener stop hooks** - `stop_model_listener()` added across sync/async transport traits, dispatcher paths, and ZMQ/NATS implementations to support explicit listener shutdown during runtime teardown

### Changed
- **Beta support contract** - `README.md`, crate docs, and client module docs now define `0.5.0-beta` as the local/default client runtime release; `zmq-transport`, `nats-transport`, and server-backed workflows are documented as experimental
- **Feature defaults and docs.rs surface** - Default features changed to `["client", "tch-backend", "metrics", "logging"]`; docs.rs feature coverage expanded, and transports now opt in separately instead of being enabled by default
- **Dependency and feature wiring** - `relayrl_algorithms`, `relayrl_env_trait`, and `generic-array` added; `relayrl_types/codec-full` moved behind `zmq-transport` / `nats-transport`; `tokio-stream` made optional under `nats-transport`; `async-nats` bumped to 0.47.0; `opentelemetry-otlp` bumped to 0.31.1 with `grpc-tonic`; dev dependencies now use `tch`, `burn-ndarray`, and `burn-tch` while `gym` was removed
- **Agent startup and transport params** - `RelayRLAgent::start()` / `restart()` now consume `AgentStartParameters<B>`; `AgentBuilder` stores `ClientModes` directly instead of `Option<ClientModes>`; `InferenceParams` / `TrainingParams` now derive `Default`, and `TrainingParams` is available whenever either transport feature is enabled
- **Runtime coordination and routing** - Coordinator, lifecycle, state, scale, actor, buffer, receiver, filter, and router dispatcher paths were reworked for shared router state, receiver-to-router dispatch, actor-count/max-traj-length-driven semaphore backpressure, tighter model-version routing, and the move of `router_dispatcher` into `runtime/router/`
- **Experimental transport behavior** - ZMQ and NATS model-listener behavior was aligned around persistent listening, explicit shutdown, and model-version propagation; transport dispatch now coordinates model update routing and cleanup more consistently across both backends
- **Observability runtime plumbing** - Metrics manager state is now rebound from config and threaded through coordinator, actor, scale, state, dispatcher, retry, and exporter paths; logging was simplified around a smaller `log4rs`-based module and broad `println!`/`eprintln!` usage was replaced with `log` macros
- **Docs, examples, and assets** - Quick-start examples now use the beta import paths and `agent.start(params)` pattern; the example server model metadata and artifact were renamed from `client_model.pt` to `server_model.pt`

### Removed
- **Legacy network presets** - Removed `full-zmq-network`, `zmq-training-network`, `zmq-inference-network`, `full-nats-network`, `nats-training-network`, `nats-inference-network`, and the old `*-training-server` / `*-inference-server` feature flags from `Cargo.toml`
- **In-crate templates and tokio utility module** - The old framework `templates` module was dropped in favor of external prelude re-exports from `relayrl_env_trait` / `relayrl_algorithms`, and `utilities::tokio` was removed from the crate
- **Legacy logging helpers and runtime statistics stub** - Deleted the old logging builder/filter/sink submodules and removed the placeholder `RelayRLAgent::runtime_statistics()` API

### Fixed
- **Configuration schema and defaults** - Corrected `hyperparameter_args` naming, renamed default transport JSON keys to `*_address`, changed the default local trajectory output type to valid `Csv`, added default metrics fields, and corrected the default training server model name to `training_server_model`
- **Local trajectory directory handling** - `LocalTrajectoryFileParams::new()` now retries directory creation, verifies that the resolved path is actually a directory, and `Default` no longer assumes validation cannot fail
- **Config hot-reload behavior** - Lifecycle polling now rebuilds its interval only when `config_update_polling_seconds` changes, uses a unified `handle_config_change()` path, and keeps shared `max_traj_length` state synchronized as configuration updates are applied
- **Shutdown and listener cleanup** - Actor shutdown now surfaces timeout failures more cleanly, and transport/model listeners can be stopped explicitly during receiver shutdown instead of depending on best-effort teardown
- **Metrics exporter rebinding and gated builds** - Prometheus collector reuse, OTLP provider rebinding, docs.rs compilation, and transport-specific feature-gated builds were tightened across metrics and transport codepaths

### Breaking
- Default features changed from `["client", "zmq-transport", "nats-transport"]` to `["client", "tch-backend", "metrics", "logging"]`; consumers must now opt into `zmq-transport` or `nats-transport` explicitly
- Feature flags renamed/removed: legacy network presets and `zmq-training-server` / `zmq-inference-server` / `nats-training-server` / `nats-inference-server` were removed in favor of explicit transport flags plus `training-server` / `inference-server`
- `RelayRLAgent::start()` / `restart()` now take `AgentStartParameters<B>` instead of the previous positional argument lists
- `AgentBuilder::client_modes` is no longer `Option<ClientModes>`
- Prelude imports moved from flat `prelude::{tensor,action,trajectory,model,templates}` exports to `prelude::types::*`, `prelude::algorithms`, and `prelude::templates::{algorithms,environment}`; the in-crate `templates` module is gone
- `RuntimeStatisticsReturnType` is now gated behind `metrics` or `logging`, and the placeholder `runtime_statistics()` API was removed
- `ClientError` is no longer `#[non_exhaustive]`, and new variants were added for model validation and local model update support
- `TransportConfigParams::max_traj_length` and related builder setters now use `usize` instead of `u128`
- Generated/default config JSON changed shape: transport keys now use `*_address` names, metrics fields were added, `trajectory_file_output.file_type` defaults to `Csv`, and the example server model filename changed to `server_model.pt`

## [0.5.0-alpha.3] - 2026-03-13

### Added
- **NATS Implementation** - Full NATS transport implementation replacing alpha.2 stubs; `nats/ops.rs` added (~1459 lines); `nats/interface.rs` and `nats/policies.rs` substantially completed with working send/receive logic, stream handling, and authentication policy enforcement
- **`tokio-stream` dependency** - Added `tokio-stream 0.1.18` to support async NATS stream processing
- **`combine_results` utility** - `combine_results` added to `transport_sink/mod.rs` for aggregating multi-transport results; `client_transport_factory` made `async`
- **`is_local_file_writing_enabled` helper** - Crate-level helper on `RelayRLAgent` to query whether local file writing is active

### Changed
- **NATS `SharedTransportAddresses` init** - `lifecycle_manager.rs` updated to properly initialize `SharedTransportAddresses` when constructing a NATS transport client
- **Transport dispatcher param ownership** - `send_client_ids` and `send_inference_model_init_request` parameters changed from references to owned values
- **ZMQ model listener** - Model listener updated to receive multipart ZMQ messages carrying both the serialized model and `actor_id`
- **Example config NATS addresses** - All three example configs updated to include the NATS address namespace
- **`async_trait` on async transport traits** - `#[async_trait]` applied uniformly across async transport traits in `transport_sink`

### Fixed
- **Feature flag compilation** - Resolved compilation errors under `client` and `client zmq-transport` feature flag selections across `coordinator.rs`, `scale_manager.rs`, `transport_sink/mod.rs`, `zmq/ops.rs`, and `router/buffer.rs`
- **Trailing comma in example configs** - Stray JSON trailing comma removed from all three example configs and `configuration.rs` that would have caused config parsing failures at runtime

---

## [0.5.0-alpha.2] - 2026-03-07

### Changed
- **relayrl_types** - Dependency updated from 0.5.2 to 0.5.3; fixes ort-related compilation error
- **Workspace dependency hierarchy** - Dependency versions and shared crates (e.g. `relayrl_types`, `tokio`, `serde`, `burn-tensor`, `arrow`, etc.) reworked in root [Cargo.toml](Cargo.toml) via `[workspace.dependencies]`; framework crate uses `relayrl_types = { workspace = true, ... }` and other workspace-inherited deps
- **ZMQ ops** - Removed `unwrap`/`expect` in [crates/relayrl_framework/src/network/client/runtime/data/transport_sink/zmq/ops.rs](crates/relayrl_framework/src/network/client/runtime/data/transport_sink/zmq/ops.rs); errors now propagated via `Result` and `map_err` where appropriate

### Added
- **NATS scaffolding** - Additional scaffold for NATS transport: [nats/policies.rs](crates/relayrl_framework/src/network/client/runtime/data/transport_sink/nats/policies.rs) placeholder (`NatsAuthentication`); [nats/interface.rs](crates/relayrl_framework/src/network/client/runtime/data/transport_sink/nats/interface.rs) implements `AsyncClientTransportInterface`, `AsyncClientScalingTransportOps`, `AsyncClientInferenceTransportOps`, and related execution traits with stub method bodies ready for implementation

---

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
