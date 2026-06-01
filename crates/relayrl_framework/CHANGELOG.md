# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0-beta.5] - 2026-06-01

### Added
- **`AgentBuilder` extracted into `builder.rs`** - `AgentBuilder`, `AgentStartParameters`, `ClientModes`, `ActorInferenceMode`, `ActorTrainingDataMode`, `ModelMode`, `LocalTrajectoryFileParams`, `LocalTrajectoryFileType`, `AlgorithmInitArgs`, `DefaultHyperparameterArgs`, `InferenceParams`, `TrainingParams`, and all transport address args are moved from `agent.rs` into a new `builder.rs` module and re-exported from `agent.rs`.
- **`run_env_with_ppo`, `run_env_with_ippo`, `run_env_with_mappo` on `RelayRLActorEnv`** - `RelayRLActorEnv` now exposes three PPO-specific `run_env_with_*` methods alongside `run_env_eval` so callers can drive integrated PPO/IPPO/MAPPO training directly from the environment loop without supplying a generic kernel.
- **`TrainingInterface` and `runtime/data/training/mod.rs`** - New module introduces `TrainingInterface<B>`, `TrainingError`, `train_ppo`, and a `train_mappo` stub that own the per-actor async PPO training loop: observation collection, GAE finalization, background SGD through `PPOTrainer`, and refreshing the pi/vf `ModelModule` back into the actor runtime after each epoch.
- **`ErasedActorRuntime` trait** - New object-safe `ErasedActorRuntime<B>` trait in `actor.rs` allows `StateManager` and coordinator to store actor runtimes as `Arc<dyn ErasedActorRuntime<B>>`, eliminating the `D_IN`/`D_OUT` const-generic parameters from manager structs and impl blocks. `ActorShape` and `ActorDTypes` helper structs accompany the trait.
- **`ToAnyBurnTensor` trait** - Added to `coordinator.rs` with implementations for `Tensor<B, D, Float>`, `Tensor<B, D, Int>`, and `Tensor<B, D, Bool>`, enabling typed tensor inputs to `request_action` without requiring callers to pre-wrap in `AnyBurnTensor`.
- **`router_buffer_size_per_actor` configuration field** - `ClientConfigParams` and the transport builder now expose `router_buffer_size_per_actor` (default `1000`); the config JSON template includes the field and the loader round-trips it.
- **`flat_mask_bytes` on environment interfaces** - `EnvironmentInterface`, `ScalarVecEnv`, `BatchVecEnv`, and `SingleVecEnv` now expose `flat_mask_bytes()` returning `Option<Vec<u8>>`, propagating action masks from environments through the runtime. `ScalarVecEnv::new` gathers masks at construction time.
- **`rand_distr` workspace dependency** - Added to `relayrl_framework` as a direct dependency.
- **Transport features now on by default** - `nats-transport` and `zmq-transport` added to the crate's `default` feature set.

### Changed
- **`RelayRLAgent` and `RelayRLAgentActors` de-generified** - `RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>` and the `RelayRLAgentActors` trait are simplified to `RelayRLAgent<B>` / `RelayRLAgentActors<B>`; dimension and kind type parameters move to individual `new_actor<const D_IN, const D_OUT>()` and `new_actors<const D_IN, const D_OUT>()` call sites.
- **`request_action` now accepts plain `Tensor` inputs** - `request_action<D_IN, D_OUT, KindIn, KindOut>` accepts `Tensor<B, D_IN, KindIn>` and `Option<Tensor<B, D_OUT, KindOut>>` directly via `ToAnyBurnTensor` bounds, instead of `Arc<AnyBurnTensor<B, D>>`.
- **`RelayRLActorEnv` simplified** - `run_env()` with its wide generic kernel bounds (`StepKernelTrait`, `REINFORCEKernelTrait`, `DDPGKernelTrait`, …) is replaced by `run_env_eval()` for pure inference loops and the typed `run_env_with_ppo/ippo/mappo` methods for integrated training.
- **`StateManager` de-generified** - `StateManager<B, D_IN, D_OUT>` collapses to `StateManager<B>`; dimension parameters move to individual `new_actor<D_IN, D_OUT>` and `restart_actor<D_IN, D_OUT>` call sites, consistent with the `ErasedActorRuntime` change.
- **`ClientInterface` and `ClientEnvironments` de-generified** - Both coordinator traits drop their `D_IN`/`D_OUT` const-generic parameters; `new_actor` becomes `new_actor<const D_IN, const D_OUT>`.
- **`AlgorithmCfg` replaced by `AlgorithmInitArgs`** - `AlgorithmCfg` (covering DDPG, TD3, REINFORCE, PPO, and custom) is replaced by `AlgorithmInitArgs { PPO, IPPO, MAPPO }`, reflecting the algorithm scope reduction in `relayrl_algorithms` 0.4.0; `configuration.rs` hyperparameter defaults are updated accordingly.
- **Config JSON and `ClientConfigParams` updated** - `algorithm_name` field renamed to `algorithm`; DDPG/TD3/REINFORCE/custom hyperparameter blocks removed from the generated config JSON; PPO defaults adjusted (`traj_per_epoch: 1`, `clip_ratio: 0.1`, `train_pi_iters: 40`, `vf_lr: 3e-4`); `max_traj_length` removed from `ClientConfigParams`; `get_algorithm_name()` / `set_algorithm_name()` removed; `set_router_buffer_size_per_actor()` added to the builder.
- **`step_bytes` return type extended** - `EnvironmentInterface::step_bytes`, `ScalarVecEnv::step_bytes`, and the `VectorEnvironment`/`ScalarEnvironment` trait methods now return `(obs_bytes, Option<mask_bytes>, rewards, dones, truncateds)`, adding truncation signals and optional mask bytes alongside the existing done flags.
- **`env_dtype_to_dtype`, `decode_argmax`, `decode_continuous_bytes` consolidated** - These internal helpers are removed from `actor.rs` and moved to `state_manager.rs`.
- **Crate description updated** - `relayrl_framework` description changed to "A heterogeneous RL runtime control platform for concurrent multi-actor execution."
- **Legacy server files removed** - `server/legacy/training_grpc.rs`, `training_server_wrapper.rs`, and `training_zmq.rs` deleted.
- **`relayrl_types` updated to `0.8.0`, `relayrl_algorithms` updated to `0.4.0`** - Workspace dependency versions bumped in root `Cargo.toml`.
- **`relayrl_env_trait` updated** - Self-documenting type aliases (`Observation`, `Mask`, `Reward`, `Done`, `Truncated`) added to the trait surface; `build_mask()` and `flat_mask_bytes()` added to `Environment`; `DynScalarEnvironment`/`DynVectorEnv` formatting cleaned up; `ScalarEnvironment`/`VectorEnvironment` step signatures updated to match the extended return type.

### Breaking
- **`RelayRLAgent` generic parameters changed** - `RelayRLAgent<B, D_IN, D_OUT, KindIn, KindOut>` is now `RelayRLAgent<B>`; dimension and kind generics move to individual actor methods.
- **`RelayRLActorEnv::run_env()` removed** - The broad-kernel `run_env()` is gone; use `run_env_eval()` for inference-only loops or the new `run_env_with_ppo/ippo/mappo` for integrated training.
- **`AlgorithmCfg` removed** - The old multi-algorithm enum is replaced by `AlgorithmInitArgs { PPO, IPPO, MAPPO }`.
- **`ClientConfigParams::algorithm_name` renamed** - The config field is now `algorithm` in both the Rust struct and the client config JSON; `get_algorithm_name()` is removed.
- **`ClientConfigParams::max_traj_length` removed** - The field no longer exists at the config level; max trajectory length is provided per actor at `new_actor` call time.
- **`ReplayBufferSize` and `SaveModelPath` type aliases removed** - Previously in `agent.rs`; callers should use `usize` and `PathBuf` directly.
- **`step_bytes` return type** - Environment `step_bytes` now returns a 5-tuple `(obs, Option<mask>, rewards, dones, truncateds)` instead of a 3-tuple; custom environment implementations must be updated.

## [0.5.0-beta.4] - 2026-05-06

### Added
- **Config-driven algorithm selection** - Client configuration now carries `algorithm_name` plus structured initial hyperparameters for PPO, REINFORCE, DDPG, TD3, and their independent / multi-agent variants.
  - Added `Algorithm` / `AlgorithmCfg` coverage for `DDPG`, `IDDPG`, `MADDPG`, `TD3`, `ITD3`, and `MATD3`
  - Default config JSON generation moved into `utilities::config_json` and now includes per-algorithm defaults for the expanded algorithm set
- **Optional training during local environment runs** - `run_env()` can now accept an algorithm configuration, save path, replay-buffer size, device, and kernel so local environment stepping can feed trajectories into `relayrl_algorithms` trainers.
  - Trained model modules are acquired from the selected trainer and refreshed back into the actor runtime for subsequent local inference
- **Actor runtime model refresh helper** - Added `ActorRuntime::perform_refresh_model()` to initialize or reload the hot-reloadable local model from a `ModelModule` and `DeviceType`.

### Changed
- **Local action hot paths** - `request_action()` and `flag_last_action()` now use cached hot-path state for local runtimes, reducing router/task boundaries for local inference and terminal-action handling.
  - `flag_last_action()` can return completed trajectories internally for training while still preserving the public send-to-buffer behavior
- **Vector environment execution** - `ScalarVecEnv` now uses Rayon-backed parallel reset and `step_bytes()` paths for larger environment batches.
- **Router message model** - Routed messages now carry only `actor_id` and a nested `RoutingProtocol`, with control and data payloads split into `ControlPayload` and `DataPayload`.
  - Router timeouts and actor dispatch now match against the nested protocol variants instead of a separate routed payload field
- **Package metadata and dependencies** - `relayrl_framework` crate version is now `0.5.0-beta.4`, `rayon` is a direct dependency, and the direct `half` dependency no longer enables a misspelled `feature = ["bytemuck"]` entry.

### Fixed
- **Config hyperparameter field mapping** - Generated and loaded algorithm hyperparameters now align with the current `relayrl_algorithms` parameter structs, including actor/critic learning-rate names and TD3/DDPG training fields.
- **Local terminal-action handling** - Actor `perform_flag_last_action()` now discards the optional returned trajectory for message-driven calls and still reports success after processing the terminal action.

### Breaking
- **`run_env()` algorithm-training signature** - `RelayRLActorEnv::run_env()` and coordinator/state-manager environment execution now take an optional algorithm-training tuple when callers want integrated training.
  - Callers using the trait directly may need to update method signatures and generic kernel bounds to include the expanded algorithm kernel traits and `WeightProvider`
- **Internal router protocol API** - Crate-internal users of `RoutedMessage`, `RoutingProtocol`, or `RoutedPayload` must migrate to the new `ControlPayload` / `DataPayload` protocol shape.

## [0.5.0-beta.3] - 2026-04-26

### Added
- **Actor runtime direct-operation handle** - Added `ActorRuntime` ownership for local model state, trajectory state, max-trajectory length, buffer sending, and metrics so coordinator/state-manager paths can execute selected actor operations directly without routing every operation through the actor inbox
- **Flattened byte inference path for environments** - `ActorRuntime::perform_local_byte_inference()` converts flat environment observation bytes into `TensorData`, calls `ModelModule::flat_batch_inference()`, falls back to `flat_batch_zeros()` on inference failure, and decodes model outputs back to action bytes for discrete and continuous action spaces
- **Environment byte-buffer execution support** - Environment interfaces and scalar/vector wrappers now expose `n_envs_dims()`, `flat_observation_bytes()`, `step_bytes()`, `flat_env_ids()`, and `action_is_discrete()` so `run_env()` can drive scalar and vector environments through the `relayrl_env_trait` 1.2 flattened API
- **Action decoding utilities** - Added dtype mapping plus `decode_argmax()` and `decode_continuous_bytes()` helpers for NdArray and Tch environment action dtypes, including half/bfloat16 handling where supported
- **Action-routing regression coverage** - Added coordinator/state-manager tests for direct `flag_last_action` behavior and request-action routing through the global dispatcher

### Changed
- **Package metadata and dependencies** - `relayrl_framework` crate version is now `0.5.0-beta.3`, and `half` is a direct dependency for half/bfloat16 byte decoding
- **Environment integration API alignment** - `RelayRLActorEnv::set_env()`, `ClientEnvironments::set_env()`, and state-manager environment storage now accept `Box<dyn relayrl_env_trait::Environment>` instead of the old backend/dimension/tensor-kind-generic environment trait object
- **Environment run loop** - `run_env()` now supports the local inference mode by pulling an `ActorRuntime` and shared environment map, then executing `step_count` iterations of observation bytes -> local model inference -> environment `step_bytes()` -> per-environment `flag_last_action` when done
- **Scalar and vector environment wrappers** - `ScalarVecEnv` now maintains stable environment ordering and flat observation/action byte strides; `BatchVecEnv` delegates flattened observation, step, dimension, id, and action-space queries to the wrapped vector environment
- **Router protocol shape** - Environment batching now happens through the local byte path, so router-level `RequestInferenceBatch` routing was removed from timeout handling and dispatch
- **Runtime locking and handle storage** - State-manager environment operations use shared/read access with interior concurrent maps, actor runtimes are stored alongside actor model handles, and runtime handles are updated/cleared during actor id changes, removals, and shutdown
- **Buffer worker throughput constants** - Trajectory buffer worker batching changed from `10` items every `100ms` to `10_000` items every `1ms` as a temporary throughput patch
- **Internal sink module layout** - The memory sink module moved under `runtime/data/sinks/memory_sink`, aligning it with the transport sink layout

### Fixed
- **Online memory mode detection** - `uses_in_memory_data()` now recognizes transport-enabled `OnlineWithMemory(TrainingParams)` and `OnlineWithFilesAndMemory(TrainingParams, Option<LocalTrajectoryFileParams>)`
- **Training address extraction** - Coordinator startup now includes `OnlineWithFilesAndMemory` when deriving training server address arguments, and treats offline memory/file-and-memory modes as local-only
- **Transport sink import paths** - Internal transport imports now point at `runtime/data/sinks/transport_sink`, matching the current sink module location
- **Optional codec startup wiring** - Agent `start()` and `restart()` now pass transport codecs as `Some(codec)` into coordinator startup when transport features are enabled

### Removed
- **Router-level batched inference messages** - Removed `RoutingProtocol::RequestInferenceBatch`, `RoutedPayload::RequestInferenceBatch`, and `BatchedInferenceRequest`
- **Old benchmark files** - Deleted the stale `benches/old` network/runtime benchmark files that no longer matched the current runtime architecture
- **Legacy environment tensor conversion path** - Removed the framework-side `IntoAnyTensorKind` requirement and old `TensorData`-to-`AnyBurnTensor` environment action conversion path in favor of byte-buffer environment stepping

### Breaking
- **Custom environment integration** - Code passing generic `Environment<B, D_IN, D_OUT, KindIn, KindOut>` objects into `RelayRLActorEnv::set_env()` must migrate to the `relayrl_env_trait` 1.2 object-safe `Environment` API and provide flattened byte observation/action methods
- **Internal router integrations** - Any crate-internal or downstream code depending on `RequestInferenceBatch`/`BatchedInferenceRequest` must use the live `request_action` path or the new local environment byte path instead
- **Internal module paths** - References to the old `runtime/data/memory_sink` location must move to `runtime/data/sinks/memory_sink`

## [0.5.0-beta.2] - 2026-04-23

### Added
- **Actor environment control API** - `RelayRLActorEnv` adds `run_env`, `set_env`, `remove_env`, `get_env_count`, and `set_env_count` so local actors can manage scalar and vector environment lifecycles through the coordinator
- **Batched local inference routing** - `RequestInferenceBatch` and `BatchedInferenceRequest` add batched observation dispatch for vectorized local environments
  - `FlagLastInference` now carries optional `env_id` and `env_label` so finalized trajectories can preserve environment identity across batched runs

### Changed
- **Environment trait alignment** - `relayrl_framework` now inherits `relayrl_env_trait` from the workspace and targets the 1.1 environment API surface used by the new actor environment plumbing
- **Runtime routing and vectorized environment handling** - Coordinator, state-manager, and router paths were reworked around shared router state, batched environment execution, and explicit routing timeouts for `RequestInferenceBatch` and `FlagLastInference`
- **Hot-path runtime optimizations** - Cache padding and ordering refinements were applied across shared actor counts, router flags, backpressure permits, circuit-breaker counters, and shutdown state to reduce contention and improve responsiveness under load

### Fixed
- **Action request coordination** - `request_action()` now acquires shared dispatcher and valid-id state in one path and tightens the routing window to reduce actor reply races
- **Runtime ordering and recovery behavior** - Actor distribution/removal ordering, backpressure wakeups, and circuit-breaker state transitions were tightened to behave more predictably under load

### Breaking
- **Environment integration surface** - Consumers integrating custom environments must adapt to the newer `relayrl_env_trait` 1.1 generics and method requirements now used by framework environment APIs
  - Projects that pinned `relayrl_env_trait` `1.0.x` alongside `0.5.0-beta.1` need to align with the workspace-managed env-trait dependency before moving to `0.5.0-beta.2`

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
