//! A Rust-native, deep reinforcement learning **runtime** for concurrent actor execution and data collection, all in a single process.
//!
//! RelayRL's agent provides a full actor lifecycle, live router scaling, per-actor model hot-swap, trajectory data collection, and both **step-driven** and **environment-driven** control patterns.
//!
//! Unlike other RL frameworks that scale replicas of one policy across multiple processes, RelayRL focuses on **heterogeneous** policy execution.
//! Each actor can bind its own environment and its own independent (or device-shared), **hot-swappable** model. This makes it well suited for embedding RL
//! inside a native application, running distinct policies side by side, or swapping a policy into a subset of actors with **near zero downtime**.
//!
//! RelayRL also outperforms (expectedly) other popular, GIL-bound RL frameworks in terms of raw throughput and latency, memory consumption, and scalability. See [relayrl.dev](https://relayrl.dev) for more details.
//!
//! This crate is a thin facade re-exporting the **most recent stable release** of [`relayrl_framework`].
//!
//! RL algorithms, data types, and the environment trait live in [`relayrl_algorithms`](https://docs.rs/relayrl_algorithms/latest/relayrl_algorithms/),
//! [`relayrl_types`](https://docs.rs/relayrl_types/latest/relayrl_types/), and [`relayrl_env_trait`](https://docs.rs/relayrl_env_trait/latest/relayrl_env_trait/) respectively.
//!
//! # Prerequisites
//!
//! - A [Tokio](https://docs.rs/tokio/1.52.3/tokio/) runtime. On a current-threaded runtime, actors will execute concurrently, but not in parallel. To enable parallel execution, use a multi-threaded runtime.
//! - A compatible [Burn](https://docs.rs/burn/0.21.0/burn/) backend. Currently, only [Burn-NdArray](https://docs.rs/burn-ndarray/0.21.0/burn_ndarray/) and [Burn-Tch](https://docs.rs/burn-tch/0.21.0/burn_tch/) are supported.
//! - A compatible inference runtime. Currently, only **LibTorch** and the **ONNX Runtime (ORT)** are supported.
//!
//! # Building the Agent
//!
//! An agent is constructed with [`AgentBuilder<B>`](crate::network::AgentBuilder), which separates *runtime-invariant*
//! configuration from the startup parameters returned alongside the agent. The builder
//! uses a chained-setter style: each setter returns the updated builder,
//! and [`AgentBuilder::build`](crate::network::AgentBuilder::build) consumes it and yields ([`RelayRLAgent<B>`](crate::network::RelayRLAgent), [`AgentStartParameters<B>`](crate::network::AgentStartParameters)).
//!
//! On creation of the builder, it is necessary to specify a `Backend` generic type **B** that matches the expected inference runtime for both
//! tensor operations and model inference. By default, RelayRL uses burn_ndarray's **NdArray** for tensor operations and **ORT** for model inference. To use **LibTorch** for both,
//! you can enable the `tch-backend` feature flag and pass burn_tch's `Tch` as the `Backend` generic type.
//!
//! The builder defaults to `ActorInferenceMode::Client(ModelMode::Independent)` inference, `ActorTrainingDataMode::OfflineWithMemory`
//! trajectory recording, a router scale of `1`, no default model, a buffer size of `1024` per actor, and no config path.
//!
//! ```rust
//! use relayrl::network::client::{AgentBuilder, ActorInferenceMode, ActorTrainingDataMode, ModelMode};
//! use relayrl::types::model::ModelModule;
//! use burn_ndarray::NdArray;
//!
//! # async fn build() -> Result<(), Box<dyn std::error::Error>> {
//! let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
//!
//! let (mut agent, params) = AgentBuilder::<NdArray>::builder()
//!     .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Shared))
//!     .actor_training_data_mode(ActorTrainingDataMode::OfflineWithFilesAndMemory(None))
//!     .router_scale(2)
//!     .router_buffer_size_per_actor(2048)
//!     .default_model(default_model)
//!     .config_path(std::path::PathBuf::from("client_config.json"))
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! There are a series of client modes and parameters that can be set on the builder when `zmq-transport` or `nats-transport` feature flags are enabled;
//! see [Experimental Network Transport](#experimental-network-transport).
//!
//! #### Inference Configuration
//!
//! Inference mode is set via [`ActorInferenceMode`](crate::network::ActorInferenceMode), where a
//! [`ModelMode`](crate::network::ModelMode) enum controls model loading semantics for all actors in the runtime.
//!
//! - `Independent` *(default)*: each actor holds its own model handle loaded into memory independently.
//!   This allows different actors to run genuinely different policies simultaneously.
//! - `Shared`: actors on the same device share a single model handle, reducing memory consumption.
//!   For instance, a group of CPU actors and a group of GPU actors each share one handle for their respective device.
//!   When `update_model` is called in Shared mode, the runtime refreshes one representative actor per device so
//!   each shared handle is updated exactly once.
//!
//! #### Training Data Configuration
//!
//! Trajectories produced by actors are recorded according to the
//! [`ActorTrainingDataMode`](crate::network::ActorTrainingDataMode) selected on the builder:
//!
//! - `OfflineWithFiles`: write to local files,
//!   either `Csv` or `Arrow`,
//!   via [`LocalTrajectoryFileParams`](crate::network::LocalTrajectoryFileParams). `LocalTrajectoryFileParams::new` validates the target path
//!   and creates the directory if it does not exist.
//! - `OfflineWithMemory` *(default)*: keep trajectories in
//!   an in-memory buffer, retrievable at any time with `RelayRLAgent::get_trajectory_cache`.
//! - `OfflineWithFilesAndMemory`: write to local files and keep in memory simultaneously.
//! - `Disabled`: trajectory recording is disabled entirely; useful when
//!   data collection is managed externally or the actors are used purely for inference.
//!
//! #### JSON Configuration
//!
//! Every agent is backed by a JSON configuration file that is used to load the runtime's operational settings while it is live. The path defaults to
//! `client_config.json` in the current working directory and can be overridden with
//! [`AgentBuilder::config_path`](crate::network::AgentBuilder). If the file does not exist it is
//! created on first use, pre-populated with defaults; if it exists it is read and parsed
//! into a [`ClientConfigLoader`](crate::config::ClientConfigLoader).
//!
//! A malformed file does not abort
//! startup, the loader logs the error and falls back to built-in defaults.
//!
//! The file has two top-level sections, `client_config` and `transport_config`:
//!
//! ```json
//! {
//!     "client_config": {
//!         "config_update_polling_seconds": 10.0,
//!         "init_hyperparameters": { "PPO": { "gamma": 0.99, "lam": 0.97 } },
//!         "router_buffer_size_per_actor": 1000,
//!         "trajectory_file_output": { "directory": "experiment_data", "file_type": "Csv" },
//!         "metrics_meter_name": "relayrl-client",
//!         "metrics_otlp_endpoint": { "prefix": "http://", "host": "127.0.0.1", "port": "4317" }
//!     },
//!     "transport_config": {
//!         "nats_addresses": { "...": "..." },
//!         "zmq_addresses": { "...": "..." },
//!         "local_model_module": { "directory": "model_module", "model_name": "client_model", "format": "pt" }
//!     }
//! }
//! ```
//!
//! ##### Defaults vs. arguments vs. config changes
//!
//! A setting can be supplied from three places, resolved in a fixed order of precedence at startup
//! (highest first):
//!
//! 1. **Builder and runtime arguments (highest).** Values passed programmatically win wherever they
//!    overlap a file setting. The one knob present in both is `router_buffer_size_per_actor`:
//!    [`AgentBuilder::router_buffer_size_per_actor`](crate::network::AgentBuilder) overrides the file
//!    when set, and the file value is used when it is left unset. Per-call runtime arguments are also
//!    in this tier — `new_actors`/`new_actor` take their `device`, `max_traj_length`, and `model`
//!    directly, and `update_model` swaps a model explicitly; none of these are sourced from the file.
//! 2. **Config file values (middle).** When no argument overrides them, the file supplies the
//!    operational settings: `config_update_polling_seconds`, `router_buffer_size_per_actor`,
//!    `trajectory_file_output` (directory + `Csv`/`Arrow`), the metrics meter/endpoint, the transport
//!    server addresses, and `local_model_module` (the on-disk location the runtime loads a model from).
//! 3. **Built-in defaults (lowest).** If a value is absent from both the arguments and the file — or
//!    the file fails to parse — the loader falls back to hard-coded defaults (for example a router
//!    buffer of `1000` and a `10.0`-second poll interval).
//!
//! Some settings are *builder-only* and never read from the file (for example `router_scale`,
//! `default_model`, and the inference / training-data modes), while others are *file-only* with no
//! `AgentBuilder` equivalent on the local path (for example `config_update_polling_seconds`, the
//! metrics endpoint, and the transport addresses).
//!
//! ##### Config changes at runtime
//!
//! After `start`, a background task polls the file every `config_update_polling_seconds` and, when
//! the file's modification time changes, reloads it and applies a subset of settings to the running
//! agent with no restart: the trajectory-file output, the resolved local model path, and the
//! per-actor router buffer size (plus the metrics meter/endpoint under the `metrics` feature, and the
//! transport addresses and default hyperparameters under a transport feature). Note that a live
//! reload of `router_buffer_size_per_actor` is applied unconditionally, so it overrides whatever the
//! builder set at startup. The active configuration can be re-read on demand with
//! [`RelayRLAgent::get_config`](crate::network::RelayRLAgent).
//!
//! # The Agent Runtime
//!
//! Building an agent does not start it. [`RelayRLAgent::start`](crate::network::RelayRLAgent::start) spins up the coordinator, managers,
//! routers, and supporting runtime tasks designated by the `AgentStartParameters`; you then
//! create one or more actors with [`RelayRLAgentActors::new_actors`](crate::network::RelayRLAgentActors::new_actors) (each on
//! a chosen [`DeviceType`](crate::types::tensor::relayrl::DeviceType) with its own trajectory length and optional model). Once actors
//! exist there are two ways to [`execute`](#Heterogeneous-Actor-Execution) them, and [`RelayRLAgent::shutdown`](crate::network::RelayRLAgent::shutdown) tears everything
//! down gracefully. [`RelayRLAgent::restart`](crate::network::RelayRLAgent::restart) is also available to tear down and reinitialise
//! the runtime without destroying the agent handle — useful when you want to reload the agent using
//! a different set of `AgentStartParameters` without building a new agent.
//!
//! Actors are created with `new_actors::<D_IN, D_OUT>(count, device, max_traj_length, model)`,
//! where the const generics `D_IN` and `D_OUT` declare the observation tensor rank and the
//! action tensor rank respectively. These must be consistent with the environment and model
//! that will be used with those actors. `new_actor` (singular) creates exactly one actor and
//! accepts the same signature. Each actor is assigned to a device, given an independent
//! trajectory buffer of `max_traj_length` steps, and, if `model` is `Some`, pre-loads that
//! model into its handle; otherwise the actor waits for a model to be provided via
//! `update_model` before it can perform inference.
//!
//! ```rust
//! # async fn run(mut nd_agent: RelayRLAgent<burn_ndarray::NdArray>, tch_agent: RelayRLAgent<burn_tch::Tch>, params: AgentStartParameters<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
//! use types::model::ModelModule;
//! use burn_ndarray::NdArray;
//!
//! nd_agent.start(params).await?;
//! tch_agent.start(params).await?;
//!
//! const ENV1_OBS_IN: usize = 2;
//! const ENV1_ACT_OUT: usize = 2;
//! let env1_actors = 4;
//! let env1_max_traj_length = 1_000;
//!
//! const ENV2_OBS_IN: usize = 6;
//! const ENV2_ACT_OUT: usize = 1;
//! let env2_actors = 3;
//! let env2_max_traj_length = 2_000;
//!
//! let no_default_model: Option<ModelModule<NdArray>> = None;
//!
//! // Create two groups of actors with different obs/action ranks on different devices.
//! let env1_actor_ids = nd_agent.new_actors::<ENV1_OBS_IN, ENV1_ACT_OUT>(
//!     env1_actors, DeviceType::Cpu, env1_max_traj_length, no_default_model.clone()
//! ).await?;
//! let env2_actor_ids = tch_agent.new_actors::<ENV2_OBS_IN, ENV2_ACT_OUT>(
//!     env2_actors, DeviceType::Gpu(0), env2_max_traj_length, no_default_model.clone()
//! ).await?;
//!
//! let all_nd_actor_ids = nd_agent.get_actor_ids()?;
//! let all_tch_actor_ids = tch_agent.get_actor_ids()?;
//!
//! // ... interact ...
//!
//! nd_agent.remove_actors(all_nd_actor_ids).await?;
//! tch_agent.remove_actors(all_tch_actor_ids).await?;
//! nd_agent.shutdown().await?;
//! tch_agent.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Introspection methods after `start`:
//!
//! - `get_model_version(actor_ids)` — returns `(ActorUuid, version: i64)` pairs reflecting how
//!   many times each actor's model has been hot-swapped since startup.
//! - `get_trajectory_cache()` — returns a shared `DashMap<ActorUuid, Vec<Arc<RelayRLTrajectory>>>`
//!   of all in-memory trajectories collected so far (only populated under `OfflineWithMemory` or
//!   `OfflineWithFilesAndMemory` modes for `ActorTrainingDataMode`).
//! - `get_config()` — fetches the active `ClientConfigLoader` being watched by the lifecycle manager.
//! - `set_config_path(path)` — hot-swap the configuration file path without restarting the runtime.
//!
//! #### Actor Management
//!
//! Every actor created by `new_actor` / `new_actors` runs as its own Tokio task and is tracked
//! in three places: the **namespaced UUID registry** (the source of truth for which IDs are
//! live), the coordinator's state manager (which owns each actor's task handle, device, bound
//! environment, model handle, and trajectory buffer), and the router layer (which maps each
//! actor to a routing worker and an inbox channel). The methods below manipulate that
//! bookkeeping while the runtime is live, without tearing the agent down.
//!
//! - `get_actor_ids()` — returns the UUIDs of all live actors by reading the client's slice of
//!   the namespaced UUID registry. Because it is the registry view, callers should not rely on
//!   any particular ordering. Returns `NoRuntimeInstanceError` if the agent has not been started.
//! - `remove_actor(id)` / `remove_actors(ids)` — de-register one or more actors. Under the hood
//!   each removal aborts the actor's task, drops its environment, device, model handle, runtime
//!   handle, and router route, decrements the live-actor count, and frees the UUID back to the
//!   registry. `remove_actors` is a convenience wrapper: an empty list is rejected with
//!   `NoopActorCount`, a single ID delegates to `remove_actor`, and larger lists are removed
//!   one by one.
//! - `set_actor_id(current_id, new_id)` — rename a live actor in place. The runtime moves the
//!   actor's task handle, inbox, router assignment, device, environment, and runtime handle from
//!   `current_id` to `new_id` and updates the registry; the underlying task keeps running and its
//!   inbox is preserved, so in-flight routing is not interrupted. It fails if `current_id` is not
//!   found or if `new_id` is already taken. Useful for aligning an actor's identity with an
//!   external system (for example a session or player ID).
//!
//! ```rust
//! # async fn manage(agent: &mut RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
//! use uuid::Uuid;
//!
//! // Enumerate live actors (registry order; do not assume sorting).
//! let ids = agent.get_actor_ids()?;
//!
//! // Give the first actor a stable, externally-meaningful identity. The task keeps
//! // running under the new ID with its inbox intact.
//! let session_id = Uuid::new_v4();
//! agent.set_actor_id(ids[0], session_id).await?;
//!
//! // Retire the remaining actors; their tasks are aborted and their UUIDs freed.
//! agent.remove_actors(ids[1..].to_vec()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Router Scaling
//!
//! Between the coordinator and the actors sits a pool of data routing workers that dispatch
//! messages to actors and drain trajectory sink buffers (see [Training Data Configuration](#training-data-configuration)). The initial pool size is set with
//! `AgentBuilder::router_scale`, and the per-actor channel capacity is set with
//! `AgentBuilder::router_buffer_size_per_actor` (defaults to `1024`). Both can be tuned
//! to match the expected message volume: a larger buffer absorbs bursty workloads without
//! backpressure, while a smaller buffer surfaces overload conditions more quickly.
//!
//! The router pool can be adjusted while the runtime is live with
//! `RelayRLAgent::scale_throughput` — a positive value scales out, a negative value
//! scales in. This lets you grow routing capacity under load without restarting the agent.
//!
//! ```rust
//! # async fn scale(agent: &mut RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
//! agent.scale_throughput(2).await?;   // add two more routing workers
//! agent.scale_throughput(-1).await?;  // remove one
//! # Ok(())
//! # }
//! ```
//!
//! Note that this will consume the runtime for the duration of the operation, thus it's **not** recommended to perform a scale operation
//! while actors are actively running unless your application can tolerate a pause in execution.
//!
//! Performing a `scale_throughput` operation will distribute all actors across all available routers upon completion. Assuming there is an equal number of actors and routers,
//! the runtime's `ScaleManager` will designate a 1:1 ratio of routers-to-actors. Any excess routers will be left idle. If there are more actors than routers, actors will be
//! distributed as evenly as possible across the available routers with more actors per router.
//!
//! **Static** workloads should initialize with a router scale equal to the expected number of actors. Under **dynamic** workloads where actors are **consistently added and removed**, initializing
//! with a higher router scale than the expected number of actors is recommended.
//! This minimizes contention for router resources and avoids unnecessary scaling operations needed to maintain a stable router throughput across actors.
//!
//! #### Config Lifecycle
//!
//! After `start`, a background `LifecycleManager` task watches the config file and automatically reloads it whenever its
//! modification time changes. On the local/default path a reload refreshes the
//! trajectory-file output, the resolved local model path, and the per-actor router buffer size (plus
//! the metrics meter/endpoint under the `metrics` feature); transport addresses and default
//! hyperparameters are additionally refreshed when a transport feature is enabled.
//!
//! If `config_update_polling_seconds` itself changes, the polling interval is rebuilt to match.
//!
//! The watched path can be swapped at runtime with [`RelayRLAgent::set_config_path`](crate::network::RelayRLAgent),
//! and the active loader is retrievable with [`RelayRLAgent::get_config`](crate::network::RelayRLAgent).
//!
//! # Heterogeneous Actor Execution
//!
//! Actors are independent units of execution. Each runs in its own task, and in
//! `ModelMode::Independent` each owns its own model handle, so different
//! actors can run *different* policies on *different* environments simultaneously. On a
//! multi-threaded Tokio runtime their inference runs in parallel (no GIL); on a
//! current-thread runtime they run concurrently.
//!
//! Models are hot-swappable at runtime. [`RelayRLAgent::update_model(model, actor_ids)`](crate::network::RelayRLAgent::update_model) can target
//! a subset of actors by passing `Some(vec![id_a, id_b])`, letting you roll a freshly trained
//! policy into specific actors while the rest keep serving the previous one, with no restart and
//! no downtime. Passing `None` updates all live actors. In `ModelMode::Shared`, the runtime
//! refreshes one representative actor per device so each shared handle is updated exactly once.
//!
//! Note that `update_model` is rejected (returns `ModelUpdateNotSupported`) when the agent is
//! configured with any `Online` training data mode, since model updates are managed server-side
//! in that case.
//!
//! After a hot-swap, `get_model_version(actor_ids)` returns the current swap count for each
//! actor as `(ActorUuid, i64)` pairs, which can be used to confirm that the update propagated.
//!
//! ```rust
//! use relayrl::types::model::ModelModule;
//! use relayrl::network::RelayRLAgent;
//! use burn_ndarray::NdArray;
//!
//! # async fn swap(agent: &RelayRLAgent<NdArray>, new_model: ModelModule<NdArray>) -> Result<(), Box<dyn std::error::Error>> {
//! let ids = agent.get_actor_ids()?;
//! // Swap a new policy into actors 0 and 2 only; actor 1 keeps the old policy.
//! agent.update_model(new_model, Some(vec![ids[0], ids[2]])).await?;
//! // Verify the swap landed.
//! let versions = agent.get_model_version(vec![ids[0], ids[2]]).await?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Step-driven Integration
//!
//! In the step-driven pattern, *your* code owns the loop. You hold the observations and
//! ask specific actors for actions one step at a time via [`RelayRLAgent::request_action`](crate::network::RelayRLAgent::request_action),
//! marking episode boundaries with [`RelayRLAgent::flag_last_action`](crate::network::RelayRLAgent::flag_last_action). This is the right
//! fit for embedding RelayRL inside an existing simulator, game engine, or control loop.
//!
//! `request_action` is generic over `<D_IN, D_OUT, KindIn, KindOut>`: the two const generics
//! must match the observation and action tensor ranks declared when the actors were created,
//! and `KindIn`/`KindOut` are the tensor element kinds (e.g. `Float`). It accepts an observation
//! tensor, an optional action mask tensor, and a `reward: f32` for the previous step, and returns
//! `Vec<(ActorUuid, Arc<RelayRLAction>)>` — one entry per actor in the `ids` list. You can
//! target any subset of live actors by constructing the `ids` vector accordingly.
//!
//! `flag_last_action(ids, reward: Option<f32>)` appends a terminal action (`done = true`) to
//! each named actor's current trajectory, signalling the end of an episode. After calling it,
//! the actor begins a fresh trajectory on the next `request_action`.
//!
//! ```rust
//! # async fn control_loop_step(agent: &RelayRLAgent<burn_ndarray::NdArray>) -> Result<(), Box<dyn std::error::Error>> {
//! use burn_ndarray::NdArray;
//! use burn_tensor::{Tensor, Float};
//!
//! let ids = agent.get_actor_ids()?;
//! let obs = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//! let mask = None;
//! let reward = 0.0;
//!
//! let _actions = agent.request_action(ids.clone(), obs, mask, reward).await?;
//!
//! // Mark end of episode for all actors with the terminal reward.
//! agent.flag_last_action(ids, Some(reward + 1.0)).await?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Environment-driven Integration
//!
//! In the environment-driven pattern, the [`RelayRLAgent`](crate::network::RelayRLAgent) owns the loop. You implement either the
//! `ScalarEnvironment` or `VectorEnvironment` trait (both extend the base `Environment` trait
//! from [`relayrl_env_trait`](https://docs.rs/relayrl_env_trait/1.3.1/relayrl_env_trait/)) for your environment, bind it to an actor with
//! `RelayRLActorEnv::set_env(actor_id, Box<dyn Environment>, count)`, and let the runtime
//! drive the rollout. The `count` argument controls how many logical environment copies are
//! associated with that actor: when `count < 8` the runtime steps them sequentially; when
//! `count >= 8` rayon data parallelism is engaged across the copies. The count can be
//! adjusted after binding with `set_env_count`, queried with `get_env_count`, and the
//! environment can be removed entirely with `remove_env`.
//!
//! Only one `run_env_*` loop may be active per actor at a time; attempting to start a second
//! returns `ClientError::RunEnvActive` immediately.
//!
//! - `run_env_eval(actor_id, loop_iters)` — runs `loop_iters` evaluation steps on the bound
//!   environment; no training update is applied.
//! - `run_env_with_ppo(actor_id, loop_iters, max_traj_length, trainer_spec)` — runs a
//!   single-agent PPO training rollout. Requires a `PPOTrainerSpec<B, KindIn, KindOut, Pi>`
//!   where **Pi** is a [`NeuralNetwork<B, KindIn, KindOut>`](crate::algorithms::NeuralNetwork). See [`relayrl_algorithms`](https://docs.rs/relayrl_algorithms/0.4.1/relayrl_algorithms/) for details on
//!   constructing the trainer spec.
//! - `run_env_with_ippo` and `run_env_with_mappo` — independent and multi-agent PPO training
//!   rollouts respectively; coming soon.
//!
//! ```rust
//! # async fn drive(mut agent: RelayRLAgent<burn_ndarray::NdArray>, env: Box<dyn relayrl_env_trait::traits::Environment>) -> Result<(), Box<dyn std::error::Error>> {
//! let (actor_id1, actor_id2) = {
//!     let ids = agent.get_actor_ids()?;
//!     (ids[0], ids[1])
//! };
//!
//! // sequential env stepping is enabled when env count < 8
//! agent.set_env(actor_id1, env, 7).await?;       // 7 vectorized env copies on this actor
//! agent.run_env_eval(actor_id1, 10_000).await?;  // run 10k loop iterations
//!
//! // rayon data parallelism is enabled when env count >= 8
//! agent.set_env(actor_id2, env, 1024).await?;    // 1024 vectorized env copies on this actor
//! agent.run_env_eval(actor_id2, 1_000).await?;   // run 1k loop iterations
//!
//! let count = agent.get_env_count(actor_id2).await?;
//! agent.set_env_count(actor_id2, count / 2).await?;  // halve the env count live
//! agent.remove_env(actor_id1).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Experimental Network Transport
//!
//! <div class="warning">
//! Transport- and server-backed workflows are <strong>experimental</strong> in this 0.5.x,
//! even when their feature flags are enabled. The current supported path is the local runtime.
//! </div>
//!
//! With the `zmq-transport` and/or `nats-transport` features enabled, an agent can be configured
//! for server-backed workflows. The following **transport-gated** surface becomes available:
//!
//! #### Builder setters:
//! - `transport_type(TransportType)` — selects `TransportType::ZMQ` or `TransportType::NATS`; defaults to ZMQ when zmq-transport is enabled.
//! - `default_ppo_params(PPOParams)` / `default_ippo_params(IPPOParams)` / `default_mappo_params(MAPPOParams)` — supply
//!   hyperparameters forwarded to the training server at handshake time.
//!
//! #### Inference modes:
//! - `ActorInferenceMode::Server(InferenceParams)` — all actor inference is routed to a remote inference server.
//!   `InferenceParams` holds the `ModelMode`, an optional `CodecConfig` (compression, encryption, integrity), and
//!   the server addresses via `InferenceAddressesArgs` (`ZMQ` or `NATS` variants, wrapping `ZmqInferenceAddressesArgs`
//!   or a NATS subject string).
//! - `ActorInferenceMode::ClientFallback(ModelMode, InferenceParams)` — all actors performs inference locally
//!   as a fallback while the rest route to the server.
//!
//! #### Training data modes:
//! - `ActorTrainingDataMode::Online(TrainingParams)` — trajectories are streamed to a training server.
//! - `ActorTrainingDataMode::OnlineWithFiles(TrainingParams, ...)` — stream to server and write to local files.
//! - `ActorTrainingDataMode::OnlineWithMemory(TrainingParams)` — stream to server and keep in memory.
//! - `ActorTrainingDataMode::OnlineWithFilesAndMemory(TrainingParams, ...)` — all three simultaneously.
//!
//! `TrainingParams` mirrors `InferenceParams` with the addition of optional hyperparameter args and
//! training-specific addresses via `TrainingAddressesArgs` (`ZMQ` wrapping `ZmqTrainingAddressesArgs`,
//! or `NATS`). `ZmqTrainingAddressesArgs` exposes the agent listener, model server, trajectory server,
//! and scaling server endpoints individually.
//!
//! These paths are under active development and are not covered by the 0.5.x support promise.
//!

pub mod algorithms {
    pub use relayrl_framework::prelude::algorithms::*;
}

pub mod config {
    pub use relayrl_framework::prelude::config::*;
}

pub mod network {
    pub use relayrl_framework::prelude::network::*;
}

pub mod types {
    pub use relayrl_framework::prelude::types::*;
}
