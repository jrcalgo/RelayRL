//! Build concurrent reinforcement learning systems from specialized actors.
//!
//! A Rust-native, concurrent actor-system runtime for heterogeneous RL workloads
//!
//! RelayRL's agent provides a full actor-system lifecycle, live router scaling, per-actor model hot-swap, trajectory data collection, and both **step-driven** and **environment-driven** inference control patterns.
//!
//! Unlike other RL frameworks that scale replicas of one policy across multiple processes, RelayRL focuses on **heterogeneous** policy execution via many actors.
//! Each actor can bind its own environment and its own independent (or device-shared), **hot-swappable** model. This makes it well suited for embedding RL
//! inside a native application, running distinct policies side by side, or swapping a policy into a subset of actors with **near zero downtime**.
//!
//! RelayRL also outperforms (expectedly) other popular, GIL-bound RL frameworks in terms of raw throughput and latency, memory consumption, and horizontal scalability.
//! For benchmarks and other system details, visit [relayrl.dev](https://relayrl.dev).
//!
//! This crate is a thin facade re-exporting the **most recent stable release** of [`relayrl_framework`].
//! RL algorithms, data types, and the environment trait live in [`relayrl_algorithms`](https://docs.rs/relayrl_algorithms/0.4.1/relayrl_algorithms/),
//! [`relayrl_types`](https://docs.rs/relayrl_types/0.8.1/relayrl_types/), and [`relayrl_env_trait`](https://docs.rs/relayrl_env_trait/1.3.1/relayrl_env_trait/) respectively.
//!
//! # Prerequisites
//!
//! - A [Tokio](https://docs.rs/tokio/1.52.3/tokio/) runtime. On a current-threaded runtime, actors will execute concurrently, but not in parallel. To enable parallel execution, use a multi-threaded runtime.
//! - A compatible inference runtime. Currently, only **ONNX Runtime (ORT) 1.26.0** and the **LibTorch 2.9.0** are supported.
//!
//! # Quick Start
//!
//! ```ignore, rust
//! use relayrl::agent::*;
//! use relayrl::types::model::ModelModule;
//! use relayrl::types::tensor::relayrl::DeviceType;
//! use relayrl::types::tensor::burn::{Tensor, Float, ndarray::NdArray};
//!
//! [tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Build the agent handle and its startup parameters.
//!     let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
//!     let (mut agent, params) = AgentBuilder::<NdArray>::builder()
//!         .params()
//!         .data_routers(2)
//!         .default_model(default_model)
//!         .build()
//!         .await?;
//!
//!     // Start the agent runtime: coordinator, managers, and data routing workers.
//!     agent.start(params).await?;
//!
//!     // Create four actors with rank-2 observations and rank-1 actions.
//!     let actor_info = agent
//!         .new_actors::<2, 1>(4, DeviceType::Cpu, 1_000, None, None)
//!         .await?;
//!     let actor_ids: Vec<_> = actor_info.iter().map(|(id, _)| *id).collect();
//!
//!     // Request actions for all actors. The const rank generics must match creation.
//!     let observation = Tensor::<NdArray, 2, Float>::zeros(
//!         [1, 4],
//!         &Default::default(),
//!     );
//!     let actions: Vec<_> = agent
//!         .request_actions::<2, 1, Float, Float>(actor_ids.clone(), observation, None, 0.0)
//!         .await?;
//!
//!     // Use actions in your simulator, then mark the episode boundary across actors.
//!     some_env_steps(actions);
//!     agent.flag_last_actions(actor_ids, Some(1.0)).await?;
//!
//!     agent.shutdown().await?;
//!     Ok(())
//! }
//! ```
//!
//! # Building the Agent
//!
//! An agent is constructed with [`AgentBuilder<B>`](crate::agent::AgentBuilder), which separates *runtime-invariant*
//! configuration from the startup parameters returned alongside the agent. The builder
//! uses a chained-setter style: each invariant and parameter setter returns the updated builder,
//! and [`AgentBuilder::build`](crate::agent::AgentBuilder::build) consumes it and yields ([`RelayRLAgent<B>`](crate::agent::RelayRLAgent), [`AgentStartParameters<B>`](crate::agent::AgentStartParameters)).
//!
//! On creation of the builder, it is necessary to specify a `Backend` generic type **B** that matches the expected inference runtime for both
//! tensor operations and model inference. By default, RelayRL *expects* burn_ndarray's **NdArray** for tensor operations and **ORT** for model inference. To use **LibTorch** for both,
//! you can enable the `tch-backend` feature flag and pass burn_tch's `Tch` as the `Backend` generic type.
//!
//! The builder defaults to `ActorInferenceMode::Client(ModelMode::Independent)` inference, `ActorDataMode::OfflineWithCache(1000)`
//! trajectory recording, a router scale of `1`, no default model, a buffer size of `1024` per actor, and no config path.
//!
//! ```rust
//! use relayrl::agent::AgentBuilder;
//! use relayrl::agent::{ActorInferenceMode, ActorDataMode, ModelMode};
//! use relayrl::types::model::ModelModule;
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//!
//! async fn build() -> Result<(), Box<dyn std::error::Error>> {
//!     let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
//!
//!     let (mut agent, params) = AgentBuilder::<NdArray>::builder()
//!         .modes() // init invariants
//!         .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Shared))
//!         .actor_data_mode(ActorDataMode::OfflineWithFilesAndCache(None, 1000))
//!         .params() // runtime parameters
//!         .data_routers(2)
//!         .data_buffer_size(100)
//!         .default_model(default_model)
//!         .config_path(std::path::PathBuf::from("client_config.json"))
//!         .build()
//!         .await?;
//!
//!     // ...
//!
//!     Ok(())
//! }
//! ```
//!
//! Setting the default model is optional, but it is **recommended** to do so to avoid the overhead of loading the model into memory for each actor.
//!
//! There are a series of client modes and parameters that can be set on the builder when `zmq-transport` or `nats-transport` feature flags are enabled;
//! see [Experimental Network Transport](#experimental-network-transport).
//!
//! #### Inference Configuration
//!
//! Inference mode is set via [`ActorInferenceMode`](crate::agent::ActorInferenceMode), where a
//! [`ModelMode`](crate::agent::ModelMode) enum controls model loading semantics for all actors in the runtime.
//!
//! - `Independent` *(default)*: each actor holds its own model handle loaded into memory independently.
//!   This allows different actors to run genuinely different policies simultaneously.
//! - `Shared`: actors on the same device share a single model handle, reducing memory consumption.
//!   For instance, a group of CPU actors and a group of GPU actors each share one handle for their respective device.
//!   When `update_models` is called in Shared mode, the runtime refreshes one representative actor per device so
//!   each shared handle is updated exactly once.
//!
//! #### Step-Driven Training Data Configuration
//!
//! Trajectories produced by actors are recorded according to the
//! [`ActorDataMode`](crate::agent::ActorDataMode) selected on the builder:
//!
//! - `OfflineWithCache(size)` *(default)*: keep trajectories in
//!   an in-memory buffer, retrievable at any time with `RelayRLAgent::drain_trajectory_caches(actor_ids)`.
//! - `OfflineWithFiles`: write to local files,
//!   either `Csv` or `Arrow`,
//!   via [`LocalTrajectoryFileParams`](crate::agent::LocalTrajectoryFileParams). `LocalTrajectoryFileParams::new` validates the target path
//!   and creates the directory if it does not exist.
//! - `OfflineWithFilesAndCache`: write to local files and keep in memory simultaneously.
//! - `Disabled`: trajectory recording is disabled entirely; useful when
//!   data collection is managed externally or the actors are used purely for inference.
//!
//! #### Agent-wide JSON Configuration
//!
//! Every agent is backed by a JSON configuration file that is used to load the runtime's operational settings while it is live. The path defaults to
//! `client_config.json` in the current working directory and can be overridden with
//! [`AgentBuilder::config_path`](crate::agent::AgentBuilder). If the file does not exist it is
//! created on first use, pre-populated with defaults; if it exists it is read and parsed
//! into a [`ClientConfigLoader`](crate::utils::config::ClientConfigLoader).
//!
//! A malformed file does not abort
//! startup, the loader logs the error and falls back to built-in defaults.
//!
//! The file has two top-level sections, `client_config` and `transport_config`:
//!
//! ```json
//! {
//!     "client_config": {
//!         "config_polling_seconds": 10,
//!         "init_hyperparameters": { "PPO": { "gamma": 0.99, "lam": 0.97, ... }, ... },
//!         "trajectory_file_output": { "directory": "experiment_data", "file_type": "Csv" },
//!         "local_model_module": { "directory": "model_module", "model_name": "client_model", "format": "onnx" },
//!         "metrics": { "meter_name": "relayrl-client", "otlp_endpoint": { "prefix": "http://", "host": "127.0.0.1", "port": "4317" } }
//!     },
//!     "transport_config": {
//!         "nats_addresses": { "...": "..." },
//!         "zmq_addresses": { "...": "..." }
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
//!    overlap a file setting (for example, a one knob present in both is `data_buffer_size`:
//!    [`AgentBuilder::data_buffer_size`](crate::agent::AgentBuilder) overrides the file
//!    when set, and the file value is used when it is left unset). Per-call runtime arguments are also
//!    in this tier - `new_actors`/`new_actor` take their `device`, `max_traj_length`, and `model`
//!    directly, and `update_models` swaps a model explicitly; none of these are sourced from the file.
//! 2. **Config file values (middle).** When no argument overrides them, the file supplies the
//!    operational settings: `config_polling_seconds`,
//!    `trajectory_file_output` (directory + `Csv`/`Arrow`), the metrics meter/endpoint, the transport
//!    server addresses, and `local_model_module` (the on-disk location the runtime loads a model from).
//! 3. **Built-in defaults (lowest).** If a value is absent from both the arguments and the file, or
//!    the file fails to parse, the loader falls back to hard-coded defaults (for example a data buffer
//!    of `1024` and a `10`-second poll interval).
//!
//! Some settings are *builder-only* and never read from the file (for example `data_routers`,
//! `default_model`, and the inference / training-data modes), while others are *file-only* with no
//! `AgentBuilder` equivalent on the local, offline path (for example `config_polling_seconds`, the
//! metrics endpoint, and the transport addresses).
//!
//! ##### Config changes at runtime
//!
//! After [`RelayRLAgent::start`](crate::agent::RelayRLAgent::start), a background task polls the file every `config_polling_seconds` and, when
//! the file's modification time changes, reloads it and applies a subset of settings to the running
//! agent with no restart: the trajectory-file output and the resolved local model path (plus the metrics meter/endpoint under the `metrics` feature, and the
//! transport addresses and default hyperparameters under a transport feature).
//!
//! # The Agent Runtime
//!
//! Building an agent does **not** start it. [`RelayRLAgent::start`](crate::agent::RelayRLAgent::start) spins up the coordinator, managers,
//! routers, and supporting control/data runtime tasks designated by the `AgentStartParameters`; you then
//! create one or more actors with [`RelayRLActors::new_actors`](crate::agent::RelayRLActors::new_actors) (each on
//! a chosen [`DeviceType`](crate::types::tensor::relayrl::DeviceType) with its own trajectory length and optional model). Once actors
//! exist there are two ways to [`execute`](#Heterogeneous-Actor-Execution) them, and [`RelayRLAgent::shutdown`](crate::agent::RelayRLAgent::shutdown) tears everything
//! down gracefully. [`RelayRLAgent::restart`](crate::agent::RelayRLAgent::restart) is also available to tear down and reinitialise
//! the runtime without destroying the agent handle. This is useful when you want to reload the agent using
//! a different set of `AgentStartParameters` without building a new agent.
//!
//!
//!
//! Actors are created with `new_actors::<D_IN, D_OUT>(count, device, max_traj_length, nametag, model)`,
//! where the const generics `D_IN` and `D_OUT` declare the observation tensor rank and the
//! action/mask tensor rank respectively. These must be consistent with the environment and model
//! that will be used with those actors. `new_actor` (singular) creates exactly one actor and
//! accepts the same signature. Each actor is assigned to a device, given an independent
//! trajectory buffer of `max_traj_length` steps, an optional `nametag` for tracking, and, if `model` is `Some`, pre-loads that
//! model into its handle; otherwise the actor waits for a model to be provided via
//! `update_models` before it can perform inference.
//!
//! If multiple actors are initialized with the same `nametag`, each successive actor's `nametag` will be post-fixed with `_#` where # is some number.
//!
//! ```rust
//! use relayrl::agent::*;
//! use relayrl::types::model::ModelModule;
//! use relayrl::types::tensor::burn::{tch::Tch, ndarray::NdArray};
//!
//! async fn run(
//!     mut nd_agent: RelayRLAgent<NdArray>,
//!     tch_agent: RelayRLAgent<Tch>,
//!     params: AgentStartParameters<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!
//!     nd_agent.start(params).await?;
//!     tch_agent.start(params).await?;
//!
//!     const ENV1_OBS_IN: usize = 2;
//!     const ENV1_ACT_OUT: usize = 2;
//!     let env1_actors = 4;
//!     let env1_max_traj_length = 1_000;
//!
//!     const ENV2_OBS_IN: usize = 6;
//!     const ENV2_ACT_OUT: usize = 1;
//!     let env2_actors = 3;
//!     let env2_max_traj_length = 2_000;
//!
//!     let no_nd_model: Option<ModelModule<NdArray>> = None;
//!     let no_tch_model: Option<ModelModule<Tch>> = None;
//!
//!     // Create two groups of actors with different obs/action ranks on different devices.
//!     let env1_actor_info = nd_agent.new_actors::<ENV1_OBS_IN, ENV1_ACT_OUT>(
//!         env1_actors, DeviceType::Cpu, env1_max_traj_length, Some("routing_decision".to_string()), no_nd_model
//!     ).await?;
//!     let env2_actor_info = tch_agent.new_actors::<ENV2_OBS_IN, ENV2_ACT_OUT>(
//!         env2_actors, DeviceType::Gpu(0), env2_max_traj_length, Some("compute_decision".to_string()), no_tch_model
//!     ).await?;
//!
//!     let all_nd_actor_ids: Vec<_> = env1_actor_info.iter().map(|(id, _)| *id).collect();
//!     let all_tch_actor_ids: Vec<_> = env2_actor_info.iter().map(|(id, _)| *id).collect();
//!
//!     // ... interact ...
//!
//!     nd_agent.remove_actors(all_nd_actor_ids).await?;
//!     tch_agent.remove_actors(all_tch_actor_ids).await?;//!     nd_agent.shutdown().await?;
//!     tch_agent.shutdown().await?;
//!     Ok(())
//! }
//! ```
//!
//! #### Introspection methods after [`RelayRLAgent::start`](crate::agent::RelayRLAgent::start)
//!
//! - `get_model_versions(actor_ids)` - returns `(ActorUuid, version: i64)` pairs reflecting how
//!   many times each actor's model has been hot-swapped since startup.
//! - `drain_trajectory_caches(actor_ids)` - returns and drains the in-memory trajectory cache for the given
//!   actor IDs (only populated under `OfflineWithCache` or `OfflineWithFilesAndCache` modes).
//! - `get_config()` - fetches the active `ClientConfigLoader` being watched by the lifecycle manager.
//! - `set_config_path(path)` - hot-swap the configuration file path without restarting the runtime.
//!
//! #### Actor Management
//!
//! Every actor created by `new_actor` / `new_actors` runs as its own Tokio task and is tracked
//! in three places: the **namespaced UUID registry** (the source of truth for which IDs are
//! live), the coordinator's state manager (which owns each actor's task handle, device, bound
//! environment, model handle, and trajectory buffer), and the data router layer (which maps each
//! actor to a routing worker and an inbox channel). The methods below manipulate that
//! bookkeeping while the runtime is live, without tearing the agent down.
//!
//! - `get_actor_info()` - returns the UUIDs and nametags of all live actors by reading the client's slice of
//!   the namespaced UUID registry. Because it is the registry view, callers should not rely on
//!   any particular ordering. Returns `NoRuntimeInstanceError` if the agent has not been started.
//! - `get_actor_info_by_rank<D_IN, D_OUT>()` - returns the UUIDs & nametags of all valid live actors that
//!   match the generic `D_IN`, `D_OUT` rank inputs.
//! - `remove_actor(actor_id)` / `remove_actors(actor_ids)` - de-register one or more actors. Under the hood
//!   each removal aborts the actor's task, drops its environment, device, model handle, runtime
//!   handle, and router route, decrements the live-actor count, and frees the UUID back to the
//!   registry. `remove_actors` is a convenience wrapper: an empty list is rejected with
//!   `NoopActorCount`, a single ID delegates to `remove_actor`, and larger lists are removed
//!   one by one.
//! - `set_actor_id(current_id, new_id)` - rename a live actor's UUID in place. The runtime moves the
//!   actor's task handle, inbox, router assignment, device, environment, and runtime handle from
//!   `current_id` to `new_id` and updates the registry; the underlying task keeps running and its
//!   inbox is preserved, so in-flight routing is not interrupted. It fails if `current_id` is not
//!   found or if `new_id` is already taken. Useful for aligning an actor's identity with an
//!   external system (for example a session or player ID).
//! - `set_actor_nametag(actor_id, new_nametag)` - rename a live actor's nametag in place. The runtime
//!   replaces the `actor_id`'s existing `nametag` with the `new_nametag` value.
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLActors};
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//! use relayrl::utils::uuid::Uuid;
//!
//! async fn manage(
//!     agent: &mut RelayRLAgent<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!
//!     // Enumerate live actors (registry order; do not assume sorting).
//!     let actor_info: Vec<(Uuid, Option<Arc<str>>)> = agent.get_actor_info().await?;//!
//!     // Give the first actor a stable, externally-meaningful identity. The task keeps
//!     // running under the new ID with its inbox intact.
//!     let session_id = Uuid::new_v4();
//!     agent.set_actor_id(actor_info[0].0, session_id).await?;
//!
//!     // Retire the remaining actors; their tasks are aborted and their UUIDs freed.
//!     agent.remove_actors(actor_info[1..].iter().map(|(id, _)| id.clone()).collect()).await?;
//!     Ok(())
//! }
//! ```
//!
//! #### Router Scaling
//!
//! Between the coordinator and the actors sits a pool of data routing workers that dispatch
//! messages to actors and drain trajectory sink buffers (see [Step-Driven Training Data Configuration](#step-driven-training-data-configuration)). The initial pool size is set with
//! `AgentBuilder::data_routers`, and the per-actor channel capacity is set with
//! `AgentBuilder::data_buffer_size` (defaults to `1024`). Both can be tuned
//! to match the expected message volume: a larger buffer absorbs bursty workloads without
//! backpressure, while a smaller buffer surfaces overload conditions more quickly.
//!
//! The router pool can be adjusted while the runtime is live with
//! `RelayRLAgent::scale_data_routers` - a positive value scales out, a negative value
//! scales in. This lets you grow routing capacity under load without restarting the agent.
//!
//! ```rust
//! use relayrl::agent::RelayRLAgent;
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//!
//! async fn scale(
//!     agent: &mut RelayRLAgent<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     agent.scale_data_routers(2).await?;   // add two more routing workers
//!     agent.scale_data_routers(-1).await?;  // remove one
//!     Ok(())
//! }
//! ```
//!
//! **Note:** this will consume the runtime for the duration of the operation, thus it's **not recommended** to perform a scale operation
//! while actors are actively running unless your application can tolerate a temporary pause in execution.
//!
//! Performing a `scale_data_routers` operation will distribute all actors across all available routers upon completion. Assuming there is an equal number of actors and routers,
//! the runtime's `ScaleManager` will designate a 1:1 ratio of routers-to-actors. Any excess routers will be left idle. If there are more actors than routers, actors will be
//! distributed as evenly as possible across the available routers with more actors per router.
//!
//! **Static** workloads should initialize with a router scale equal to the expected number of actors. Under **dynamic** workloads where actors are **consistently added and removed**, initializing
//! with a higher router scale than the expected number of actors is **recommended**.
//! This minimizes contention for router resources and avoids unnecessary scaling operations needed to maintain a stable router throughput across actors.
//!
//! #### Config Lifecycle
//!
//! After `start`, a background `LifecycleManager` task watches the config file and automatically reloads it whenever its
//! modification time changes. On the local/default path a reload refreshes the
//! trajectory-file output and the resolved local model path (plus
//! the metrics meter/endpoint under the `metrics` feature); transport addresses and default
//! hyperparameters are additionally refreshed when a transport feature is enabled.
//!
//! If `config_polling_seconds` itself changes, the polling interval is rebuilt to match.
//!
//! The watched path can be swapped at runtime with [`RelayRLAgent::set_config_path`](crate::agent::RelayRLAgent),
//! and the active loader is retrievable with [`RelayRLAgent::get_config`](crate::agent::RelayRLAgent).
//!
//! # Heterogeneous Actor Execution
//!
//! Actors are independent units of execution. Each runs in its own task, and in
//! `ModelMode::Independent` each owns its own model handle, so different
//! actors can run *different* policies on *different* environments simultaneously. On a
//! multi-threaded Tokio runtime their inference runs in parallel (no GIL); on a
//! current-thread runtime they run concurrently.
//!
//! Models are hot-swappable at runtime. [`RelayRLAgent::update_models::<D_IN, D_OUT>(actor_ids, model)`](crate::agent::RelayRLAgent::update_models) can target
//! a subset of actors by passing `Some(vec![id_a, id_b])`, letting you roll a freshly trained
//! policy into specific actors while the rest keep serving the previous one, with no restart and
//! no downtime. Passing `None` signifies no particular actor IDs for the call, thus the function updates all live actors. In `ModelMode::Shared`, the runtime
//! refreshes one representative actor per device so each shared handle is updated exactly once.
//!
//! After a hot-swap, `get_model_versions(actor_ids)` returns the current swap count for each
//! actor as `(ActorUuid, i64)` pairs, which can be used to confirm that the update propagated.
//!
//! **Note:** `update_models` is rejected (returns `ModelUpdateNotSupported`) when the agent is
//! configured with any `Online` training data mode, since model updates are managed server-side
//! in that case. This is to preserve model versioning consistency by enabling only models sent
//! over the network to the client serving as the only source of truth.
//!
//! ```rust
//! use relayrl::agent::*;
//! use relayrl::types::model::ModelModule;
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//!
//! async fn swap(
//!     agent: &RelayRLAgent<NdArray>,
//!     new_model: ModelModule<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     let actor_info = agent.get_actor_info().await?;
//!     let (ids, _nametags): (Vec<_>, Vec<_>) = actor_info.into_iter().unzip();
//!     // Swap a new policy into actors 0 and 2 only; actor 1 keeps the old policy.
//!     agent.update_models::<2, 1>(Some(vec![ids[0], ids[2]]), new_model).await?;
//!     // Verify the swap landed.
//!     let versions = agent.get_model_versions(vec![ids[0], ids[2]]).await?;
//!     Ok(())
//! }
//! ```
//!
//! #### Step-driven Integration
//!
//! In the step-driven pattern, *your* code owns the loop. You hold the observations and
//! ask specific actors for actions one step at a time via [`RelayRLAgent::request_action`](crate::agent::RelayRLAgent::request_action),
//! marking episode boundaries with [`RelayRLAgent::flag_last_action`](crate::agent::RelayRLAgent::flag_last_action). This is the right
//! fit for embedding RelayRL inside an existing simulator, game engine, or control loop.
//!
//! `request_action` is generic over `<D_IN, D_OUT, KindIn, KindOut>`: just as elsewhere, the two const generics
//! must match the observation and action tensor ranks declared when the actors were created,
//! whereas `KindIn`/`KindOut` are the tensor element kinds (e.g. `Float`). It accepts an observation
//! tensor, an optional action mask tensor, and a `reward: f32` for the previous step, and returns
//! `Vec<(ActorUuid, Arc<RelayRLAction>)>` - one entry per actor in the `actor_ids` list. You can
//! target any subset of live actors by constructing the `actor_ids` vector accordingly.
//!
//! `flag_last_action(ids, reward: Option<f32>)` appends a terminal action (`done = true`) to
//! each named actor's current trajectory, signalling the end of an episode. After calling it,
//! the actor begins a fresh trajectory on the next `request_action`.
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLStepDriven, RelayRLActors};
//! use relayrl::types::tensor::burn::{Tensor, Float, ndarray::NdArray};
//! use relayrl::utils::uuid::Uuid;
//!
//! async fn control_loop_step(
//!     agent: &RelayRLAgent<NdArray>,
//!     relevant_actor: Uuid,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!
//!     let obs = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//!     let mask = None;
//!     let reward = 0.0;
//!
//!     // Request the `relevant_actor` to perform inference and return a `RelayRLAction`.
//!     let _actions = agent.request_action(relevant_actor, obs, mask, reward).await?;
//!
//!     // Mark end of episode for `relevant_actor` with the terminal reward.
//!     agent.flag_last_action(relevant_actor, Some(reward + 1.0)).await?;
//!     Ok(())
//! }
//! ```
//!
//! Just like with `update_models`, there are plural forms of both `request_action` and `flag_last_action` for targeting more than a single actor at a time.
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLStepDriven, RelayRLActors};
//! use relayrl::types::tensor::burn::{Tensor, Float, ndarray::NdArray};
//!
//! async fn control_loop_step(
//!     agent: &RelayRLAgent<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!
//!     let actor_info = agent.get_actor_info().await?;
//!     let (all_ids, _all_nametags): (Vec<_>, Vec<_>) = actor_info.into_iter().unzip();
//!     let obs = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//!     let mask = None;
//!     let reward = 0.0;
//!
//!     // Request all actor IDs to perform inference and return a `Vec<RelayRLAction`.
//!     let _actions = agent.request_actions(all_ids.clone(), obs, mask, reward).await?;
//!
//!     // Mark end of episode for all actors with the terminal reward.
//!     agent.flag_last_actions(all_ids, Some(reward + 1.0)).await?;
//!     Ok(())
//! }
//! ```
//!
//! In the step-driven pattern, trajectory collection is automatic: every
//! `request_action` / `flag_last_action` cycle appends experience to an
//! in-memory buffer (when `ActorDataMode::OfflineWithCache` or
//! `OfflineWithFilesAndCache` is selected).
//!
//! To perform a PPO training update
//! after enough episodes have accumulated in-memory, drain the cache, feed the
//! trajectories to a [`PPOTrainer`](crate::algorithms::PPO::PPOTrainer) (or any other compatible RL algorithm procedure), run an
//! epoch, then hot-swap the updated policy back into the actors:
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLStepDriven, RelayRLActors};
//! use relayrl::algorithms::PPO::{PPOTrainer, PPOTrainerSpec};
//! use relayrl::algorithms::GenericMlp;
//! use relayrl::types::tensor::relayrl::{DType, NdArrayDType, DeviceType};
//! use relayrl::types::tensor::burn::{Float, ndarray::NdArray};
//! use std::path::PathBuf;
//!
//! async fn step_driven_training(
//!     agent: &mut RelayRLAgent<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Build a trainer spec matching the environment's obs/act dimensions.
//!     let spec = PPOTrainerSpec::<NdArray, Float, Float,
//!         GenericMlp<NdArray, Float, Float>>::default(
//!         PathBuf::from("env_dir"),
//!         PathBuf::from("policy.onnx"),
//!         4,  // obs_dim
//!         DType::NdArray(NdArrayDType::F32),
//!         2,  // act_dim
//!         DType::NdArray(NdArrayDType::F32),
//!         1_000,
//!         DeviceType::Cpu,
//!     )?;
//!     let mut trainer = PPOTrainer::new(spec)?;
//!
//!     // 2. Get actor ids by rank
//!     let actor_rank_info = agent.get_actor_info_by_rank::<2, 1>().await?;
//!     let (actor_ids, _actor_nametags): (Vec<_>, Vec<_>) = actor_rank_info.into_iter().unzip();
//!
//!     // 3. Drain the trajectory cache collected during the step-driven loop.
//!     if let Some(cache) = agent.drain_trajectory_caches(actor_ids.clone()) {
//!         for (_, trajs) in cache.iter() {
//!             for traj in trajs {
//!                 trainer.receive_trajectory((**traj).clone()).await?;
//!             }
//!         }
//!     }
//! 
//!     // 4. Run one training epoch and apply the result.
//!     if let Some(handle) = trainer.start_epoch_training() {
//!         let output = handle.await?;
//!         trainer.apply_epoch_result(output);
//!         trainer.log_epoch();
//!     }
//!
//!     // 5. Push the updated policy into all live actors.
//!     if let Some(new_model) = trainer.acquire_pi_module() {
//!         agent.update_models::<2, 1>(Some(actor_ids), new_model).await?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! #### Environment-driven Integration
//!
//! In the environment-driven pattern, the [`RelayRLAgent`](crate::agent::RelayRLAgent) owns the loop. You implement either the
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
//! - `run_env_eval(actor_id, env_steps)` - runs evaluation transitions on the bound
//!   environment; no training update is applied. Vectorized env copies are stepped as a batch,
//!   so the final batch may exceed `env_steps` by at most `count - 1` transitions.
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLBatchEnv};
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//!
//! async fn drive(
//!     mut agent: RelayRLAgent<NdArray>,
//!     env: Box<dyn relayrl::env::Environment>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     let (actor_id1, actor_id2) = {
//!         let actor_info = agent.get_actor_info().await?;
//!         (actor_info[0].0, actor_info[1].0)
//!     };
//!
//!     // sequential env stepping is enabled when env count < 8
//!     agent.set_env(actor_id1, env, 7).await?;       // 7 vectorized env copies on this actor
//!     agent.run_env_eval(actor_id1, 10_000).await?;  // run at least 10k env transitions
//!
//!     // rayon data parallelism is enabled when env count >= 8
//!     agent.set_env(actor_id2, env, 1024).await?;    // 1024 vectorized env copies on this actor
//!     agent.run_env_eval(actor_id2, 1_000).await?;   // run at least 1k env transitions
//!
//!     let count = agent.get_env_count(actor_id2).await?;
//!     agent.set_env_count(actor_id2, count / 2).await?;  // halve the env count live
//!     agent.remove_env(actor_id1).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! - `run_env_with_ppo(actor_id, loop_iters, max_traj_length, trainer_spec)` - runs a
//!   single-agent PPO training rollout until `loop_iters` complete. Requires a `PPOTrainerSpec<B, KindIn, KindOut, Pi>`
//!   where **Pi** is a [`NeuralNetwork<B, KindIn, KindOut>`](crate::algorithms::NeuralNetwork). See [`relayrl_algorithms`](https://docs.rs/relayrl_algorithms/0.4.1/relayrl_algorithms/) for details on
//!   constructing the trainer spec.
//! - `run_env_with_ippo` and `run_env_with_mappo` - independent and multi-agent PPO training
//!   rollouts respectively; coming soon (tm).
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLBatchEnv};
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//!
//! async fn train(
//!     mut agent: RelayRLAgent<NdArray>,
//!     env: Box<dyn relayrl::env::Environment>,
//!     ppo_trainer: PPOTrainer<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     let actor_id1 = {
//!         let rank_info = agent.get_actor_info_by_rank::<3, 1>().await?;
//!         rank_info[0].0
//!     };
//!     agent.set_env(actor_id1, env, 7).await?;
//!
//!     let new_model = agent.run_env_with_ppo(actor_id1, 10_000, 10_000, ppo_trainer).await?.export_to_path("model.mpk")?;
//!     agent.update_models::<3, 1>(Some(vec![actor_id1]), new_model).await?;
//!
//!     Ok(())
//! }
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
//! #### Builder setters
//! - `transport_type(TransportMode)` - selects `TransportMode::ZMQ` or `TransportMode::NATS`; defaults to ZMQ when zmq-transport is enabled.
//! - `default_ppo_params(PPOParams)` / `default_ippo_params(IPPOParams)` / `default_mappo_params(MAPPOParams)` - supply
//!   hyperparameters forwarded to the training server at handshake time.
//!
//! #### Inference modes
//! - `ActorInferenceMode::Server(InferenceParams)` - all actor inference is routed to a remote inference server.
//!   `InferenceParams` holds the `ModelMode`, an optional `CodecConfig` (compression, encryption, integrity), and
//!   the server addresses via `InferenceAddressesArgs` (`ZMQ` or `NATS` variants, wrapping `ZmqInferenceAddressesArgs`
//!   or a NATS subject string).
//! - `ActorInferenceMode::ClientFallback(ModelMode, InferenceParams)` - all actors performs inference locally
//!   as a fallback while the rest route to the server.
//!
//! #### Training data modes
//! - `ActorDataMode::Online(TrainingParams)` - trajectories are streamed to a training server.
//! - `ActorDataMode::OnlineWithFiles(TrainingParams, ...)` - stream to server and write to local files.
//! - `ActorDataMode::OnlineWithCache(TrainingParams)` - stream to server and keep in memory.
//! - `ActorDataMode::OnlineWithFilesAndCache(TrainingParams, ...)` - all three simultaneously.
//!
//! `TrainingParams` mirrors `InferenceParams` with the addition of optional hyperparameter args and
//! training-specific addresses via `TrainingAddressesArgs` (`ZMQ` wrapping `ZmqTrainingAddressesArgs`,
//! or `NATS`). `ZmqTrainingAddressesArgs` exposes the agent listener, model server, trajectory server,
//! and scaling server endpoints individually.
//!
//! These paths are under active development and are not covered by the 0.5.x support promise.

pub mod agent {
    pub use relayrl_framework::network::client::agent::*;
}

pub mod algorithms {
    /// The Proximal Policy Optimization family: [`PPOTrainerSpec`](PPO::PPOTrainerSpec) /
    /// [`PPOTrainer`](PPO::PPOTrainer), the policy-value kernel, and the `PPO`/`IPPO`/`MAPPO`
    /// algorithms and their hyperparameters.
    #[allow(non_snake_case)]
    pub mod PPO {
        pub use relayrl_framework::prelude::algorithms::PPO::*;
    }

    /// Neural-network building blocks: the [`NeuralNetwork`] trait family,
    /// [`GenericMlp`], [`ConvNetPolicy`],
    /// [`ValueFunction`], activations, and model-export helpers.
    pub mod nn {
        pub use relayrl_framework::prelude::algorithms::nn::*;
    }

    /// Hand-rolled ONNX `ModelProto` builder that serializes MLP policies without an external
    /// protobuf dependency, for loading via the ONNX Runtime.
    pub mod onnx_builder {
        pub use relayrl_framework::prelude::algorithms::onnx_builder::*;
    }

    pub use relayrl_framework::prelude::algorithms::{
        ActivationKind, ArchLayer, GenericMlp, LayerSpecs, NeuralNetwork, NeuralNetworkError,
        NeuralNetworkForward, NeuralNetworkSpec, ValueFunction, WeightProvider,
        acquire_conv_model_module, acquire_model_module, conv_policy, convert_byte_dtype_to_f32,
        convert_byte_dtype_to_i64, dtype_to_byte_count,
    };
}

pub mod env {
    pub use relayrl_framework::prelude::templates::environment::*;
}

// pub mod servers {
//
// }

pub mod types {
    /// Per-timestep experience: [`RelayRLAction`](action::RelayRLAction), its auxiliary
    /// [`RelayRLData`](action::RelayRLData), and the [`EncodedAction`](action::EncodedAction) codec wrapper.
    pub mod action {
        pub use relayrl_framework::prelude::types::action::*;
    }

    /// Hot-reloadable inference models: [`ModelModule`](model::ModelModule),
    /// [`HotReloadableModel`](model::HotReloadableModel), and [`ModelError`](model::ModelError).
    pub mod model {
        pub use relayrl_framework::prelude::types::model::*;
    }

    /// On-disk trajectory adapters for persisting episodes as Arrow IPC or CSV files.
    pub mod records {
        pub use relayrl_framework::prelude::types::records::*;
    }

    /// Backend-neutral tensors: the serializable `TensorData` container, dtypes/devices, and the
    /// Burn backend bridge, exposed through the `relayrl` and `burn` views.
    pub mod tensor {
        pub use relayrl_framework::prelude::types::tensor::*;
    }

    /// Episodes of experience: [`RelayRLTrajectory`](trajectory::RelayRLTrajectory), its trait, and
    /// the [`EncodedTrajectory`](trajectory::EncodedTrajectory) codec wrapper.
    pub mod trajectory {
        pub use relayrl_framework::prelude::types::trajectory::*;
    }
}

pub mod utils {
    /// JSON configuration loaders and builders (`ClientConfigLoader`, `TransportConfigParams`, etc.),
    /// the [`HyperparameterArgs`](config::HyperparameterArgs) input shape, and the transport-gated
    /// `network_codec` helpers.
    pub mod config {
        pub use relayrl_framework::prelude::utilities::config::*;
    }

    /// Namespaced UUID types used to identify actors and environments within the runtime registry.
    pub mod uuid {
        pub use relayrl_framework::prelude::utilities::uuid::*;
    }
}
