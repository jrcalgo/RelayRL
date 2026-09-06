//! *Build concurrent reinforcement learning systems with composable Rust agent runtimes*
//!
//! RelayRL is an actor-oriented Rust runtime for building concurrent, heterogeneous
//! reinforcement learning systems inside a single application. It provides the
//! infrastructure to compose multiple independent inference workloads,
//! environments, and data-collection pipelines through isolated runtime components
//! coordinated by a host application.
//!
//! Rather than organizing reinforcement learning around a single training loop or
//! monolithic agent process, RelayRL models learning systems as independent runtime
//! components with explicit execution boundaries. Each agent runtime manages its own actors,
//! state, lifecycle, and trajectory collection while remaining coordinated
//! through application-level control. Each actor can operate independently
//! while contributing experience data to larger reinforcement learning workflows.
//!
//! At the implementation level, RelayRL executes actors as independent Tokio tasks under
//! a layered async control plane. A public [`RelayRLAgent`](crate::agent::RelayRLAgent)
//! facade interfaces with a coordinator responsible for lifecycle management, actor state,
//! live data scaling, and runtime orchestration.
//! Each actor is an independently addressed execution unit capable of local inference,
//! environment interaction, and trajectory assembly.
//!
//! Actors may optionally bind to independent environments, devices, and model handles.
//! Model execution supports per-actor **Independent** ownership or **Shared** ownership
//! per device with atomic hot-reload capabilities. The runtime supports both step-driven
//! control from external loops and environment-driven actor loops where actors manage
//! their own environment interaction.
//!
//! RelayRL provides a different scaling model from replica-oriented reinforcement learning
//! systems. Instead of primarily duplicating a single policy across processes, RelayRL
//! enables heterogeneous policies, environments, and workloads to coexist concurrently
//! within one agent runtime while collecting experience through configurable
//! trajectory sinks.
//!
//! This crate is a thin facade re-exporting the **most recent released version** of [`relayrl_framework`].
//! RL algorithms, data types, and the environment trait live in [`relayrl_algorithms`](https://docs.rs/relayrl_algorithms/0.5.0/relayrl_algorithms/),
//! [`relayrl_types`](https://docs.rs/relayrl_types/0.9.1/relayrl_types/), and [`relayrl_env_trait`](https://docs.rs/relayrl_env_trait/1.3.1/relayrl_env_trait/) respectively.
//!
//! For benchmarks and other system details, visit [relayrl.dev](https://relayrl.dev).
//!
//! # Prerequisites
//!
//! - A [Tokio](https://docs.rs/tokio/latest/tokio/) runtime. On a current-threaded runtime, actors will execute concurrently, but not in parallel. To enable parallel execution, use a multi-threaded runtime.
//! - A compatible inference runtime. Currently, only **ONNX Runtime (ORT) 1.24.x** and the **LibTorch 2.9.0** are supported.
//!
//! # Quick Start
//!
//! ```rust
//! use relayrl::agent::*;
//! use relayrl::types::model::ModelModule;
//! use relayrl::types::tensor::DeviceType;
//! use relayrl::types::tensor::burn::{Tensor, Float, ndarray::NdArray};
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
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
//!
//!     // Request actions for all actors. The const rank generics must match creation.
//!     let observation = Tensor::<NdArray, 2, Float>::zeros(
//!         [1, 4],
//!         &Default::default(),
//!     );
//!     let actions: Vec<_> = agent
//!         .request_actions::<2, 1, Float, Float>(&actor_info, observation, None, 0.0)
//!         .await?;
//!
//!     // Use actions in your simulator, then mark the episode boundary across actors.
//!     // some_env_steps(actions);
//!     agent.flag_last_actions(&actor_info, Some(1.0)).await?;
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
//! The builder defaults to `ActorInferenceMode::Client(ModelMode::Independent)`, `ActorDataMode::OfflineWithCache(1000)`
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
//! - `OfflineWithCache` *(default)*: keep trajectories in
//!   an in-memory buffer, retrievable at any time with `RelayRLAgent::drain_trajectory_caches(actors)`.
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
//! [`AgentBuilder::config_path`](crate::agent::AgentBuilder). If the *default* path does not exist it is
//! created on first use, pre-populated with defaults. A custom path supplied via `config_path` is not
//! auto-created in the same way: it must already exist, since the loader reads it directly. If the file
//! exists it is read and parsed into a [`ClientConfigLoader`](crate::utils::config::ClientConfigLoader).
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
//!    overlap a file setting. Two examples: [`AgentBuilder::data_buffer_size`](crate::agent::AgentBuilder)
//!    overrides the file's per-actor buffer size when set (the file value is used when it is left unset),
//!    and [`AgentBuilder::config_polling_seconds`](crate::agent::AgentBuilder) overrides the file's
//!    `config_polling_seconds` when set - once supplied at startup, the runtime keeps that argument's
//!    value fixed for its lifetime and ignores later file changes to the field. Per-call runtime
//!    arguments are also in this tier - `new_actors`/`new_actor` take their `device`, `max_traj_length`,
//!    and `model` directly, and `update_models` swaps a model explicitly; none of these are sourced
//!    from the file.
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
//! `AgentBuilder` equivalent on the local, offline path (for example the metrics endpoint and the
//! transport addresses).
//!
//! ##### Config changes at runtime
//!
//! After [`RelayRLAgent::start`](crate::agent::RelayRLAgent::start), a background task polls the file every `config_polling_seconds` and, when
//! the file's modification time changes, reloads it and applies a subset of settings to the running
//! agent with no restart: the trajectory-file output and the resolved local model path (plus the metrics meter/endpoint under the `metrics` feature, and the
//! transport addresses and default hyperparameters under a transport feature).
//!
//! # Running the Agent
//!
//! Building an agent does **not** start it. [`RelayRLAgent::start`](crate::agent::RelayRLAgent::start) spins up the coordinator, managers,
//! routers, and supporting control/data runtime tasks designated by the `AgentStartParameters`; you then
//! create one or more actors with [`RelayRLActors::new_actors`](crate::agent::RelayRLActors::new_actors) (each on
//! a chosen [`DeviceType`](crate::types::tensor::DeviceType) with its own trajectory length and optional model). Once actors
//! exist there are two ways to [`execute`](#Heterogeneous-Actor-Execution) them.
//!
//! Starting the `RelayRLAgent` necessitates that the agent eventually be
//! shutdown via [`RelayRLAgent::shutdown`](crate::agent::RelayRLAgent::shutdown), which tears everything
//! down gracefully. [`RelayRLAgent::restart`](crate::agent::RelayRLAgent::restart) is also available to tear down and reinitialise
//! the runtime without destroying the agent handle. This is useful when you want to reload the agent using
//! a different set of `AgentStartParameters` without building a new agent.
//!
//! #### Declaring new actors
//!
//! Actors are created with `new_actors::<D_IN, D_OUT>(count, device, max_traj_length, nametag, model)`,
//! where the const generics `D_IN` and `D_OUT` declare the observation tensor rank and the
//! action/mask tensor rank respectively. These must be consistent with the environment and model
//! that will be used with those actors.
//!
//! `new_actor` (singular) creates exactly one actor and
//! accepts the same signature. Each actor is assigned to a device, given an independent
//! trajectory buffer of `max_traj_length` steps, an optional `nametag` for tracking, and, if `model` is `Some`, pre-loads that
//! model into its handle; otherwise the actor waits for a model to be provided via
//! `update_models` before it can perform inference.
//!
//! A supplied `nametag` string is stored internally as a `{ tag, duplicate }` pair, not concatenated
//! into a single string; `ActorInfo::nametag()` returns this pair opaquely. When a `nametag` is
//! supplied, the runtime scans all currently live actors for the highest existing `duplicate` value
//! under that same `tag` and assigns `duplicate` values starting one past it (`0` if the tag is unused
//! yet). This lookup spans every live actor, not just the current call, so a batch of `new_actors` and
//! later `new_actor`/`new_actors` calls that reuse the same tag string never collide with each other.
//! `get_actors_by_tag` matches only on the `tag` part, ignoring `duplicate`, so it returns every actor
//! sharing a base tag regardless of their individual `duplicate` values.
//!
//! ```rust
//! use relayrl::agent::*;
//! use relayrl::types::model::ModelModule;
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//! use relayrl::types::tensor::DeviceType;
//!
//! async fn run_simulation(
//!     mut npc_team_1: RelayRLAgent<NdArray>,
//!     mut npc_team_2: RelayRLAgent<NdArray>,
//!     params_1: AgentStartParameters<NdArray>,
//!     params_2: AgentStartParameters<NdArray>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     npc_team_1.start(params_1).await?;
//!     npc_team_2.start(params_2).await?;
//!
//!     const ENV1_OBS_IN: usize = 2;
//!     const ENV1_ACT_OUT: usize = 2;
//!     let team1_actors = 4;
//!     let team1_max_traj_length = 1_000;
//!
//!     const ENV2_OBS_IN: usize = 6;
//!     const ENV2_ACT_OUT: usize = 1;
//!     let team2_actors = 3;
//!     let team2_max_traj_length = 2_000;
//!
//!     let no_nd_model: Option<ModelModule<NdArray>> = None;
//!
//!     // Create two groups of actors with different obs/action ranks on different devices.
//!     // The trailing `None` is an optional `AlgorithmInitArgs`, only present when a
//!     // transport feature (`nats-transport` / `zmq-transport`) is enabled.
//!     let team1_actor_info = npc_team_1.new_actors::<ENV1_OBS_IN, ENV1_ACT_OUT>(
//!         team1_actors, DeviceType::Cpu, team1_max_traj_length, Some("routing_decision"), no_nd_model.clone(),
//!         #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
//!         None,
//!     ).await?;
//!     let team2_actor_info = npc_team_2.new_actors::<ENV2_OBS_IN, ENV2_ACT_OUT>(
//!         team2_actors, DeviceType::Cpu, team2_max_traj_length, Some("compute_decision"), no_nd_model,
//!         #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
//!         None,
//!     ).await?;
//!
//!     // ... interact ...
//!
//!     npc_team_1.remove_actors(&team1_actor_info).await?;
//!     npc_team_2.remove_actors(&team2_actor_info).await?;
//!
//!     npc_team_1.shutdown().await?;
//!     npc_team_2.shutdown().await?;
//!     Ok(())
//! }
//! ```
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
//! - `get_actor(id)` - returns the [`ActorInfo`](crate::agent::ActorInfo) handle for a single live
//!   actor by its current `ActorUuid`.
//! - `get_all_actors()` - returns the `ActorInfo` handles of all live actors by reading the client's
//!   slice of the namespaced UUID registry. Because it is the registry view, callers should not rely on
//!   any particular ordering. Returns an error if the agent has not been started (no runtime instance).
//! - `get_actors_by_rank<D_IN, D_OUT>()` - returns the `ActorInfo` handles of all live actors that
//!   match the generic `D_IN`, `D_OUT` rank inputs.
//! - `get_actors_by_tag(nametag)` - returns the `ActorInfo` handles of all live actors whose nametag's
//!   `tag` string matches `nametag` (or all untagged actors when `nametag` is `None`), regardless of
//!   each actor's `duplicate` value (see [Declaring new actors](#declaring-new-actors)).
//! - `remove_actor(actor)` / `remove_actors(actors)` - de-register one or more actors. Under the hood
//!   each removal aborts the actor's task, drops its environment, device, model handle, runtime
//!   handle, and router route, decrements the live-actor count, and frees the UUID back to the
//!   registry. `remove_actors` is a convenience wrapper: an empty slice is rejected with
//!   `NoopActorCount`, a single actor delegates to `remove_actor`, and larger slices are removed
//!   one by one.
//! - `set_actor_id(actor, new_id)` - rename a live actor's UUID in place. The runtime moves the
//!   actor's task handle, inbox, router assignment, device, environment, and runtime handle from
//!   `actor.id` to `new_id` and updates the registry, and `actor` is updated in place to reflect
//!   the new ID; the underlying task keeps running and its inbox is preserved, so in-flight
//!   routing is not interrupted. It fails if `actor.id` is not found or if `new_id` is already
//!   taken. Useful for aligning an actor's identity with an external system (for example a
//!   session or player ID).
//! - `set_actor_nametag(actor, new_nametag)` - rename a live actor's nametag in place. The runtime
//!   replaces the actor's existing `nametag` with the `new_nametag` value, and `actor` is updated
//!   in place to reflect it.
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
//!     let actor_info = agent.get_all_actors().await?;
//!
//!     // Give the first actor a stable, externally-meaningful identity. The task keeps
//!     // running under the new ID with its inbox intact; `actor_info[0]` observes the new
//!     // id immediately, since it shares the same underlying slot as every other clone.
//!     let session_id = Uuid::new_v4();
//!     agent.set_actor_id(&actor_info[0], session_id).await?;
//!
//!     // Retire the remaining actors; their tasks are aborted and their UUIDs freed.
//!     agent.remove_actors(&actor_info[1..]).await?;
//!     Ok(())
//! }
//! ```
//!
//! #### Data Scaling
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
//! If `config_polling_seconds` itself changes in the file, the polling interval is rebuilt to match -
//! unless [`AgentBuilder::config_polling_seconds`](crate::agent::AgentBuilder) was supplied at startup,
//! in which case that argument's value is fixed for the runtime's lifetime and further file changes to
//! the field are ignored.
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
//! Models are hot-swappable at runtime. [`RelayRLActors::update_models::<D_IN, D_OUT>(actors, model)`](crate::agent::RelayRLActors::update_models) can target
//! a subset of actors by passing `Some(&[actor_a, actor_b])`, letting you roll a freshly trained
//! policy into specific actors while the rest keep serving the previous one, with no restart and
//! no downtime. Passing `None` signifies no particular actors for the call, thus the function updates all live actors. In `ModelMode::Shared`, the runtime
//! refreshes one representative actor per device so each shared handle is updated exactly once.
//!
//! After a hot-swap, `get_model_versions(actors)` returns the current swap count for each
//! actor as `(ActorInfo, i64)` pairs, which can be used to confirm that the update propagated.
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
//!     let actor_info = agent.get_all_actors().await?;
//!     // Swap a new policy into actors 0 and 2 only; actor 1 keeps the old policy.
//!     let target_actors = vec![actor_info[0].clone(), actor_info[2].clone()];
//!     agent.update_models::<2, 1>(Some(&target_actors), new_model).await?;
//!     // Verify the swap landed.
//!     let versions = agent.get_model_versions(&target_actors).await?;
//!     Ok(())
//! }
//! ```
//!
//! #### Step-driven Integration
//!
//! In the step-driven pattern, *your* code owns the loop. You hold the observations and
//! ask specific actors for actions one step at a time via [`RelayRLStepDriven::request_action`](crate::agent::RelayRLStepDriven::request_action),
//! marking episode boundaries with [`RelayRLStepDriven::flag_last_action`](crate::agent::RelayRLStepDriven::flag_last_action). This is the right
//! fit for embedding RelayRL inside an existing simulator, game engine, or control loop.
//!
//! `request_action` is generic over `<D_IN, D_OUT, KindIn, KindOut>`: just as elsewhere, the two const generics
//! must match the observation and action tensor ranks declared when the actors were created,
//! whereas `KindIn`/`KindOut` are the tensor element kinds (e.g. `Float`). It accepts an observation
//! tensor, an optional action mask tensor, and a `reward: f32` for the previous step, and returns
//! `Vec<(ActorInfo, Arc<RelayRLAction>)>` - one entry per actor in the `actors` slice. You can
//! target any subset of live actors by constructing the `actors` slice accordingly.
//!
//! `flag_last_action(actor, reward: Option<f32>)` appends a terminal action (`done = true`) to
//! the named actor's current trajectory, signalling the end of an episode. After calling it,
//! the actor begins a fresh trajectory on the next `request_action`.
//!
//! ```rust
//! use relayrl::agent::{ActorInfo, RelayRLAgent, RelayRLStepDriven, RelayRLActors};
//! use relayrl::types::tensor::burn::{Tensor, Float, ndarray::NdArray};
//!
//! async fn control_loop_step(
//!     agent: &RelayRLAgent<NdArray>,
//!     relevant_actor: &ActorInfo,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!
//!     let obs = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//!     let mask: Option<Tensor<NdArray, 2, Float>> = None;
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
//!     let actor_info = agent.get_all_actors().await?;
//!     let obs = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//!     let mask = Option::<Tensor<NdArray, 2, Float>>::None;
//!     let reward = 0.0;
//!
//!     // Request all actors to perform inference and return a `Vec<(ActorInfo, Arc<RelayRLAction>)>`.
//!     let _actions = agent.request_actions(&actor_info, obs, mask, reward).await?;
//!
//!     // Mark end of episode for all actors with the terminal reward.
//!     agent.flag_last_actions(&actor_info, Some(reward + 1.0)).await?;
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
//! use relayrl::types::tensor::{DType, NdArrayDType, DeviceType};
//! use relayrl::types::tensor::burn::{Float, ndarray::NdArray};
//!
//! # use std::path::PathBuf;
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
//!     // 2. Get actors by rank
//!     let actor_rank_info = agent.get_actors_by_rank::<2, 1>().await?;
//!
//!     // 3. Drain the trajectory cache collected during the step-driven loop.
//!     // The returned map is keyed by each actor's stable `ActorUuid`, not by `ActorInfo`.
//!     if let Some(cache) = agent.drain_trajectory_caches(&actor_rank_info) {
//!         for (_actor_id, trajs) in cache.iter() {
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
//!         agent.update_models::<2, 1>(Some(&actor_rank_info), new_model).await?;
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
//! `RelayRLBatchEnv::set_env(actor, Box<dyn Environment>, count)`, and let the runtime
//! drive the rollout. The `count` argument controls how many logical environment copies are
//! associated with that actor: when `count < 8` the runtime steps them sequentially; when
//! `count >= 8` rayon data parallelism is engaged across the copies. The count can be
//! adjusted after binding with `set_env_count`, queried with `get_env_count`, and the
//! environment can be removed entirely with `remove_env`.
//!
//! Only one `run_env_*` loop may be active per actor at a time; attempting to start a second
//! returns `ClientError::RunEnvActive` immediately.
//!
//! ##### Evaluation
//!
//! `run_env_eval(actor, loop_iters)` - runs `loop_iters` evaluation iterations on the bound
//!   environment; no training update is applied. Each iteration steps every bound env copy once, so
//!   the total number of transitions performed is `loop_iters * count`, not `loop_iters` itself.
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLBatchEnv, RelayRLActors};
//! use relayrl::types::tensor::burn::ndarray::NdArray;
//!
//! async fn drive_car(
//!     mut agent: RelayRLAgent<NdArray>,
//!     pedals: Box<dyn relayrl::env::Environment>,
//!     steering_wheel: Box<dyn relayrl::env::Environment>,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     let (actor1, actor2) = {
//!         let actor_info = agent.get_all_actors().await?;
//!         (actor_info[0].clone(), actor_info[1].clone())
//!     };
//!
//!     // sequential env stepping is enabled when env count < 8
//!     agent.set_env(&actor1, pedals, 7).await?;       // 7 vectorized env copies on this actor
//!     agent.run_env_eval(&actor1, 10_000).await?;  // 10k loop iters -> 70k env-copy transitions
//!
//!     // rayon data parallelism is enabled when env count >= 8
//!     agent.set_env(&actor2, steering_wheel, 1024).await?;    // 1024 vectorized env copies on this actor
//!     agent.run_env_eval(&actor2, 1_000).await?;   // 1k loop iters -> 1.024m env-copy transitions
//!
//!     let count = agent.get_env_count(&actor2).await?;
//!     agent.set_env_count(&actor2, count / 2).await?;  // halve the env count live
//!     agent.remove_env(&actor1).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ##### Training
//!
//! `run_env_with_ppo(actor, loop_iters, max_traj_length, trainer_spec)` - runs a
//!   single-agent PPO training rollout until `loop_iters` complete. Requires a `PPOTrainerSpec<B, KindIn, KindOut, Pi>`
//!   where **Pi** is a [`NeuralNetwork<B, KindIn, KindOut>`](crate::algorithms::NeuralNetwork). See [`relayrl_algorithms`](https://docs.rs/relayrl_algorithms/0.5.0/relayrl_algorithms/) for details on
//!   constructing the trainer spec.
//! `run_env_with_ippo` and `run_env_with_mappo` - independent and multi-agent PPO training
//!   rollouts respectively; coming soon (tm).
//!
//! ```rust
//! use relayrl::agent::{RelayRLAgent, RelayRLBatchEnv, RelayRLActors};
//! use relayrl::types::tensor::burn::{Float, ndarray::NdArray};
//! use relayrl::algorithms::{PPO::PPOTrainerSpec, GenericMlp};
//!
//! async fn train(
//!     mut agent: RelayRLAgent<NdArray>,
//!     video_game: Box<dyn relayrl::env::Environment>,
//!     ppo_spec: PPOTrainerSpec<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>,
//!     save_model_path: std::path::PathBuf,
//! ) -> Result<(), Box<dyn std::error::Error>> {
//!     let actor1 = {
//!         let rank_info = agent.get_actors_by_rank::<3, 1>().await?;
//!         rank_info[0].clone()
//!     };
//!     agent.set_env(&actor1, video_game, 7).await?;
//!
//!     let new_model = agent.run_env_with_ppo(&actor1, 10_000, 10_000, ppo_spec).await?;
//!     new_model.save(save_model_path)?;
//!     agent.update_models::<3, 1>(Some(&[actor1]), new_model).await?;
//!
//!     Ok(())
//! }
//! ```
//!

// # Experimental Network Transport
//
// <div class="warning">
// Transport- and server-backed workflows are <strong>experimental</strong> in this 0.5.x,
// even when their feature flags are enabled. The current supported path is the local runtime.
// </div>
//
// With the `zmq-transport` and/or `nats-transport` features enabled, an agent can be configured
// for server-backed workflows. The following **transport-gated** surface becomes available:
//
// #### Builder setters
//
// - `transport_type(TransportMode)` - selects `TransportMode::ZMQ` or `TransportMode::NATS`; defaults to ZMQ when zmq-transport is enabled.
// - `default_ppo_params(PPOParams)` / `default_ippo_params(IPPOParams)` / `default_mappo_params(MAPPOParams)` - supply
//   hyperparameters forwarded to the training server at handshake time.
//
// #### Inference modes
// - `ActorInferenceMode::Server(InferenceParams)` - all actor inference is routed to a remote inference server.
//   `InferenceParams` holds the `ModelMode`, an optional `CodecConfig` (compression, encryption, integrity), and
//   the server addresses via `InferenceAddressesArgs` (`ZMQ` or `NATS` variants, wrapping `ZmqInferenceAddressesArgs`
//   or a NATS subject string).
// - `ActorInferenceMode::ClientFallback(ModelMode, InferenceParams)` - all actors performs inference locally
//   as a fallback while the rest route to the server.
//
// #### Training data modes
// - `ActorDataMode::Online(TrainingParams)` - trajectories are streamed to a training server.
// - `ActorDataMode::OnlineWithFiles(TrainingParams, ...)` - stream to server and write to local files.
// - `ActorDataMode::OnlineWithCache(TrainingParams)` - stream to server and keep in memory.
// - `ActorDataMode::OnlineWithFilesAndCache(TrainingParams, ...)` - all three simultaneously.
//
// `TrainingParams` mirrors `InferenceParams` with the addition of optional hyperparameter args and
// training-specific addresses via `TrainingAddressesArgs` (`ZMQ` wrapping `ZmqTrainingAddressesArgs`,
// or `NATS`). `ZmqTrainingAddressesArgs` exposes the agent listener, model server, trajectory server,
// and scaling server endpoints individually.
//
// These paths are under active development and are not covered by the 0.5.x support promise.

pub mod agent {
    pub use relayrl_framework::agent::process::*;
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
    /// [`GenericMlp`], [`ConvNetPolicy`](nn::ConvNetPolicy),
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
