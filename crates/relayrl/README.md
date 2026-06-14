<p align="center">
  <h1 align="center">RelayRL</h1>
</p>

A Rust-native runtime for concurrent, multi-actor deep reinforcement learning.
RelayRL runs many actors in a single process, performs local inference, collects
trajectory data, and hot-swaps policies with near-zero downtime. It is:

* **Heterogeneous**: each actor can bind its own environment and its own
  independent (or device-shared), hot-swappable model. Run distinct policies
  side by side, or roll a freshly trained policy into a subset of actors while
  the rest keep serving the old one.

* **Concurrent**: actors run as independent Tokio tasks. On a multi-threaded
  runtime their inference runs in parallel with no GIL; on a current-thread
  runtime they run concurrently.

* **Embeddable**: a small builder-driven API over a layered, async runtime makes
  it well suited for embedding RL inside a native application, simulator, game
  engine, or control loop.

<p align="center">
  <a href="https://crates.io/crates/relayrl">
    <img src="https://img.shields.io/crates/v/relayrl.svg" alt="Crates.io" />
  </a>
  <a href="https://docs.rs/relayrl">
    <img src="https://img.shields.io/docsrs/relayrl" alt="Docs.rs" />
  </a>
  <a href="https://github.com/jrcalgo/relayrl/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache 2.0 licensed" />
  </a>
</p>

<p align="center">
  <a href="https://relayrl.dev">Website</a> &nbsp;|&nbsp;
  <a href="https://docs.rs/relayrl">API Docs</a> &nbsp;|&nbsp;
  <a href="../relayrl_framework/README.md">Framework Runtime</a> &nbsp;|&nbsp;
  <a href="https://github.com/jrcalgo/relayrl">Repository</a>
</p>

## 0.5.0

This is the first major release since `0.4.52`, and a complete rewrite of the
client runtime. Highlights of the `0.5.0` line:

* **Multi-actor native runtime.** A layered, async client built for concurrent
  actor execution, live router scaling, per-actor model hot-swap, and
  trajectory collection.
* **Rust-first design.** PyO3 and direct Python dependencies have been removed
  from the framework; all core components are written in Rust.
* **Backend independence.** The hard `tch` dependency was replaced with
  [Burn](https://burn.dev), enabling generic tensor interfacing. CPU
  (`burn-ndarray`) and LibTorch (`burn-tch`) backends are supported, with
  `TorchScript` and `ONNX` model inference.
* **Decoupled crates.** Data types, algorithms, and the environment contract now
  live in dedicated crates (`relayrl_types`, `relayrl_algorithms`,
  `relayrl_env_trait`).

The supported path in `0.5.0` is the local/default client runtime. Network
transport (ZMQ/NATS) and server-backed inference/training workflows are
implemented as **experimental** and are not covered by the `0.5.x` support
promise. See [Feature flags](#feature-flags) and
[Current support](#current-support).

 ## Overview

`relayrl` is a thin facade that re-exports the most recent stable release of
[`relayrl_framework`], the multi-actor client runtime. The rest of the stack is
split into focused crates:

* [`relayrl_framework`]: the runtime. Composes the data model and learning logic
  into a controllable, scalable client (coordinator, router workers, actors, and
  data sinks).
* [`relayrl_types`]: the lowest-level crate. Owns backend-agnostic tensors,
  actions, trajectories, on-disk record adapters (Arrow/CSV), and the codec
  pipeline.
* [`relayrl_algorithms`]: the learning logic. Policy and value networks, rollout
  buffering, and the PPO family (`PPO`, `IPPO`, `MAPPO`).
* [`relayrl_env_trait`]: the environment abstraction. Defines the
  `Environment`, `ScalarEnvironment`, and `VectorEnvironment` contracts the
  runtime drives.

The facade groups these behind four modules: `relayrl::network` (agent API),
`relayrl::types` (actions, tensors, trajectories, records, models),
`relayrl::algorithms` (PPO and neural-network building blocks), and
`relayrl::utilities` (configuration and UUID registry types).

[`relayrl_framework`]: https://docs.rs/relayrl_framework
[`relayrl_types`]: https://docs.rs/relayrl_types
[`relayrl_algorithms`]: https://docs.rs/relayrl_algorithms
[`relayrl_env_trait`]: https://docs.rs/relayrl_env_trait

## Prerequisites

* A [Tokio](https://docs.rs/tokio) runtime. Use a multi-threaded runtime for
  parallel actor execution.
* A compatible [Burn](https://docs.rs/burn/0.21.0/) backend. Currently `burn-ndarray`
  (CPU) and `burn-tch` (LibTorch, CPU/CUDA/MPS) are supported.
* A compatible inference runtime: **LibTorch 2.9.0** or the
  **ONNX Runtime (ORT) 1.26.0**.

## Quick start

Add `relayrl` and a Burn backend to your `Cargo.toml`:

```toml
[dependencies]
relayrl = "0.5.0"
burn-ndarray = "0.20.1"
burn-tensor = "0.20.1"
tokio = { version = "1", features = ["full"] }
```

Build the agent, start the runtime, create actors, request actions, mark the
episode boundary, and shut down:

```rust,no_run
use relayrl::network::*;
use relayrl::types::model::ModelModule;
use relayrl::types::tensor::relayrl::DeviceType;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build the agent handle and its startup parameters.
    let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .router_scale(2)
        .default_model(default_model)
        .build()
        .await?;

    // Start the runtime: coordinator, lifecycle manager, and router workers.
    agent.start(params).await?;

    // Create four actors with rank-2 observations and rank-2 actions.
    let actor_ids = agent
        .new_actors::<2, 2>(4, DeviceType::Cpu, 1_000, None)
        .await?;

    // Request actions. The const generics must match actor creation.
    let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
    let _actions = agent
        .request_action::<2, 2, Float, Float>(actor_ids.clone(), observation, None, 0.0)
        .await?;

    // Mark the episode boundary, then tear everything down gracefully.
    agent.flag_last_action(actor_ids, Some(1.0)).await?;
    agent.shutdown().await?;
    Ok(())
}
```

RelayRL supports two integration patterns: **step-driven** (your code owns the
loop and calls `request_action` per step) and **environment-driven** (the agent
owns the loop and drives a bound `Environment`). See the
[API docs](https://docs.rs/relayrl) for builder configuration, inference and
training-data modes, router scaling, model hot-swap, and environment binding.

## Feature flags

* `client` (default): core client runtime.
* `logging` (default): log4rs logging.
* `metrics` (default): Prometheus/OpenTelemetry metrics.
* `tch-backend`: LibTorch (`tch`) backend and model support.
* `zmq-transport` / `nats-transport`: experimental network transports.
* `training-server` / `inference-server`: experimental server integrations.
* `profile`: flamegraph and tokio-console profiling.

## Current support

The supported `0.5.0` path is the local/default client runtime, including:

* local inference and actor lifecycle management
* live router scaling
* local Arrow/CSV trajectory writing and in-memory trajectory retrieval
* parallelized environment batching
* PPO training rollouts

Experimental in `0.5.0` (enabled by feature flags but not covered by the
support promise):

* `zmq-transport` and `nats-transport`
* server-backed inference or training workflows

## Changelog

Each crate keeps its own changelog. See the [crates/relayrl_framework changelog](../relayrl_framework/CHANGELOG.md) 
for runtime changes.

## Contributing

Contributions are welcome. Please open issues or pull requests for bug reports,
feature requests, or improvements.

## License

This project is licensed under the [Apache License 2.0](../../LICENSE).
