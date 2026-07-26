# RelayRL Framework

The multi-actor reinforcement learning client runtime that powers RelayRL.
This crate is the top-level runtime: it composes the data model from
`relayrl_types` and the learning logic from `relayrl_algorithms` into a
controllable, scalable client that runs many actors, performs local inference,
and streams trajectories to data sinks. It is:

* **Heterogeneous**: each actor runs as its own task and (in `Independent`
  mode) owns its own hot-swappable model, so different actors can serve
  different policies on different environments at the same time.

* **Concurrent**: the runtime is Tokio-based. Routers can be scaled live with
  `scale_data_routers`, and actors run concurrently with interior-mutable shared
  state, in parallel on a multi-threaded runtime.

* **Layered**: a small public API (`RelayRLAgent` + `AgentBuilder`) sits over an
  internal coordination, routing, and data-sink stack, keeping the surface
  ergonomic while the runtime stays modular.

[![Crates.io][crates-badge]][crates-url]
[![Docs.rs][docs-badge]][docs-url]
[![Apache 2.0 licensed][license-badge]][license-url]

[crates-badge]: https://img.shields.io/crates/v/relayrl_framework.svg
[crates-url]: https://crates.io/crates/relayrl_framework
[docs-badge]: https://img.shields.io/docsrs/relayrl_framework
[docs-url]: https://docs.rs/relayrl_framework
[license-badge]: https://img.shields.io/badge/license-Apache--2.0-blue.svg
[license-url]: https://github.com/jrcalgo/relayrl/blob/main/LICENSE

[API Docs](https://docs.rs/relayrl_framework) |
[relayrl crate](../relayrl/README.md) |
[Changelog](CHANGELOG.md) |
[Repository](https://github.com/jrcalgo/relayrl)

## Most users should use the `relayrl` crate

[`relayrl`](../relayrl/README.md) is the higher-level workspace facade that
re-exports this runtime under a single namespace (`relayrl::agent`,
`relayrl::types`, `relayrl::algorithms`, `relayrl::utils`). Prefer depending
on `relayrl` once published unless you specifically need the runtime crate
directly.

```toml
[dependencies]
relayrl = "0.5.0-rc.1"
```

## Overview

`relayrl_framework` is the runtime layer of the RelayRL stack. It pulls the rest
of the stack together:

* `relayrl_types`: backend-agnostic tensors, actions, trajectories, on-disk
  record adapters (Arrow/CSV), and the codec pipeline.
* `relayrl_algorithms`: policy and value networks, rollout buffering, and the
  PPO family (`PPO`, `IPPO`, `MAPPO`).
* `relayrl_env_trait`: the `Environment`, `ScalarEnvironment`, and
  `VectorEnvironment` contracts the runtime drives.

The supported path in `0.5.0` is the local/default client runtime. Client
network transport (ZMQ/NATS) is **experimental**. Server-backed
inference/training runtimes are **not shipped** in this branch. See
[Feature flags](#feature-flags) and [Current support](#current-support).

## Architecture

The client runtime is layered, with a small public API over an internal,
concurrency-oriented runtime:

```text
Public API ......... RelayRLAgent + AgentBuilder
       |
Coordination ....... ClientCoordinator (orchestrator)
       |             ScaleManager (router scaling)
       |             StateManager (actor state)
       |             LifecycleManager (config, shutdown)
       |
Routing ............ RouterDispatcher + scalable Router workers
       |
Actors ............. concurrent actors, local model inference, trajectory building
       |
Data sinks ......... file sink (Arrow/CSV), transport sink (ZMQ/NATS, experimental)
```

The local/default control flow is:
`AgentBuilder -> RelayRLAgent -> ClientCoordinator -> actors/data routers -> data sinks`.

## Module structure

* `network`: the runtime.
  * `network::client`: the multi-actor client runtime (rewritten in v0.5.0). The
    public `agent` module holds the `RelayRLAgent` facade, `AgentBuilder`
    construction API, and the `ActorInfo` actor handle; the internal `runtime`
    holds `control` (coordinator, lifecycle, scaling, state), `data::router`
    (message routing), and `data` (file sinks plus experimental transport
    sinks).
  * Server runtime is not shipped; `training-server` / `inference-server`
    feature flags are reserved/no-op in this branch.
* `utilities`: JSON configuration loading/builders, logging (log4rs), and
  metrics (Prometheus/OpenTelemetry).
* `prelude`: grouped re-exports spanning this crate plus `relayrl_types`,
  `relayrl_algorithms`, and `relayrl_env_trait`.

## Quick start

Add `relayrl_framework` and a Burn backend to your `Cargo.toml`:

```toml
[dependencies]
relayrl_framework = "0.5.0-rc.1"
tokio = { version = "1", features = ["full"] }
```

Build the agent, start the runtime, create actors, request actions, and shut
down. The example is `no_run` because it expects a model directory and config
on disk:

```rust,no_run
use relayrl_framework::prelude::network::*;
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::DeviceType;
use relayrl_framework::prelude::types::tensor::burn::{Tensor, Float, ndarray::NdArray};

use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construct the agent and its startup parameters (single backend type parameter).
    let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .params()
        .data_routers(2)
        .default_model(default_model)
        .config_path(PathBuf::from("client_config.json"))
        .build()
        .await?;

    // Start the coordinator and router workers.
    agent.start(params).await?;

    // Create four actors with rank-2 observations and rank-2 actions.
    let actor_info = agent
        .new_actors::<2, 2>(4, DeviceType::Cpu, 1_000, None, None)
        .await?;

    // Request actions for all actors. The const generics must match actor creation.
    let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
    let _actions = agent
        .request_actions::<2, 2, Float, Float>(&actor_info, observation, None, 0.0)
        .await?;

    // Mark the episode boundary for all actors, then tear everything down gracefully.
    agent.flag_last_actions(&actor_info, Some(1.0)).await?;
    agent.shutdown().await?;
    Ok(())
}
```

## Feature flags

* `client` (default): core client runtime.
* `logging-init`: log4rs logging.
* `tch-backend`: LibTorch (`tch`) backend support via `relayrl_types`.
* `metrics`: Prometheus/OpenTelemetry metrics.
* `profile`: flamegraph and tokio-console profiling.
* `zmq-transport` / `nats-transport`: experimental client network transports.
* `inference-server` / `training-server`: reserved/no-op feature flags; no
  server runtime ships in this branch.

## Current support

* **Supported:** the local/default client runtime, including local inference and
  actor lifecycle management, live router scaling, local Arrow/CSV trajectory
  writing, in-memory trajectory retrieval, parallelized environment batching,
  and PPO training rollouts.
* **Experimental:** client ZMQ/NATS transport paths, even when their feature
  flags are enabled.
* **Not shipped:** server-backed inference or training runtimes.

## Release Notes / Changelog

[CHANGELOG](CHANGELOG.md)

## Contributing

Contributions are welcome. Please open issues or pull requests for bug reports,
feature requests, or improvements.

## License

This project is licensed under the [Apache License 2.0](../../LICENSE).
