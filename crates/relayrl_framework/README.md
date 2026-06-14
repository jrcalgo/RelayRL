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
  `scale_throughput`, and actors run concurrently with interior-mutable shared
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

[`relayrl`](../relayrl/README.md) is the stable, higher-level facade that
re-exports the most recent release of this runtime under a single namespace
(`relayrl::network`, `relayrl::types`, `relayrl::algorithms`,
`relayrl::utilities`). Prefer depending on `relayrl` unless you specifically
need to depend on the runtime crate directly.

```toml
[dependencies]
relayrl = "0.5.0"
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

The supported path in `0.5.0` is the local/default client runtime. Network
transport (ZMQ/NATS) and server-backed inference/training workflows are
implemented as **experimental** and remain experimental even when their feature
flags are enabled. See [Feature flags](#feature-flags) and
[Current support](#current-support).

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
`AgentBuilder -> RelayRLAgent -> ClientCoordinator -> routers/actors -> data sinks`.

## Module structure

* `network`: the runtime.
  * `network::client`: the multi-actor client runtime (rewritten in v0.5.0). The
    public `agent` module holds the `RelayRLAgent` facade and `AgentBuilder`
    construction API; the internal `runtime` holds `coordination` (coordinator,
    lifecycle, scaling, state), `router` (message routing), and `data` (file
    sinks plus experimental transport sinks).
  * `network::server`: optional, experimental training/inference servers behind
    feature flags.
* `utilities`: JSON configuration loading/builders, logging (log4rs), and
  metrics (Prometheus/OpenTelemetry).
* `prelude`: grouped re-exports spanning this crate plus `relayrl_types`,
  `relayrl_algorithms`, and `relayrl_env_trait`.

## Quick start

Add `relayrl_framework` and a Burn backend to your `Cargo.toml`:

```toml
[dependencies]
relayrl_framework = "0.5.0"
burn-ndarray = "0.20.1"
burn-tensor = "0.20.1"
tokio = { version = "1", features = ["full"] }
```

Build the agent, start the runtime, request actions, and shut down. The example
is `no_run` because it expects a model directory and config on disk:

```rust,no_run
use relayrl_framework::prelude::network::*;
use relayrl_framework::prelude::types::model::ModelModule;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, Float};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construct the agent and its startup parameters (single backend type parameter).
    let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .router_scale(2)
        .default_model(default_model)
        .config_path(PathBuf::from("client_config.json"))
        .build()
        .await?;

    // Start the coordinator, routers, and actors.
    agent.start(params).await?;

    // Request actions: const generics are the observation/action tensor ranks.
    let ids = agent.get_actor_ids()?;
    let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
    let _actions = agent
        .request_action::<2, 2, Float, Float>(ids, observation, None, 0.0)
        .await?;

    // Tear everything down gracefully.
    agent.shutdown().await?;
    Ok(())
}
```

## Feature flags

* `client` (default): core client runtime.
* `logging` (default): log4rs logging.
* `tch-backend`: LibTorch (`tch`) backend support via `relayrl_types`.
* `metrics`: Prometheus/OpenTelemetry metrics.
* `profile`: flamegraph and tokio-console profiling.
* `zmq-transport` / `nats-transport`: experimental network transports.
* `inference-server` / `training-server`: experimental server integrations.

Note that, unlike the `relayrl` crate, the framework's default feature set is
`["client", "logging"]` and does not enable `metrics`.

## Current support

The supported `0.5.0` path is the local/default client runtime, including:

* local inference and actor lifecycle management
* live router scaling
* local Arrow/CSV trajectory writing and in-memory trajectory retrieval
* parallelized environment batching
* PPO training rollouts

Transport-backed workflows remain experimental even when the corresponding
feature flags are enabled:

* `zmq-transport` and `nats-transport`
* server-backed inference or training workflows

## Changelog

[CHANGELOG](CHANGELOG.md)

## Contributing

Contributions are welcome. Please open issues or pull requests for bug reports,
feature requests, or improvements.

## License

This project is licensed under the [Apache License 2.0](../../LICENSE).
