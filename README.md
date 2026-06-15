<div align = "center">
  
# RelayRL

[![RelayRL crate](https://img.shields.io/crates/v/relayrl.svg)](https://crates.io/crates/relayrl)
[![RelayRL documentation](https://docs.rs/relayrl/badge.svg)](https://docs.rs/relayrl)
[![Apache 2.0 licensed](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)

RelayRL is a Rust-native runtime for concurrent, multi-actor deep
reinforcement learning. It is designed for embedding RL inside native
applications, simulators, games, and control loops: run many actors in one
Tokio process, perform local model inference, collect trajectories, and
hot-swap policies while the runtime is live.

The top-level [`relayrl`](crates/relayrl/) crate is a facade over the runtime
and the rest of the stack. It re-exports the agent API, data types, PPO
training pieces, and environment traits from the focused crates in this
workspace.
</div>

## What RelayRL Provides

RelayRL focuses on the local/default client runtime in the `0.5` line:

- **Heterogeneous actors**: each actor can run its own environment and its own
  independent or device-shared model.
- **Concurrent execution**: actors run as Tokio tasks, with parallel execution
  on a multi-threaded runtime.
- **Hot-swappable policies**: update all actors or a selected subset without
  tearing the runtime down.
- **Trajectory collection**: store trajectories in memory or write them as
  Arrow/CSV records for offline training.
- **Environment-driven rollouts**: bind scalar or vector environments and let
  the runtime drive evaluation or PPO rollouts.

Network transports (`zmq-transport`, `nats-transport`) and server-backed
inference/training workflows are experimental and are not part of the current
support promise.

## Crate Layout

- [`relayrl`](crates/relayrl/): the stable-release, recommended crate.
- [`relayrl_framework`](crates/relayrl_framework/): the async multi-actor
  client runtime.
- [`relayrl_types`](crates/relayrl_types/): tensors, actions, trajectories,
  model modules, records, and codec utilities.
- [`relayrl_algorithms`](crates/relayrl_algorithms/): PPO/IPPO/MAPPO trainers
  and neural-network building blocks.
- [`relayrl_env_trait`](crates/relayrl_env_trait/): scalar and vector
  environment contracts.

## Using RelayRL

Add `relayrl` and a Burn backend to your `Cargo.toml`:

```toml
[dependencies]
relayrl = "0.5.0"
burn-ndarray = "0.20.1"
burn-tensor = "0.20.1"
tokio = { version = "1", features = ["full"] }
```

Build an agent, create actors, request actions, and shut down:

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

    // Start the coordinator, lifecycle manager, and router workers.
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

RelayRL also supports an environment-driven pattern where the agent owns the
loop and drives a bound `Environment`. 

## Documentation

 - [API documentation][api-docs]: details builder
configuration, model modes, router scaling, file sinks, trajectory caches,
PPO rollouts. 
 - [Learner's guide][website-docs]: provides a high-level overview of each crate in this repository and their public API surfaces.

[api-docs]: https://docs.rs/relayrl
[website-docs]: https://relayrl.dev/learn

## Feature Flags

- `client` (default): core client runtime.
- `logging` (default): log4rs logging.
- `metrics`: Prometheus/OpenTelemetry metrics.
- `tch-backend`: LibTorch-backed tensors and model support.
- `zmq-transport` / `nats-transport`: experimental network transports.
- `training-server` / `inference-server`: experimental server integrations.
- `profile`: flamegraph and tokio-console profiling.

## Contributing

Contributions are welcome. Please open issues or pull requests for bug reports,
feature requests, or improvements.

## License

RelayRL is licensed under the [Apache License 2.0](LICENSE).
