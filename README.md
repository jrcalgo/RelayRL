<div align = "center">
  
# RelayRL

[![RelayRL crate](https://img.shields.io/crates/v/relayrl.svg)](https://crates.io/crates/relayrl)
[![RelayRL documentation](https://docs.rs/relayrl/badge.svg)](https://docs.rs/relayrl)
[![Apache 2.0 licensed](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)

RelayRL is a Rust-native runtime for concurrent, deep
reinforcement learning actor system. It is designed for embedding RL inside native
applications, simulators, games, and control loops: run many actors in one
Tokio process, perform local model inference, collect trajectories, and
hot-swap policies while the runtime is live.
</div>

## What RelayRL Provides

RelayRL focuses on the local/default client runtime in the `0.5.0` line:

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
support promise in the `relayrl` crate.

## Crate Layout

- [`relayrl`](crates/relayrl/): the recommended crate; release updates.
- [`relayrl_framework`](crates/relayrl_framework/): the async multi-actor
  client runtime; release + development updates.
- [`relayrl_types`](crates/relayrl_types/): tensors, actions, trajectories,
  model modules, records, and codec utilities.
- [`relayrl_algorithms`](crates/relayrl_algorithms/): PPO/IPPO/MAPPO trainers
  and neural-network building blocks.
- [`relayrl_env_trait`](crates/relayrl_env_trait/): scalar and vector
  environment contracts.

The top-level [`relayrl`](crates/relayrl/) crate is a facade over the runtime
and the rest of the stack. It re-exports the agent API, data types, PPO
training pieces, and environment traits from the focused crates in this
workspace.

## Using RelayRL

Add `relayrl` and `tokio` to your dependencies:

```toml
[dependencies]
relayrl = "0.5.0-rc.1"
tokio = { version = "1", features = ["full"] }
```

Build a step-driven agent by creating actors, requesting actions, and eventually shutting down:

```rust,no_run
use relayrl::agent::*;
use relayrl::types::model::ModelModule;
use relayrl::types::tensor::DeviceType;
use relayrl::types::tensor::burn::{Float, Tensor, ndarray::NdArray};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build the agent handle and its startup parameters.
    let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .modes()
        .actor_inference_mode(ActorInferenceMode::Client(ModelMode::Shared))
        .actor_data_mode(ActorDataMode::OfflineWithFilesAndCache(None, 1000))
        .params()
        .data_routers(2)
        .default_model(default_model)
        .build()
        .await?;

    // Start the coordinator, managers, and router workers.
    agent.start(params).await?;

    // Create four actors with rank-2 observations and rank-1 actions, CPU device type, 
    // a maximum trajectory length of 1000, a nametag, and no overridden default model.
    let actor_info: Vec<ActorInfo> = agent
        .new_actors::<2, 1>(4, DeviceType::Cpu, 1_000, Some("subsystem-actors"), None)
        .await?;

    // Create a rank-2 observation tensor based on the relevant environment.
    let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());

    // Request actions. The const generics must match actor creation.
    let _actions = agent
        .request_actions::<2, 1, Float, Float>(&actor_info, observation, None, 0.0)
        .await?;

    // Mark the episode boundary, then tear everything down gracefully.
    agent.flag_last_actions(&actor_info, Some(1.0)).await?;
    agent.shutdown().await?;

    Ok(())
}
```

RelayRL also supports an environment-driven pattern where each actor can own its own loop
 and drive a bound `Environment` trait implementation:

```rust,no_run
async fn batch_env_exec(
  mut agent: RelayRLAgent<NdArray>,
  env1: Box<dyn Environment>,
  env2: Box<dyn Environment>,
  trainer: PPOTrainerSpec<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>,
) -> Result<(), Box<dyn std::error::Error>> {
  let actor_info = agent.get_all_actors().await?;
  let (actor1, actor2) = (&actor_info[0], &actor_info[1]);

  let env1_count = 64;
  let env2_count = 1024;

  agent.set_env(actor1, env1, env1_count).await?;
  agent.set_env(actor2, env2, env2_count).await?;

  let loop_iters = 1000;
  let max_traj_length = 10_000;

  agent.run_env_eval(actor1, loop_iters).await?;
  agent.run_env_with_ppo(actor2, loop_iters, max_traj_length, trainer).await?;

  Ok(())
}
```

## Documentation

 - [Learner's guide][website-docs]: provides a high-level overview of each crate in this repository and their public API surfaces.
 - [API documentation][api-docs]: details builder
configuration, model modes, router scaling, file sinks, trajectory caches,
PPO rollouts. 

[api-docs]: https://docs.rs/relayrl
[website-docs]: https://relayrl.dev/learn

## Feature Flags

- `client` (default): core client/agent runtime.
- `logging-init`: log4rs logging initialization.
- `metrics`: Prometheus/OpenTelemetry metrics.
- `tch-backend`: LibTorch-backed tensors and model support.
- `profile`: flamegraph and tokio-console profiling.

## Contributing

Contributions are welcome. Please open issues or pull requests for bug reports,
feature requests, or improvements.

## License

RelayRL is licensed under the [Apache License 2.0](LICENSE).
