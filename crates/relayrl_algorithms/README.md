# RelayRL Algorithms

**Single- and Multi-Agent Deep Reinforcement Learning Algorithms**

---
**Status:** Under active development, numerous changes and refinements in the coming updates will be made to the algorithms based on integration testing and benchmarking!

## Overview

`relayrl_algorithms` is the training-focused crate in the RelayRL ecosystem. It provides Burn-based deep reinforcement learning algorithms and trainer facades for PPO/IPPO/MAPPO, along with the shared abstractions needed to ingest trajectories, run training steps, log epochs, and persist checkpoints. In practice, it is the place where the algorithm runtime lives, while `relayrl_types` supplies the common tensor, action data, and trajectory types used throughout the project.

Within the larger RelayRL project, this crate is designed to pair naturally with `relayrl_framework` when you want RelayRL's runtime and utilization story: multi-actor orchestration, data collection, and the broader client-side workflow. At the same time, `relayrl_algorithms` is not coupled to the framework crate itself. It can be used independently in custom Rust training pipelines, as long as you provide the surrounding environment loop and trajectory flow expected by the trainer APIs.

This crate is still early-stage and under active development. The current `0.x.x` surface is intended to be useful for integration work, experimentation, and benchmarking, but readers should expect continued API refinement as the framework integration story matures and additional algorithms are stabilized.

The only algorithm currently proven to work is PPO; results may vary with IPPO and (which is currently a stub) MAPPO.

## Quick start

Construction follows a **spec-then-build** flow: assemble a `PPOTrainerSpec` (its
`default` constructor builds matching policy/value networks for you), then hand it to
`PPOTrainer::new` to validate and instantiate a runnable trainer. `PPOTrainerSpec::ppo`
/ `::ippo` / `::mappo` are available instead of `::default` when you want to supply your
own `PPONetworkArgs` or hyperparameters.

```rust,ignore
use relayrl_algorithms::prelude::ppo::trainer::{PPOTrainer, PPOTrainerSpec};
use relayrl_algorithms::prelude::nn::GenericMlp;
use relayrl_types::prelude::tensor::relayrl::{DType, NdArrayDType, DeviceType};
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use burn_ndarray::NdArray;
use burn_tensor::Float;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build a spec: `default` constructs matching policy/value networks for you.
    let spec = PPOTrainerSpec::<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>::default(
        PathBuf::from("env_dir"),
        PathBuf::from("model.mpk"),
        8, DType::NdArray(NdArrayDType::F32),   // observation dim + dtype
        4, DType::NdArray(NdArrayDType::F32),   // action dim + dtype
        1_000,                                   // rollout buffer size
        DeviceType::Cpu,
    )?;

    // 2. Validate the spec and construct the runnable trainer.
    let mut trainer = PPOTrainer::new(spec)?;

    // 3. Feed trajectories collected elsewhere (e.g. from relayrl_framework actors)
    //    until an epoch's worth of data has accumulated.
    let trajectory: RelayRLTrajectory = /* ...collected rollout... */;
    let epoch_ready = trainer.receive_trajectory(trajectory).await?;

    if epoch_ready {
        // 4. Training runs on a background task; await the join handle for the result.
        if let Some(handle) = trainer.start_epoch_training() {
            let output = handle.await?;
            trainer.apply_epoch_result(output);
        }
        trainer.log_epoch();
    }

    // 5. Export the trained policy as a `ModelModule` for inference or hot-swap
    //    (e.g. via `RelayRLAgent::update_models` in relayrl_framework).
    let policy_module = trainer.acquire_pi_module();

    Ok(())
}
```

Notes:

- `receive_trajectory`, `start_epoch_training`, `apply_epoch_result`, `log_epoch`, and
  `acquire_pi_module`/`acquire_vf_module` are inherent `PPOTrainer` methods that delegate
  to `IndependentPPOAlgorithm` for the `PPO`/`IPPO` variants. `MAPPO` currently
  `unimplemented!()`s on all of these — see the status note above.
- `receive_trajectory` returns `Ok(true)` once enough trajectories have accumulated to
  train an epoch (per `TrainerArgs::buffer_size` / hyperparameters); training itself is
  triggered explicitly via `start_epoch_training`, not automatically.
- `acquire_pi_module`/`acquire_vf_module` return `Option<relayrl_types::model::ModelModule<B>>`,
  built from the trained kernel's layer specs — `None` until at least one training epoch
  has completed.

## License
[Apache License 2.0](../../LICENSE)

