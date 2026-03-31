# RelayRL Algorithms

**MultiAgent Deep Reinforcement Learning Algorithms**

---
**Status:** Under active development, numerous changes and refinements in the coming updates will be made to the algorithms based on integration testing and benchmarking!

## Overview

`relayrl_algorithms` is the training-focused crate in the RelayRL ecosystem. It provides Burn-based deep reinforcement learning algorithms and trainer facades for PPO/IPPO/MAPPO and REINFORCE/IREINFORCE/MAREINFORCE, along with the shared abstractions needed to ingest trajectories, run training steps, log epochs, and persist checkpoints. In practice, it is the place where the algorithm runtime lives, while `relayrl_types` supplies the common tensor, action data, and trajectory types used throughout the project.

Within the larger RelayRL project, this crate is designed to pair naturally with `relayrl_framework` when you want RelayRL's runtime and utilization story: multi-actor orchestration, data collection, and the broader client-side workflow. At the same time, `relayrl_algorithms` is not coupled to the framework crate itself. It can be used independently in custom Rust training pipelines, as long as you provide the surrounding environment loop and trajectory flow expected by the trainer APIs.

This crate is still early-stage and under active development. The current `0.1.x` surface is intended to be useful for integration work, experimentation, and benchmarking, but readers should expect continued API refinement as the framework integration story matures and additional algorithms are stabilized.

## License
[Apache License 2.0](../../LICENSE)

