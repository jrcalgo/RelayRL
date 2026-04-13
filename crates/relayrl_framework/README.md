# RelayRL Framework

**Core Library for Deep Multi-Agent Reinforcement Learning**

---
**Version:** 0.5.0-beta.1

**Status:** Under active development, expect breaking changes.

**Tested Platform Support:** macOS (Silicon), Linux (Ubuntu), Windows 10 (x86_64)

## Overview

With v0.5.0 being a complete rewrite of v0.4.5's client implementation, the `relayrl_framework` crate now provides a **multi-actor native** client runtime for deep reinforcement learning experiments. The training server (and new inference server) are under development and remain unavailable in this update.

As of now, the supported beta path is the local/default client runtime. Provide your own
`TorchScript` or `ONNX` model formatted to the framework's standardized `ModelModule` interface.
Transport-backed and server-backed workflows remain experimental in `0.5.0-beta`.

All feature flags other than `client` are (more) **unstable** - if not entirely unimplemented - and unsuitable for RL experiment usage. Use at your own risk!

**Key Features:**

- **Multi-actor native architecture** with concurrent actor execution
- Local Arrow file sink for **offline trajectory data collection** and training
- **In-memory** trajectory retrieval for the last 1,000 trajectories collected
- **Scalable** router-based message dispatching for actor runtimes
- **Ergonomic builder pattern** API for agent construction
- **Multiple device type support** via `NdArray` for CPU exclusively and `Tch` for CPU/CUDA/MPS

**Current Limitations:**

- **Transport Layer:** Network transport (ZMQ/NATS) is implemented as experimental, however no complementary server is available at this time

**Major Changes:**

- **Architecture Redesign:** Monolithic design of v0.4.5 abstracted into a decoupled layered architecture, enhancing modularity, maintainability, and testability.
- **Rust-First Design Philosophy:** Complete removal of PyO3 and its Python code dependencies from framework; all core components written entirely in Rust.
- **Backend Independence:** Replacement of direct `Tch` crate dependency with `Burn`, enabling generic Tensor interfacing with the framework (currently supports Burn's `Tch` and `NdArray` Tensor backends, as well as `TorchScript` and `ONNX` model inference).
- **Data Persistence:** Trajectory data can be optionally persisted in-memory and/or via `Csv` or `Arrow` file formats.
- **Improved Error Handling:** Near complete removal of panics and replacement with proper error handling (retries, branches, etc.) and upstream propagation.
- **Tonic/gRPC Removal:** All Tonic-related code has been removed with focus being cast on building strong `ZMQ` and `NATS` transport implementations.
- **Type System:** Moved to a separate crate (`relayrl_types`).
- **RL Algorithms:** Moved to a separate crate (`relayrl_algorithms`), which remains unimplemented for now.
- **Python Bindings:** Moved to a separate crate (`relayrl_python`), which remains unimplemented for now.

## Quick Start

### 0.5.0-beta Scope

Supported in `0.5.0-beta.1`:

- local inference
- actor lifecycle management
- router scaling
- local Arrow/CSV trajectory writing
- in-memory trajectory retrieval

Experimental in `0.5.0-beta`:

- `zmq-transport`
- `nats-transport`
- server-backed inference or training workflows
- server-side crates and scaffolding

```rust
use relayrl_framework::prelude::network::{AgentBuilder, RelayRLAgentActors};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, Float};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build and Start
    const OBS_RANK: usize = 2;
    const ACT_RANK: usize = 2;

    let model_path = PathBuf::from("dummy_model");
    
    let (mut agent, params) = AgentBuilder::<NdArray, OBS_RANK, ACT_RANK, Float, Float>::builder()
        .actor_count(4)
        .default_model(ModelModule::<NdArray>::load_from_path(model_path)?)
        .build().await?;

    agent.start(params).await?;

    // 2. Interact (using Burn Tensors)
    let reward: f32 = 1.0;
    let obs = Tensor::<NdArray, OBS_RANK, Float>::zeros([1, 4], &Default::default());
    
    let ids = agent.get_actor_ids()?; 
    
    let acts = agent.request_action(ids.clone(), obs, None, reward).await?;
    let versions = agent.get_model_version(ids.clone()).await?;

    // 3. Actor Runtime Management
    agent.new_actor(DeviceType::Cpu, None).await?;
    
    let new_actor_count: u32 = 10;
    agent.new_actors(new_actor_count, DeviceType::Mps, None).await?;
    
    let ids = agent.get_actor_ids()?;
    if ids.len() >= 2 {
        agent.set_actor_id(ids[0], uuid::Uuid::new_v4()).await?;
        agent.remove_actor(ids[1]).await?;
    }

    // 4. Agent Management and Shutdown
    let last_reward: Option<f32> = Some(3.0);
    let ids = agent.get_actor_ids()?;
    agent.flag_last_action(ids.clone(), last_reward).await?;
    
    agent.scale_throughput(2).await?; 
    agent.scale_throughput(-2).await?;
    
    agent.shutdown().await?;
    
    Ok(())
}
```

## Usage Instructions

[View this guide for agent usage :)](../../CLIENT_GUIDE.md)

## Roadmap

- ### **v0.5.x:**
  - Local/default client runtime beta polish
  - Comprehensive client testing and benchmarking on common RL environments
  - Transport-backed client workflows remain experimental during the beta period

- ### **v0.6.0:**
  - Training Server implementation with support for Online/Offline training workflows
  - `relayrl_algorithms` crate integration to enable deep RL algorithmic training and Client `ModelModule` acquisition
  - Comprehensive Training Server testing and benchmarking
  - Comprehensive Client-Training Server network testing and benchmarking on common RL environments
  - Momentary Training Server stabilization

- ### **v0.7.0:**
  - Inference Server implementation to provide client with remote inference capabilities
  - Inference Server and Training Server communication for updating Inference Server's inference model(s)
  - Comprehensive Inference Server testing and benchmarking
  - Comprehensive Client-Inference Server-Training Server network testing and benchmarking on common RL environments

- ### **v0.8.0:**
  - Full Client-Training Server-Inference Server integration
  - Performance optimizations
  - API stabilization
  - Possibly breaking changes

- ### **v0.9.0 / v1.0.0:**
  - **v0.9.0** if still refining APIs and features
  - **v1.0.0** if ready for production stability guarantees
  - The version bump choice between these two depends on API stability and feature completeness

- ### **Beyond this crate:**
  - `relayrl_algorithms` crate creation and publication for training workflows
  - `relayrl_types` updates to minimize serialization overhead and to reduce tensor copy towards zero-copy (as much as possible)
  - `relayrl_cli` for ease-of-use, deployability, and language agnostic execution via a deployable gRPC pipeline for external CLI process interfacing

## Contributing

Contributions are welcomed! Please open issues or pull requests for bug reports, feature requests, or improvements. I'll be glad to work with you!

## License

[Apache License 2.0](../../LICENSE)
