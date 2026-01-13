<div align="center">

# RelayRL

**Multi-Agent Distributed Reinforcement Learning Framework**

[![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Alpha-red.svg)]()

*A Rust-native framework for scalable deep reinforcement learning experiments*

</div>

---

## Overview

RelayRL is a **monorepo** containing a suite of Rust crates designed for distributed multi-agent reinforcement learning. Built with a Rust-first philosophy, the framework prioritizes performance, type safety, and scalability while maintaining an ergonomic API for single- and multi-agent learning environments.

> **Alpha Software:** This project is under active development. Expect breaking changes and incomplete functionality.

### Key Highlights

- **Pure Rust Core** — No Python dependencies in the framework layer
- **Multi-Actor Native** — Concurrent actor execution with router-based message dispatching
- **Backend Agnostic** — Generic tensor interface via Burn (supports `NdArray`, `Tch` for CPU/CUDA/MPS)
- **Modular Architecture** — Decoupled layers for client, server, transport, and data handling
- **Robust Error Handling** — Proper error propagation instead of panics

## Crates

| Crate | Version | Description |
|-------|---------|-------------|
| [`relayrl_framework`](crates/relayrl_framework/) | `0.5.0-alpha` | Core library with client runtime, server scaffolding, and utilities |
| [`relayrl_types`](crates/relayrl_types/) | `0.4.0` | Data types, tensor containers, and codec pipeline (compression, encryption, integrity) |
| [`relayrl_algorithms`](crates/relayrl_algorithms/) | `0.1.0` | RL algorithms (PPO, REINFORCE) — *scaffolding only* |
| [`relayrl_python`](crates/relayrl_python/) | `0.1.0` | Python bindings via PyO3 — *scaffolding only* |
| [`relayrl_cli`](crates/relayrl_cli/) | `0.1.0` | Command-line interface with gRPC — *scaffolding only* |

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Tested |
| Linux (Ubuntu) | Tested |
| Windows 10 (x86_64) | Tested |
| Windows 11 (x86_64) | Not tested (yet) |

## Quick Start

### Prerequisites

- Rust 2024 edition (`rustup update`)
- For GPU support: CUDA toolkit or MPS-compatible macOS

### Installation

In your Cargo.toml:
```toml
relayrl_framework = "0.5.0-alpha"
relayrl_types = "0.3.21"
```

### Basic Usage

```rust
use relayrl_framework::prelude::network::{RelayRLAgent, AgentBuilder};
use relayrl_framework::prelude::types::{ModelModule, DeviceType};
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, Float};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build agent with 4 concurrent actors
    let (mut agent, params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
        .actor_count(4)
        .default_model(ModelModule::<NdArray>::load_from_path("model.pt".into()))
        .build().await?;

    // Start the agent runtime
    agent.start(
        params.actor_count, 
        params.router_scale, 
        params.default_device, 
        params.default_model, 
        params.config_path
    ).await?;

    // Request actions from actors
    let obs = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
    let ids = agent.get_actor_ids()?;
    let actions = agent.request_action(ids, obs, None, 1.0).await?;

    // Graceful shutdown
    agent.shutdown().await?;
    Ok(())
}
```

For more usage details, see the [Framework README](crates/relayrl_framework/README.md) and the [Client Guide](CLIENT_GUIDE.md).

## Framework Roadmap

### Near Term
- **v0.5.0** — Client ZMQ transport, PostgreSQL/SQLite database layer, comprehensive testing
- **v0.6.0** — Training Server with online/offline workflows, algorithm integration

### Medium Term
- **v0.7.0** — Inference Server for remote inference capabilities
- **v0.8.0** — Full system integration, performance optimizations, API stabilization

### Long Term
- **v0.9.0 / v1.0.0** — Production stability guarantees
- `relayrl_algorithms` — Complete RL algorithm implementations
- `relayrl_cli` — Language-agnostic deployable gRPC interface

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting PRs.

## License

Licensed under [Apache License 2.0](LICENSE).
