# RelayRL OpenTTD Example

This example maps RelayRL onto an OpenTTD-style simulation as a system of
adaptive subsystems rather than one monolithic game-playing agent.

## OpenTTD version

- Upstream repository: <https://github.com/OpenTTD/OpenTTD>
- Latest stable GitHub release cited by this example: `15.3`, published
  `2026-04-04`: <https://github.com/OpenTTD/OpenTTD/releases/tag/15.3>
- Newest prerelease tag observed while adding the example: `16.0-beta1`:
  <https://github.com/OpenTTD/OpenTTD/releases/tag/16.0-beta1>

The `native-openttd` feature is written against an OpenTTD 15.3 integration
build that exports the C ABI described in `native/openttd_relayrl_shim.h`.
OpenTTD itself is a C++ application and does not provide a stable public C ABI,
so real integrations should implement that shim inside an OpenTTD fork or
plugin boundary where game-state APIs are available.

## Mental model

RelayRL actors are modeled as independent control processes. A runtime models a
bounded adaptive subsystem, and the host application coordinates subsystems
through snapshots and structured signals.

```text
OpenTTD host
├── Transport actors
├── Economy actors
├── Industry actors
├── Town-growth actors
└── Infrastructure actors
```

Each actor optimizes a local subsystem objective. The broader game behavior
emerges when the host applies their commands to the shared OpenTTD world state.

## Subsystem actors

| Subsystem | Actors | Local objectives |
| --- | --- | --- |
| Transport | route planner, dispatch, congestion, vehicle allocation | latency, utilization, bottlenecks, cargo backlog |
| Economy | profit model, demand estimator, investment priority | profit delta, demand accuracy, capital allocation |
| Industry | supply chain, production balancer | shortages, output saturation, wasted production |
| Town growth | growth predictor, rating optimizer | population growth, station ratings, cargo satisfaction |
| Infrastructure | expansion planner, cost optimizer, topology | bottleneck relief, construction efficiency, connectivity |

Runtimes do not call each other directly. They emit commands and signals to the
host, and the host folds those results into the next observation projection.

## Build

Default build, using the deterministic mock bridge:

```bash
cargo run -p openttd-relayrl-example
```

Native OpenTTD bridge build:

```bash
export OPENTTD_SOURCE_DIR=/path/to/OpenTTD-15.3
export OPENTTD_BUILD_DIR=/path/to/OpenTTD-15.3/build-with-relayrl-shim
cargo build -p openttd-relayrl-example --features native-openttd
```

The native build expects `OPENTTD_BUILD_DIR` to contain
`libopenttd_relayrl_shim` implementing:

- `relayrl_openttd_create`
- `relayrl_openttd_destroy`
- `relayrl_openttd_reset`
- `relayrl_openttd_step`
- `relayrl_openttd_snapshot`
- `relayrl_openttd_apply_command`

The reference `native/openttd_relayrl_shim.cpp` is intentionally small. Replace
its placeholder state extraction with calls into an OpenTTD 15.3 integration
point.

## RelayRL integration points

- `environment::OpenTtdEnvironment` implements `Environment` and
  `ScalarEnvironment`. Cloning creates an independent OpenTTD bridge instance,
  so `agent.set_env(actor_id, env, count)` can scale parallel environments.
- `host::OpenTtdHost` demonstrates the step-driven pattern with
  `request_action` and `flag_last_action`.
- `training::train_then_freeze_subsystems` demonstrates sequential
  `PPOTrainerSpec` setup, `run_env_with_ppo`, model persistence, and frozen
  model installation before online cross-subsystem interaction.
