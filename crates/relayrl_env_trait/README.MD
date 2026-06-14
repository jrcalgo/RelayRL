# RelayRL Environment Traits

The environment abstraction layer for RelayRL. This crate is intentionally tiny and
dependency-light (`thiserror` + `uuid` only): it defines *how* the runtime talks to an
environment, but contains no simulator, no tensor backend, and no runtime logic. The
`relayrl_framework` client drives any type that implements these traits.

## System layout

All public items live in the `traits` module and are re-exported at the crate root:

- `Environment`: the base contract every environment shares (observation/mask building,
  dtypes and dimensions, flat-bytes accessors, discreteness, and conversion into an
  `EnvironmentHandle`).
- `ScalarEnvironment`: one object per logical environment, stepped with a single action.
- `VectorEnvironment`: one object that owns a *batch* of logical environments, stepped
  with a batch of actions in a single call.
- `EnvironmentHandle`: a runtime-facing enum unifying boxed scalar and vector
  environments, with `DynScalarEnvironment` providing object-safe, clonable scalar envs.
- Supporting types: `EnvironmentUuid` (stable per-env identity), `EnvDType` /
  `EnvironmentKind`, `ScalarEnvReset` / `VectorEnvReset`, the byte aliases
  `Observation` / `Mask` / `Reward` / `Done` / `Truncated`, and
  `TrainingPerformanceReturnFn` for custom training signals.

## Scalar vs. vector execution

The framework may run **many logical environments in parallel** (one
`ScalarEnvironment` per worker) or a single **batched** simulator that implements
`VectorEnvironment`:

- Use `ScalarEnvironment` when each sub-environment is its own object with a scalar
  step. A parallel runner holds many handles, assigns one stable `EnvironmentUuid` per
  sub-env, and steps each worker independently.
- Use `VectorEnvironment` when one implementation can apply a batch of actions keyed by
  `EnvironmentUuid` in a single call (GPU batching, vectorized physics, a remote batched
  service, etc.).

## Design notes and implementor contracts

- **`Send + Sync` everywhere.** All traits require `Send + Sync`, so mutable simulation
  state should live behind interior mutability (e.g. `Mutex`, atomics) rather than `&mut
  self` — every method takes `&self`.
- **Opaque identity.** Treat `EnvironmentUuid` as opaque; the same uuid must refer to
  one logical env across `reset`/`step` and any runtime routing.
- **Ordering.** Unless your concrete type documents otherwise, callers should not assume
  `VectorEnvironment::step_bytes` output order matches input order; key results by
  `EnvironmentUuid`.
- **Errors are whole-operation.** `EnvironmentError` describes the entire call; partial
  success is not expressed in the type system. Surface per-env failures inside your info
  payloads if you need them.
- **Type-erased observations.** `Environment::build_observation` returns
  `std::any::Any` for framework integration; pair it with a documented downcasting
  convention. The `flat_*_bytes` accessors provide the byte-oriented path the runtime uses.

## Quick start

A minimal scalar environment skeleton. Note that every method takes `&self`, so any
mutable state must use interior mutability:

```rust,no_run
use relayrl_env_trait::*;
use std::any::Any;

#[derive(Clone)]
struct MyEnv;

impl Environment for MyEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> { Ok(()) }
    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(vec![0u8; self.observation_dim()]))
    }
    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> { Ok(Box::new(())) }
    fn observation_dtype(&self) -> EnvDType { EnvDType::NdArray(EnvNdArrayDType::F32) }
    fn action_dtype(&self) -> EnvDType { EnvDType::NdArray(EnvNdArrayDType::I64) }
    fn observation_dim(&self) -> usize { 8 }
    fn action_dim(&self) -> usize { 4 }
    fn flat_observation_bytes(&self) -> Observation { vec![0u8; self.observation_dim()] }
    fn flat_mask_bytes(&self) -> Mask { None }
    fn action_is_discrete(&self) -> bool { true }
    fn kind(&self) -> EnvironmentKind { EnvironmentKind::Scalar }
    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(Box::new(*self))
    }
}

impl ScalarEnvironment for MyEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        Ok(ScalarEnvReset { observation: self.flat_observation_bytes(), info: None })
    }
    fn step_bytes(
        &self,
        _action: &[u8],
    ) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        Some((self.flat_observation_bytes(), None, 0.0, false, false))
    }
}
```

## License

This project is licensed under the [Apache License 2.0](../../LICENSE).
