# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-06-01

### Added
- **PPO-focused trainer and spec APIs** - Introduced `PPOTrainer`, `PPOTrainerSpec`, `PPONetworkArgs`, and `MAPPOTrainerSpec` as the new top-level entry points for building PPO, IPPO, and MAPPO training loops.
  - `PPOTrainerSpec` variants (`PPO`, `IPPO`, `MAPPO`) accept typed network args and optional hyperparameter overrides
  - `PPOTrainer::new()` validates and constructs the appropriate algorithm from a spec
  - `register_first_slot_with_key()`, `start_epoch_training()`, and `apply_epoch_result()` compose an async-friendly epoch cycle where collection and SGD run concurrently
- **Generic neural-network abstractions** - Added `NeuralNetwork`, `NeuralNetworkSpec`, `NeuralNetworkForward`, `NeuralNetworkError`, `WeightProvider`, `GenericMlp`, `ValueFunction`, and `ActivationKind` to `algorithms::mod` as reusable building blocks for custom policy networks.
  - `GenericMlp` implements `NeuralNetwork` out of the box and can be dropped into any PPO trainer without writing a custom kernel
- **PPO kernel and policy-head types** - Exposed `PPOKernel`, `PPOKernelSnapshot`, `PPOKernelFactory`, `PPOKernelTraining`, `PPOKernelOps`, `PPOKernelTrainingArgs`, `DiscretePPOKernel`, `ContinuousPPOKernel`, `PPOPolicyHead`, `DiscretePPOPolicyHead`, and `ContinuousPPOPolicyHead` as public API.
  - `PPOKernelSnapshot` gives inference tasks a thread-safe read-only view of the current kernel through `clone_for_inference()` and `to_arc_snapshot()`
  - `get_ppo_actor_kernel()` and `get_ippo_actor_kernel()` on `PPOTrainer` surface the live kernel for inference callers
- **Separate policy and value-function model export** - `IndependentPPOAlgorithm` and `PPOTrainer` now expose `acquire_pi_module()` and `acquire_vf_module()` which return `relayrl_types::model::ModelModule` values built from the kernel's layer specs.
- **Episode-aware PPO replay buffer** - Rewrote `PPOReplayBuffer` with structured episode tracking, GAE finalization, and episodic drain helpers.
  - `finalize_and_drain_first_n_blocking()` drains exactly the first `n` complete episodes and filters stale ones by `max_version_lag`
  - `purge_stale_episodes()`, `get_episode_count()`, `get_complete_step_count()`, `is_full()`, and `episodes_needed_for_steps()` give callers fine-grained buffer introspection
  - `get_obs_for_first_n_episodes()` retrieves observations for GAE value estimation without draining the buffer
- **Expanded `IPPOParams` hyperparameters** - Added `max_version_lag`, `normalize_obs`, `normalize_returns`, `max_episode_steps`, `minibatch`, `min_steps_per_epoch`, `max_buffered_episodes`, and `rollout_len` to `IPPOParams`/`PPOParams`.
- **Fresh value and log-prob recomputation before SGD** - `start_epoch_training()` re-runs value forward and policy log-prob passes on the current Burn model before each SGD round, eliminating stale-value bias from previously stored ORT/TorchScript estimates.
- **PPO prelude** - Added `prelude::ppo` and `prelude::nn` module re-exports in `lib.rs` so callers can import the full PPO surface and generic neural-network traits without navigating internal module paths.
- **`torch_builder` module** - Added `algorithms::torch_builder` (behind the `tch-model` feature) as a renamed and updated replacement for the prior `pt_builder`.

### Changed
- **PPO training internals** - `IndependentPPOAlgorithm` now runs SGD on flat observation/action/advantage/return arrays via `PPOActorCriticMlp` with orthogonal weight initialisation, separate pi/vf layer stacks, gradient clipping, optional minibatch shuffling, and target-KL early stopping.
  - Training metrics (`LossPi`, `DeltaLossPi`, `LossV`, `DeltaLossV`, `KL`, `Entropy`, `ClipFrac`, `StopIter`) are stored through `apply_epoch_result()` and accumulated in `EpochLogger`
- **Crate description and keywords** - Updated `Cargo.toml` description to reflect the PPO-only focus; replaced the `simulation` and `training` categories/keywords with `artificial-intelligence` and `single-agent`.
- **Feature flags simplified** - Removed the `ndarray-backend` and `onnx-model` feature flags from this crate; `ndarray-backend` and `onnx-model` are now activated unconditionally through the `relayrl_types` dependency, and `burn-ndarray` is a non-optional dependency.
  - `tempfile` moved from an optional dep gated on `tch-model` to a mandatory workspace dependency
  - `rand_distr`, `half`, and `rayon` added as mandatory workspace dependencies

### Breaking
- **DDPG, TD3, and REINFORCE algorithm families removed** - All `DDPG`, `TD3`, `REINFORCE`, and `CUSTOM` submodules, their kernels, replay buffers, and trainer facades have been deleted.
  - Public exports `DDPGAlgorithm`, `IDDPGAlgorithm`, `MADDPGAlgorithm`, `TD3Algorithm`, `ITD3Algorithm`, `MATD3Algorithm`, `ReinforceAlgorithm`, `IREINFORCEAlgorithm`, `MAREINFORCEAlgorithm`, and all associated kernel traits are no longer available
  - Callers using any of these types must migrate to external implementations for this release cycle
- **Old `RelayRLTrainer` and multi-algorithm facade removed** - The prior `RelayRLTrainer` struct and its `ppo()`, `mappo()`, `reinforce()`, `ddpg()`, `td3()`, and `iddpg()` constructors no longer exist; use `PPOTrainerSpec` and `PPOTrainer` instead.
- **`MultiagentTrainerSpec` and multi-agent kernel generics removed** - The previous `MultiagentTrainerSpec<K>` and `MultiagentKernelTrait`-based construction are gone; MAPPO is now constructed via `PPOTrainerSpec::MAPPO`.
- **`ndarray-backend` and `onnx-model` feature flags removed** - Consumers that enabled these features explicitly on `relayrl_algorithms` must remove those feature declarations; the functionality is now unconditional.
- **`pt_builder` module renamed** - `algorithms::pt_builder` is now `algorithms::torch_builder`; update any direct module paths accordingly.

## [0.3.0] - 2026-05-06

### Added
- **DDPG and TD3 algorithm families** - Added independent and multi-agent DDPG/TD3 implementations with public aliases, params, kernels, replay buffers, and trainer facades.
  - New exports include `DDPGAlgorithm`, `IDDPGAlgorithm`, `MADDPGAlgorithm`, `TD3Algorithm`, `ITD3Algorithm`, and `MATD3Algorithm`
  - `RelayRLTrainer` now exposes `ddpg()`, `iddpg()`, `td3()`, and `itd3()` constructors alongside the existing PPO and REINFORCE helpers
- **Custom multi-agent kernels** - Multi-agent PPO and REINFORCE now accept user-supplied kernels through `MultiagentTrainerSpec<K>` and the shared `MultiagentKernelTrait` interface.
  - Added algorithm-specific multi-agent kernel traits for PPO, REINFORCE, DDPG, and TD3 so framework callers can plug in training kernels consistently
- **Cross-backend model acquisition** - Added centralized `acquire_model_module()` support for exporting trained policies as `relayrl_types::model::ModelModule` values.
  - ONNX export is enabled by default for ndarray builds through the new `onnx-model` feature
  - Optional TorchScript export is available through the new `tch-model` feature using LibTorch-backed model bytes
- **Model byte builders** - Added ONNX and TorchScript MLP builders that serialize `WeightProvider` layer specs into loadable model artifacts.
  - The ONNX builder preserves Burn's `[in, out]` linear weight layout for `Gemm` nodes
- **Serializable hyperparameters** - Added `Debug`, `Clone`, `PartialEq`, `Serialize`, and `Deserialize` derives to algorithm parameter structs for configuration loading and persistence.

### Changed
- **Trainer model export API** - `AlgorithmTrait` now exposes `acquire_model()` for feature-gated in-memory policy export.
  - PPO, REINFORCE, DDPG, TD3, and their multi-agent variants delegate export through the shared model acquisition path when their kernels implement `WeightProvider`

### Breaking
- **Multi-agent trainer construction** - `MultiagentTrainerSpec` and the `mappo()` / `mareinforce()` constructors now require an explicit kernel argument.
  - Existing callers using the old built-in multi-agent PPO or REINFORCE kernel construction must pass an appropriate kernel value instead
- **Kernel trait bounds** - Trainer and algorithm implementations now require `WeightProvider` for model-export-capable kernels, and multi-agent algorithms are generic over their kernel type.
  - Custom kernels may need to implement the new export and multi-agent base traits to compile against 0.3.0

## [0.2.0] - 2026-04-13

### Added
- **Trainer lifecycle helpers** - Added `reset_epoch()` across trainer facades to standardize epoch management around async training loops
  - `PpoTrainer` resets per-actor trajectory counts, while `ReinforceTrainer` and `MultiagentTrainer` expose matching no-op helpers for a consistent API
- **Policy export support** - Added the `WeightProvider` trait and new `acquire_model_module()` methods for exporting trained PPO-family policies as in-memory `relayrl_types::model::ModelModule` values
  - `PpoTrainer` and `MultiagentTrainer` expose policy export on `ndarray-backend`, while `ReinforceTrainer` now provides a consistent `acquire_model_module()` surface that returns `None`
- **ONNX MLP byte builder** - Added `algorithms::onnx_builder::build_onnx_mlp_bytes()` for generating serialized ONNX `ModelProto` payloads from extracted policy-layer weights
  - The builder emits opset 17 models that can be loaded directly through ORT in-memory model-loading paths without an external protobuf dependency
  - Layer specs from `WeightProvider::get_pi_layer_specs()` are encoded using Burn's `[in, out]` weight layout for ONNX `Gemm` nodes

### Fixed
- **Async replay-buffer sampling** - PPO replay-buffer sampling no longer panics when called from within an active Tokio runtime
  - Independent PPO now samples via a scoped helper thread with its own runtime, and multi-agent PPO uses `tokio::task::block_in_place()` when already running inside Tokio
- **Per-actor PPO kernel initialization** - Additional independent-PPO actor slots now initialize kernels with `KN::new_for_actor(obs_dim, act_dim)` instead of `Default::default()`
  - Extra actors now receive correctly shaped kernels instead of placeholder dimensions

### Breaking
- **Default backend features** - Default builds now enable only `ndarray-backend`
  - `tch-backend` is no longer pulled in automatically, so consumers that relied on LibTorch via defaults must enable it explicitly
- **PPO kernel trait constructor** - `PPOKernelTrait` now requires `new_for_actor(obs_dim, act_dim)` for correctly shaped per-actor kernel creation
  - Custom PPO kernels must implement the new constructor in addition to the existing PPO loss hooks

## [0.1.0] - 2026-03-31

### Added
- **Initial multi-agent RL algorithms crate** - Introduced the first public `relayrl_algorithms` release for Burn-based reinforcement learning training.
  - Exported PPO-family algorithms and aliases including `PPOAlgorithm`, `IPPOAlgorithm`, and `MAPPOAlgorithm`
  - Exported REINFORCE-family algorithms and aliases including `ReinforceAlgorithm`, `IREINFORCEAlgorithm`, and `MAREINFORCEAlgorithm`
- **Trainer facade and constructor specs** - Added ergonomic entry points for building training loops without reaching through internal modules.
  - `RelayRLTrainer`, `PpoTrainer`, `ReinforceTrainer`, and `MultiagentTrainer`
  - `TrainerArgs`, `PpoTrainerSpec`, `ReinforceTrainerSpec`, and `MultiagentTrainerSpec`
- **Shared integration and kernel traits** - Added reusable abstractions for trajectory ingestion, stepping, and pluggable training kernels.
  - `AlgorithmTrait` and `TrajectoryData` for RelayRL-native, CSV, and Arrow trajectory wrappers from `relayrl_types`
  - `PPOKernelTrait`, `StepKernelTrait`, and `TrainableKernelTrait` for custom kernel implementations
- **Runtime support utilities** - Added baseline logging and backend feature support for algorithm experimentation.
  - `EpochLogger` for tabular epoch metrics
  - Default `ndarray-backend` and `tch-backend` feature support for Burn backends
