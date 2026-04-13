# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-13

### Added
- **Trainer lifecycle helpers** - Added `reset_epoch()` across trainer facades to standardize epoch management around async training loops
  - `PpoTrainer` resets per-actor trajectory counts, while `ReinforceTrainer` and `MultiagentTrainer` expose matching no-op helpers for a consistent API
- **Policy export support** - Added the `WeightProvider` trait and new `acquire_model_module()` methods for exporting trained PPO-family policies as in-memory `relayrl_types::model::ModelModule` values
  - `PpoTrainer` and `MultiagentTrainer` expose policy export on `ndarray-backend`, while `ReinforceTrainer` now provides a consistent `acquire_model_module()` surface that returns `None`

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
