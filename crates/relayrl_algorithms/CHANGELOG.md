# Changelog

All notable changes to this project will be documented in this file.

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
