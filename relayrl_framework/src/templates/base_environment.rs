//! This module defines traits for training and testing environments in the RL4Sys framework.
//! These traits provide a common interface for building observations, running the environment loop,
//! and (in the case of training) calculating performance metrics during model training.

/// The `EnvironmentTrainingTrait` defines the interface that must be implemented by any
/// environment where a model is trained. Implementing this trait allows an environment to
/// interact with the RL4Sys framework's training pipeline.
///
/// # Methods
///
/// * `run_environment(&self)` - Executes the main training loop, which includes environment interaction,
///   action selection, and model updates.
/// * `build_observation(&self)` - Constructs an observation from the environment, typically by processing
///   raw sensor data or state information.
/// * `calculate_performance_return(&self)` - Computes a performance metric (e.g., total reward) for the
///   current episode or training iteration, which is used to evaluate and improve the model.
pub trait EnvironmentTrainingTrait {
    fn run_environment(&self);
    fn build_observation(&self);
    fn calculate_performance_return(&self);
}

/// The `EnvironmentTestingTrait` defines the interface that must be implemented by any
/// environment where a trained model is evaluated for its inference performance. Implementing
/// this trait allows an environment to run in a test or inference mode.
///
/// # Methods
///
/// * `run_environment(&self)` - Executes the main inference or testing loop, where the trained model
///   is used to interact with the environment and produce actions.
/// * `build_observation(&self)` - Constructs an observation from the environment, preparing it for
///   input into the trained model.
pub trait EnvironmentTestingTrait {
    fn run_environment(&self);
    fn build_observation(&self);
}
