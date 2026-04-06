//! This module defines traits for training and testing environments in the RelayRL framework.
//! These traits provide a common interface for building observations, running the environment loop,
//! and (in the case of training) calculating performance metrics during model training.
//! 
//! Each trait's methods assume that the necessary variables are set somewhere else, typically in the environment implementation itself,
//! thus the methods are parameterless.

pub mod environment_traits {
    use std::any::Any;
    pub use thiserror::Error;

    #[derive(Debug, Error, Clone)]
    pub enum EnvironmentError {
        #[error("Environment error: {0}")]
        EnvironmentError(String),
        #[error("Observation building error: {0}")]
        ObservationBuildingError(String),
        #[error("Training performance return error: {0}")]
        TrainingPerformanceReturnError(String),
    }

    /// The `EnvironmentTrait` defines the interface that must be implemented by any
    /// environment where a model can be trained or evaluated for its inference performance. Implementing
    /// this trait allows an environment to run in a test or inference mode.
    ///
    /// # Methods
    ///
    /// * `run_environment(&self)` - Executes the main inference or testing loop, where the trained model
    ///   is used to interact with the environment and produce actions.
    /// * `build_observation(&self)` - Constructs an observation from the environment, preparing it for
    ///   input into the trained model.
    pub trait EnvironmentTrait {
        fn run_environment(&self) -> Result<(), EnvironmentError>;
        fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError>;
}

    /// The `TrainingPerfReturnFn` defines the interface that must be implemented by any
    /// function that computes a performance metric (e.g., total reward) for the
    /// current episode or training iteration, which is used to evaluate and improve the model during training.
    ///
    /// # Method
    /// 
    /// * `calculate_performance_return(&self)` - Computes a performance metric (e.g., total reward) for the
    ///   current episode or training iteration, which is used to evaluate and improve the model.
    pub trait TrainingPerformanceReturnFn {
        fn calculate_performance_return(&self) -> Result<Box<dyn Any>, EnvironmentError>;
    }
}