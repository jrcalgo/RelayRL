//! This module defines a trait that must be implemented by any learning algorithm
//! (such as DQN, PPO, etc.) that is integrated with the RelayRL framework. The trait
//! specifies the required functionality for saving models, receiving trajectories,
//! training the model, and logging training epochs.

use burn_tensor::backend::Backend;
use relayrl_types::prelude::records::{ArrowTrajectory, CsvTrajectory};
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum AlgorithmError {
    #[error("Initialization failed: {0}")]
    InitializationError(String),
    #[error("Insertion of trajectory failed: {0}")]
    TrajectoryInsertionError(String),
    #[error("Buffer sampling failed: {0}")]
    BufferSamplingError(String),
    #[error("Kernel registration failed: {0}")]
    KernelRegistrationError(String),
    #[error("Invalid specification: {0}")]
    InvalidSpec(String),
    #[error(transparent)]
    NeuralNetworkError(#[from] crate::algorithms::NeuralNetworkError),
}

#[allow(clippy::large_enum_variant)]
pub enum TrajectoryType {
    RelayRL(RelayRLTrajectory),
    Csv(CsvTrajectory),
    Arrow(ArrowTrajectory),
}

pub trait TrajectoryData {
    fn into_relayrl(self) -> Option<RelayRLTrajectory>;
}

impl TrajectoryData for RelayRLTrajectory {
    fn into_relayrl(self) -> Option<RelayRLTrajectory> {
        Some(self)
    }
}

impl TrajectoryData for CsvTrajectory {
    fn into_relayrl(self) -> Option<RelayRLTrajectory> {
        self.trajectory
    }
}

impl TrajectoryData for ArrowTrajectory {
    fn into_relayrl(self) -> Option<RelayRLTrajectory> {
        self.trajectory
    }
}

/// The `AlgorithmTrait` defines the interface that every algorithm implementation must fulfill.
///
/// # Associated Types
///
/// * `Action`: Represents the type of action that the algorithm produces. This type must implement
///   the [`RelayRLActionTrait`].
///
/// * `Trajectory`: Represents the type of trajectory (a sequence of actions) that the algorithm uses
///   for training. This type must implement [`RelayRLTrajectoryTrait`] with its `Action` type matching `Self::Action`.
///
/// # Required Methods
///
/// * `save(&self, filename: &str)`:
///   Save the current model to the specified file. This allows persistence of model state.
/// * `receive_trajectory(&self, trajectory: Self::Trajectory)`:
///   Process a received trajectory for training. This method is called when new experience data
///   is available.
///
/// * `train_model(&self)`:
///   Trigger the training process of the model. The implementation should update the model based
///   on the accumulated trajectories or experiences.
///
/// * `log_epoch(&self)`:
///   Log the training status or results for the current epoch. This may include metrics such as loss,
///   reward averages, etc.
pub trait AlgorithmTrait<T: TrajectoryData> {
    /// Receives a trajectory of actions and incorporates it into the training process.
    ///
    /// # Arguments
    ///
    /// * `trajectory` - A trajectory containing a sequence of actions experienced by the agent.
    #[allow(async_fn_in_trait)]
    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError>;

    /// Triggers the training process of the model.
    ///
    /// This function should implement the logic to update the model based on received trajectories.
    fn train_model(&mut self);

    /// Logs the training progress for the current epoch.
    ///
    /// This method can be used to print or store metrics such as loss, accuracy, rewards, etc.
    fn log_epoch(&mut self);

    /// Saves the current model to a file specified by `filename`.
    ///
    /// # Arguments
    ///
    /// * `filename` - The path where the model should be saved.
    fn save_model(&self, filename: &str);

    /// Acquires the trained model as a ModelModule for inference or export.
    ///
    /// Returns `None` if no model has been trained yet, if weight export is not supported,
    /// or if the required feature flags are not enabled.
    ///
    /// # Type Parameters
    ///
    /// * `B` - The Burn backend type (e.g., NdArray or LibTorch)
    fn acquire_model<B: Backend + BackendMatcher<Backend = B>>(
        &self,
    ) -> Option<relayrl_types::model::ModelModule<B>>;
}
