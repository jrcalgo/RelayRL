//! This module defines a trait that must be implemented by any learning algorithm
//! (such as DQN, PPO, etc.) that is integrated with the RL4Sys framework. The trait
//! specifies the required functionality for saving models, receiving trajectories,
//! training the model, and logging training epochs.

use crate::types::action::RL4SysActionTrait;
use crate::types::trajectory::RL4SysTrajectoryTrait;

/// The `AlgorithmTrait` defines the interface that every algorithm implementation must fulfill.
///
/// # Associated Types
///
/// * `Action`: Represents the type of action that the algorithm produces. This type must implement
///   the [`RL4SysActionTrait`].
///
/// * `Trajectory`: Represents the type of trajectory (a sequence of actions) that the algorithm uses
///   for training. This type must implement [`RL4SysTrajectoryTrait`] with its `Action` type matching `Self::Action`.
///
/// # Required Methods
///
/// * `save(&self, filename: &str)`:
///   Save the current model to the specified file. This allows persistence of model state.
///
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
pub trait AlgorithmTrait {
    type Action: RL4SysActionTrait;
    type Trajectory: RL4SysTrajectoryTrait<Action = Self::Action>;

    /// Saves the current model to a file specified by `filename`.
    ///
    /// # Arguments
    ///
    /// * `filename` - The path where the model should be saved.
    fn save(&self, filename: &str);

    /// Receives a trajectory of actions and incorporates it into the training process.
    ///
    /// # Arguments
    ///
    /// * `trajectory` - A trajectory containing a sequence of actions experienced by the agent.
    fn receive_trajectory(&self, trajectory: Self::Trajectory);

    /// Triggers the training process of the model.
    ///
    /// This function should implement the logic to update the model based on received trajectories.
    fn train_model(&self);

    /// Logs the training progress for the current epoch.
    ///
    /// This method can be used to print or store metrics such as loss, accuracy, rewards, etc.
    fn log_epoch(&self);
}
