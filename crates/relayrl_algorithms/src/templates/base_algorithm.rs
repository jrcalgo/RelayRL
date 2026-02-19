//! This module defines a trait that must be implemented by any learning algorithm
//! (such as DQN, PPO, etc.) that is integrated with the RelayRL framework. The trait
//! specifies the required functionality for saving models, receiving trajectories,
//! training the model, and logging training epochs.

use burn_tensor::backend::Backend;
use burn_tensor::{Float, Int, TensorKind};
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, TensorData, TensorError};
use relayrl_types::prelude::tensor::burn::Tensor;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::records::{CsvTrajectory, ArrowTrajectory};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum AlgorithmError {
    #[error("Insertion of trajectory failed: {0}")]
    TrajectoryInsertionError(String),
    #[error("Buffer sampling failed: {0}")]
    BufferSamplingError(String),
}

pub enum TrajectoryType {
    RelayRL(RelayRLTrajectory),
    Csv(CsvTrajectory),
    Arrow(ArrowTrajectory),
}

pub trait TrajectoryData {
    fn into_relayrl(self) -> RelayRLTrajectory;
}

impl TrajectoryData for RelayRLTrajectory {
    fn into_relayrl(self) -> RelayRLTrajectory {
        self
    }
}

impl TrajectoryData for CsvTrajectory {
    fn into_relayrl(self) -> RelayRLTrajectory {
        self.trajectory
    }
}

impl TrajectoryData for ArrowTrajectory {
    fn into_relayrl(self) -> RelayRLTrajectory {
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
/// w
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
    async fn receive_trajectory(&self, trajectory: T) -> Result<bool, AlgorithmError>;

    /// Triggers the training process of the model.
    ///
    /// This function should implement the logic to update the model based on received trajectories.
    fn train_model(&self);

    /// Logs the training progress for the current epoch.
    ///
    /// This method can be used to print or store metrics such as loss, accuracy, rewards, etc.
    fn log_epoch(&self);
}

pub enum ForwardOutput<B: Backend + BackendMatcher, const OUT_D: usize> {
    Discrete {
        probs: Tensor<B, OUT_D, Float>,
        logits: Tensor<B, OUT_D, Float>,
        logp_a: Option<Tensor<B, OUT_D, Float>>,
    },
    Continuous {
        mean: Tensor<B, OUT_D, Float>,
        std: Tensor<B, 2, Float>,
        logp_a: Option<Tensor<B, OUT_D, Float>>,
    },
}

pub enum StepAction<B: Backend + BackendMatcher> {
    Discrete(Tensor<B, 2, Int>),
    Continuous(Tensor<B, 2, Float>),
}

pub trait ForwardKernelTrait<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, InK>,
        mask: Tensor<B, OUT_D, OutK>,
        act: Option<Tensor<B, OUT_D, OutK>>,
    ) -> ForwardOutput<B, OUT_D>;
}

pub trait StepKernelTrait<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    fn step<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, InK>,
        mask: Tensor<B, OUT_D, OutK>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError>;

    fn get_input_dim(&self) -> usize;
    fn get_output_dim(&self) -> usize;
}
