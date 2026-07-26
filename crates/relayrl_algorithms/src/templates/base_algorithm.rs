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
    #[error("Invalid model save path: {0}")]
    InvalidSavePath(String),
    #[error("Model export failed: {0}")]
    ModelExportError(String),
    #[error("Model save failed: {0}")]
    ModelSaveError(#[from] relayrl_types::model::ModelError),
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
/// * `Backend`: The Burn backend this algorithm trains and exports models on.
///
/// # Required Methods
///
/// * `receive_trajectory(&mut self, trajectory: T)`:
///   Process a received trajectory for training. This method is called when new experience data
///   is available.
///
/// * `train_model(&mut self)`:
///   Trigger the training process of the model. The implementation should update the model based
///   on the accumulated trajectories or experiences.
///
/// * `log_epoch(&mut self)`:
///   Log the training status or results for the current epoch. This may include metrics such as loss,
///   reward averages, etc.
///
/// * `save_model(&self, output_dir: &str)`:
///   Persist the current exportable model (the same module returned by [`Self::acquire_model`])
///   into an output directory containing `metadata.json` and the backend-specific model artifact.
///   The saved directory is reloadable via [`relayrl_types::model::ModelModule::load_from_path`].
///
/// * `acquire_model(&self)`:
///   Export the current model as a [`relayrl_types::model::ModelModule`] for inference or hot-swap.
pub trait AlgorithmTrait<T: TrajectoryData> {
    /// The Burn backend this algorithm implementation trains and exports models on (e.g.
    /// `NdArray` or `LibTorch`). Tying the backend to the implementor instead of to
    /// `acquire_model` lets that method return `ModelModule<Self::Backend>` directly, with no
    /// need to reconcile two independently-chosen backend types at runtime.
    type Backend: Backend + BackendMatcher<Backend = Self::Backend>;

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

    /// Saves the current exportable model into `output_dir`.
    ///
    /// The output directory will contain `metadata.json` and the backend-specific model artifact
    /// named by the exported [`relayrl_types::model::ModelModule`] metadata. This is the same
    /// model returned by [`Self::acquire_model`].
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory where `metadata.json` and the model artifact should be written.
    fn save_model(&self, output_dir: &str) -> Result<(), AlgorithmError>;

    /// Acquires the trained model as a ModelModule for inference or export.
    ///
    /// Returns `None` if no model has been trained yet, if weight export is not supported,
    /// or if the required feature flags are not enabled.
    fn acquire_model(&self) -> Option<relayrl_types::model::ModelModule<Self::Backend>>;
}
