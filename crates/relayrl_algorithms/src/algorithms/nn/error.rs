/// Errors from neural network construction, dimension validation, or device/dtype resolution.
#[derive(thiserror::Error, Debug, Clone)]
pub enum NeuralNetworkError {
    #[error("Unsupported device: {0}")]
    UnsupportedDevice(String),
    #[error("Unsupported DType: {0}")]
    UnsupportedDType(String),
    #[error("Unsupported output params: {0}")]
    UnsupportedOutputParams(String, String),
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("Input dimension mismatch: {0} != {1}")]
    InputDimMismatch(usize, usize),
    #[error("Invalid distribution")]
    InvalidDistribution,
    #[error("Invalid continuous policy output dim (must be even and positive): {output_dim}")]
    InvalidContinuousOutputDim { output_dim: usize },
    #[error("Continuous policy output dim mismatch: expected {expected}, got {actual}")]
    ContinuousOutputDimMismatch { expected: usize, actual: usize },
    #[error("Invalid continuous action dtype (must be floating): {0}")]
    InvalidContinuousActionDType(String),
    #[error("Model output too short: expected at least {expected} elements, got {actual}")]
    ModelOutputTooShort { expected: usize, actual: usize },
}
