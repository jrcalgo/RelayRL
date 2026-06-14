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
}
