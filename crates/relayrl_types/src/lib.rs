use dashmap::DashMap;

pub mod types {
    pub mod action;
    pub mod tensor;
    pub mod trajectory;
}

pub mod utilities {
    #[cfg(feature = "compression")]
    pub mod compress;

    #[cfg(feature = "integrity")]
    pub mod integrity;

    #[cfg(feature = "encryption")]
    pub mod encrypt;

    #[cfg(feature = "metadata")]
    pub mod metadata;

    #[cfg(feature = "quantization")]
    pub mod quantize;

    #[cfg(feature = "integrity")]
    pub mod chunking;
}

pub mod prelude {
    pub use crate::types::action::{
        ActionError, CodecConfig, EncodedAction, RelayRLAction, RelayRLData,
    };

    pub use crate::types::tensor::{
        BackendMatcher, BoolTensor, DType, DeviceType, FloatTensor, IntTensor,
        SupportedTensorBackend, TensorData, TensorError,
    };

    pub use crate::types::trajectory::{
        EncodedTrajectory, RelayRLTrajectory, RelayRLTrajectoryTrait, TrajectoryError,
    };

    #[cfg(feature = "compression")]
    pub use crate::utilities::compress::{CompressedData, CompressionScheme};

    #[cfg(feature = "integrity")]
    pub use crate::utilities::integrity::{VerifiedData, compute_checksum};

    #[cfg(feature = "encryption")]
    pub use crate::utilities::encrypt::{EncryptedData, EncryptionKey};

    #[cfg(feature = "metadata")]
    pub use crate::utilities::metadata::TensorMetadata;

    #[cfg(feature = "quantization")]
    pub use crate::utilities::quantize::{QuantizationScheme, QuantizedData};

    #[cfg(feature = "integrity")]
    pub use crate::utilities::chunking::{ChunkedTensor, TensorChunk};
}

/// Hyperparams enum represents hyperparameter inputs
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(DashMap<String, String>),
    Args(Vec<String>),
}
