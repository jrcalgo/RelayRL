use dashmap::DashMap;

pub mod types {
    pub mod data;
    pub mod model;
}

pub mod data_prelude {
    pub use crate::types::data::action::{
        ActionError, CodecConfig, EncodedAction, RelayRLAction, RelayRLData,
    };

    pub use crate::types::data::tensor::{
        BackendMatcher, BoolTensor, DType, DeviceType, FloatTensor, IntTensor,
        SupportedTensorBackend, TensorData, TensorError,
    };

    pub use crate::types::data::trajectory::{
        EncodedTrajectory, RelayRLTrajectory, RelayRLTrajectoryTrait, TrajectoryError,
    };

    #[cfg(feature = "compression")]
    pub use crate::types::data::utilities::compress::{CompressedData, CompressionScheme};

    #[cfg(feature = "integrity")]
    pub use crate::types::data::utilities::integrity::{VerifiedData, compute_checksum};

    #[cfg(feature = "encryption")]
    pub use crate::types::data::utilities::encrypt::{EncryptedData, EncryptionKey};

    #[cfg(feature = "metadata")]
    pub use crate::types::data::utilities::metadata::TensorMetadata;

    #[cfg(feature = "quantization")]
    pub use crate::types::data::utilities::quantize::{QuantizationScheme, QuantizedData};

    #[cfg(feature = "integrity")]
    pub use crate::types::data::utilities::chunking::{ChunkedTensor, TensorChunk};
}

pub mod model_prelude {

}

/// Hyperparams enum represents hyperparameter inputs
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(DashMap<String, String>),
    Args(Vec<String>),
}
