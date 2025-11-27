use dashmap::DashMap;

pub mod types {
    pub mod data;
    #[cfg(any(feature = "tch-model", feature = "onnx-model"))]
    pub mod model;
}

pub mod prelude {
    pub use crate::types::data::action::{
        ActionError, CodecConfig, EncodedAction, RelayRLAction, RelayRLData,
    };

    pub use crate::types::data::tensor::{
        AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
        IntBurnTensor, SupportedTensorBackend, TensorData, TensorError,
    };

    pub use crate::types::data::trajectory::{
        EncodedTrajectory, RelayRLTrajectory, RelayRLTrajectoryTrait, TrajectoryError,
    };

    #[cfg(any(feature = "tch-model", feature = "onnx-model"))]
    pub use crate::types::model::{HotReloadableModel, ModelError, ModelModule};

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

/// Hyperparams enum represents hyperparameter inputs
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(DashMap<String, String>),
    Args(Vec<String>),
}
