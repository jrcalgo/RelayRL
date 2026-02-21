pub mod data;
#[cfg(any(feature = "tch-model", feature = "onnx-model"))]
pub mod model;

pub mod prelude {
    pub mod action {
        pub use crate::data::action::{
            ActionError, CodecConfig, EncodedAction, RelayRLAction, RelayRLData,
        };
    }

    pub mod tensor {
        pub mod relayrl {
            pub use crate::data::tensor::{
                AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
                IntBurnTensor, SupportedTensorBackend, TensorData, TensorError,
            };
        }

        pub mod burn {
            pub use burn_tensor::*;
        }
    }

    pub mod trajectory {
        pub use crate::data::trajectory::{
            EncodedTrajectory, RelayRLTrajectory, RelayRLTrajectoryTrait, TrajectoryError,
        };
    }

    pub mod records {
        pub use crate::data::records::arrow::{ArrowTrajectory, ArrowTrajectoryError};
        pub use crate::data::records::csv::{CsvTrajectory, CsvTrajectoryError};
    }

    pub mod model {
        #[cfg(any(feature = "tch-model", feature = "onnx-model"))]
        pub use crate::model::{HotReloadableModel, ModelError, ModelModule};
    }

    pub mod codec {
        #[cfg(feature = "compression")]
        pub use crate::data::utilities::compress::{CompressedData, CompressionScheme};

        #[cfg(feature = "integrity")]
        pub use crate::data::utilities::integrity::{compute_checksum, VerifiedData};

        #[cfg(feature = "encryption")]
        pub use crate::data::utilities::encrypt::{EncryptedData, EncryptionKey};

        #[cfg(feature = "metadata")]
        pub use crate::data::utilities::metadata::TensorMetadata;

        #[cfg(feature = "quantization")]
        pub use crate::data::utilities::quantize::{QuantizationScheme, QuantizedData};

        #[cfg(feature = "integrity")]
        pub use crate::data::utilities::chunking::{ChunkedTensor, TensorChunk};
    }
}

use std::collections::HashMap;

/// Hyperparams enum represents hyperparameter inputs
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HyperparameterArgs {
    Map(HashMap<String, String>),
    List(Vec<String>),
}
