use dashmap::DashMap;

pub mod types {
    pub mod action;
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
        RelayRLAction, RelayRLData, TensorData, TensorBackend, DType,
        ActionError, CodecConfig, EncodedAction,
    };
    pub use crate::types::trajectory::{
        RelayRLTrajectory, RelayRLTrajectoryTrait,
        TrajectoryError, EncodedTrajectory,
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
    pub use crate::utilities::quantize::{QuantizedData, QuantizationScheme};
    
    #[cfg(feature = "integrity")]
    pub use crate::utilities::chunking::{ChunkedTensor, TensorChunk};
}

/// Hyperparams enum represents hyperparameter inputs
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(DashMap<String, String>),
    Args(Vec<String>),
}
