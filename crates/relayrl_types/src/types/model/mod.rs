pub mod hot_reloadable;
pub mod utils;

use std::collections::HashMap;
use std::path::PathBuf;

use burn_tensor::{Tensor, backend::Backend};

use crate::types::data::action::RelayRLData;

use crate::types::data::tensor::{BackendMatcher, DeviceType};

pub use hot_reloadable::HotReloadableModel;
pub use burn_tensor::Shape;
pub use utils::{
    convert_generic_dict, deserialize_model, serialize_model, validate_model, validate_model_simple,
};

#[derive(Debug, Clone)]
pub enum ModelError {
    SerializationError(String),
    DeserializationError(String),
    BackendError(String),
    DTypeError(String),
    InvalidInputDimension(String),
    InvalidOutputDimension(String),
    UnsupportedRank(String,)
}

/// Placeholder ModelModule wrapper. Will be fully implemented once we have proper Burn module definitions.
#[derive(Clone)]
pub struct ModelModule<B: Backend + BackendMatcher> {
    pub path: PathBuf,
    pub model: Model<B>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub default_device: DeviceType,
}

impl<B: Backend + BackendMatcher> ModelModule<B> {
    /// Load a model from a `.pt` or `.onnx` file using burn-import. 
    pub fn load_from_path(path: impl Into<PathBuf>) -> Result<Self, ModelError> {
        let path: PathBuf = path.into();
        // TODO: Implement actual loading with burn-import once we have the right API
        Ok(Self { path })
    }

    /// Run the model forward.  Placeholder implementation.
    pub fn step<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: Tensor<B, D_IN>,
        _mask: Tensor<B, D_OUT>,
    ) -> (Tensor<B, D_OUT>, HashMap<String, RelayRLData>) {
        // TODO: call into generated module; placeholder echoes observation.
        let action = self.model.step::<D_IN, D_OUT>(observation, mask);
        (action, HashMap::new())
    }

    pub fn save(&self, _path: impl Into<PathBuf>) -> Result<(), String> {
        // TODO: Implement actual saving
        Ok(())
    }
}
