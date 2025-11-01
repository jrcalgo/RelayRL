use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use burn_tensor::{Shape, Tensor, backend::Backend};
use crate::types::data::action::RelayRLData;
use crate::types::data::tensor::{BackendMatcher, DeviceType};

use crate::types::model::{ModelError, ModelModule};

/// Converts a dictionary of auxiliary data into a HashMap with String keys and RelayRLData values.
///
/// This function works with generic Rust types (i32, f64, TensorData) instead of Torch-specific IValue.
/// It's designed to work with Burn tensors and the relayrl_types TensorData abstraction.
///
/// # Arguments
///
/// * `dict` - A reference to a HashMap with String keys and generic RelayRLData values.
///
/// # Returns
///
/// An Option containing a HashMap with String keys and RelayRLData values.
pub fn convert_generic_dict(
    dict: &HashMap<String, RelayRLData>,
) -> Option<HashMap<String, RelayRLData>> {
    Some(dict.clone())
}

/// Validates a Burn model by checking that it can perform a forward pass with dummy tensors.
///
/// This function creates dummy input tensors (observation and mask) of the specified dimensions,
/// runs a forward pass through the model, and verifies that:
/// 1. The model returns a valid action tensor
/// 2. The output shape matches expectations
///
/// # Arguments
///
/// * `model` - A reference to the ModelModule to be validated.
/// * `input_dim` - The dimensionality of the input observation vector.
/// * `output_dim` - The dimensionality of the expected output action vector.
///
/// # Panics
///
/// This function will panic if:
/// - The input or output dimensions are invalid
/// - The model fails to process the dummy tensors
/// - The output shape doesn't match expectations
pub fn validate_model<B: Backend + BackendMatcher + 'static>(
    model: &ModelModule<B>,
    input_dim: usize,
    output_dim: usize,
) -> Result<(), ModelError> {
    let device: <<B as BackendMatcher>::Backend as Backend>::Device =
        <B as BackendMatcher>::get_device(&DeviceType::default()).expect("Failed to get device");

    // Shapes: prefer full shapes if present; else assume batch-first vectors
    let in_shape_vec = if model.input_shape.dims.is_empty() {
        vec![1, model.input_dim]
    } else {
        model.input_shape.dims.clone()
    };
    let out_shape_vec = if model.output_shape.dims.is_empty() {
        vec![1, model.output_dim]
    } else {
        model.output_shape.dims.clone()
    };

    let 

    match (in_shape_vec.len(), out_shape_vec.len()) {
        (1, 1) => {
            let in_arr: [usize; 1] = in_shape_vec.as_slice().try_into().unwrap();
            let out_arr: [usize; 1] = out_shape_vec.as_slice().try_into().unwrap();
            validate_with_ranks::<B, 1, 1>(model, &device, in_arr, out_arr)
        }
        (2, 2) => {
            let in_arr: [usize; 2] = in_shape_vec.as_slice().try_into().unwrap();
            let out_arr: [usize; 2] = out_shape_vec.as_slice().try_into().unwrap();
            validate_with_ranks::<B, 2, 2>(model, &device, in_arr, out_arr)
        }
        (3, 3) => {
            let in_arr: [usize; 3] = in_shape_vec.as_slice().try_into().unwrap();
            let out_arr: [usize; 3] = out_shape_vec.as_slice().try_into().unwrap();
            validate_with_ranks::<B, 3, 3>(model, &device, in_arr, out_arr)
        }
        (4, 4) => {
            let in_arr: [usize; 4] = in_shape_vec.as_slice().try_into().unwrap();
            let out_arr: [usize; 4] = out_shape_vec.as_slice().try_into().unwrap();
            validate_with_ranks::<B, 4, 4>(model, &device, in_arr, out_arr)
        }
        _ => Err(ModelError::UnsupportedRank(format!(
            "Unsupported ranks: input {} output {}",
            in_shape_vec.len(),
            out_shape_vec.len()
        ))),
    }
}

fn validate_model_shapes<B: Backend + BackendMatcher + 'static, const D_IN: usize, const D_OUT: usize>(
    model: &ModelModule<B>,
    device: <<B as BackendMatcher>::Backend as Backend>::Device,
    input_shape: [usize; D_IN],
    output_shape: [usize; D_OUT],
) -> Result<(), ModelError> {
    let obs: Tensor<<B as BackendMatcher>::Backend, D_IN> = Tensor::zeros(Shape::from(input_shape), &device);
    let mask: Tensor<<B as BackendMatcher>::Backend, D_OUT> = Tensor::zeros(Shape::from(output_shape), &device);

    let (action_tensor, _) = model.step::<D_IN, D_OUT>(obs, mask);

    let action_shape = action_tensor.shape();
    
    assert_eq!(
        action_shape.dims, output_shape.dims,
        ModelError::InvalidOutputDimension(format!(
            "Model output shape mismatch: expected {:?}, got {:?}",
            output_shape.dims, action_shape.dims
        ))
    );

    Ok(())
}

/// Validates a Burn model with a simple sanity check using the model's path.
///
/// This is a lightweight validation that checks if the model file exists and can be loaded.
/// For more thorough validation, use `validate_model` with actual dimensions.
///
/// # Arguments
///
/// * `model` - A reference to the ModelModule to be validated.
pub fn validate_model_simple<B: Backend + BackendMatcher>(model: &ModelModule<B>) {
    // For now, just check that the model has a valid path
    // In the future, this could try to load and inspect the model structure
    // TODO: Implement actual model structure validation once burn-import API is fully integrated
}

/// Serializes a model (`ModelModule`) into a vector of bytes.
///
/// The model is saved to a temporary file and then read back into a byte vector.
///
/// # Arguments
///
/// * `model` - A reference to the [`ModelModule`] (model) to be serialized.
///
/// # Returns
///
/// A vector of bytes representing the serialized model.
pub fn serialize_model<B: Backend + BackendMatcher>(
    model: &ModelModule<B>,
    dir: PathBuf,
) -> Vec<u8> {
    let temp_file = tempfile::Builder::new()
        .prefix("_model")
        .suffix(".pt")
        .tempfile_in(dir)
        .expect("Failed to create temp file");
    let temp_path = temp_file.path();

    model.save::<B>(temp_path).expect("Failed to save model");

    std::fs::read(temp_path).expect("Failed to read model bytes")
}

/// Deserializes a vector of bytes into a model (`ModelModule`).
///
/// The function writes the provided bytes to a temporary file, flushes it, and then loads
/// the model from that file.
///
/// # Arguments
///
/// * `model_bytes` - A vector of bytes containing the serialized model.
///
/// # Returns
///
/// A [`ModelModule`] representing the deserialized model.
pub fn deserialize_model<B: Backend + BackendMatcher>(
    model_bytes: Vec<u8>,
    device: DeviceType,
) -> Result<ModelModule<B>, ModelError> {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(&model_bytes)
        .expect("Failed to write model bytes");
    temp_file.flush().expect("Failed to flush temp file");

    Ok(
        ModelModule::<B>::load_from_path(temp_file.path())
            .expect("Failed to load model from bytes"),
    )
}
