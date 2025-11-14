use std::collections::HashMap;
use std::convert::TryInto;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use crate::types::data::action::RelayRLData;
use crate::types::data::tensor::{
    AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
    IntBurnTensor, NdArrayDType, TchDType,
};
use burn_tensor::{Shape, Tensor, backend::Backend};

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
/// This function creates dummy tensors whose shapes match the metadata, runs a forward pass, and
/// verifies that the produced action tensor matches the expected shape. Supports input and output
/// ranks from 1 to 8 (independently).
pub fn validate_module<B: Backend + BackendMatcher<Backend = B> + 'static>(
    module: &ModelModule<B>,
) -> Result<(), ModelError> {
    let device = module.resolve_device();

    let input_shape = &module.metadata.input_shape;
    let output_shape = &module.metadata.output_shape;

    if !(1..=8).contains(&input_shape.len()) || !(1..=8).contains(&output_shape.len()) {
        return Err(ModelError::UnsupportedRank(format!(
            "Unsupported ranks: input {} output {}",
            input_shape.len(),
            output_shape.len()
        )));
    }

    match input_shape.len() {
        1 => validate_with_input::<B, 1>(module, &device, input_shape, output_shape),
        2 => validate_with_input::<B, 2>(module, &device, input_shape, output_shape),
        3 => validate_with_input::<B, 3>(module, &device, input_shape, output_shape),
        4 => validate_with_input::<B, 4>(module, &device, input_shape, output_shape),
        5 => validate_with_input::<B, 5>(module, &device, input_shape, output_shape),
        6 => validate_with_input::<B, 6>(module, &device, input_shape, output_shape),
        7 => validate_with_input::<B, 7>(module, &device, input_shape, output_shape),
        8 => validate_with_input::<B, 8>(module, &device, input_shape, output_shape),
        9 => validate_with_input::<B, 9>(module, &device, input_shape, output_shape),
        _ => unreachable!(),
    }
}

fn validate_with_input<B: Backend + BackendMatcher<Backend = B> + 'static, const D_IN: usize>(
    module: &ModelModule<B>,
    device: &<B as Backend>::Device,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<(), ModelError> {
    match output_shape.len() {
        1 => call_validate::<B, D_IN, 1>(module, device, input_shape, output_shape),
        2 => call_validate::<B, D_IN, 2>(module, device, input_shape, output_shape),
        3 => call_validate::<B, D_IN, 3>(module, device, input_shape, output_shape),
        4 => call_validate::<B, D_IN, 4>(module, device, input_shape, output_shape),
        5 => call_validate::<B, D_IN, 5>(module, device, input_shape, output_shape),
        6 => call_validate::<B, D_IN, 6>(module, device, input_shape, output_shape),
        7 => call_validate::<B, D_IN, 7>(module, device, input_shape, output_shape),
        8 => call_validate::<B, D_IN, 8>(module, device, input_shape, output_shape),
        9 => call_validate::<B, D_IN, 9>(module, device, input_shape, output_shape),
        _ => Err(ModelError::UnsupportedRank(format!(
            "Unsupported ranks: input {} output {}",
            input_shape.len(),
            output_shape.len()
        ))),
    }
}

fn call_validate<
    B: Backend + BackendMatcher<Backend = B> + 'static,
    const D_IN: usize,
    const D_OUT: usize,
>(
    module: &ModelModule<B>,
    device: &<B as Backend>::Device,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<(), ModelError> {
    let input_array: [usize; D_IN] = slice_to_array::<D_IN>(input_shape)?;
    let output_array: [usize; D_OUT] = slice_to_array::<D_OUT>(output_shape)?;
    let input_shape = Shape::from(input_array);
    let output_shape = Shape::from(output_array);

    validate_model_shapes::<B, D_IN, D_OUT>(module, device, &input_shape, &output_shape)
}

fn slice_to_array<const N: usize>(shape: &[usize]) -> Result<[usize; N], ModelError> {
    shape.try_into().map_err(|_| {
        ModelError::InvalidMetadata(format!(
            "Expected dimension of length {N}, but got {}",
            shape.len()
        ))
    })
}

fn validate_model_shapes<
    B: Backend + BackendMatcher<Backend = B> + 'static,
    const D_IN: usize,
    const D_OUT: usize,
>(
    module: &ModelModule<B>,
    device: &<B as Backend>::Device,
    input_shape: &Shape,
    output_shape: &Shape,
) -> Result<(), ModelError> {
    let obs: AnyBurnTensor<B, D_IN> = match &module.metadata.input_dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => AnyBurnTensor::Float(
                FloatBurnTensor::empty(input_shape, &DType::NdArray(nd.clone()), device),
            ),
            NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                AnyBurnTensor::Int(IntBurnTensor::empty(
                    input_shape,
                    &DType::NdArray(nd.clone()),
                    device,
                ))
            }
            NdArrayDType::Bool => AnyBurnTensor::Bool(BoolBurnTensor::empty(
                input_shape,
                &DType::NdArray(nd.clone()),
                device,
            )),
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => AnyBurnTensor::Float(
                FloatBurnTensor::empty(input_shape, &DType::Tch(tch.clone()), device),
            ),
            TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                AnyBurnTensor::Int(IntBurnTensor::empty(
                    input_shape,
                    &DType::Tch(tch.clone()),
                    device,
                ))
            }
            TchDType::Bool => AnyBurnTensor::Bool(BoolBurnTensor::empty(
                input_shape,
                &DType::Tch(tch.clone()),
                device,
            )),
        },
    };

    let mask: AnyBurnTensor<B, D_OUT> = match &module.metadata.output_dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => AnyBurnTensor::Float(
                FloatBurnTensor::empty(output_shape, &DType::NdArray(nd.clone()), device),
            ),
            NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                AnyBurnTensor::Int(IntBurnTensor::empty(
                    output_shape,
                    &DType::NdArray(nd.clone()),
                    device,
                ))
            }
            NdArrayDType::Bool => AnyBurnTensor::Bool(BoolBurnTensor::empty(
                output_shape,
                &DType::NdArray(nd.clone()),
                device,
            )),
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => AnyBurnTensor::Float(
                FloatBurnTensor::empty(output_shape, &DType::Tch(tch.clone()), device),
            ),
            TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                AnyBurnTensor::Int(IntBurnTensor::empty(
                    output_shape,
                    &DType::Tch(tch.clone()),
                    device,
                ))
            }
            TchDType::Bool => AnyBurnTensor::Bool(BoolBurnTensor::empty(
                output_shape,
                &DType::Tch(tch.clone()),
                device,
            )),
        },
    };

    let (action_tensor, _, _) = module.step::<D_IN, D_OUT>(obs, Some(mask));

    let action_dims: &Vec<usize> = &action_tensor.shape;
    let output_dims: &Vec<usize> = &output_shape.dims;

    for (a, o) in action_dims.iter().zip(output_dims.iter()) {
        if *a != *o {
            return Err(ModelError::InvalidOutputDimension(format!(
                "Model output shape mismatch: expected {:?}, got {:?}",
                output_dims, action_dims
            )));
        }
    }
    Ok(())
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
pub fn serialize_model_module<B: Backend + BackendMatcher<Backend = B>>(
    model: &ModelModule<B>,
    dir: PathBuf,
) -> Vec<u8> {
    let temp_file = tempfile::Builder::new()
        .prefix("_model")
        .suffix(".pt")
        .tempfile_in(dir)
        .expect("Failed to create temp file");
    let temp_path = temp_file.path();

    ModelModule::<B>::save(model, temp_path).expect("Failed to save model");
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
pub fn deserialize_model_module<B: Backend + BackendMatcher<Backend = B>>(
    model_bytes: Vec<u8>,
    _device: DeviceType,
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
