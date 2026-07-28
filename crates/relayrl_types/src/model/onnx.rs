//! Schema-agnostic ONNX ingestion for a single tensor input / single tensor output graph.
//!
//! RelayRL never hard-codes ONNX I/O names. Instead, [`OnnxSignature::introspect`] discovers
//! the real input/output names, element types, and shapes directly from the loaded
//! [`Session`], and [`validate_metadata_against_signature`] checks the RelayRL-supplied
//! [`ModelMetadata`] against that discovered signature at construction time. Graphs that
//! expose more than one input/output, or any non-tensor I/O (sequences, maps, optionals),
//! are rejected up front instead of failing later inside `Session::run`.

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Outlet, Value as OrtValue, ValueType};
use ort::value::{IntoTensorElementType, PrimitiveTensorElementType, TensorElementType};

#[cfg(feature = "tch-backend")]
use half::bf16;
use half::f16;

#[cfg(feature = "ndarray-backend")]
use crate::data::tensor::NdArrayDType;
#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
use crate::data::tensor::{DType, TensorData};

use super::{ModelError, ModelMetadata};

/// Discovered name, element type, and shape of a single ONNX tensor input or output.
///
/// `dims` mirrors ONNX Runtime's convention: a dynamic/symbolic dimension is reported as `-1`.
#[derive(Debug, Clone)]
pub struct OnnxTensorIo {
    pub name: String,
    pub element_type: TensorElementType,
    pub dims: Vec<i64>,
    pub symbols: Vec<String>,
}

/// The single-input/single-output tensor signature of an ONNX graph, discovered at load time.
#[derive(Debug, Clone)]
pub struct OnnxSignature {
    pub input: OnnxTensorIo,
    pub output: OnnxTensorIo,
}

impl OnnxSignature {
    /// Introspects `session`'s declared inputs/outputs.
    ///
    /// Fails if the graph does not expose exactly one tensor input and one tensor output.
    pub fn introspect(session: &Session) -> Result<Self, ModelError> {
        let inputs = session.inputs();
        let outputs = session.outputs();

        if inputs.len() != 1 {
            return Err(ModelError::UnsupportedModelType(format!(
                "ONNX graphs must declare exactly one input to be usable with RelayRL; found {} ({})",
                inputs.len(),
                describe_names(inputs)
            )));
        }
        if outputs.len() != 1 {
            return Err(ModelError::UnsupportedModelType(format!(
                "ONNX graphs must declare exactly one output to be usable with RelayRL; found {} ({})",
                outputs.len(),
                describe_names(outputs)
            )));
        }

        Ok(Self {
            input: describe_tensor_io(inputs[0].name(), inputs[0].dtype())?,
            output: describe_tensor_io(outputs[0].name(), outputs[0].dtype())?,
        })
    }
}

fn describe_names(outlets: &[Outlet]) -> String {
    outlets
        .iter()
        .map(Outlet::name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_tensor_io(name: &str, dtype: &ValueType) -> Result<OnnxTensorIo, ModelError> {
    match dtype {
        ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } => Ok(OnnxTensorIo {
            name: name.to_string(),
            element_type: *ty,
            dims: shape.to_vec(),
            symbols: dimension_symbols.to_vec(),
        }),
        other => Err(ModelError::UnsupportedModelType(format!(
            "ONNX I/O '{name}' must be a tensor to be usable with RelayRL; found {other:?} \
             (sequences, maps, and optional values are not supported)"
        ))),
    }
}

/// Builds an ONNX Runtime session from a file and introspects its single-input/single-output signature.
pub(crate) fn commit_and_introspect_from_file(
    path: &Path,
) -> Result<(Arc<Mutex<Session>>, OnnxSignature), ModelError> {
    let session = Session::builder()
        .map_err(|err| ModelError::BackendError(err.to_string()))?
        .commit_from_file(path)
        .map_err(|err| ModelError::BackendError(err.to_string()))?;
    let signature = OnnxSignature::introspect(&session)?;
    Ok((Arc::new(Mutex::new(session)), signature))
}

/// Builds an ONNX Runtime session from in-memory bytes and introspects its signature.
pub(crate) fn commit_and_introspect_from_memory(
    bytes: &[u8],
) -> Result<(Arc<Mutex<Session>>, OnnxSignature), ModelError> {
    let session = Session::builder()
        .map_err(|err| ModelError::BackendError(err.to_string()))?
        .commit_from_memory(bytes)
        .map_err(|err| ModelError::BackendError(err.to_string()))?;
    let signature = OnnxSignature::introspect(&session)?;
    Ok((Arc::new(Mutex::new(session)), signature))
}

/// Maps a RelayRL `DType` to the ONNX Runtime tensor element type it must bind to.
pub fn onnx_element_type_for(dtype: &DType) -> TensorElementType {
    match dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => TensorElementType::Float16,
            NdArrayDType::F32 => TensorElementType::Float32,
            NdArrayDType::F64 => TensorElementType::Float64,
            NdArrayDType::I8 => TensorElementType::Int8,
            NdArrayDType::I16 => TensorElementType::Int16,
            NdArrayDType::I32 => TensorElementType::Int32,
            NdArrayDType::I64 => TensorElementType::Int64,
            NdArrayDType::Bool => TensorElementType::Bool,
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => TensorElementType::Float16,
            TchDType::Bf16 => TensorElementType::Bfloat16,
            TchDType::F32 => TensorElementType::Float32,
            TchDType::F64 => TensorElementType::Float64,
            TchDType::I8 => TensorElementType::Int8,
            TchDType::I16 => TensorElementType::Int16,
            TchDType::I32 => TensorElementType::Int32,
            TchDType::I64 => TensorElementType::Int64,
            TchDType::U8 => TensorElementType::Uint8,
            TchDType::Bool => TensorElementType::Bool,
        },
    }
}

/// Validates that `metadata`'s declared dtypes/shapes are compatible with the ONNX graph's
/// discovered `signature`. Element types must match exactly; every *fixed* graph dimension
/// (i.e. not `-1`) must equal the corresponding metadata dimension, while dynamic dimensions
/// accept any metadata value.
pub fn validate_metadata_against_signature(
    metadata: &ModelMetadata,
    signature: &OnnxSignature,
) -> Result<(), ModelError> {
    validate_tensor_io(
        "input",
        &metadata.input_dtype,
        &metadata.input_shape,
        &signature.input,
    )?;
    validate_tensor_io(
        "output",
        &metadata.output_dtype,
        &metadata.output_shape,
        &signature.output,
    )?;
    Ok(())
}

fn validate_tensor_io(
    label: &str,
    dtype: &DType,
    shape: &[usize],
    io: &OnnxTensorIo,
) -> Result<(), ModelError> {
    let expected_ty = onnx_element_type_for(dtype);
    if expected_ty != io.element_type {
        return Err(ModelError::DTypeError(format!(
            "ONNX {label} '{}' has element type {}, but metadata declares {} (maps to {})",
            io.name, io.element_type, dtype, expected_ty
        )));
    }

    if shape.len() != io.dims.len() {
        return Err(ModelError::UnsupportedRank(format!(
            "ONNX {label} '{}' has rank {}, but metadata declares rank {}",
            io.name,
            io.dims.len(),
            shape.len()
        )));
    }

    for (axis, (&meta_dim, &graph_dim)) in shape.iter().zip(io.dims.iter()).enumerate() {
        if graph_dim >= 0 && graph_dim as usize != meta_dim {
            return Err(ModelError::InvalidMetadata(format!(
                "ONNX {label} '{}' has fixed dimension {axis} = {graph_dim}, but metadata declares {meta_dim}",
                io.name
            )));
        }
    }

    Ok(())
}

/// Runs the graph's single input → single output tensor pass, binding by the discovered
/// input/output names and returning the actual runtime output shape (important for graphs
/// with dynamic dimensions, e.g. a symbolic batch axis).
pub(crate) fn run_tensor(
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
    input_dtype: &DType,
    output_dtype: &DType,
    shape: &[usize],
    data: &[u8],
) -> Result<TensorData, ModelError> {
    let (out_shape, out_bytes) =
        dispatch_input(input_dtype, output_dtype, shape, data, session, signature)?;
    Ok(TensorData::new(
        out_shape,
        output_dtype.clone(),
        out_bytes,
        TensorData::get_backend_from_dtype(output_dtype),
    ))
}

fn dispatch_input(
    input_dtype: &DType,
    output_dtype: &DType,
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError> {
    match input_dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => {
                dispatch_output::<f16>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::F32 => {
                dispatch_output::<f32>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::F64 => {
                dispatch_output::<f64>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::I8 => {
                dispatch_output::<i8>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::I16 => {
                dispatch_output::<i16>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::I32 => {
                dispatch_output::<i32>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::I64 => {
                dispatch_output::<i64>(output_dtype, shape, data, session, signature)
            }
            NdArrayDType::Bool => {
                dispatch_output_bool_in(output_dtype, shape, data, session, signature)
            }
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => dispatch_output::<f16>(output_dtype, shape, data, session, signature),
            TchDType::Bf16 => {
                dispatch_output::<bf16>(output_dtype, shape, data, session, signature)
            }
            TchDType::F32 => dispatch_output::<f32>(output_dtype, shape, data, session, signature),
            TchDType::F64 => dispatch_output::<f64>(output_dtype, shape, data, session, signature),
            TchDType::I8 => dispatch_output::<i8>(output_dtype, shape, data, session, signature),
            TchDType::I16 => dispatch_output::<i16>(output_dtype, shape, data, session, signature),
            TchDType::I32 => dispatch_output::<i32>(output_dtype, shape, data, session, signature),
            TchDType::I64 => dispatch_output::<i64>(output_dtype, shape, data, session, signature),
            TchDType::U8 => dispatch_output::<u8>(output_dtype, shape, data, session, signature),
            TchDType::Bool => {
                dispatch_output_bool_in(output_dtype, shape, data, session, signature)
            }
        },
    }
}

fn dispatch_output<IN>(
    output_dtype: &DType,
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError>
where
    IN: IntoTensorElementType + PrimitiveTensorElementType + Debug + Clone + bytemuck::Pod,
{
    match output_dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => convert_pod::<IN, f16>(shape, data, session, signature),
            NdArrayDType::F32 => convert_pod::<IN, f32>(shape, data, session, signature),
            NdArrayDType::F64 => convert_pod::<IN, f64>(shape, data, session, signature),
            NdArrayDType::I8 => convert_pod::<IN, i8>(shape, data, session, signature),
            NdArrayDType::I16 => convert_pod::<IN, i16>(shape, data, session, signature),
            NdArrayDType::I32 => convert_pod::<IN, i32>(shape, data, session, signature),
            NdArrayDType::I64 => convert_pod::<IN, i64>(shape, data, session, signature),
            NdArrayDType::Bool => convert_bool_out::<IN>(shape, data, session, signature),
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => convert_pod::<IN, f16>(shape, data, session, signature),
            TchDType::Bf16 => convert_pod::<IN, bf16>(shape, data, session, signature),
            TchDType::F32 => convert_pod::<IN, f32>(shape, data, session, signature),
            TchDType::F64 => convert_pod::<IN, f64>(shape, data, session, signature),
            TchDType::I8 => convert_pod::<IN, i8>(shape, data, session, signature),
            TchDType::I16 => convert_pod::<IN, i16>(shape, data, session, signature),
            TchDType::I32 => convert_pod::<IN, i32>(shape, data, session, signature),
            TchDType::I64 => convert_pod::<IN, i64>(shape, data, session, signature),
            TchDType::U8 => convert_pod::<IN, u8>(shape, data, session, signature),
            TchDType::Bool => convert_bool_out::<IN>(shape, data, session, signature),
        },
    }
}

fn dispatch_output_bool_in(
    output_dtype: &DType,
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError> {
    match output_dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => convert_bool_in::<f16>(shape, data, session, signature),
            NdArrayDType::F32 => convert_bool_in::<f32>(shape, data, session, signature),
            NdArrayDType::F64 => convert_bool_in::<f64>(shape, data, session, signature),
            NdArrayDType::I8 => convert_bool_in::<i8>(shape, data, session, signature),
            NdArrayDType::I16 => convert_bool_in::<i16>(shape, data, session, signature),
            NdArrayDType::I32 => convert_bool_in::<i32>(shape, data, session, signature),
            NdArrayDType::I64 => convert_bool_in::<i64>(shape, data, session, signature),
            NdArrayDType::Bool => convert_bool_bool(shape, data, session, signature),
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => convert_bool_in::<f16>(shape, data, session, signature),
            TchDType::Bf16 => convert_bool_in::<bf16>(shape, data, session, signature),
            TchDType::F32 => convert_bool_in::<f32>(shape, data, session, signature),
            TchDType::F64 => convert_bool_in::<f64>(shape, data, session, signature),
            TchDType::I8 => convert_bool_in::<i8>(shape, data, session, signature),
            TchDType::I16 => convert_bool_in::<i16>(shape, data, session, signature),
            TchDType::I32 => convert_bool_in::<i32>(shape, data, session, signature),
            TchDType::I64 => convert_bool_in::<i64>(shape, data, session, signature),
            TchDType::U8 => convert_bool_in::<u8>(shape, data, session, signature),
            TchDType::Bool => convert_bool_bool(shape, data, session, signature),
        },
    }
}

/// Locks the session, feeds `session_input` under the discovered input name, runs the graph,
/// and hands the discovered output value to `extract`.
fn run_and_extract<F>(
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
    session_input: SessionInputValue<'_>,
    extract: F,
) -> Result<(Vec<usize>, Vec<u8>), ModelError>
where
    F: FnOnce(&DynValue) -> Result<(Vec<usize>, Vec<u8>), ModelError>,
{
    let mut feeds = HashMap::with_capacity(1);
    feeds.insert(signature.input.name.clone(), session_input);

    let mut session_guard = session
        .lock()
        .map_err(|e| ModelError::BackendError(format!("Failed to lock ONNX session: {e}")))?;

    let outputs = session_guard
        .run(feeds)
        .map_err(|e| ModelError::BackendError(format!("ONNX session run failed: {e}")))?;

    let value = outputs.get(signature.output.name.as_str()).ok_or_else(|| {
        ModelError::BackendError(format!(
            "ONNX session did not produce an output named '{}'",
            signature.output.name
        ))
    })?;

    extract(value)
}

fn extract_pod<OUT>(
    value: &DynValue,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError>
where
    OUT: IntoTensorElementType + PrimitiveTensorElementType + Debug + Clone + bytemuck::Pod,
{
    let (out_shape, out_slice) = value.try_extract_tensor::<OUT>().map_err(|e| {
        ModelError::BackendError(format!(
            "Failed to extract ONNX output '{}': {e:?}",
            signature.output.name
        ))
    })?;
    let shape_usize = out_shape.iter().map(|&d| d as usize).collect();
    Ok((shape_usize, bytemuck::cast_slice(out_slice).to_vec()))
}

fn extract_bool(
    value: &DynValue,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError> {
    let (out_shape, out_slice) = value.try_extract_tensor::<bool>().map_err(|e| {
        ModelError::BackendError(format!(
            "Failed to extract ONNX output '{}': {e:?}",
            signature.output.name
        ))
    })?;
    let shape_usize = out_shape.iter().map(|&d| d as usize).collect();
    let bytes: Vec<u8> = out_slice.iter().map(|&b| u8::from(b)).collect();
    Ok((shape_usize, bytes))
}

fn convert_pod<IN, OUT>(
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError>
where
    IN: IntoTensorElementType + PrimitiveTensorElementType + Debug + Clone + bytemuck::Pod,
    OUT: IntoTensorElementType + PrimitiveTensorElementType + Debug + Clone + bytemuck::Pod,
{
    let typed_in: Vec<IN> = bytemuck::cast_slice::<u8, IN>(data).to_vec();
    let ort_shape = ort::value::Shape::from(shape);
    let ort_value = OrtValue::from_array((ort_shape, typed_in))
        .map_err(|e| ModelError::BackendError(format!("Failed to build ONNX input tensor: {e}")))?;
    let session_input = SessionInputValue::from(ort_value);

    run_and_extract(session, signature, session_input, |value| {
        extract_pod::<OUT>(value, signature)
    })
}

fn convert_bool_in<OUT>(
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError>
where
    OUT: IntoTensorElementType + PrimitiveTensorElementType + Debug + Clone + bytemuck::Pod,
{
    let bool_data: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    let ort_shape = ort::value::Shape::from(shape);
    let ort_value = OrtValue::from_array((ort_shape, bool_data))
        .map_err(|e| ModelError::BackendError(format!("Failed to build ONNX input tensor: {e}")))?;
    let session_input = SessionInputValue::from(ort_value);

    run_and_extract(session, signature, session_input, |value| {
        extract_pod::<OUT>(value, signature)
    })
}

fn convert_bool_out<IN>(
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError>
where
    IN: IntoTensorElementType + PrimitiveTensorElementType + Debug + Clone + bytemuck::Pod,
{
    let typed_in: Vec<IN> = bytemuck::cast_slice::<u8, IN>(data).to_vec();
    let ort_shape = ort::value::Shape::from(shape);
    let ort_value = OrtValue::from_array((ort_shape, typed_in))
        .map_err(|e| ModelError::BackendError(format!("Failed to build ONNX input tensor: {e}")))?;
    let session_input = SessionInputValue::from(ort_value);

    run_and_extract(session, signature, session_input, |value| {
        extract_bool(value, signature)
    })
}

fn convert_bool_bool(
    shape: &[usize],
    data: &[u8],
    session: &Arc<Mutex<Session>>,
    signature: &OnnxSignature,
) -> Result<(Vec<usize>, Vec<u8>), ModelError> {
    let bool_data: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    let ort_shape = ort::value::Shape::from(shape);
    let ort_value = OrtValue::from_array((ort_shape, bool_data))
        .map_err(|e| ModelError::BackendError(format!("Failed to build ONNX input tensor: {e}")))?;
    let session_input = SessionInputValue::from(ort_value);

    run_and_extract(session, signature, session_input, |value| {
        extract_bool(value, signature)
    })
}
