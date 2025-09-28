//! This module provides functionality for converting between internal RL4Sys tensor/action types
//! and their serialized representations using safetensors. It also defines error types and conversion
//! functions to support these operations, as well as integration with Python via pyo3.

use bytemuck::cast_slice;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensorError as tensorerror, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::{From, TryFrom};
use std::fmt;
use std::fmt::Debug;
use tch::{Device, Kind, Tensor};

#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::PyRL4SysAction;
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use pyo3::{Bound, PyAny};

/// An enum representing errors that may occur when working with safetensors.
///
/// This enum provides variants for errors coming from safetensors, bytemuck conversions,
/// unsupported data types, serialization issues, and tensor conversion errors.
#[derive(Debug)]
pub enum SafeTensorError {
    SafeTensorError(tensorerror),
    BytemuckError(String),
    UnsupportedDType(String),
    SerializationError(String),
    TensorConversionError(String),
    // Add other error variants as needed
}

/// Implements the std::error::Error trait for SafeTensorError.
impl std::error::Error for SafeTensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SafeTensorError::SafeTensorError(e) => Some(e),
            SafeTensorError::BytemuckError(_) => None,
            SafeTensorError::UnsupportedDType(_) => None,
            SafeTensorError::SerializationError(_) => None,
            SafeTensorError::TensorConversionError(_) => None,
        }
    }
}

/// Implements Display for SafeTensorError to provide formatted error messages.
impl fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafeTensorError::SafeTensorError(e) => write!(f, "SafeTensorError: {}", e),
            SafeTensorError::BytemuckError(e) => write!(f, "BytemuckError: {}", e),
            SafeTensorError::UnsupportedDType(e) => write!(f, "UnsupportedDType: {}", e),
            SafeTensorError::SerializationError(e) => write!(f, "SerializationError: {}", e),
            SafeTensorError::TensorConversionError(e) => write!(f, "TensorConversionError: {}", e),
        }
    }
}

/// Converts a safetensors error into a SafeTensorError.
impl From<tensorerror> for SafeTensorError {
    fn from(error: tensorerror) -> Self {
        SafeTensorError::SafeTensorError(error)
    }
}

/// Converts a bytemuck PodCastError into a SafeTensorError.
impl From<bytemuck::PodCastError> for SafeTensorError {
    fn from(error: bytemuck::PodCastError) -> Self {
        SafeTensorError::BytemuckError(error.to_string())
    }
}

/// An enum representing supported data types for tensor serialization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DType {
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    Bool,
}

/// Implements Display for DType to format the enum as a string.
impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Byte => write!(f, "Byte"),
            DType::Short => write!(f, "Short"),
            DType::Int => write!(f, "Int"),
            DType::Long => write!(f, "Long"),
            DType::Float => write!(f, "Float"),
            DType::Double => write!(f, "Double"),
            DType::Bool => write!(f, "Bool"),
        }
    }
}

/// Converts our DType enum to the safetensors library Dtype.
impl From<DType> for safetensors::tensor::Dtype {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::Byte => safetensors::tensor::Dtype::U8,
            DType::Short => safetensors::tensor::Dtype::I16,
            DType::Int => safetensors::tensor::Dtype::I32,
            DType::Long => safetensors::tensor::Dtype::I64,
            DType::Float => safetensors::tensor::Dtype::F32,
            DType::Double => safetensors::tensor::Dtype::F64,
            DType::Bool => safetensors::tensor::Dtype::U8,
        }
    }
}

/// Attempts to convert a string slice to a DType enum.
impl TryFrom<&str> for DType {
    type Error = SafeTensorError;

    fn try_from(dtype: &str) -> Result<Self, Self::Error> {
        match dtype {
            "Byte" => Ok(DType::Byte),
            "Short" => Ok(DType::Short),
            "Int" => Ok(DType::Int),
            "Long" => Ok(DType::Long),
            "Float" => Ok(DType::Float),
            "Double" => Ok(DType::Double),
            "Bool" => Ok(DType::Bool),
            _ => Err(SafeTensorError::UnsupportedDType(
                dtype.parse().expect("dtype parse error"),
            )),
        }
    }
}

/// Converts a safetensors Dtype into our DType enum.
impl TryFrom<safetensors::tensor::Dtype> for DType {
    type Error = SafeTensorError;

    fn try_from(dtype: safetensors::tensor::Dtype) -> Result<Self, Self::Error> {
        match dtype {
            safetensors::tensor::Dtype::U8 => Ok(DType::Byte),
            safetensors::tensor::Dtype::I16 => Ok(DType::Short),
            safetensors::tensor::Dtype::I32 => Ok(DType::Int),
            safetensors::tensor::Dtype::I64 => Ok(DType::Long),
            safetensors::tensor::Dtype::F32 => Ok(DType::Float),
            safetensors::tensor::Dtype::F64 => Ok(DType::Double),
            safetensors::tensor::Dtype::BOOL => Ok(DType::Bool),
            _ => Err(SafeTensorError::UnsupportedDType(format!("{:?}", dtype))),
        }
    }
}

/// Converts a tch::Kind enum into its string representation.
///
/// # Arguments
///
/// * `kind` - A tch::Kind value.
///
/// # Returns
///
/// A String representing the kind.
fn kind_to_string(kind: Kind) -> String {
    match kind {
        Kind::Uint8 => "Uint8".to_string(),
        Kind::Int16 => "Int16".to_string(),
        Kind::Int => "Int32".to_string(),
        Kind::Int64 => "Int64".to_string(),
        Kind::Float => "Float".to_string(),
        Kind::Double => "Double".to_string(),
        Kind::Bool => "Bool".to_string(),
        _ => "Unsupported".to_string(),
    }
}

/// A struct representing serialized tensor data and its attributes.
///
/// This structure holds the tensor shape, data type, and the serialized data bytes.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorData {
    shape: Vec<i64>,
    dtype: DType,
    pub(crate) data: Vec<u8>,
}

/// An enum representing various types of data that may be associated with a tensor.
///
/// This enum includes variants for tensors (as [TensorData]) and for common primitive data types.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum RL4SysData {
    Tensor(TensorData),
    Byte(u8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    String(String),
    Bool(bool),
    // Add other variants if ever needed
}

/// Implements conversion from a Python dictionary (PyDict) bound reference representing tensor data
/// into a [TensorData] Rust object.
///
/// The dictionary is expected to have the keys "shape", "dtype", and "data".
///
/// # Arguments
///
/// * `tensor_data_dict` - A bound reference to a PyDict containing tensor data.
///
/// # Returns
///
/// A [Result] containing the constructed [TensorData] on success, or a [SafeTensorError] on failure.
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
impl TryFrom<Bound<'_, PyDict>> for TensorData {
    type Error = SafeTensorError;

    fn try_from(tensor_data_dict: Bound<'_, PyDict>) -> Result<Self, SafeTensorError> {
        let shape: Bound<PyAny> = tensor_data_dict
            .get_item("shape")
            .expect("`shape` key not found")
            .expect("`shape` value not found");
        let shape_i64: Vec<i64> = shape
            .extract::<Vec<i64>>()
            .expect("failed to extract shape as i64");

        let dtype: Bound<PyAny> = tensor_data_dict
            .get_item("dtype")
            .expect("`dtype` key not found")
            .expect("`dtype` value not found");
        let dtype_converted: DType = DType::try_from(
            dtype
                .extract::<String>()
                .expect("failed to extract dtype as string")
                .as_str(),
        )?;

        let data: Bound<PyAny> = tensor_data_dict
            .get_item("data")
            .expect("`data` key not found")
            .expect("`data` value not found");
        let data_bytes: Vec<u8> = data.extract().expect("failed to extract data as bytes");

        Ok(TensorData {
            shape: shape_i64,
            dtype: dtype_converted,
            data: data_bytes,
        })
    }
}

/// Implements conversion from a tch::Tensor reference to a [TensorData] object.
///
/// The tensor is moved to the CPU, made contiguous, and its shape and data are extracted.
/// Safetensors is then used to serialize the tensor into a byte vector.
///
/// # Arguments
///
/// * `tensor` - A reference to a tch::Tensor to be converted.
///
/// # Returns
///
/// A [Result] containing the constructed [TensorData] on success, or a [SafeTensorError] on failure.
impl TryFrom<&Tensor> for TensorData {
    type Error = SafeTensorError;

    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        let tensor = tensor.to_device(Device::Cpu).contiguous();

        let shape_i64 = tensor.size().to_vec();
        let shape: Vec<usize> = shape_i64.iter().map(|&dim| dim as usize).collect();

        let numel = tensor.numel();
        let (dtype, data_bytes) = match tensor.kind() {
            Kind::Float => {
                let mut buffer = vec![0f32; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Float, cast_slice(&buffer).to_vec())
            }
            Kind::Double => {
                let mut buffer = vec![0f64; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Double, cast_slice(&buffer).to_vec())
            }
            Kind::Uint8 => {
                let mut buffer = vec![0u8; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Byte, cast_slice(&buffer).to_vec())
            }
            Kind::Int16 => {
                let mut buffer = vec![0i16; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Short, cast_slice(&buffer).to_vec())
            }
            Kind::Int64 => {
                let mut buffer = vec![0i64; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Long, cast_slice(&buffer).to_vec())
            }
            Kind::Int => {
                let mut buffer = vec![0i32; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Int, cast_slice(&buffer).to_vec())
            }
            Kind::Bool => {
                let mut buffer = vec![0u8; numel];
                tensor.copy_data(&mut buffer, numel);
                (DType::Bool, cast_slice(&buffer).to_vec())
            }
            _ => {
                return Err(SafeTensorError::UnsupportedDType(kind_to_string(
                    tensor.kind(),
                )));
            }
        };

        let dtype_safe: safetensors::tensor::Dtype = dtype.clone().into();

        let tensor_view = TensorView::new(dtype_safe, shape, &data_bytes)
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;

        let bytes = safetensors::serialize([("tensor", &tensor_view)], &None)
            .map_err(|e| SafeTensorError::SerializationError(e.to_string()))?;

        Ok(TensorData {
            shape: shape_i64,
            dtype,
            data: bytes,
        })
    }
}

/// Implements conversion from a [TensorData] object back into a tch::Tensor.
///
/// The safetensors byte data is deserialized to extract the tensor's shape and data,
/// and then a new tch::Tensor is constructed from the extracted data.
///
/// # Arguments
///
/// * `tensor_data` - A [TensorData] instance containing serialized tensor data.
///
/// # Returns
///
/// A [Result] containing the reconstructed tch::Tensor or a [SafeTensorError] on failure.
impl TryFrom<TensorData> for Tensor {
    type Error = SafeTensorError;

    fn try_from(tensor_data: TensorData) -> Result<Self, Self::Error> {
        let tensors: SafeTensors =
            SafeTensors::deserialize(&tensor_data.data).expect("Failed to deserialize");

        let tensor_view: TensorView = tensors
            .tensor("tensor")
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;

        let shape: Vec<i64> = tensor_view.shape().iter().map(|&dim| dim as i64).collect();
        let dtype: Dtype = tensor_view.dtype();
        let data: &[u8] = tensor_view.data();

        let dtype_converted: DType = DType::try_from(dtype)?;

        let tensor: Tensor = match dtype_converted {
            DType::Byte => {
                let slice: &[u8] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
            DType::Short => {
                let slice: &[i16] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
            DType::Int => {
                let slice: &[i32] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
            DType::Long => {
                let slice: &[i64] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
            DType::Float => {
                let slice: &[f32] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
            DType::Double => {
                let slice: &[f64] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
            DType::Bool => {
                let slice: &[u8] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape)
            }
        };

        Ok(tensor.to_device(Device::Cpu).contiguous())
    }
}

/// A marker trait for RL4SysAction, used strictly for type checking.
pub trait RL4SysActionTrait {}

/// The core RL4SysAction struct that represents an action in the RL4Sys framework.
///
/// An RL4SysAction holds optional serialized tensor data for observation, action, and mask,
/// a reward value, optional auxiliary data as a HashMap, and flags indicating whether the action
/// is terminal and whether the reward was updated.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RL4SysAction {
    pub obs: Option<TensorData>,
    pub act: Option<TensorData>,
    pub mask: Option<TensorData>,
    pub rew: f32,
    pub data: Option<HashMap<String, RL4SysData>>,
    pub done: bool,
}

/// Implementation of RL4SysAction.
///
/// This implementation provides:
/// - A constructor for creating a new action,
/// - Getter methods for accessing individual fields,
/// - A setter method for updating the reward,
/// - And methods for converting between raw tensors and serialized actions,
///   as well as converting the action to a Python-compatible representation.
impl RL4SysAction {
    /// Constructs a new [`RL4SysAction`] from the provided parameters.
    ///
    /// # Arguments
    ///
    /// * `obs` - Optional serialized observation as [TensorData].
    /// * `act` - Optional serialized action as [TensorData].
    /// * `mask` - Optional serialized mask as [TensorData].
    /// * `rew` - A float representing the reward.
    /// * `data` - Optional auxiliary data as a HashMap.
    /// * `done` - A boolean indicating if the action terminates an episode.
    /// * `reward_update_flag` - A boolean flag indicating whether the reward was updated.
    ///
    /// # Returns
    ///
    /// A new [`RL4SysAction`] instance.
    pub fn new(
        obs: Option<TensorData>,
        act: Option<TensorData>,
        mask: Option<TensorData>,
        rew: f32,
        data: Option<HashMap<String, RL4SysData>>,
        done: bool,
    ) -> Self {
        RL4SysAction {
            obs,
            act,
            mask,
            rew,
            data,
            done,
        }
    }

    /* ** Getter Functions ** */

    /// Returns a reference to the serialized observation, if available.
    pub fn get_obs(&self) -> Option<&TensorData> {
        self.obs.as_ref()
    }

    /// Returns a reference to the serialized action, if available.
    pub fn get_act(&self) -> Option<&TensorData> {
        self.act.as_ref()
    }

    /// Returns a reference to the serialized mask, if available.
    pub fn get_mask(&self) -> Option<&TensorData> {
        self.mask.as_ref()
    }

    /// Returns the reward value.
    pub fn get_rew(&self) -> f32 {
        self.rew
    }

    /// Returns a reference to the auxiliary data hashmap, if available.
    pub fn get_data(&self) -> Option<&HashMap<String, RL4SysData>> {
        self.data.as_ref()
    }

    /// Returns the done flag indicating whether this action terminates an episode.
    pub fn get_done(&self) -> bool {
        self.done
    }

    /* ** Setter Function ** */

    /// Updates the reward value of the action.
    ///
    /// # Arguments
    ///
    /// * `reward` - The new reward value.
    pub fn update_reward(&mut self, reward: f32) {
        self.rew = reward;
    }
}

/// A marker trait implementation for RL4SysAction used strictly for type checking.
impl RL4SysActionTrait for RL4SysAction {}

/// Implementation of additional methods for RL4SysData.
impl RL4SysData {
    /// Converts a tch::Tensor to an [`RL4SysData::Tensor`] variant by converting it to [TensorData].
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to a tch::Tensor.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing [`RL4SysData::Tensor(TensorData)`] on success, or a [SafeTensorError] on failure.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, SafeTensorError> {
        let tensor_data = TensorData::try_from(tensor)?;
        Ok(RL4SysData::Tensor(tensor_data))
    }
}

/// Additional conversion functions and utilities for RL4SysAction.
impl RL4SysAction {
    /// Converts raw tch::Tensor objects (observation, action, mask) into serialized [TensorData] objects
    /// and constructs an [`RL4SysAction`].
    ///
    /// This function attempts to convert each provided tensor into a [TensorData] object using [RL4SysData::from_tensor].
    /// If a conversion fails, an error is returned.
    ///
    /// # Arguments
    ///
    /// * `obs` - Optional reference to the observation tensor.
    /// * `action` - Optional reference to the action tensor.
    /// * `mask` - Optional reference to the mask tensor.
    /// * `rew` - A float representing the reward.
    /// * `data` - Optional auxiliary data as a HashMap.
    /// * `done` - A boolean flag indicating if this is the terminal action.
    /// * `reward_update_flag` - A boolean flag indicating whether the reward was updated.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the constructed [`RL4SysAction`] on success, or a [SafeTensorError] on failure.
    pub fn from_tensors(
        obs: Option<&Tensor>,
        action: Option<&Tensor>,
        mask: Option<&Tensor>,
        rew: f32,
        data: Option<HashMap<String, RL4SysData>>,
        done: bool,
    ) -> Result<Self, SafeTensorError> {
        let obs_data: Option<RL4SysData> = obs
            .map(RL4SysData::from_tensor)
            .transpose()
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;

        let act_data: Option<RL4SysData> = action
            .map(RL4SysData::from_tensor)
            .transpose()
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;

        let mask_data: Option<RL4SysData> = mask
            .map(RL4SysData::from_tensor)
            .transpose()
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;

        Ok(RL4SysAction {
            obs: obs_data.map(|data| {
                if let RL4SysData::Tensor(tensor_data) = data {
                    tensor_data
                } else {
                    TensorData {
                        shape: vec![],
                        dtype: DType::Float,
                        data: vec![],
                    }
                }
            }),
            act: act_data.map(|data| {
                if let RL4SysData::Tensor(tensor_data) = data {
                    tensor_data
                } else {
                    TensorData {
                        shape: vec![],
                        dtype: DType::Float,
                        data: vec![],
                    }
                }
            }),
            mask: mask_data.map(|data| {
                if let RL4SysData::Tensor(tensor_data) = data {
                    tensor_data
                } else {
                    TensorData {
                        shape: vec![],
                        dtype: DType::Float,
                        data: vec![],
                    }
                }
            }),
            rew,
            data,
            done,
        })
    }

    /// Creates a [TensorData] object from a vector of serialized safetensors bytes.
    ///
    /// This function deserializes the provided bytes using safetensors, extracts the tensor's shape and data type,
    /// and returns a [TensorData] containing these attributes along with the original byte data.
    ///
    /// # Arguments
    ///
    /// * `bytes` - A vector of bytes representing the serialized tensor.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing the [TensorData] on success, or a [SafeTensorError] on failure.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<TensorData, SafeTensorError> {
        let safe_tensors: SafeTensors = SafeTensors::deserialize(&bytes)
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;
        let tensor_view: TensorView = safe_tensors
            .tensor("tensor")
            .map_err(|e| SafeTensorError::TensorConversionError(e.to_string()))?;

        let shape_i64: Vec<i64> = tensor_view.shape().iter().map(|&dim| dim as i64).collect();
        let dtype_safe: Dtype = tensor_view.dtype();
        let dtype_converted = DType::try_from(dtype_safe)?;

        Ok(TensorData {
            shape: shape_i64,
            dtype: dtype_converted,
            data: bytes,
        })
    }

    /// Converts the current [`RL4SysAction`] into its Python representation.
    ///
    /// This function wraps the current action in a [`PyRL4SysAction`], making it accessible to Python code.
    ///
    /// # Returns
    ///
    /// A [`PyRL4SysAction`] containing the same internal action data.
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network",
        feature = "python_bindings"
    ))]
    pub fn into_py(self) -> PyRL4SysAction {
        PyRL4SysAction {
            inner: RL4SysAction {
                obs: self.obs,
                act: self.act,
                mask: self.mask,
                rew: self.rew,
                data: self.data,
                done: self.done,
            },
        }
    }
}
