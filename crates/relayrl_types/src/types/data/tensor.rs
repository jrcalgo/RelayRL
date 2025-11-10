use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[cfg(feature = "ndarray-backend")]
use burn_ndarray::NdArray;
#[cfg(feature = "tch-backend")]
use burn_tch::LibTorch as Tch;

use burn_tensor::{
    BasicOps, Bool, Float, Int, Shape, Tensor, TensorData as BurnTensorData, TensorKind,
    backend::Backend,
};

#[derive(Debug, Clone)]
pub enum TensorError {
    SerializationError(String),
    DeserializationError(String),
    BackendError(String),
    DTypeError(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationError(e) => write!(f, "[TensorError] Serialization error: {}", e),
            Self::DeserializationError(e) => {
                write!(f, "[TensorError] Deserialization error: {}", e)
            }
            Self::BackendError(e) => write!(f, "[TensorError] Backend error: {}", e),
            Self::DTypeError(e) => write!(f, "[TensorError] DType error: {}", e),
        }
    }
}

/// Tensor backend enumeration for runtime backend selection
/// Constrains burn-tensor backends to tch and ndarray
#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SupportedTensorBackend {
    None,
    /// CPU-based NdArray backend
    #[cfg(feature = "ndarray-backend")]
    NdArray,
    /// LibTorch backend (GPU/CPU)
    #[cfg(feature = "tch-backend")]
    Tch,
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
impl Default for SupportedTensorBackend {
    fn default() -> Self {
        #[cfg(feature = "ndarray-backend")]
        return SupportedTensorBackend::NdArray;

        #[cfg(feature = "tch-backend")]
        return SupportedTensorBackend::Tch;
    }
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "tch-backend")]
    Cuda(usize),
    #[cfg(feature = "tch-backend")]
    Mps,
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
pub trait BackendMatcher {
    type Backend: Backend + 'static;

    fn matches_backend(supported: &SupportedTensorBackend) -> bool;
    fn get_device(device: &DeviceType) -> Result<burn_tensor::Device<Self::Backend>, TensorError>;
}

#[cfg(feature = "ndarray-backend")]
impl BackendMatcher for NdArray {
    type Backend = NdArray;

    fn matches_backend(supported: &SupportedTensorBackend) -> bool {
        *supported == SupportedTensorBackend::NdArray
    }

    fn get_device(device: &DeviceType) -> Result<burn_tensor::Device<Self::Backend>, TensorError> {
        match device {
            DeviceType::Cpu => Ok(burn_tensor::Device::<Self::Backend>::Cpu),
            _ => Err(TensorError::BackendError(
                "Unsupported device type".to_string(),
            )),
        }
    }
}

#[cfg(feature = "tch-backend")]
impl BackendMatcher for Tch {
    type Backend = Tch;

    fn matches_backend(supported: &SupportedTensorBackend) -> bool {
        *supported == SupportedTensorBackend::Tch
    }

    fn get_device(device: &DeviceType) -> Result<burn_tensor::Device<Self::Backend>, TensorError> {
        match device {
            DeviceType::Cpu => Ok(burn_tensor::Device::<Self::Backend>::Cpu),
            #[cfg(feature = "tch-backend")]
            DeviceType::Cuda(index) => Ok(burn_tensor::Device::<Self::Backend>::Cuda(*index)),
            #[cfg(feature = "tch-backend")]
            DeviceType::Mps => Ok(burn_tensor::Device::<Self::Backend>::Mps),
        }
    }
}

/// Data type enumeration for tensor serialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    #[cfg(feature = "ndarray-backend")]
    NdArray(NdArrayDType),
    #[cfg(feature = "tch-backend")]
    Tch(TchDType),
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(ndarray) => write!(f, "NdArray({})", ndarray),
            #[cfg(feature = "tch-backend")]
            DType::Tch(tch) => write!(f, "Tch({})", tch),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TchDType {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    Bool,
}

impl std::fmt::Display for TchDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TchDType::F16 => write!(f, "F16"),
            TchDType::Bf16 => write!(f, "Bf16"),
            TchDType::F32 => write!(f, "F32"),
            TchDType::F64 => write!(f, "F64"),
            TchDType::I8 => write!(f, "I8"),
            TchDType::I16 => write!(f, "I16"),
            TchDType::I32 => write!(f, "I32"),
            TchDType::I64 => write!(f, "I64"),
            TchDType::U8 => write!(f, "U8"),
            TchDType::Bool => write!(f, "Bool"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NdArrayDType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    Bool,
}

impl std::fmt::Display for NdArrayDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NdArrayDType::F16 => write!(f, "F16"),
            NdArrayDType::F32 => write!(f, "F32"),
            NdArrayDType::F64 => write!(f, "F64"),
            NdArrayDType::I8 => write!(f, "I8"),
            NdArrayDType::I16 => write!(f, "I16"),
            NdArrayDType::I32 => write!(f, "I32"),
            NdArrayDType::I64 => write!(f, "I64"),
            NdArrayDType::Bool => write!(f, "Bool"),
        }
    }
}

/// Wraps dtype-wrapped BurnTensor objects defined in this namespace for easy storage, conversion, and retrieval
#[derive(Debug)]
pub enum AnyBurnTensor<B: Backend + 'static, const D: usize> {
    Float(FloatBurnTensor<B, D>),
    Int(IntBurnTensor<B, D>),
    Bool(BoolBurnTensor<B, D>),
}

impl<B: Backend + 'static, const D: usize> Clone for AnyBurnTensor<B, D> {
    fn clone(&self) -> Self {
        match self {
            AnyBurnTensor::Float(wrapper) => AnyBurnTensor::Float(wrapper.clone()),
            AnyBurnTensor::Int(wrapper) => AnyBurnTensor::Int(wrapper.clone()),
            AnyBurnTensor::Bool(wrapper) => AnyBurnTensor::Bool(wrapper.clone()),
        }
    }
}

impl<B: Backend + 'static, const D: usize> AnyBurnTensor<B, D> {
    /// Helper function to extract tensor and determine backend from dtype for Float conversions
    fn extract_tensor_and_backend_float(self) -> (Tensor<B, D, Float>, SupportedTensorBackend) {
        match self {
            AnyBurnTensor::Float(wrapper) => {
                let supported_backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor, supported_backend)
            }
            AnyBurnTensor::Int(wrapper) => {
                let supported_backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor.float(), supported_backend)
            }
            AnyBurnTensor::Bool(wrapper) => {
                let supported_backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor.float(), supported_backend)
            }
        }
    }

    /// Helper function to extract tensor and determine backend from dtype for Int conversions
    fn extract_tensor_and_backend_int(self) -> (Tensor<B, D, Int>, SupportedTensorBackend) {
        match self {
            AnyBurnTensor::Float(wrapper) => {
                let supported_backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor.int(), supported_backend)
            }
            AnyBurnTensor::Int(wrapper) => {
                let supported_backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor, supported_backend)
            }
            AnyBurnTensor::Bool(wrapper) => {
                let supported_backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor.int(), supported_backend)
            }
        }
    }

    /// Helper function to extract tensor and determine backend from dtype for Bool conversions
    fn extract_tensor_and_backend_bool(self) -> (Tensor<B, D, Bool>, SupportedTensorBackend) {
        match self {
            AnyBurnTensor::Float(wrapper) => {
                let backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor.bool(), backend)
            }
            AnyBurnTensor::Int(wrapper) => {
                let backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor.bool(), backend)
            }
            AnyBurnTensor::Bool(wrapper) => {
                let backend = TensorData::get_backend_from_dtype(&wrapper.dtype);
                (wrapper.tensor, backend)
            }
        }
    }

    pub fn get_tensor_type(self) -> (String, DType) {
        match self {
            AnyBurnTensor::Float(wrapper) => (String::from("float"), wrapper.dtype),
            AnyBurnTensor::Int(wrapper) => (String::from("int"), wrapper.dtype),
            AnyBurnTensor::Bool(wrapper) => (String::from("bool"), wrapper.dtype),
        }
    }

    pub fn into_f16_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_float();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F16),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::F16),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_bf16_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_float();
        let conversion_dtype = match backend {
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::Bf16),
            _ => {
                return Err(TensorError::DTypeError(
                    "Bf16 is only supported for Tch backend".to_string(),
                ));
            }
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_f32_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_float();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_f64_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_float();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F64),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::F64),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_i8_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_int();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::I8),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::I8),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_i16_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_int();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::I16),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::I16),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_i32_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_int();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::I32),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::I32),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_i64_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_int();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::I64),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::I64),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_u8_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_int();
        let conversion_dtype = match backend {
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::U8),
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => {
                return Err(TensorError::DTypeError(
                    "U8 is only supported for Tch backend".to_string(),
                ));
            }
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }

    pub fn into_bool_data(self) -> Result<TensorData, TensorError> {
        let (tensor, backend) = self.extract_tensor_and_backend_bool();
        let conversion_dtype = match backend {
            #[cfg(feature = "ndarray-backend")]
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::Bool),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::Bool),
            _ => return Err(TensorError::BackendError("Unsupported backend".to_string())),
        };
        let conversion_tensor = ConversionBurnTensor {
            inner: tensor,
            conversion_dtype,
        };
        TensorData::try_from(conversion_tensor)
    }
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
#[derive(Debug, Clone)]
pub struct FloatBurnTensor<B: Backend + 'static, const D: usize> {
    pub tensor: Tensor<B, D, Float>,
    pub dtype: DType,
}

impl<B: Backend + 'static, const D: usize> FloatBurnTensor<B, D> {
    pub fn empty(shape: &Shape, dtype: &DType, device: &<B as Backend>::Device) -> Self {
        let tensor = Tensor::<B, D, Float>::empty(shape.clone(), &device);
        Self {
            tensor,
            dtype: dtype.clone(),
        }
    }
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
#[derive(Debug, Clone)]
pub struct IntBurnTensor<B: Backend + 'static, const D: usize> {
    pub tensor: Tensor<B, D, Int>,
    pub dtype: DType,
}

impl<B: Backend + 'static, const D: usize> IntBurnTensor<B, D> {
    pub fn empty(shape: &Shape, dtype: &DType, device: &<B as Backend>::Device) -> Self {
        let tensor = Tensor::<B, D, Int>::empty(shape.clone(), &device);
        Self {
            tensor,
            dtype: dtype.clone(),
        }
    }
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
#[derive(Debug, Clone)]
pub struct BoolBurnTensor<B: Backend + 'static, const D: usize> {
    pub tensor: Tensor<B, D, Bool>,
    pub dtype: DType,
}

impl<B: Backend + 'static, const D: usize> BoolBurnTensor<B, D> {
    pub fn empty(shape: &Shape, dtype: &DType, device: &<B as Backend>::Device) -> Self {
        let tensor = Tensor::<B, D, Bool>::empty(shape.clone(), &device);
        Self {
            tensor,
            dtype: dtype.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
    pub supported_backend: SupportedTensorBackend,
}

impl TensorData {
    pub fn new(
        shape: Vec<usize>,
        dtype: DType,
        data: Vec<u8>,
        supported_backend: SupportedTensorBackend,
    ) -> Self {
        Self {
            shape,
            dtype,
            data,
            supported_backend,
        }
    }

    pub fn num_el(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn get_backend_from_dtype(dtype: &DType) -> SupportedTensorBackend {
        match dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(_) => SupportedTensorBackend::NdArray,
            #[cfg(feature = "tch-backend")]
            DType::Tch(_) => SupportedTensorBackend::Tch,
        }
    }
}

impl TensorData {
    /// Convert TensorData to a Float Tensor
    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub fn to_float_tensor<B: BackendMatcher + 'static, const D: usize>(
        &self,
        device: &DeviceType,
    ) -> Result<FloatBurnTensor<B::Backend, D>, TensorError> {
        let device: <<B as BackendMatcher>::Backend as Backend>::Device = B::get_device(device)?;

        let shape: Shape = Shape::from(self.shape.as_slice());

        match &self.dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => {
                match dtype {
                    #[cfg(feature = "quantization")]
                    NdArrayDType::F16 => {
                        let values: &[f16] = bytemuck::cast_slice(&self.data);
                        // Convert f16 to f32 for processing
                        let f32_values: Vec<f32> = values.iter().map(|&v| v.to_f32()).collect();
                        let data = BurnTensorData::new(f32_values, shape);
                        Ok(FloatBurnTensor {
                            tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                            dtype: DType::NdArray(NdArrayDType::F16),
                        })
                    }
                    NdArrayDType::F32 => {
                        let values: &[f32] = bytemuck::cast_slice(&self.data);
                        let data = BurnTensorData::new(values.to_vec(), shape);
                        Ok(FloatBurnTensor {
                            tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                            dtype: DType::NdArray(NdArrayDType::F32),
                        })
                    }
                    NdArrayDType::F64 => {
                        let values: &[f64] = bytemuck::cast_slice(&self.data);
                        let data = BurnTensorData::new(values.to_vec(), shape);
                        Ok(FloatBurnTensor {
                            tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                            dtype: DType::NdArray(NdArrayDType::F64),
                        })
                    }
                    _ => Err(TensorError::DTypeError(format!(
                        "Cannot convert {:?} to Float tensor",
                        dtype
                    ))),
                }
            }
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let values: &[f16] = bytemuck::cast_slice(&self.data);
                    let f32_values: Vec<f32> = values.iter().map(|&v| v.to_f32()).collect();
                    let data = BurnTensorData::new(f32_values, shape);
                    Ok(FloatBurnTensor {
                        tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::F16),
                    })
                }
                TchDType::Bf16 => {
                    let values: &[bf16] = bytemuck::cast_slice(&self.data);
                    let f32_values: Vec<f32> = values.iter().map(|&v| v.to_f32()).collect();
                    let data = BurnTensorData::new(f32_values, shape);
                    Ok(FloatBurnTensor {
                        tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::Bf16),
                    })
                }
                TchDType::F32 => {
                    let values: &[f32] = bytemuck::cast_slice(&self.data);
                    let data = BurnTensorData::new(values.to_vec(), shape);
                    Ok(FloatBurnTensor {
                        tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::F32),
                    })
                }
                TchDType::F64 => {
                    let values: &[f64] = bytemuck::cast_slice(&self.data);
                    let data = BurnTensorData::new(values.to_vec(), shape);
                    Ok(FloatBurnTensor {
                        tensor: Tensor::<B::Backend, D, Float>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::F64),
                    })
                }
                _ => Err(TensorError::DTypeError(format!(
                    "Cannot convert {:?} to Float tensor",
                    dtype
                ))),
            },
        }
    }

    /// Convert TensorData to an Int Tensor
    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub fn to_int_tensor<B: BackendMatcher + 'static, const D: usize>(
        &self,
        device: &DeviceType,
    ) -> Result<IntBurnTensor<B::Backend, D>, TensorError> {
        let device: <<B as BackendMatcher>::Backend as Backend>::Device = B::get_device(device)?;

        let shape: Shape = Shape::from(self.shape.as_slice());

        match &self.dtype {
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::I8 => {
                    let values: &[i8] = bytemuck::cast_slice(&self.data);
                    let i32_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                    let data = BurnTensorData::new(i32_values, shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::NdArray(NdArrayDType::I8),
                    })
                }
                NdArrayDType::I16 => {
                    let values: &[i16] = bytemuck::cast_slice(&self.data);
                    let i32_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                    let data = BurnTensorData::new(i32_values, shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::NdArray(NdArrayDType::I16),
                    })
                }
                NdArrayDType::I32 => {
                    let values: &[i32] = bytemuck::cast_slice(&self.data);
                    let data = BurnTensorData::new(values.to_vec(), shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::NdArray(NdArrayDType::I32),
                    })
                }
                NdArrayDType::I64 => {
                    let values: &[i64] = bytemuck::cast_slice(&self.data);
                    let data = BurnTensorData::new(values.to_vec(), shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::NdArray(NdArrayDType::I64),
                    })
                }
                _ => Err(TensorError::DTypeError(format!(
                    "Cannot convert {:?} to Int tensor",
                    dtype
                ))),
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::U8 => {
                    let values: &[u8] = bytemuck::cast_slice(&self.data);
                    let i32_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                    let data = BurnTensorData::new(i32_values, shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::U8),
                    })
                }
                TchDType::I8 => {
                    let values: &[i8] = bytemuck::cast_slice(&self.data);
                    let i32_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                    let data = BurnTensorData::new(i32_values, shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::I8),
                    })
                }
                TchDType::I16 => {
                    let values: &[i16] = bytemuck::cast_slice(&self.data);
                    let i32_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                    let data = BurnTensorData::new(i32_values, shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::I16),
                    })
                }
                TchDType::I32 => {
                    let values: &[i32] = bytemuck::cast_slice(&self.data);
                    let data = BurnTensorData::new(values.to_vec(), shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::I32),
                    })
                }
                TchDType::I64 => {
                    let values: &[i64] = bytemuck::cast_slice(&self.data);
                    let data = BurnTensorData::new(values.to_vec(), shape);
                    Ok(IntBurnTensor {
                        tensor: Tensor::<B::Backend, D, Int>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::I64),
                    })
                }
                _ => Err(TensorError::DTypeError(format!(
                    "Cannot convert {:?} to Int tensor",
                    dtype
                ))),
            },
        }
    }

    /// Convert TensorData to a Bool Tensor
    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub fn to_bool_tensor<B: BackendMatcher + 'static, const D: usize>(
        &self,
        device: &DeviceType,
    ) -> Result<BoolBurnTensor<B::Backend, D>, TensorError> {
        let device: <<B as BackendMatcher>::Backend as Backend>::Device = B::get_device(device)?;

        let shape: Shape = Shape::from(self.shape.as_slice());

        match &self.dtype {
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::Bool => {
                    let values: &[u8] = bytemuck::cast_slice(&self.data);
                    let bool_values: Vec<bool> = values.iter().map(|&v| v != 0).collect();
                    let data = BurnTensorData::new(bool_values, shape);
                    Ok(BoolBurnTensor {
                        tensor: Tensor::<B::Backend, D, Bool>::from_data(data, &device),
                        dtype: DType::NdArray(NdArrayDType::Bool),
                    })
                }
                _ => Err(TensorError::DTypeError(format!(
                    "Cannot convert {:?} to Bool tensor",
                    dtype
                ))),
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::Bool => {
                    let values: &[u8] = bytemuck::cast_slice(&self.data);
                    let bool_values: Vec<bool> = values.iter().map(|&v| v != 0).collect();
                    let data = BurnTensorData::new(bool_values, shape);
                    Ok(BoolBurnTensor {
                        tensor: Tensor::<B::Backend, D, Bool>::from_data(data, &device),
                        dtype: DType::Tch(TchDType::Bool),
                    })
                }
                _ => Err(TensorError::DTypeError(format!(
                    "Cannot convert {:?} to Bool tensor",
                    dtype
                ))),
            },
        }
    }
}

/// Converts a BurnTensor to a RelayRL TensorData structure
#[derive(Debug, Clone)]
pub struct ConversionBurnTensor<B: Backend + 'static, const D: usize, K: TensorKind<B>> {
    pub inner: Tensor<B, D, K>,
    pub conversion_dtype: DType,
}

impl<B: Backend + 'static, const D: usize, K: TensorKind<B> + BasicOps<B>>
    TryFrom<ConversionBurnTensor<B, D, K>> for TensorData
{
    type Error = TensorError;

    fn try_from(t: ConversionBurnTensor<B, D, K>) -> Result<Self, Self::Error> {
        let data = t.inner.to_data();
        let shape = data.shape.clone();

        fn pack_bytes<E: burn_tensor::Element>(
            data: &burn_tensor::TensorData,
        ) -> Result<Vec<u8>, TensorError> {
            let v: Vec<E> = data
                .to_vec::<E>()
                .map_err(|e| TensorError::DTypeError(format!("Element cast failed: {:?}", e)))?;
            Ok(bytemuck::cast_slice(&v).to_vec())
        }

        fn pack_bools(data: &burn_tensor::TensorData) -> Result<Vec<u8>, TensorError> {
            let v: Vec<bool> = data
                .to_vec::<bool>()
                .map_err(|e| TensorError::DTypeError(format!("Bool cast failed: {:?}", e)))?;
            Ok(v.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect())
        }

        let (supported_backend, bytes) = match &t.conversion_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(nd) => {
                use super::tensor::NdArrayDType::*;
                let bytes = match nd {
                    #[cfg(feature = "quantization")]
                    F16 => pack_bytes::<half::f16>(&data)?,
                    F32 => pack_bytes::<f32>(&data)?,
                    F64 => pack_bytes::<f64>(&data)?,
                    I8 => pack_bytes::<i8>(&data)?,
                    I16 => pack_bytes::<i16>(&data)?,
                    I32 => pack_bytes::<i32>(&data)?,
                    I64 => pack_bytes::<i64>(&data)?,
                    Bool => pack_bools(&data)?,
                    #[cfg(not(feature = "quantization"))]
                    F16 => {
                        return Err(TensorError::DTypeError(
                            "F16 requires 'quantization' feature".into(),
                        ));
                    }
                };
                (SupportedTensorBackend::NdArray, bytes)
            }
            #[cfg(feature = "tch-backend")]
            DType::Tch(td) => {
                use super::tensor::TchDType::*;
                let bytes = match td {
                    #[cfg(feature = "quantization")]
                    F16 => pack_bytes::<half::f16>(&data)?,
                    #[cfg(feature = "quantization")]
                    Bf16 => pack_bytes::<half::bf16>(&data)?,
                    F32 => pack_bytes::<f32>(&data)?,
                    F64 => pack_bytes::<f64>(&data)?,
                    I8 => pack_bytes::<i8>(&data)?,
                    I16 => pack_bytes::<i16>(&data)?,
                    I32 => pack_bytes::<i32>(&data)?,
                    I64 => pack_bytes::<i64>(&data)?,
                    U8 => pack_bytes::<u8>(&data)?,
                    Bool => pack_bools(&data)?,
                    #[cfg(not(feature = "quantization"))]
                    F16 => {
                        return Err(TensorError::DTypeError(
                            "F16 requires 'quantization' feature".into(),
                        ));
                    }
                    #[cfg(not(feature = "quantization"))]
                    Bf16 => {
                        return Err(TensorError::DTypeError(
                            "Bf16 requires 'quantization' feature".into(),
                        ));
                    }
                };
                (SupportedTensorBackend::Tch, bytes)
            }
            _ => {
                return Err(TensorError::BackendError(
                    "Unsupported or missing target backend for conversion".into(),
                ));
            }
        };

        Ok(TensorData {
            shape,
            dtype: t.conversion_dtype,
            data: bytes,
            supported_backend,
        })
    }
}
