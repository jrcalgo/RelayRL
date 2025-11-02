pub mod hot_reloadable;
pub mod utils;

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn_tensor::{Tensor, TensorData, TensorKind, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::types::data::action::RelayRLData;
use crate::types::data::tensor::{BackendMatcher, DType, DeviceType, NdArrayDType, TchDType};
use half::{bf16, f16};

#[cfg(feature = "tch-model")]
use tch::{CModule, Kind, Tensor as TchTensor, no_grad};

#[cfg(feature = "ndarray-backend")]
use ndarray::{ArrayD, CowArray, IxDyn};

#[cfg(feature = "onnx-model")]
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    tensor::OrtOwnedTensor,
    value::Value as OrtValue,
};

#[cfg(feature = "onnx-model")]
use uuid::Uuid;

pub use burn_tensor::Shape;
pub use hot_reloadable::HotReloadableModel;

#[derive(Debug, Clone)]
pub enum ModelError {
    SerializationError(String),
    DeserializationError(String),
    BackendError(String),
    DTypeError(String),
    InvalidInputDimension(String),
    InvalidOutputDimension(String),
    UnsupportedRank(String),
    UnsupportedBackend(String),
    IoError(String),
    JsonError(String),
    UnsupportedModelType(String),
    InvalidMetadata(String),
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::IoError(e.to_string())
    }
}

impl From<serde_json::Error> for ModelError {
    fn from(e: serde_json::Error) -> Self {
        ModelError::JsonError(e.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelFileType {
    Pt,
    Onnx,
}

impl ModelFileType {
    pub fn from_path(path: &Path) -> Result<Self, ModelError> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
        {
            "pt" => Ok(ModelFileType::Pt),
            "onnx" => Ok(ModelFileType::Onnx),
            other => Err(ModelError::UnsupportedModelType(format!(
                "Unsupported extension: {}",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_file: String,
    pub model_type: ModelFileType,
    pub input_dtype: DType,
    pub output_dtype: DType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub default_device: Option<DeviceType>,
}

impl ModelMetadata {
    pub fn load_from_dir(dir: impl Into<PathBuf>) -> Result<Self, ModelError> {
        let dir: PathBuf = dir.into();
        let meta_path: PathBuf = dir.join("metadata.json");
        let mut s = String::new();
        fs::File::open(&meta_path)?.read_to_string(&mut s)?;
        let meta: ModelMetadata = serde_json::from_str(&s)?;

        if meta.model_file.trim().is_empty() {
            return Err(ModelError::InvalidMetadata(
                "metadata.model_file is empty".to_string(),
            ));
        }
        if meta.input_shape.is_empty() || meta.output_shape.is_empty() {
            return Err(ModelError::InvalidMetadata(
                "metadata input_shape/output_shape cannot be empty".to_string(),
            ));
        }
        Ok(meta)
    }

    pub fn save_to_dir(&self, dir: impl Into<PathBuf>) -> Result<(), ModelError> {
        let dir: PathBuf = dir.into();
        fs::create_dir_all(&dir)?;
        let meta_path: PathBuf = dir.join("metadata.json");
        let s = serde_json::to_string_pretty(self)?;
        fs::write(meta_path, s)?;
        Ok(())
    }

    pub fn resolve_model_path(&self, dir: &Path) -> PathBuf {
        dir.join(&self.model_file)
    }
}

#[derive(Debug, Clone)]
pub enum InferenceModel {
    #[cfg(feature = "tch-model")]
    Pt(Arc<CModule>),
    #[cfg(feature = "onnx-model")]
    Onnx {
        environment: Arc<Environment>,
        session: Arc<Session>,
    },
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct Model<B: Backend + BackendMatcher<Backend = B>> {
    pub file_type: ModelFileType,
    raw_bytes: Arc<[u8]>,
    inference: InferenceModel,
    _phantom: PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> Model<B> {
    fn load_from_file(file_type: ModelFileType, path: &Path) -> Result<Self, ModelError> {
        let raw_bytes: Arc<[u8]> = fs::read(path)?.into();
        let inference: InferenceModel = Self::build_inference(file_type.clone(), path)?;
        Ok(Self {
            file_type,
            raw_bytes,
            inference,
            _phantom: PhantomData,
        })
    }

    fn build_inference(
        file_type: ModelFileType,
        path: &Path,
    ) -> Result<InferenceModel, ModelError> {
        match file_type {
            ModelFileType::Pt => {
                #[cfg(feature = "tch-model")]
                {
                    let module = CModule::load(path)
                        .map_err(|err| ModelError::BackendError(err.to_string()))?;
                    Ok(InferenceModel::Pt(Arc::new(module)))
                }
                #[cfg(not(feature = "tch-model"))]
                {
                    Ok(InferenceModel::Unsupported)
                }
            }
            ModelFileType::Onnx => {
                #[cfg(feature = "onnx-model")]
                {
                    let env = Arc::new(
                        Environment::builder()
                            .with_name(format!("relayrl-env-{}", Uuid::new_v4()))
                            .build()
                            .map_err(|err| ModelError::BackendError(err.to_string()))?,
                    );
                    let session = Arc::new(
                        SessionBuilder::new(&env)
                            .map_err(|err| ModelError::BackendError(err.to_string()))?
                            .with_model_from_file(path)
                            .map_err(|err| ModelError::BackendError(err.to_string()))?,
                    );
                    Ok(InferenceModel::Onnx {
                        environment: env,
                        session: session,
                    })
                }
                #[cfg(not(feature = "onnx-model"))]
                {
                    Ok(InferenceModel::Unsupported)
                }
            }
        }
    }

    fn save_to_path(&self, path: &Path) -> Result<(), ModelError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.raw_bytes.as_ref())?;
        Ok(())
    }

    fn inference(&self) -> &InferenceModel {
        &self.inference
    }
}

#[derive(Clone)]
pub enum AnyTensor<B: Backend + 'static, const D: usize> {
    Float(Tensor<B, D, burn_tensor::Float>),
    Int(Tensor<B, D, burn_tensor::Int>),
    Bool(Tensor<B, D, burn_tensor::Bool>),
}

impl<B: Backend + 'static, const D: usize> AnyTensor<B, D> {
    /// Convert to Float tensor (will cast if needed)
    pub fn into_float(self) -> Tensor<B, D, burn_tensor::Float> {
        match self {
            AnyTensor::Float(t) => t,
            AnyTensor::Int(t) => t.float(),
            AnyTensor::Bool(t) => t.float(),
        }
    }
    
    /// Convert to Int tensor (will cast if needed)
    pub fn into_int(self) -> Tensor<B, D, burn_tensor::Int> {
        match self {
            AnyTensor::Float(t) => t.int(),
            AnyTensor::Int(t) => t,
            AnyTensor::Bool(t) => t.int(),
        }
    }
    
    /// Convert to Bool tensor (will cast if needed)  
    pub fn into_bool(self) -> Tensor<B, D, burn_tensor::Bool> {
        match self {
            AnyTensor::Float(t) => t.bool(),
            AnyTensor::Int(t) => t.bool(),
            AnyTensor::Bool(t) => t,
        }
    }
}

#[derive(Clone)]
pub struct ModelModule<B: Backend + BackendMatcher<Backend = B>> {
    pub model: Model<B>,
    pub metadata: ModelMetadata,
}

impl<B: Backend + BackendMatcher<Backend = B>> ModelModule<B> {
    /// Load from a directory containing `metadata.json` and the model file, or from a `metadata.json` path.
    pub fn load_from_path(path: impl Into<PathBuf>) -> Result<Self, ModelError> {
        let path: PathBuf = path.into();
        let dir = if path.is_dir() {
            path
        } else if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.eq_ignore_ascii_case("metadata.json"))
            .unwrap_or(false)
        {
            path.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            let meta_path = dir.join("metadata.json");
            if !meta_path.exists() {
                return Err(ModelError::InvalidMetadata(format!(
                    "metadata.json not found at {}",
                    meta_path.display()
                )));
            }
            dir
        };

        let metadata = ModelMetadata::load_from_dir(&dir)?;
        let model_path = metadata.resolve_model_path(&dir);
        let file_type = ModelFileType::from_path(&model_path)?;
        let model = Model::<B>::load_from_file(file_type, &model_path)?;

        Ok(Self { model, metadata })
    }

    /// Save `metadata.json` and the model file into `dir`.
    pub fn save(&self, dir: impl Into<PathBuf>) -> Result<(), ModelError> {
        let dir: PathBuf = dir.into();
        self.metadata.save_to_dir(&dir)?;
        let model_path = self.metadata.resolve_model_path(&dir);
        self.model.save_to_path(&model_path)?;
        Ok(())
    }

    /// Generic forward; dispatches to ONNX or LibTorch paths based on metadata.
    pub fn step<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: AnyTensor<B, D_IN>,
        mask: Option<AnyTensor<B, D_OUT>>,
    ) -> (AnyTensor<B, D_OUT>, HashMap<String, RelayRLData>) {
        let base_action = self
            .run_inference::<D_IN, D_OUT>(observation)
            .unwrap_or_else(|| self.zeros_action::<D_OUT>());

        let action = match mask {
            Some(mask_tensor) => {
                // Convert both to float for multiplication, then convert back based on output dtype
                let base_float = base_action.clone().into_float();
                let mask_float = mask_tensor.into_float();
                AnyTensor::Float(base_float * mask_float)
            }
            None => base_action,
        };

        (action, HashMap::new())
    }

    fn resolve_device(&self) -> <B as Backend>::Device {
        let preferred = self.metadata.default_device.clone().unwrap_or_default();
        <B as BackendMatcher>::get_device(&preferred)
            .or_else(|_| <B as BackendMatcher>::get_device(&DeviceType::default()))
            .expect("Failed to resolve backend device")
    }

    fn zeros_action<const D_OUT: usize>(&self) -> AnyTensor<B, D_OUT> {
        let device = self.resolve_device();
        let shape = Shape::from(self.metadata.output_shape.clone());
        
        // Create zeros tensor based on output dtype
        match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => {
                    AnyTensor::Float(Tensor::<B, D_OUT, Float>::zeros(shape, &device))
                }
                NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                    AnyTensor::Int(Tensor::<B, D_OUT, Int>::zeros(shape, &device))
                }
                NdArrayDType::Bool => {
                    AnyTensor::Bool(Tensor::<B, D_OUT, Bool>::zeros(shape, &device))
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => {
                    AnyTensor::Float(Tensor::<B, D_OUT, Float>::zeros(shape, &device))
                }
                TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                    AnyTensor::Int(Tensor::<B, D_OUT, Int>::zeros(shape, &device))
                }
                TchDType::Bool => {
                    AnyTensor::Bool(Tensor::<B, D_OUT, Bool>::zeros(shape, &device))
                }
            },
        }
    }

    fn run_inference<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: AnyTensor<B, D_IN>,
    ) -> Option<AnyTensor<B, D_OUT>> {
        match self.model.inference() {
            #[cfg(feature = "tch-model")]
            InferenceModel::Pt(module) => {
                self.run_libtorch_step::<D_IN, D_OUT>(module, observation)
            }
            #[cfg(feature = "onnx-model")]
            InferenceModel::Onnx { session, .. } => {
                self.run_onnx_step::<D_IN, D_OUT>(session, observation)
            }
            _ => None,
        }
    }

    #[cfg(feature = "tch-model")]
    fn run_libtorch_step<const D_IN: usize, const D_OUT: usize>(
        &self,
        module: &Arc<CModule>,
        observation: AnyTensor<B, D_IN>,
    ) -> Option<AnyTensor<B, D_OUT>> {
        use crate::types::data::tensor::{ConversionTensor, TensorData};
        
        // Step 1: Convert observation to the appropriate tensor kind for conversion
        let obs_tensor_converted = match &self.metadata.input_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => {
                    observation.into_float()
                }
                NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                    observation.into_int()
                }
                NdArrayDType::Bool => observation.into_bool(),
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => {
                    observation.into_float()
                }
                TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                    observation.into_int()
                }
                TchDType::Bool => observation.into_bool(),
            },
        };
        
        // Step 2: Convert to TensorData (need to handle each kind separately)
        let obs_tensor_data: TensorData = {
            match &self.metadata.input_dtype {
                #[cfg(feature = "ndarray-backend")]
                DType::NdArray(dtype) => match dtype {
                    NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => {
                        ConversionTensor {
                            tensor: obs_tensor_converted,
                            conversion_dtype: self.metadata.input_dtype.clone(),
                        }
                        .try_into()
                        .ok()?
                    }
                    NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                        ConversionTensor {
                            tensor: obs_tensor_converted,
                            conversion_dtype: self.metadata.input_dtype.clone(),
                        }
                        .try_into()
                        .ok()?
                    }
                    NdArrayDType::Bool => {
                        ConversionTensor {
                            tensor: obs_tensor_converted,
                            conversion_dtype: self.metadata.input_dtype.clone(),
                        }
                        .try_into()
                        .ok()?
                    }
                },
                #[cfg(feature = "tch-backend")]
                DType::Tch(dtype) => match dtype {
                    TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => {
                        ConversionTensor {
                            tensor: obs_tensor_converted,
                            conversion_dtype: self.metadata.input_dtype.clone(),
                        }
                        .try_into()
                        .ok()?
                    }
                    TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                        ConversionTensor {
                            tensor: obs_tensor_converted,
                            conversion_dtype: self.metadata.input_dtype.clone(),
                        }
                        .try_into()
                        .ok()?
                    }
                    TchDType::Bool => {
                        ConversionTensor {
                            tensor: obs_tensor_converted,
                            conversion_dtype: self.metadata.input_dtype.clone(),
                        }
                        .try_into()
                        .ok()?
                    }
                },
            }
        };
        
        // Step 3-5: Create torch tensor, run inference (same as before)
        let obs_shape_i64: Vec<i64> = obs_tensor_data
            .shape
            .iter()
            .map(|&d| d as i64)
            .collect();
        
        let obs_tensor: TchTensor = match &obs_tensor_data.dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 => {
                    let values: &[f16] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::F32 => {
                    let values: &[f32] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::F64 => {
                    let values: &[f64] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I8 => {
                    let values: &[i8] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I16 => {
                    let values: &[i16] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I32 => {
                    let values: &[i32] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I64 => {
                    let values: &[i64] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::Bool => {
                    let values: &[u8] = bytemuck::cast_slice(&obs_tensor_data.data);
                    let bool_values: Vec<bool> = values.iter().map(|&v| v != 0).collect();
                    TchTensor::from_slice(&bool_values).reshape(obs_shape_i64.as_slice())
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let values: &[f16] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::Bf16 => {
                    let values: &[bf16] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::F32 => {
                    let values: &[f32] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::F64 => {
                    let values: &[f64] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::I8 => {
                    let values: &[i8] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::I16 => {
                    let values: &[i16] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::I32 => {
                    let values: &[i32] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::I64 => {
                    let values: &[i64] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::U8 => {
                    let values: &[u8] = bytemuck::cast_slice(&obs_tensor_data.data);
                    TchTensor::from_slice(values).reshape(obs_shape_i64.as_slice())
                }
                TchDType::Bool => {
                    let values: &[u8] = bytemuck::cast_slice(&obs_tensor_data.data);
                    let bool_values: Vec<bool> = values.iter().map(|&v| v != 0).collect();
                    TchTensor::from_slice(&bool_values).reshape(obs_shape_i64.as_slice())
                }
            },
        };
        
        let act_tensor: TchTensor = no_grad(|| module.forward_ts(&[&obs_tensor]))
            .ok()?
            .to_kind(Kind::Float);
        
        let flattened_act: TchTensor = act_tensor.flatten(0, -1);
        
        // Steps 5-6: Convert back to bytes and TensorData (same as before)
        let act_bytes: Vec<u8> = match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 => {
                    let vec: Vec<f16> = Vec::<f16>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::F32 => {
                    let vec: Vec<f32> = Vec::<f32>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::F64 => {
                    let vec: Vec<f64> = Vec::<f64>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I8 => {
                    let vec: Vec<i8> = Vec::<i8>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I16 => {
                    let vec: Vec<i16> = Vec::<i16>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I32 => {
                    let vec: Vec<i32> = Vec::<i32>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I64 => {
                    let vec: Vec<i64> = Vec::<i64>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::Bool => {
                    let vec: Vec<bool> = Vec::<bool>::try_from(flattened_act).ok()?;
                    vec.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect()
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let vec: Vec<f16> = Vec::<f16>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::Bf16 => {
                    let vec: Vec<bf16> = Vec::<bf16>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::F32 => {
                    let vec: Vec<f32> = Vec::<f32>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::F64 => {
                    let vec: Vec<f64> = Vec::<f64>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I8 => {
                    let vec: Vec<i8> = Vec::<i8>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I16 => {
                    let vec: Vec<i16> = Vec::<i16>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I32 => {
                    let vec: Vec<i32> = Vec::<i32>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I64 => {
                    let vec: Vec<i64> = Vec::<i64>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::U8 => {
                    let vec: Vec<u8> = Vec::<u8>::try_from(flattened_act).ok()?;
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::Bool => {
                    let vec: Vec<bool> = Vec::<bool>::try_from(flattened_act).ok()?;
                    vec.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect()
                }
            },
        };
        
        let act_tensor_data = TensorData::new(
            self.metadata.output_shape.clone(),
            self.metadata.output_dtype.clone(),
            act_bytes,
            TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
        );
        
        // Step 7: Convert TensorData back to the appropriate AnyTensor variant
        let device_type = self.metadata.default_device.clone().unwrap_or_default();
        
        match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => {
                    let float_tensor = act_tensor_data.to_float_tensor::<B, D_OUT>(&device_type).ok()?;
                    Some(AnyTensor::Float(float_tensor.tensor))
                }
                NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                    let int_tensor = act_tensor_data.to_int_tensor::<B, D_OUT>(&device_type).ok()?;
                    Some(AnyTensor::Int(int_tensor.tensor))
                }
                NdArrayDType::Bool => {
                    let bool_tensor = act_tensor_data.to_bool_tensor::<B, D_OUT>(&device_type).ok()?;
                    Some(AnyTensor::Bool(bool_tensor.tensor))
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => {
                    let float_tensor = act_tensor_data.to_float_tensor::<B, D_OUT>(&device_type).ok()?;
                    Some(AnyTensor::Float(float_tensor.tensor))
                }
                TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                    let int_tensor = act_tensor_data.to_int_tensor::<B, D_OUT>(&device_type).ok()?;
                    Some(AnyTensor::Int(int_tensor.tensor))
                }
                TchDType::Bool => {
                    let bool_tensor = act_tensor_data.to_bool_tensor::<B, D_OUT>(&device_type).ok()?;
                    Some(AnyTensor::Bool(bool_tensor.tensor))
                }
            },
        }
    }

    #[cfg(feature = "onnx-model")]
    fn run_onnx_step<const D_IN: usize, const D_OUT: usize>(
        &self,
        session: &Arc<Session>,
        observation: AnyTensor<B, D_IN>,
    ) -> Option<AnyTensor<B, D_OUT>> {
        let obs_data = observation.into_float().to_data().convert::<f32>();
        let obs_vec = obs_data.into_vec::<f32>().ok()?;

        let input_shape = self.metadata.input_shape.clone();
        let array = ArrayD::from_shape_vec(input_shape, obs_vec).ok()?;

        let cow_array: CowArray<'_, f32, IxDyn> = CowArray::from(array);
        let input_value = OrtValue::from_array(session.allocator(), &cow_array).ok()?;

        let mut outputs = session.run(vec![input_value]).ok()?;
        let first = outputs.into_iter().next()?;
        let owned: OrtOwnedTensor<'_, f32, IxDyn> = first.try_extract().ok()?;

        let y_vec: Vec<f32> = owned.view().iter().copied().collect();

        let device = self.resolve_device();
        let tensor_data = TensorData::new(y_vec, self.metadata.output_shape.clone());
        
        // Convert back to the appropriate AnyTensor variant based on output dtype
        match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 | NdArrayDType::F32 | NdArrayDType::F64 => {
                    let float_tensor = tensor_data.to_float_tensor::<B, D_OUT>(&device).ok()?;
                    Some(AnyTensor::Float(float_tensor.tensor))
                }
                NdArrayDType::I8 | NdArrayDType::I16 | NdArrayDType::I32 | NdArrayDType::I64 => {
                    let int_tensor = tensor_data.to_int_tensor::<B, D_OUT>(&device).ok()?;
                    Some(AnyTensor::Int(int_tensor.tensor))
                }
                NdArrayDType::Bool => {
                    let bool_tensor = tensor_data.to_bool_tensor::<B, D_OUT>(&device).ok()?;
                    Some(AnyTensor::Bool(bool_tensor.tensor))
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 | TchDType::Bf16 | TchDType::F32 | TchDType::F64 => {
                    let float_tensor = tensor_data.to_float_tensor::<B, D_OUT>(&device).ok()?;
                    Some(AnyTensor::Float(float_tensor.tensor))
                }
                TchDType::I8 | TchDType::I16 | TchDType::I32 | TchDType::I64 | TchDType::U8 => {
                    let int_tensor = tensor_data.to_int_tensor::<B, D_OUT>(&device).ok()?;
                    Some(AnyTensor::Int(int_tensor.tensor))
                }
                TchDType::Bool => {
                    let bool_tensor = tensor_data.to_bool_tensor::<B, D_OUT>(&device).ok()?;
                    Some(AnyTensor::Bool(bool_tensor.tensor))
                }
            },
        }
    }
}
