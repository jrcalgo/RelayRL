pub mod hot_reloadable;
pub mod utils;

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{ArrayBase, CowRepr, Dim, IxDynImpl};
use ort::value::DynArrayRef;
use ort::tensor::TensorDataToType;

use burn_tensor::{Tensor, TensorData as BurnTensorData, TensorKind, backend::Backend};
use ort::tensor::IntoTensorElementDataType;
use serde::{Deserialize, Serialize};

use crate::types::data::action::RelayRLData;
use crate::types::data::tensor::{
    AnyBurnTensor, BackendMatcher, BoolBurnTensor, ConversionBurnTensor, DType, DeviceType,
    FloatBurnTensor, IntBurnTensor, NdArrayDType, SupportedTensorBackend, TchDType, TensorData,
};
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
#[cfg(all(
    any(feature = "tch-model", feature = "onnx-model"),
    any(feature = "ndarray-backend", feature = "tch-backend")
))]
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
    #[cfg(all(
        any(feature = "tch-model", feature = "onnx-model"),
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    pub fn step<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: AnyBurnTensor<B, D_IN>,
        mask: Option<AnyBurnTensor<B, D_OUT>>,
    ) -> (TensorData, HashMap<String, RelayRLData>) {
        let base_action = self
            .run_inference::<D_IN, D_OUT>(observation)
            .unwrap_or_else(|_| {
                self.zeros_action::<D_OUT>()
                    .expect("Failed to create zeros action")
            });

        let action: TensorData = match mask {
            Some(mask_tensor) => {
                // Convert both to float for multiplication, then convert back based on output dtype
                let mask_td: TensorData = match mask_tensor {
                    AnyBurnTensor::Float(wrapper) => TensorData::try_from(ConversionBurnTensor {
                        inner: wrapper.tensor,
                        conversion_dtype: self.metadata.output_dtype.clone(),
                    }),
                    AnyBurnTensor::Int(wrapper) => TensorData::try_from(ConversionBurnTensor {
                        inner: wrapper.tensor,
                        conversion_dtype: self.metadata.output_dtype.clone(),
                    }),
                    AnyBurnTensor::Bool(wrapper) => TensorData::try_from(ConversionBurnTensor {
                        inner: wrapper.tensor,
                        conversion_dtype: self.metadata.output_dtype.clone(),
                    }),
                }
                .expect("Failed to convert mask tensor to TensorData");

                let action_data: Vec<u8> = base_action
                    .data
                    .iter()
                    .zip(mask_td.data.iter())
                    .map(|(a, m)| a * m)
                    .collect();
                TensorData {
                    shape: base_action.shape.clone(),
                    dtype: base_action.dtype.clone(),
                    data: action_data,
                    supported_backend: base_action.supported_backend.clone(),
                }
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

    fn zeros_action<const D_OUT: usize>(&self) -> Result<TensorData, ModelError> {
        let device = self.resolve_device();
        let shape = Shape::from(self.metadata.output_shape.clone());
        let data = vec![0_u8; shape.num_elements()];

        // Create zeros tensor based on output dtype
        match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 => {
                    let data_vec = vec![f16::ZERO; shape.dims.iter().product()];
                    let data: &[f16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f16, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::F32 => {
                    let data_vec = vec![0_f32; shape.dims.iter().product()];
                    let data: &[f32] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f32, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::F64 => {
                    let data_vec = vec![0_f64; shape.dims.iter().product()];
                    let data: &[f64] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f64, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::I8 => {
                    let data_vec = vec![0_i8; shape.dims.iter().product()];
                    let data: &[i8] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i8, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::I16 => {
                    let data_vec = vec![0_i16; shape.dims.iter().product()];
                    let data: &[i16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i16, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::I32 => {
                    let data_vec = vec![0_i32; shape.dims.iter().product()];
                    let data: &[i32] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i32, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::I64 => {
                    let data_vec = vec![0_i64; shape.dims.iter().product()];
                    let data: &[i64] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i64, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                NdArrayDType::Bool => {
                    let data_vec = vec![false; shape.dims.iter().product()];
                    let data: &[bool] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<bool, u8>(data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::NdArray(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::NdArray,
                    ))
                }
                _ => Err(ModelError::UnsupportedModelType(format!(
                    "Unsupported dtype: {}",
                    dtype
                ))),
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let data_vec = vec![f16::ZERO; shape.dims.iter().product()];
                    let data: &[f16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f16, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::Bf16 => {
                    let data_vec = vec![bf16::ZERO; shape.dims.iter().product()];
                    let data: &[bf16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<bf16, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::F32 => {
                    let data_vec = vec![0_f32; shape.dims.iter().product()];
                    let data: &[f32] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f32, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::F64 => {
                    let data_vec = vec![0_f64; shape.dims.iter().product()];
                    let data: &[f64] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f64, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::I8 => {
                    let data_vec = vec![0_i8; shape.dims.iter().product()];
                    let data: &[i8] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i8, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::I16 => {
                    let data_vec = vec![0_i16; shape.dims.iter().product()];
                    let data: &[i16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i16, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::I32 => {
                    let data_vec = vec![0_i32; shape.dims.iter().product()];
                    let data: &[i32] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i32, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::I64 => {
                    let data_vec = vec![0_i64; shape.dims.iter().product()];
                    let data: &[i64] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<i64, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::U8 => {
                    let data_vec = vec![0_u8; shape.dims.iter().product()];
                    let data: &[u8] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<u8, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                TchDType::Bool => {
                    let data_vec = vec![false; shape.dims.iter().product()];
                    let data: &[bool] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<bool, u8>(&data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
                _ => Err(ModelError::UnsupportedModelType(format!(
                    "Unsupported dtype: {}",
                    dtype
                ))),
            },
        }
    }

    fn run_inference<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: AnyBurnTensor<B, D_IN>,
    ) -> Result<TensorData, ModelError> {
        match self.model.inference() {
            #[cfg(feature = "tch-model")]
            InferenceModel::Pt(module) => {
                self.run_libtorch_step::<D_IN, D_OUT>(module, observation)
            }
            #[cfg(feature = "onnx-model")]
            InferenceModel::Onnx { session, .. } => {
                self.run_onnx_step::<D_IN, D_OUT>(session, observation)
            }
            _ => Err(ModelError::UnsupportedModelType(
                "Unsupported model type".to_string(),
            )),
        }
    }

    #[cfg(all(
        feature = "tch-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn run_libtorch_step<const D_IN: usize, const D_OUT: usize>(
        &self,
        module: &Arc<CModule>,
        observation: AnyBurnTensor<B, D_IN>,
    ) -> Result<TensorData, ModelError> {
        // Step 1: Convert AnyBurnTensor to inner Tensor<B, D_IN, K> to metadata dtype using ConversionBurnTensor enum & methods
        // Step 2: Convert RelayRL TensorData to TchTensor
        // Step 3: Run CModule forward pass inference
        // Step 4: Convert TchTensor to bytes
        // Step 5: Convert bytes to RelayRL TensorData

        // Step 1 and Step 2
        let obs_tensor: TchTensor = match &self.metadata.input_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(nd) => match nd {
                NdArrayDType::F16 => {
                    let obs_tensor_data = observation.clone().into_f16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f16: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<f16>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::F32 => {
                    let obs_tensor_data = observation.clone().into_f32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f32: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<f32>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::F64 => {
                    let obs_tensor_data = observation.clone().into_f64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f64: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<f64>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I8 => {
                    let obs_tensor_data = observation.clone().into_i8_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i8: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i8>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I16 => {
                    let obs_tensor_data = observation.clone().into_i16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i16: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i16>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I32 => {
                    let obs_tensor_data = observation.clone().into_i32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i32: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i32>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I64 => {
                    let obs_tensor_data = observation.clone().into_i64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i64: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i64>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::Bool => {
                    let obs_tensor_data = observation.clone().into_bool_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to bool: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<u8>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(tch) => match tch {
                TchDType::F16 => {
                    let obs_tensor_data = observation.clone().into_f16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f16: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<f16>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::Bf16 => {
                    let obs_tensor_data = observation.clone().into_bf16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to bf16: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<bf16>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::F32 => {
                    let obs_tensor_data = observation.clone().into_f32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f32: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<f32>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::F64 => {
                    let obs_tensor_data = observation.clone().into_f64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f64: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<f64>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I8 => {
                    let obs_tensor_data = observation.clone().into_i8_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i8: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i8>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I16 => {
                    let obs_tensor_data = observation.clone().into_i16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i16: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i16>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I32 => {
                    let obs_tensor_data = observation.clone().into_i32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i32: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i32>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I64 => {
                    let obs_tensor_data = observation.clone().into_i64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i64: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<i64>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::U8 => {
                    let obs_tensor_data = observation.clone().into_u8_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to u8: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<u8>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::Bool => {
                    let obs_tensor_data = observation.clone().into_bool_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to bool: {}",
                            e
                        ))
                    })?;
                    let obs_shape_i64: Vec<i64> =
                        obs_tensor_data.shape.iter().map(|&d| d as i64).collect();
                    TchTensor::from_slice::<u8>(bytemuck::cast_slice(&obs_tensor_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
            }
        };

        // Step 3
        let act_tensor: TchTensor = no_grad(|| module.forward_ts(&[&obs_tensor]))
            .ok()
            .expect("Failed to run forward pass");

        // Step 4
        let flattened_act: TchTensor = act_tensor.flatten(0, -1);

        // Steps 5
        let act_bytes: Vec<u8> = match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 => {
                    let vec: Vec<f16> = Vec::<f16>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to f16");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::F32 => {
                    let vec: Vec<f32> = Vec::<f32>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to f32");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::F64 => {
                    let vec: Vec<f64> = Vec::<f64>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to f64");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I8 => {
                    let vec: Vec<i8> = Vec::<i8>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i8");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I16 => {
                    let vec: Vec<i16> = Vec::<i16>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i16");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I32 => {
                    let vec: Vec<i32> = Vec::<i32>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i32");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::I64 => {
                    let vec: Vec<i64> = Vec::<i64>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i64");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                NdArrayDType::Bool => {
                    let vec: Vec<bool> = Vec::<bool>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to bool");
                    vec.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect()
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let vec: Vec<f16> = Vec::<f16>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to f16");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::Bf16 => {
                    let vec: Vec<bf16> = Vec::<bf16>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to bf16");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::F32 => {
                    let vec: Vec<f32> = Vec::<f32>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to f32");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::F64 => {
                    let vec: Vec<f64> = Vec::<f64>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to f64");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I8 => {
                    let vec: Vec<i8> = Vec::<i8>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i8");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I16 => {
                    let vec: Vec<i16> = Vec::<i16>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i16");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I32 => {
                    let vec: Vec<i32> = Vec::<i32>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i32");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::I64 => {
                    let vec: Vec<i64> = Vec::<i64>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to i64");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::U8 => {
                    let vec: Vec<u8> = Vec::<u8>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to u8");
                    bytemuck::cast_slice(&vec).to_vec()
                }
                TchDType::Bool => {
                    let vec: Vec<bool> = Vec::<bool>::try_from(flattened_act)
                        .ok()
                        .expect("Failed to convert flattened_act to bool");
                    vec.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect()
                }
            },
        };

        // Step 6
        Ok(TensorData::new(
            self.metadata.output_shape.clone(),
            self.metadata.output_dtype.clone(),
            act_bytes,
            TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
        ))
    }

    #[cfg(all(
        feature = "onnx-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn run_onnx_step<const D_IN: usize, const D_OUT: usize>(
        &self,
        session: &Arc<Session>,
        observation: AnyBurnTensor<B, D_IN>,
    ) -> Result<TensorData, ModelError> {
        // Step 1: Convert AnyBurnTensor to inner Tensor<B, D_IN, K> to metadata dtype using ConversionBurnTensor enum & methods
        // Step 2: Convert RelayRL TensorData to OrtValue
        // Step 3: Run ONNX session forward pass inference
        // Step 4: Extract tensor from output
        // Step 5: Convert tensor to bytes
        // Step 6: Convert bytes to RelayRL TensorData

        fn convert_obs_to_act<IN, OUT>(
            tensor_data: TensorData,
            session_: &Arc<Session>,
        ) -> Result<Vec<u8>, ModelError>
        where
            IN: IntoTensorElementDataType + TensorDataToType + Debug + Clone + bytemuck::Pod,
            OUT: IntoTensorElementDataType + TensorDataToType + Debug + Clone + bytemuck::Pod,
            for<'a> DynArrayRef<'a>: From<ArrayBase<CowRepr<'a, IN>, Dim<IxDynImpl>>>
        {
            let typed_data: &[IN] = bytemuck::cast_slice(&tensor_data.data);
            let array = ArrayD::from_shape_vec(tensor_data.shape, typed_data.to_vec())
                .map_err(|e| ModelError::BackendError(format!("Failed to create array: {}", e)))?;
            
            // Let Rust infer the local lifetime - don't explicitly annotate
            let cow_array = CowArray::from(&array);
            let ort_value = OrtValue::from_array(session_.allocator(), &cow_array)
                .map_err(|e| ModelError::BackendError(format!("Failed to create OrtValue: {}", e)))?;
            
            let output_value = session_.run(vec![ort_value])
                .map_err(|e| ModelError::BackendError(format!("Failed to run session: {}", e)))?;
            let first = output_value
                .into_iter()
                .next()
                .ok_or_else(|| ModelError::BackendError("No output from ONNX session".to_string()))?;
            let owned: OrtOwnedTensor<'_, OUT, IxDyn> = first.try_extract().map_err(|e| {
                ModelError::BackendError(format!("Failed to extract tensor from output: {:?}", e))
            })?;
        
            let act_vec: Vec<OUT> = owned.view().iter().copied().collect();
            let act_bytes: Vec<u8> = bytemuck::cast_slice(&act_vec).to_vec();
            Ok(act_bytes)
        }
        

        fn match_obs_to_act<IN>(input_data: TensorData, output_dtype: DType, session_: &Arc<Session>) -> Result<Vec<u8>, ModelError>
        where
            IN: IntoTensorElementDataType + TensorDataToType + Debug + Clone + bytemuck::Pod,
            for<'b> DynArrayRef<'b>: From<ArrayBase<CowRepr<'b, IN>, Dim<IxDynImpl>>>
        {
            match &output_dtype {
                #[cfg(feature = "ndarray-backend")]
                DType::NdArray(nd) => match nd {
                    NdArrayDType::F16 => {
                        convert_obs_to_act::<IN, f32>(input_data, session_)
                    }
                    NdArrayDType::F32 => {
                        convert_obs_to_act::<IN, f32>(input_data, session_)
                    }
                    NdArrayDType::F64 => {
                        convert_obs_to_act::<IN, f64>(input_data, session_)
                    }
                    NdArrayDType::I8 => {
                        convert_obs_to_act::<IN, i8>(input_data, session_)
                    }
                    NdArrayDType::I16 => {
                        convert_obs_to_act::<IN, i16>(input_data, session_)
                    }
                    NdArrayDType::I32 => {
                        convert_obs_to_act::<IN, i32>(input_data, session_)
                    }
                    NdArrayDType::I64 => {
                        convert_obs_to_act::<IN, i64>(input_data, session_)
                    }
                    NdArrayDType::Bool => {
                        convert_obs_to_act::<IN, u8>(input_data, session_)
                    }
                }
                #[cfg(feature = "tch-backend")]
                DType::Tch(tch) => match tch {
                    TchDType::F16 => {
                        convert_obs_to_act::<IN, f32>(input_data, session_)
                    }
                    TchDType::Bf16 => {
                        convert_obs_to_act::<IN, f32>(input_data, session_)
                    }
                    TchDType::F32 => {
                        convert_obs_to_act::<IN, f32>(input_data, session_)
                    }
                    TchDType::F64 => {
                        convert_obs_to_act::<IN, f64>(input_data, session_)
                    }
                    TchDType::I8 => {
                        convert_obs_to_act::<IN, i8>(input_data, session_)
                    }
                    TchDType::I16 => {
                        convert_obs_to_act::<IN, i16>(input_data, session_)
                    }
                    TchDType::I32 => {
                        convert_obs_to_act::<IN, i32>(input_data, session_)
                    }
                    TchDType::I64 => {
                        convert_obs_to_act::<IN, i64>(input_data, session_)
                    }
                    TchDType::U8 => {
                        convert_obs_to_act::<IN, u8>(input_data, session_)
                    }
                    TchDType::Bool => {
                        convert_obs_to_act::<IN, u8>(input_data, session_)
                    }
                }
            }
        }

        // Step 1
        let act_bytes = match &self.metadata.input_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(nd) => match nd {
                NdArrayDType::F16 => {
                    // ONNX doesn't support f16, so convert to f32
                    let obs_tensor_data: TensorData =
                        observation.clone().into_f32_data().map_err(|e| {
                            ModelError::BackendError(format!(
                                "Failed to convert observation to f32 (from f16): {}",
                                e
                            ))
                        })?;
                    match_obs_to_act::<f32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::F32 => {
                    let obs_tensor_data = observation.clone().into_f32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f32: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<f32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::F64 => {
                    let obs_tensor_data = observation.clone().into_f64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f64: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<f64>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::I8 => {
                    let obs_tensor_data = observation.clone().into_i8_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i8: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i8>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::I16 => {
                    let obs_tensor_data = observation.clone().into_i16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i16: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i16>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::I32 => {
                    let obs_tensor_data = observation.clone().into_i32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i32: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::I64 => {
                    let obs_tensor_data = observation.clone().into_i64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i64: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i64>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                NdArrayDType::Bool => {
                    let obs_tensor_data = observation.clone().into_bool_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to bool: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<u8>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(tch) => match tch {
                TchDType::F16 => {
                    // ONNX doesn't support f16, so convert to f32
                    let obs_tensor_data = observation.clone().into_f32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f32 (from f16): {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<f32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::Bf16 => {
                    // ONNX doesn't support bf16, so convert to f32
                    let obs_tensor_data = observation.clone().into_f32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f32 (from bf16): {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<f32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::F32 => {
                    let obs_tensor_data = observation.clone().into_f32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f32: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<f32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::F64 => {
                    let obs_tensor_data = observation.clone().into_f64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to f64: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<f64>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::I8 => {
                    let obs_tensor_data = observation.clone().into_i8_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i8: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i8>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::I16 => {
                    let obs_tensor_data = observation.clone().into_i16_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i16: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i16>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::I32 => {
                    let obs_tensor_data = observation.clone().into_i32_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i32: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i32>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::I64 => {
                    let obs_tensor_data = observation.clone().into_i64_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to i64: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<i64>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::U8 => {
                    let obs_tensor_data = observation.clone().into_u8_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to u8: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<u8>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
                TchDType::Bool => {
                    let obs_tensor_data = observation.clone().into_bool_data().map_err(|e| {
                        ModelError::BackendError(format!(
                            "Failed to convert observation to bool: {}",
                            e
                        ))
                    })?;
                    match_obs_to_act::<u8>(obs_tensor_data, self.metadata.output_dtype.clone(), session)?
                }
            },
        };

        Ok(TensorData {
            shape: self.metadata.output_shape.clone(),
            dtype: self.metadata.output_dtype.clone(),
            data: act_bytes,
            supported_backend: TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
        })
    }
}
