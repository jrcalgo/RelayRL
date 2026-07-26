pub mod hot_reloadable;
#[cfg(feature = "onnx-model")]
pub mod onnx;
pub mod utils;

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "onnx-model")]
use std::sync::Mutex;

use burn_tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use thiserror::Error;

use crate::data::action::RelayRLData;
use crate::data::tensor::{
    AnyBurnTensor, BackendMatcher, ConversionBurnTensor, DType, DeviceType, SupportedTensorBackend,
    TensorData,
};

use half::f16;

#[cfg(feature = "tch-backend")]
use half::bf16;

#[cfg(feature = "ndarray-backend")]
use crate::data::tensor::NdArrayDType;
#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
#[cfg(feature = "tch-model")]
use tch::{CModule, Tensor as TchTensor, no_grad};

#[cfg(feature = "onnx-model")]
use ort::session::Session;

pub use burn_tensor::Shape;
pub use hot_reloadable::HotReloadableModel;

/// Errors from model loading, saving, format detection, and inference execution.
#[derive(Debug, Clone, Error)]
pub enum ModelError {
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    #[error("Backend error: {0}")]
    BackendError(String),
    #[error("DType error: {0}")]
    DTypeError(String),
    #[error("Invalid input dimension: {0}")]
    InvalidInputDimension(String),
    #[error("Invalid output dimension: {0}")]
    InvalidOutputDimension(String),
    #[error("Unsupported rank: {0}")]
    UnsupportedRank(String),
    #[error("Unsupported backend: {0}")]
    UnsupportedBackend(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("JSON error: {0}")]
    JsonError(String),
    #[error("Unsupported model type: {0}")]
    UnsupportedModelType(String),
    #[error("Invalid metadata: {0}")]
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

/// On-disk format of a model file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelFileType {
    Pt,
    Onnx,
}

impl ModelFileType {
    /// Infers the model file type from the file extension (`pt` or `onnx`).
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

/// JSON metadata (`metadata.json`) that describes a model directory: shapes, dtypes, file name, and default device.
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
    /// Reads and validates `metadata.json` from `dir`.
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

    /// Serialises this metadata to `metadata.json` inside `dir`, creating the directory if needed.
    pub fn save_to_dir(&self, dir: impl Into<PathBuf>) -> Result<(), ModelError> {
        let dir: PathBuf = dir.into();
        fs::create_dir_all(&dir)?;
        let meta_path: PathBuf = dir.join("metadata.json");
        let s = serde_json::to_string_pretty(self)?;
        fs::write(meta_path, s)?;
        Ok(())
    }

    /// Returns the resolved path to the model file by joining `dir` and `model_file`.
    pub fn resolve_model_path(&self, dir: &Path) -> PathBuf {
        dir.join(&self.model_file)
    }
}

#[derive(Debug, Clone)]
pub enum InferenceModel {
    #[cfg(feature = "tch-model")]
    Pt(Arc<CModule>),
    #[cfg(feature = "onnx-model")]
    Onnx(Arc<Mutex<Session>>, Arc<onnx::OnnxSignature>),
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
                    let (session, signature) = onnx::commit_and_introspect_from_file(path)?;
                    Ok(InferenceModel::Onnx(session, Arc::new(signature)))
                }
                #[cfg(not(feature = "onnx-model"))]
                {
                    Ok(InferenceModel::Unsupported)
                }
            }
        }
    }

    fn save_to_path(&self, path: &Path) -> Result<(), ModelError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.raw_bytes.as_ref())?;
        Ok(())
    }

    fn inference(&self) -> &InferenceModel {
        &self.inference
    }

    /// Build a `Model` directly from raw ONNX bytes, without touching the filesystem.
    /// Uses ORT's `commit_from_memory` for zero-copy session creation.
    pub fn from_onnx_bytes(bytes: Vec<u8>) -> Result<Self, ModelError> {
        #[cfg(feature = "onnx-model")]
        {
            let (session, signature) = onnx::commit_and_introspect_from_memory(&bytes)?;
            let raw_bytes: Arc<[u8]> = bytes.into();
            Ok(Self {
                file_type: ModelFileType::Onnx,
                raw_bytes,
                inference: InferenceModel::Onnx(session, Arc::new(signature)),
                _phantom: PhantomData,
            })
        }
        #[cfg(not(feature = "onnx-model"))]
        {
            let raw_bytes: Arc<[u8]> = bytes.into();
            Ok(Self {
                file_type: ModelFileType::Onnx,
                raw_bytes,
                inference: InferenceModel::Unsupported,
                _phantom: PhantomData,
            })
        }
    }
}

/// Validates a freshly-built `Model` against RelayRL-supplied `metadata`.
///
/// For ONNX models this checks the discovered graph signature (I/O count, tensor kind,
/// element type, and fixed dimensions) against `metadata`; other model kinds are not
/// currently introspected and pass through unchanged.
#[cfg_attr(not(feature = "onnx-model"), allow(unused_variables))]
fn validate_model_against_metadata<B: Backend + BackendMatcher<Backend = B>>(
    model: &Model<B>,
    metadata: &ModelMetadata,
) -> Result<(), ModelError> {
    match model.inference() {
        #[cfg(feature = "onnx-model")]
        InferenceModel::Onnx(_, signature) => {
            onnx::validate_metadata_against_signature(metadata, signature)
        }
        _ => Ok(()),
    }
}

/// A loaded model bundle: the inference engine and its `ModelMetadata`. Use `load_from_path` to construct.
///
/// ```ignore
/// use relayrl::types::model::ModelModule;
/// use burn_ndarray::NdArray;
///
/// let model = ModelModule::<NdArray>::load_from_path("model_dir")?;
/// ```
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

        Self::finish(model, metadata)
    }

    /// Validates `model` against `metadata` (schema/type/shape compatibility for ONNX graphs)
    /// before assembling the final `ModelModule`. All constructors route through this so no
    /// path can produce a module whose declared metadata disagrees with the loaded model.
    fn finish(model: Model<B>, metadata: ModelMetadata) -> Result<Self, ModelError> {
        validate_model_against_metadata(&model, &metadata)?;
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

    /// Build a `ModelModule` directly from raw ONNX bytes without touching the filesystem.
    /// The caller supplies the `metadata` describing input/output shapes and dtypes; it is
    /// validated against the ONNX graph's discovered signature before this returns.
    pub fn from_onnx_bytes(bytes: Vec<u8>, metadata: ModelMetadata) -> Result<Self, ModelError> {
        let model = Model::<B>::from_onnx_bytes(bytes)?;
        Self::finish(model, metadata)
    }

    /// Build a `ModelModule` from TorchScript bytes via a temporary file.
    /// Since `CModule::load` requires a filesystem path, this method writes the bytes
    /// to a temporary file, loads the model, then cleans up the temp file.
    /// The caller supplies the `metadata` describing input/output shapes and dtypes.
    #[cfg(feature = "tch-model")]
    pub fn from_pt_bytes(bytes: Vec<u8>, metadata: ModelMetadata) -> Result<Self, ModelError> {
        use std::io::Write;

        // Write bytes to a temporary file
        let mut temp_file = tempfile::NamedTempFile::new()
            .map_err(|e| ModelError::BackendError(format!("Failed to create temp file: {}", e)))?;

        temp_file
            .write_all(&bytes)
            .map_err(|e| ModelError::BackendError(format!("Failed to write temp file: {}", e)))?;

        let temp_path = temp_file.path();

        // Load the model from the temp file
        let module = CModule::load(temp_path)
            .map_err(|e| ModelError::BackendError(format!("Failed to load CModule: {}", e)))?;

        let raw_bytes: Arc<[u8]> = bytes.into();
        let model = Model {
            file_type: ModelFileType::Pt,
            raw_bytes,
            inference: InferenceModel::Pt(Arc::new(module)),
            _phantom: PhantomData,
        };

        // Temp file is automatically cleaned up when dropped
        Self::finish(model, metadata)
    }

    /// Stores TorchScript bytes without an active inference engine when the `tch-model` feature is disabled.
    #[cfg(not(feature = "tch-model"))]
    pub fn from_pt_bytes(bytes: Vec<u8>, metadata: ModelMetadata) -> Result<Self, ModelError> {
        let raw_bytes: Arc<[u8]> = bytes.into();
        let model = Model {
            file_type: ModelFileType::Pt,
            raw_bytes,
            inference: InferenceModel::Unsupported,
            _phantom: PhantomData,
        };
        Self::finish(model, metadata)
    }

    /// Fallible single-step inference. Prefer this over [`Self::step`] in runtime paths.
    ///
    /// Falls back to a zero action only for [`ModelError::UnsupportedModelType`] (no inference
    /// engine compiled in). Genuine engine/conversion failures are propagated.
    #[cfg(all(
        any(feature = "tch-model", feature = "onnx-model"),
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    #[allow(clippy::type_complexity)]
    pub fn try_step<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
    ) -> Result<(TensorData, Option<TensorData>, HashMap<String, RelayRLData>), ModelError> {
        let base_action = match self.run_inference::<D_IN, D_OUT>(observation) {
            Ok(action) => action,
            Err(ModelError::UnsupportedModelType(_)) => self.zeros_action::<D_OUT>()?,
            Err(error) => return Err(error),
        };

        let mask_td = mask
            .map(|mask_tensor| self.mask_to_tensor_data(mask_tensor))
            .transpose()?;

        let act_td = match mask_td.as_ref() {
            Some(mask) => Self::apply_mask_to_action(base_action, mask),
            None => base_action,
        };

        Ok((act_td, mask_td, HashMap::new()))
    }

    /// Compatibility wrapper around [`Self::try_step`]. Prefer `try_step` in new code.
    ///
    /// On error, logs and returns a zero action (or an empty tensor if zero construction also
    /// fails) rather than panicking.
    #[cfg(all(
        any(feature = "tch-model", feature = "onnx-model"),
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    pub fn step<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
    ) -> (TensorData, Option<TensorData>, HashMap<String, RelayRLData>) {
        match self.try_step::<D_IN, D_OUT>(observation, mask) {
            Ok(result) => result,
            Err(error) => {
                log::error!(
                    "[ModelModule::step] inference failed, returning zero-action fallback: {error}"
                );
                match self.zeros_action::<D_OUT>() {
                    Ok(action) => (action, None, HashMap::new()),
                    Err(fallback_error) => {
                        log::error!(
                            "[ModelModule::step] zero-action fallback failed: {fallback_error}"
                        );
                        (
                            TensorData::new(
                                self.metadata.output_shape.clone(),
                                self.metadata.output_dtype.clone(),
                                Vec::new(),
                                TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
                            ),
                            None,
                            HashMap::new(),
                        )
                    }
                }
            }
        }
    }

    #[cfg(all(
        any(feature = "tch-model", feature = "onnx-model"),
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    #[allow(clippy::type_complexity)]
    /// Runs batched inference over a slice of observations, returning one `(action, mask, aux)` per entry.
    pub fn step_batch<const D_IN: usize, const D_OUT: usize>(
        &self,
        observations: &[Arc<AnyBurnTensor<B, D_IN>>],
        masks: &[Option<Arc<AnyBurnTensor<B, D_OUT>>>],
    ) -> Result<Vec<(TensorData, Option<TensorData>, HashMap<String, RelayRLData>)>, ModelError>
    {
        if observations.is_empty() {
            return Ok(Vec::new());
        }

        if observations.len() != masks.len() {
            return Err(ModelError::InvalidInputDimension(format!(
                "batch observation/mask length mismatch: {} observations vs {} masks",
                observations.len(),
                masks.len()
            )));
        }

        let observation_data: Vec<TensorData> = observations
            .iter()
            .map(|observation| self.observation_to_tensor_data(observation.clone()))
            .collect::<Result<_, _>>()?;
        let mask_data: Vec<Option<TensorData>> = masks
            .iter()
            .map(|mask| {
                mask.as_ref()
                    .map(|mask_tensor| self.mask_to_tensor_data(mask_tensor.clone()))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;

        let batched_input = Self::stack_tensor_data(&observation_data)?;
        let batched_output = match self.run_inference_tensor_data(batched_input) {
            Ok(output) => output,
            Err(ModelError::UnsupportedModelType(_)) => {
                self.try_zeros_batch_action(observations.len())?
            }
            Err(error) => return Err(error),
        };
        let split_actions = Self::split_tensor_data_rows(batched_output, observations.len())?;

        Ok(split_actions
            .into_iter()
            .zip(mask_data)
            .map(|(base_action, mask_td)| {
                let act_td = match mask_td.as_ref() {
                    Some(mask) => Self::apply_mask_to_action(base_action, mask),
                    None => base_action,
                };
                (act_td, mask_td, HashMap::new())
            })
            .collect())
    }

    fn observation_to_tensor_data<const D_IN: usize>(
        &self,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
    ) -> Result<TensorData, ModelError> {
        match observation.as_ref() {
            AnyBurnTensor::Float(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: self.metadata.input_dtype.clone(),
            }),
            AnyBurnTensor::Int(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: self.metadata.input_dtype.clone(),
            }),
            AnyBurnTensor::Bool(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: self.metadata.input_dtype.clone(),
            }),
        }
        .map_err(|e| ModelError::BackendError(format!("Tensor conversion failed: {e}")))
    }

    fn mask_to_tensor_data<const D_OUT: usize>(
        &self,
        mask: Arc<AnyBurnTensor<B, D_OUT>>,
    ) -> Result<TensorData, ModelError> {
        match mask.as_ref() {
            AnyBurnTensor::Float(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: self.metadata.output_dtype.clone(),
            }),
            AnyBurnTensor::Int(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: self.metadata.output_dtype.clone(),
            }),
            AnyBurnTensor::Bool(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: self.metadata.output_dtype.clone(),
            }),
        }
        .map_err(|e| ModelError::BackendError(format!("Mask conversion failed: {e}")))
    }

    fn apply_mask_to_action(base_action: TensorData, mask: &TensorData) -> TensorData {
        let action_data: Vec<u8> = base_action
            .data
            .iter()
            .zip(mask.data.iter())
            .map(|(a, m)| a * m)
            .collect();
        TensorData {
            shape: base_action.shape,
            dtype: base_action.dtype,
            data: action_data,
            supported_backend: base_action.supported_backend,
        }
    }

    fn stack_tensor_data(rows: &[TensorData]) -> Result<TensorData, ModelError> {
        let first = rows.first().ok_or_else(|| {
            ModelError::InvalidInputDimension("cannot stack an empty tensor batch".to_string())
        })?;

        for row in rows.iter().skip(1) {
            if row.dtype != first.dtype || row.supported_backend != first.supported_backend {
                return Err(ModelError::DTypeError(
                    "all batched observations must share dtype/backend".to_string(),
                ));
            }
            if row.shape != first.shape {
                return Err(ModelError::InvalidInputDimension(format!(
                    "all batched observations must share shape: expected {:?}, got {:?}",
                    first.shape, row.shape
                )));
            }
        }

        let mut shape = Vec::with_capacity(first.shape.len() + 1);
        shape.push(rows.len());
        shape.extend(first.shape.iter().copied());

        let mut data = Vec::with_capacity(first.data.len() * rows.len());
        for row in rows {
            data.extend_from_slice(&row.data);
        }

        Ok(TensorData::new(
            shape,
            first.dtype.clone(),
            data,
            first.supported_backend.clone(),
        ))
    }

    fn split_tensor_data_rows(
        batch: TensorData,
        rows: usize,
    ) -> Result<Vec<TensorData>, ModelError> {
        if rows == 0 {
            return Ok(Vec::new());
        }

        let Some((&batch_rows, row_shape)) = batch.shape.split_first() else {
            return Err(ModelError::InvalidOutputDimension(
                "batched action tensor must have at least one dimension".to_string(),
            ));
        };

        if batch_rows != rows {
            return Err(ModelError::InvalidOutputDimension(format!(
                "batched action count mismatch: tensor has {} rows, expected {}",
                batch_rows, rows
            )));
        }

        let row_bytes = batch.data.len() / rows;
        if row_bytes * rows != batch.data.len() {
            return Err(ModelError::InvalidOutputDimension(
                "batched action data length is not divisible by row count".to_string(),
            ));
        }

        let row_shape = row_shape.to_vec();
        let mut result = Vec::with_capacity(rows);
        for index in 0..rows {
            let start = index * row_bytes;
            let end = start + row_bytes;
            result.push(TensorData::new(
                row_shape.clone(),
                batch.dtype.clone(),
                batch.data[start..end].to_vec(),
                batch.supported_backend.clone(),
            ));
        }
        Ok(result)
    }

    /// Resolves the Burn device for this model's preferred/default device.
    pub(crate) fn try_resolve_device(&self) -> Result<<B as Backend>::Device, ModelError> {
        let preferred = self.metadata.default_device.clone().unwrap_or_default();
        <B as BackendMatcher>::get_device(&preferred)
            .or_else(|_| <B as BackendMatcher>::get_device(&DeviceType::default()))
            .map_err(|error| {
                ModelError::BackendError(format!("Failed to resolve backend device: {error}"))
            })
    }

    fn try_zeros_batch_action(&self, rows: usize) -> Result<TensorData, ModelError> {
        let mut shape = Vec::with_capacity(self.metadata.output_shape.len() + 1);
        shape.push(rows);
        shape.extend(self.metadata.output_shape.iter().copied());
        let row_zero = self.zeros_action::<1>()?;
        let mut data = Vec::with_capacity(row_zero.data.len() * rows);
        for _ in 0..rows {
            data.extend_from_slice(&row_zero.data);
        }
        Ok(TensorData::new(
            shape,
            row_zero.dtype,
            data,
            row_zero.supported_backend,
        ))
    }

    fn zeros_action<const D_OUT: usize>(&self) -> Result<TensorData, ModelError> {
        let shape = Shape::from(self.metadata.output_shape.clone());

        // Create zeros tensor based on output dtype
        match &self.metadata.output_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 => {
                    let data_vec = vec![f16::ZERO; shape.dims.iter().product()];
                    let data: &[f16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f16, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<f32, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<f64, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i8, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i16, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i32, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i64, u8>(data);
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
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let data_vec = vec![f16::ZERO; shape.dims.iter().product()];
                    let data: &[f16] = data_vec.as_slice();
                    let u8_data = bytemuck::cast_slice::<f16, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<bf16, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<f32, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<f64, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i8, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i16, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i32, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<i64, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<u8, u8>(data);
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
                    let u8_data = bytemuck::cast_slice::<bool, u8>(data);
                    Ok(TensorData::new(
                        shape.dims.to_vec(),
                        DType::Tch(dtype.clone()),
                        u8_data.to_vec(),
                        SupportedTensorBackend::Tch,
                    ))
                }
            },
        }
    }

    fn run_inference<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
    ) -> Result<TensorData, ModelError> {
        match self.model.inference() {
            #[cfg(feature = "tch-model")]
            InferenceModel::Pt(module) => {
                self.run_libtorch_step::<D_IN, D_OUT>(module, observation)
            }
            #[cfg(feature = "onnx-model")]
            InferenceModel::Onnx(session, signature) => {
                let input_data = self.observation_to_tensor_data(observation)?;
                self.run_onnx_tensor(session, signature, input_data)
            }
            _ => Err(ModelError::UnsupportedModelType(
                "Unsupported model type".to_string(),
            )),
        }
    }

    /// Runs inference over a pre-stacked flat `TensorData` batch and returns the output `TensorData`.
    pub fn flat_batch_inference(&self, input_data: TensorData) -> Result<TensorData, ModelError> {
        self.run_inference_tensor_data(input_data)
    }

    /// Fallible zero-filled batch output with the model's output shape, repeated `rows` times.
    pub fn try_flat_batch_zeros(&self, rows: usize) -> Result<TensorData, ModelError> {
        self.try_zeros_batch_action(rows)
    }

    /// Returns a zero-filled output `TensorData` with the model's output shape, repeated `rows` times.
    ///
    /// Prefer [`Self::try_flat_batch_zeros`] in runtime paths. On failure this logs and returns an
    /// empty tensor rather than panicking.
    pub fn flat_batch_zeros(&self, rows: usize) -> TensorData {
        match self.try_flat_batch_zeros(rows) {
            Ok(data) => data,
            Err(error) => {
                log::error!(
                    "[ModelModule::flat_batch_zeros] {error}; returning empty tensor fallback"
                );
                let mut shape = Vec::with_capacity(self.metadata.output_shape.len() + 1);
                shape.push(rows);
                shape.extend(self.metadata.output_shape.iter().copied());
                TensorData::new(
                    shape,
                    self.metadata.output_dtype.clone(),
                    Vec::new(),
                    TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
                )
            }
        }
    }

    fn run_inference_tensor_data(&self, input_data: TensorData) -> Result<TensorData, ModelError> {
        match self.model.inference() {
            #[cfg(feature = "tch-model")]
            InferenceModel::Pt(module) => self.run_libtorch_step_data(module, input_data),
            #[cfg(feature = "onnx-model")]
            InferenceModel::Onnx(session, signature) => {
                self.run_onnx_tensor(session, signature, input_data)
            }
            _ => Err(ModelError::UnsupportedModelType(
                "Unsupported model type".to_string(),
            )),
        }
    }

    /// Runs the ONNX graph's single input → single output tensor pass, binding by the
    /// discovered input/output names and returning the actual runtime output shape.
    #[cfg(all(
        feature = "onnx-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn run_onnx_tensor(
        &self,
        session: &Arc<std::sync::Mutex<Session>>,
        signature: &Arc<onnx::OnnxSignature>,
        input_data: TensorData,
    ) -> Result<TensorData, ModelError> {
        onnx::run_tensor(
            session,
            signature,
            &input_data.dtype,
            &self.metadata.output_dtype,
            &input_data.shape,
            &input_data.data,
        )
    }

    #[cfg(all(
        feature = "tch-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn run_tch_forward(
        module: &Arc<CModule>,
        obs_tensor: &TchTensor,
    ) -> Result<TchTensor, ModelError> {
        no_grad(|| module.forward_ts(&[obs_tensor])).map_err(|error| {
            ModelError::BackendError(format!("LibTorch forward pass failed: {error}"))
        })
    }

    #[cfg(all(
        feature = "tch-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn tch_flattened_to_bytes(flattened: TchTensor, dtype: &DType) -> Result<Vec<u8>, ModelError> {
        match dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(dtype) => match dtype {
                NdArrayDType::F16 => {
                    let vec = Vec::<f16>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to f16: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::F32 => {
                    let vec = Vec::<f32>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to f32: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::F64 => {
                    let vec = Vec::<f64>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to f64: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::I8 => {
                    let vec = Vec::<i8>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i8: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::I16 => {
                    let vec = Vec::<i16>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i16: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::I32 => {
                    let vec = Vec::<i32>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i32: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::I64 => {
                    let vec = Vec::<i64>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i64: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                NdArrayDType::Bool => {
                    let vec = Vec::<bool>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to bool: {error}"
                        ))
                    })?;
                    Ok(vec.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect())
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(dtype) => match dtype {
                TchDType::F16 => {
                    let vec = Vec::<f16>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to f16: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::Bf16 => {
                    let vec = Vec::<bf16>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to bf16: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::F32 => {
                    let vec = Vec::<f32>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to f32: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::F64 => {
                    let vec = Vec::<f64>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to f64: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::I8 => {
                    let vec = Vec::<i8>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i8: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::I16 => {
                    let vec = Vec::<i16>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i16: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::I32 => {
                    let vec = Vec::<i32>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i32: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::I64 => {
                    let vec = Vec::<i64>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to i64: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::U8 => {
                    let vec = Vec::<u8>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to u8: {error}"
                        ))
                    })?;
                    Ok(bytemuck::cast_slice(&vec).to_vec())
                }
                TchDType::Bool => {
                    let vec = Vec::<bool>::try_from(flattened).map_err(|error| {
                        ModelError::BackendError(format!(
                            "Failed to convert LibTorch output to bool: {error}"
                        ))
                    })?;
                    Ok(vec.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect())
                }
            },
        }
    }

    #[cfg(all(
        feature = "tch-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn run_libtorch_step_data(
        &self,
        module: &Arc<CModule>,
        input_data: TensorData,
    ) -> Result<TensorData, ModelError> {
        let obs_shape_i64: Vec<i64> = input_data.shape.iter().map(|&d| d as i64).collect();
        let obs_tensor: TchTensor = match &input_data.dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(nd) => match nd {
                NdArrayDType::F16 => {
                    TchTensor::from_slice::<f16>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::F32 => {
                    TchTensor::from_slice::<f32>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::F64 => {
                    TchTensor::from_slice::<f64>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I8 => {
                    TchTensor::from_slice::<i8>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I16 => {
                    TchTensor::from_slice::<i16>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I32 => {
                    TchTensor::from_slice::<i32>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::I64 => {
                    TchTensor::from_slice::<i64>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                NdArrayDType::Bool => {
                    TchTensor::from_slice::<u8>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
            },
            #[cfg(feature = "tch-backend")]
            DType::Tch(tch) => match tch {
                TchDType::F16 => {
                    TchTensor::from_slice::<f16>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::Bf16 => {
                    TchTensor::from_slice::<bf16>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::F32 => {
                    TchTensor::from_slice::<f32>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::F64 => {
                    TchTensor::from_slice::<f64>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I8 => TchTensor::from_slice::<i8>(bytemuck::cast_slice(&input_data.data))
                    .reshape(obs_shape_i64.as_slice()),
                TchDType::I16 => {
                    TchTensor::from_slice::<i16>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I32 => {
                    TchTensor::from_slice::<i32>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::I64 => {
                    TchTensor::from_slice::<i64>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
                TchDType::U8 => TchTensor::from_slice::<u8>(bytemuck::cast_slice(&input_data.data))
                    .reshape(obs_shape_i64.as_slice()),
                TchDType::Bool => {
                    TchTensor::from_slice::<u8>(bytemuck::cast_slice(&input_data.data))
                        .reshape(obs_shape_i64.as_slice())
                }
            },
        };

        let act_tensor = Self::run_tch_forward(module, &obs_tensor)?;
        let output_shape: Vec<usize> = act_tensor
            .size()
            .into_iter()
            .map(|dim| dim as usize)
            .collect();
        let flattened_act: TchTensor = act_tensor.flatten(0, -1);
        let act_bytes = Self::tch_flattened_to_bytes(flattened_act, &self.metadata.output_dtype)?;

        Ok(TensorData::new(
            output_shape,
            self.metadata.output_dtype.clone(),
            act_bytes,
            TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
        ))
    }

    #[cfg(all(
        feature = "tch-model",
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn run_libtorch_step<const D_IN: usize, const D_OUT: usize>(
        &self,
        module: &Arc<CModule>,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
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
            },
        };

        // Step 3-5: forward, flatten, and convert without panicking.
        let act_tensor = Self::run_tch_forward(module, &obs_tensor)?;
        let flattened_act: TchTensor = act_tensor.flatten(0, -1);
        let act_bytes = Self::tch_flattened_to_bytes(flattened_act, &self.metadata.output_dtype)?;

        // Step 6
        Ok(TensorData::new(
            self.metadata.output_shape.clone(),
            self.metadata.output_dtype.clone(),
            act_bytes,
            TensorData::get_backend_from_dtype(&self.metadata.output_dtype),
        ))
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::marker::PhantomData;

    use crate::data::tensor::FloatBurnTensor;
    use burn_tensor::TensorData as BurnTensorData;

    use uuid::Uuid;

    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    use burn_ndarray::NdArray;
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    use burn_tensor::{Float, Tensor};

    fn temp_dir_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("relayrl-model-{label}-{}", Uuid::new_v4()))
    }

    #[test]
    fn model_file_type_parses_supported_extensions() {
        assert_eq!(
            ModelFileType::from_path(Path::new("policy.pt")).unwrap(),
            ModelFileType::Pt
        );
        assert_eq!(
            ModelFileType::from_path(Path::new("policy.onnx")).unwrap(),
            ModelFileType::Onnx
        );
        assert!(matches!(
            ModelFileType::from_path(Path::new("policy.bin")),
            Err(ModelError::UnsupportedModelType(message)) if message.contains("Unsupported extension")
        ));
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn model_metadata_save_load_round_trip_preserves_paths() {
        let dir = temp_dir_path("metadata-roundtrip");
        let metadata = ModelMetadata {
            model_file: "policy.onnx".to_string(),
            model_type: ModelFileType::Onnx,
            input_dtype: DType::NdArray(NdArrayDType::F32),
            output_dtype: DType::NdArray(NdArrayDType::F32),
            input_shape: vec![2],
            output_shape: vec![2],
            default_device: Some(DeviceType::Cpu),
        };

        metadata.save_to_dir(&dir).unwrap();
        let loaded = ModelMetadata::load_from_dir(&dir).unwrap();

        assert_eq!(loaded.model_file, "policy.onnx");
        assert_eq!(loaded.resolve_model_path(&dir), dir.join("policy.onnx"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(feature = "ndarray-backend")]
    fn model_metadata_load_rejects_invalid_fields() {
        let dir = temp_dir_path("metadata-invalid");
        let metadata = ModelMetadata {
            model_file: String::new(),
            model_type: ModelFileType::Onnx,
            input_dtype: DType::NdArray(NdArrayDType::F32),
            output_dtype: DType::NdArray(NdArrayDType::F32),
            input_shape: vec![2],
            output_shape: vec![2],
            default_device: Some(DeviceType::Cpu),
        };

        metadata.save_to_dir(&dir).unwrap();
        let err = ModelMetadata::load_from_dir(&dir)
            .expect_err("metadata with an empty model file should be rejected");

        assert!(matches!(
            err,
            ModelError::InvalidMetadata(message) if message.contains("model_file is empty")
        ));

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn stub_module(output_shape: Vec<usize>) -> ModelModule<NdArray> {
        ModelModule {
            model: Model {
                file_type: ModelFileType::Onnx,
                raw_bytes: Arc::<[u8]>::from(vec![1u8, 2, 3]),
                inference: InferenceModel::Unsupported,
                _phantom: PhantomData,
            },
            metadata: ModelMetadata {
                model_file: "test.onnx".to_string(),
                model_type: ModelFileType::Onnx,
                input_dtype: DType::NdArray(NdArrayDType::F32),
                output_dtype: DType::NdArray(NdArrayDType::F32),
                input_shape: vec![2],
                output_shape,
                default_device: Some(DeviceType::Cpu),
            },
        }
    }

    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn float_any_tensor(values: &[f32]) -> Arc<AnyBurnTensor<NdArray, 1>> {
        let device = NdArray::get_device(&DeviceType::Cpu).unwrap();
        let tensor = Tensor::<NdArray, 1, Float>::from_data(
            BurnTensorData::new(values.to_vec(), [values.len()]),
            &device,
        );

        Arc::new(AnyBurnTensor::Float(FloatBurnTensor {
            tensor: Arc::new(tensor),
            dtype: DType::NdArray(NdArrayDType::F32),
        }))
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn model_module_save_writes_metadata_and_model_bytes() {
        let dir = temp_dir_path("module-save");
        let module = stub_module(vec![2]);

        module.save(&dir).unwrap();

        assert!(dir.join("metadata.json").exists());
        assert_eq!(fs::read(dir.join("test.onnx")).unwrap(), vec![1, 2, 3]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn resolve_device_returns_cpu_for_ndarray_models() {
        let module = stub_module(vec![2]);
        assert!(matches!(
            module.try_resolve_device().expect("device should resolve"),
            burn_tensor::Device::<NdArray>::Cpu
        ));
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn zeros_action_matches_output_shape_dtype_and_backend() {
        let module = stub_module(vec![2]);
        let zero_action = module.zeros_action::<1>().unwrap();

        assert_eq!(zero_action.shape, vec![2]);
        assert_eq!(zero_action.dtype, DType::NdArray(NdArrayDType::F32));
        assert_eq!(
            zero_action.supported_backend,
            SupportedTensorBackend::NdArray
        );
        assert_eq!(zero_action.data, vec![0; 8]);
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn step_falls_back_to_zero_actions_when_inference_is_unavailable() {
        let module = stub_module(vec![2]);
        let observation = float_any_tensor(&[1.0, 2.0]);
        let mask = float_any_tensor(&[1.0, 0.0]);

        let (action, mask_data, aux) = module.step::<1, 1>(observation, Some(mask));

        assert!(aux.is_empty());
        assert_eq!(action.shape, vec![2]);
        assert_eq!(action.data, vec![0; 8]);
        assert_eq!(
            mask_data.expect("mask data should be preserved").data,
            [1.0f32, 0.0]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn try_step_falls_back_to_zero_actions_when_inference_is_unavailable() {
        let module = stub_module(vec![2]);
        let observation = float_any_tensor(&[1.0, 2.0]);
        let mask = float_any_tensor(&[1.0, 0.0]);

        let (action, mask_data, aux) = module
            .try_step::<1, 1>(observation, Some(mask))
            .expect("UnsupportedModelType should fall back to zeros");

        assert!(aux.is_empty());
        assert_eq!(action.shape, vec![2]);
        assert_eq!(action.data, vec![0; 8]);
        assert_eq!(
            mask_data.expect("mask data should be preserved").data,
            [1.0f32, 0.0]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn step_batch_falls_back_only_for_unsupported_model_type() {
        let module = stub_module(vec![2]);
        let observations = vec![float_any_tensor(&[1.0, 2.0]), float_any_tensor(&[3.0, 4.0])];
        let masks = vec![None, None];

        let steps = module
            .step_batch::<1, 1>(&observations, &masks)
            .expect("UnsupportedModelType should fall back to zeros");

        assert_eq!(steps.len(), 2);
        for (action, mask, aux) in steps {
            assert!(mask.is_none());
            assert!(aux.is_empty());
            assert_eq!(action.shape, vec![2]);
            assert_eq!(action.data, vec![0; 8]);
        }
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn try_flat_batch_zeros_returns_result() {
        let module = stub_module(vec![2]);
        let zeros = module
            .try_flat_batch_zeros(3)
            .expect("zeros batch should succeed");
        assert_eq!(zeros.shape, vec![3, 2]);
        assert_eq!(zeros.data, vec![0; 24]);
    }

    #[test]
    #[cfg(all(
        feature = "ndarray-backend",
        any(feature = "tch-model", feature = "onnx-model")
    ))]
    fn step_batch_rejects_observation_mask_length_mismatch() {
        let module = stub_module(vec![2]);
        let observations = vec![float_any_tensor(&[1.0, 2.0])];
        let masks = vec![None, None];
        let err = module
            .step_batch::<1, 1>(&observations, &masks)
            .expect_err("mismatched lengths should fail");
        assert!(matches!(err, ModelError::InvalidInputDimension(_)));
    }
}
