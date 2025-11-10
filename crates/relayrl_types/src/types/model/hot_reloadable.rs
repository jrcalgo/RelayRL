use crate::types::data::tensor::DeviceType;
use std::arch::is_aarch64_feature_detected;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use tokio::sync::RwLock;
use uuid::Uuid;

use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

use crate::types::data::action::{RelayRLAction, RelayRLData};
use crate::types::data::tensor::{
    AnyBurnTensor, BackendMatcher, ConversionBurnTensor, TensorData, TensorError,
};
use crate::types::data::tensor::{
    BoolBurnTensor, DType, FloatBurnTensor, IntBurnTensor, NdArrayDType, TchDType,
};
use crate::types::model::utils::validate_module;
use crate::types::model::{ModelError, ModelModule};

/// Wrapper that lets us swap the underlying model at runtime and run inference
/// in an async-safe way.
pub struct HotReloadableModel<B: Backend + BackendMatcher<Backend = B>> {
    inner: RwLock<Arc<ModelModule<B>>>,
    version: Arc<AtomicI64>,
    default_device: DeviceType,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend + BackendMatcher<Backend = B>> Clone for HotReloadableModel<B> {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.inner.blocking_read().clone()),
            version: self.version.clone(),
            default_device: self.default_device.clone(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> HotReloadableModel<B> {
    pub async fn new_from_path<P: AsRef<Path>>(
        path: P,
        device: DeviceType,
    ) -> Result<Self, ModelError> {
        let module: ModelModule<B> = ModelModule::<B>::load_from_path(path.as_ref().to_path_buf())?;
        validate_module::<B>(&module)?;

        Ok(Self {
            inner: RwLock::new(Arc::new(module.to_owned())),
            version: Arc::new(AtomicI64::new(0)),
            default_device: device,
            input_dim: module.metadata.input_shape.len(),
            output_dim: module.metadata.output_shape.len(),
        })
    }

    pub async fn new_from_module(
        module: ModelModule<B>,
        device: DeviceType,
    ) -> Result<Self, ModelError> {
        validate_module::<B>(&module)?;

        Ok(Self {
            inner: RwLock::new(Arc::new(module.to_owned())),
            version: Arc::new(AtomicI64::new(0)),
            default_device: device,
            input_dim: module.metadata.input_shape.len(),
            output_dim: module.metadata.output_shape.len(),
        })
    }

    /// Atomically swap the model from disk and bump version.
    pub async fn reload_from_path(&self, path: PathBuf, version: i64) -> Result<i64, ModelError> {
        let new_module = Arc::new(ModelModule::<B>::load_from_path(path)?);
        {
            let mut guard = self.inner.write().await;
            *guard = new_module;
        }
        self.version.store(version, Ordering::SeqCst);
        Ok(version)
    }

    pub async fn reload_from_module(
        &self,
        module: ModelModule<B>,
        version: i64,
    ) -> Result<i64, ModelError> {
        {
            let mut guard = self.inner.write().await;
            *guard = Arc::new(module);
        }
        self.version.store(version, Ordering::SeqCst);
        Ok(version)
    }

    pub fn version(&self) -> i64 {
        self.version.load(Ordering::SeqCst)
    }

    /// Generic forward that works for any backend / rank.
    pub fn forward<const I: usize, const O: usize>(
        &self,
        observation: AnyBurnTensor<B, I>,
        mask: Option<AnyBurnTensor<B, O>>,
        reward: f32,
        actor_id: Uuid,
    ) -> Result<RelayRLAction, ModelError> {
        let model_module = self.inner.blocking_read();
        let (act_td, aux) = model_module.step(observation.clone(), mask.clone());

        let obs_dtype = aux
            .get("observation_dtype")
            .and_then(|d| match d {
                RelayRLData::DType(dtype) => Some(dtype.clone()),
                _ => None,
            })
            .unwrap_or_else(default_dtype);

        let act_dtype = aux
            .get("action_dtype")
            .and_then(|d| match d {
                RelayRLData::DType(dtype) => Some(dtype.clone()),
                _ => None,
            })
            .unwrap_or_else(default_dtype);

        // Build RelayRLAction by converting tensors → TensorData
        let obs_td = match observation.clone() {
            AnyBurnTensor::Float(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor,
                conversion_dtype: obs_dtype.clone(),
            }),
            AnyBurnTensor::Int(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.float(),
                conversion_dtype: obs_dtype.clone(),
            }),
            AnyBurnTensor::Bool(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.float(),
                conversion_dtype: obs_dtype.clone(),
            }),
        }
        .map_err(|e| ModelError::BackendError(format!("Tensor conversion failed: {e}")))?;

        let mask_td: Option<TensorData> = match mask {
            Some(mask) => {
                let (tensor_kind, dtype) = mask.clone().get_tensor_type();
                let result: Result<TensorData, TensorError> = match tensor_kind.as_str() {
                    "float" => match dtype {
                        #[cfg(feature = "ndarray-backend")]
                        DType::NdArray(dtype) => match dtype {
                            NdArrayDType::F16 => mask.into_f16_data(),
                            NdArrayDType::F32 => mask.into_f32_data(),
                            NdArrayDType::F64 => mask.into_f64_data(),
                            _ => Err(TensorError::DTypeError(format!(
                                "Unsupported mask dtype: {}",
                                dtype
                            ))),
                        },
                        #[cfg(feature = "tch-backend")]
                        DType::Tch(dtype) => match dtype {
                            TchDType::F16 => mask.into_f16_data(),
                            TchDType::Bf16 => mask.into_bf16_data(),
                            TchDType::F32 => mask.into_f32_data(),
                            TchDType::F64 => mask.into_f64_data(),
                            _ => Err(TensorError::DTypeError(format!(
                                "Unsupported mask dtype: {}",
                                dtype
                            ))),
                        },
                        _ => Err(TensorError::DTypeError(format!(
                            "Unsupported mask dtype: {:?}",
                            dtype
                        ))),
                    },
                    "int" => match dtype {
                        #[cfg(feature = "ndarray-backend")]
                        DType::NdArray(dtype) => match dtype {
                            NdArrayDType::I8 => mask.into_i8_data(),
                            NdArrayDType::I16 => mask.into_i16_data(),
                            NdArrayDType::I32 => mask.into_i32_data(),
                            NdArrayDType::I64 => mask.into_i64_data(),
                            _ => Err(TensorError::DTypeError(format!(
                                "Unsupported mask dtype: {}",
                                dtype
                            ))),
                        },
                        #[cfg(feature = "tch-backend")]
                        DType::Tch(dtype) => match dtype {
                            TchDType::I8 => mask.into_i8_data(),
                            TchDType::I16 => mask.into_i16_data(),
                            TchDType::I32 => mask.into_i32_data(),
                            TchDType::I64 => mask.into_i64_data(),
                            TchDType::U8 => mask.into_u8_data(),
                            _ => Err(TensorError::DTypeError(format!(
                                "Unsupported mask dtype: {}",
                                dtype
                            ))),
                        },
                        _ => Err(TensorError::DTypeError(format!(
                            "Unsupported mask dtype: {:?}",
                            dtype
                        ))),
                    },
                    "bool" => match dtype {
                        #[cfg(feature = "ndarray-backend")]
                        DType::NdArray(dtype) => match dtype {
                            NdArrayDType::Bool => mask.into_bool_data(),
                            _ => Err(TensorError::DTypeError(format!(
                                "Unsupported mask dtype: {}",
                                dtype
                            ))),
                        },
                        #[cfg(feature = "tch-backend")]
                        DType::Tch(dtype) => match dtype {
                            TchDType::Bool => mask.into_bool_data(),
                            _ => Err(TensorError::DTypeError(format!(
                                "Unsupported mask dtype: {}",
                                dtype
                            ))),
                        },
                        _ => Err(TensorError::DTypeError(format!(
                            "Unsupported mask dtype: {:?}",
                            dtype
                        ))),
                    },
                    _ => Err(TensorError::DTypeError(format!(
                        "Unsupported tensor kind: {}",
                        tensor_kind
                    ))),
                };

                Some(result.map_err(|e| {
                    ModelError::BackendError(format!("Mask conversion failed: {e}"))
                })?)
            }
            None => None,
        };

        let r4sa = RelayRLAction::new(
            Some(obs_td),
            Some(act_td),
            mask_td,
            reward,
            false,
            Some(aux),
            Some(actor_id),
        );
        Ok(r4sa)
    }
}

fn default_dtype() -> DType {
    #[cfg(feature = "tch-backend")]
    {
        return DType::Tch(TchDType::F32);
    }

    #[cfg(all(feature = "ndarray-backend", not(feature = "tch-backend")))]
    {
        return DType::NdArray(NdArrayDType::F32);
    }

    #[cfg(all(not(feature = "tch-backend"), not(feature = "ndarray-backend")))]
    {
        panic!("No tensor backend enabled for RelayRL");
    }
}
