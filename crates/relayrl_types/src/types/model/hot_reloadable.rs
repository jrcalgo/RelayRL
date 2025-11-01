use crate::types::data::tensor::DeviceType;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use tokio::sync::RwLock;
use uuid::Uuid;

use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

use crate::types::model::utils::validate_model;
use crate::types::model::{ModelError, ModelModule};
use crate::types::data::action::{RelayRLAction, RelayRLData};
use crate::types::data::tensor::{BackendMatcher, ConversionTensor, TensorData};
use crate::types::data::tensor::{DType, NdArrayDType, TchDType};

/// Wrapper that lets us swap the underlying model at runtime and run inference
/// in an async-safe way.
pub struct HotReloadableModel<B: Backend + BackendMatcher> {
    inner: RwLock<Arc<ModelModule<B>>>,
    version: Arc<AtomicI64>,
    default_device: DeviceType,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend + BackendMatcher> Clone for HotReloadableModel<B> {
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

impl<B: Backend + BackendMatcher> HotReloadableModel<B> {
    pub async fn new_from_path<P: AsRef<Path>>(
        path: P,
        device: DeviceType,
    ) -> Result<Self, ModelError> {
        let module: ModelModule<B> = ModelModule::<B>::load_from_path(path.as_ref().to_path_buf())?;
        validate_model::<B>(&module, module.input_dim, module.output_dim)?;

        Ok(Self {
            inner: RwLock::new(Arc::new(module.to_owned())),
            version: Arc::new(AtomicI64::new(0)),
            default_device: device,
            input_dim: module.input_dim,
            output_dim: module.output_dim,
        })
    }

    pub async fn new_from_module(
        module: ModelModule<B>,
        device: DeviceType,
    ) -> Result<Self, ModelError> {
        validate_model::<B>(&module, module.input_dim, module.output_dim)?;

        Ok(Self {
            inner: RwLock::new(Arc::new(module.to_owned())),
            version: Arc::new(AtomicI64::new(0)),
            default_device: device,
            input_dim: module.input_dim,
            output_dim: module.output_dim,
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

    pub async fn reload_from_module(&self, module: ModelModule<B>, version: i64) -> Result<i64, ModelError> {
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
    pub async fn forward<const I: usize, const O: usize>(
        &self,
        observation: Tensor<B, I>,
        mask: Tensor<B, O>,
        reward: f32,
        actor_id: Uuid,
    ) -> Result<RelayRLAction, String> {
        let guard = self.inner.read().await;
        let (action, aux) = guard.step(observation.clone(), mask.clone());

        let tensor_dtype = aux
            .get("dtype")
            .and_then(|d| match d {
                RelayRLData::DType(dtype) => Some(dtype),
                _ => None,
            })
            .unwrap_or_else(|| &DType::NdArray(NdArrayDType::F32));

        // Build RelayRLAction by converting tensors → TensorData
        let obs_td = TensorData::try_from(ConversionTensor {
            tensor: observation.clone(),
            conversion_dtype: tensor_dtype.clone(),
        })
        .map_err(|e| format!("Tensor conversion failed: {e}"))?;

        let mask_td = TensorData::try_from(ConversionTensor {
            tensor: mask,
            conversion_dtype: tensor_dtype.clone(),
        })
        .map_err(|e| format!("Tensor conversion failed: {e}"))?;

        let act_td = TensorData::try_from(ConversionTensor {
            tensor: action.clone(),
            conversion_dtype: tensor_dtype.clone(),
        })
        .map_err(|e| format!("Tensor conversion failed: {e}"))?;

        let r4sa = RelayRLAction::new(
            Some(obs_td),
            Some(act_td),
            Some(mask_td),
            reward,
            false,
            Some(aux),
            Some(actor_id),
        );
        Ok(r4sa)
    }
}
