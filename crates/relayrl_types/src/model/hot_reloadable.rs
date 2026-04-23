use crate::data::tensor::DeviceType;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use uuid::Uuid;

use arc_swap::ArcSwap;
use burn_tensor::backend::Backend;

use crate::data::action::RelayRLAction;
use crate::data::tensor::DType;
use crate::data::tensor::{AnyBurnTensor, BackendMatcher, ConversionBurnTensor, TensorData};
use crate::model::utils::validate_module;
use crate::model::{ModelError, ModelModule};

/// Wrapper that lets us swap the underlying model at runtime and run inference
/// in an async-safe way.
pub struct HotReloadableModel<B: Backend + BackendMatcher<Backend = B>> {
    inner: ArcSwap<ModelModule<B>>,
    version: AtomicI64,
    default_device: DeviceType,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend + BackendMatcher<Backend = B>> HotReloadableModel<B> {
    pub async fn new_from_path<P: AsRef<Path>>(
        path: P,
        device: DeviceType,
    ) -> Result<Self, ModelError> {
        let module: ModelModule<B> = ModelModule::<B>::load_from_path(path.as_ref().to_path_buf())?;
        validate_module::<B>(&module)?;
        let input_dim = module.metadata.input_shape.len();
        let output_dim = module.metadata.output_shape.len();

        Ok(Self {
            inner: ArcSwap::new(Arc::new(module)),
            version: AtomicI64::new(0),
            default_device: device,
            input_dim,
            output_dim,
        })
    }

    pub async fn new_from_module(
        module: ModelModule<B>,
        device: DeviceType,
    ) -> Result<Self, ModelError> {
        validate_module::<B>(&module)?;
        let input_dim = module.metadata.input_shape.len();
        let output_dim = module.metadata.output_shape.len();

        Ok(Self {
            inner: ArcSwap::new(Arc::new(module)),
            version: AtomicI64::new(0),
            default_device: device,
            input_dim,
            output_dim,
        })
    }

    pub fn default_device(&self) -> &DeviceType {
        &self.default_device
    }

    pub fn version(&self) -> i64 {
        self.version.load(Ordering::SeqCst)
    }

    pub fn input_dim(&self) -> &usize {
        &self.input_dim
    }

    pub fn output_dim(&self) -> &usize {
        &self.output_dim
    }

    pub fn current_module(&self) -> Arc<ModelModule<B>> {
        self.inner.load_full()
    }

    /// Atomically swap the model from disk and bump version.
    pub async fn reload_from_path(&self, path: PathBuf, version: i64) -> Result<i64, ModelError> {
        let new_module = Arc::new(ModelModule::<B>::load_from_path(path)?);
        self.inner.store(new_module);
        self.version.store(version, Ordering::SeqCst);
        Ok(version)
    }

    pub async fn reload_from_module(
        &self,
        module: ModelModule<B>,
        version: i64,
    ) -> Result<i64, ModelError> {
        self.inner.store(Arc::new(module));
        self.version.store(version, Ordering::SeqCst);
        Ok(version)
    }

    /// Generic forward that works for any backend / rank.
    pub fn forward<const D_IN: usize, const D_OUT: usize>(
        &self,
        observation: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
        reward: f32,
        actor_id: Uuid,
    ) -> Result<RelayRLAction, ModelError> {
        let model_module = self.current_module();
        let (act_td, mask_td, aux) = model_module.step(observation.clone(), mask);

        // Build RelayRLAction by converting tensors → TensorData
        let obs_td = match observation.as_ref() {
            AnyBurnTensor::Float(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: model_module.metadata.input_dtype.clone(),
            }),
            AnyBurnTensor::Int(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: model_module.metadata.input_dtype.clone(),
            }),
            AnyBurnTensor::Bool(wrapper) => TensorData::try_from(ConversionBurnTensor {
                inner: wrapper.tensor.clone(),
                conversion_dtype: model_module.metadata.input_dtype.clone(),
            }),
        }
        .map_err(|e| ModelError::BackendError(format!("Tensor conversion failed: {e}")))?;

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

    pub fn forward_batch<const D_IN: usize, const D_OUT: usize>(
        &self,
        observations: &[Arc<AnyBurnTensor<B, D_IN>>],
        masks: &[Option<Arc<AnyBurnTensor<B, D_OUT>>>],
        rewards: &[f32],
        actor_id: Uuid,
    ) -> Result<Vec<RelayRLAction>, ModelError> {
        if observations.len() != rewards.len() {
            return Err(ModelError::InvalidInputDimension(format!(
                "batched reward count mismatch: {} observations vs {} rewards",
                observations.len(),
                rewards.len()
            )));
        }

        let model_module = self.current_module();
        let steps = model_module.step_batch::<D_IN, D_OUT>(observations, masks)?;
        if steps.len() != observations.len() {
            return Err(ModelError::InvalidOutputDimension(format!(
                "batched action count mismatch: {} actions for {} observations",
                steps.len(),
                observations.len()
            )));
        }

        Ok(steps
            .into_iter()
            .zip(observations.iter())
            .zip(rewards.iter().copied())
            .map(|(((act_td, mask_td, aux), observation), reward)| {
                let obs_td = match observation.as_ref() {
                    AnyBurnTensor::Float(wrapper) => TensorData::try_from(ConversionBurnTensor {
                        inner: wrapper.tensor.clone(),
                        conversion_dtype: model_module.metadata.input_dtype.clone(),
                    }),
                    AnyBurnTensor::Int(wrapper) => TensorData::try_from(ConversionBurnTensor {
                        inner: wrapper.tensor.clone(),
                        conversion_dtype: model_module.metadata.input_dtype.clone(),
                    }),
                    AnyBurnTensor::Bool(wrapper) => TensorData::try_from(ConversionBurnTensor {
                        inner: wrapper.tensor.clone(),
                        conversion_dtype: model_module.metadata.input_dtype.clone(),
                    }),
                }
                .map_err(|e| ModelError::BackendError(format!("Tensor conversion failed: {e}")))?;

                Ok::<RelayRLAction, ModelError>(RelayRLAction::new(
                    Some(obs_td),
                    Some(act_td),
                    mask_td,
                    reward,
                    false,
                    Some(aux),
                    Some(actor_id),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?)
    }
}

#[allow(unused)]
fn default_dtype() -> DType {
    #[cfg(feature = "tch-backend")]
    {
        DType::Tch(crate::model::TchDType::F32)
    }

    #[cfg(all(feature = "ndarray-backend", not(feature = "tch-backend")))]
    {
        DType::NdArray(crate::model::NdArrayDType::F32)
    }

    #[cfg(all(not(feature = "tch-backend"), not(feature = "ndarray-backend")))]
    {
        // effectively enforces an invariant that a backend must be enabled
        panic!("No tensor backend enabled for RelayRL"); // without a backend, we can't return anything, so we panic to be safe (bro, why would you not compile without a backend anyways?)
    }
}

#[cfg(all(
    test,
    feature = "ndarray-backend",
    any(feature = "tch-model", feature = "onnx-model")
))]
mod unit_tests {
    use super::*;
    use std::marker::PhantomData;

    use burn_ndarray::NdArray;
    use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};

    use crate::data::tensor::NdArrayDType;
    use crate::model::{InferenceModel, Model, ModelFileType, ModelMetadata, ModelModule};
    use crate::prelude::tensor::relayrl::FloatBurnTensor;

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

    #[tokio::test]
    async fn new_from_module_exposes_dimensions_and_version() {
        let reloadable = HotReloadableModel::new_from_module(stub_module(vec![2]), DeviceType::Cpu)
            .await
            .unwrap();

        assert_eq!(reloadable.version(), 0);
        assert_eq!(*reloadable.input_dim(), 1);
        assert_eq!(*reloadable.output_dim(), 1);
        assert_eq!(reloadable.default_device(), &DeviceType::Cpu);
    }

    #[tokio::test]
    async fn reload_from_module_updates_the_visible_version() {
        let reloadable = HotReloadableModel::new_from_module(stub_module(vec![2]), DeviceType::Cpu)
            .await
            .unwrap();

        let version = reloadable
            .reload_from_module(stub_module(vec![2]), 7)
            .await
            .unwrap();

        assert_eq!(version, 7);
        assert_eq!(reloadable.version(), 7);
    }

    #[tokio::test]
    async fn forward_returns_actions_with_observation_mask_and_zero_fallback() {
        let reloadable = HotReloadableModel::new_from_module(stub_module(vec![2]), DeviceType::Cpu)
            .await
            .unwrap();
        let actor_id = Uuid::new_v4();

        let action = reloadable
            .forward::<1, 1>(
                float_any_tensor(&[1.0, 2.0]),
                Some(float_any_tensor(&[1.0, 0.0])),
                3.5,
                actor_id,
            )
            .unwrap();

        assert_eq!(action.get_rew(), 3.5);
        assert!(!action.get_done());
        assert_eq!(action.get_agent_id(), Some(&actor_id));
        assert_eq!(
            action.get_obs().unwrap().data,
            [1.0f32, 2.0]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(action.get_act().unwrap().data, vec![0; 8]);
        assert_eq!(
            action.get_mask().unwrap().data,
            [1.0f32, 0.0]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert!(action.get_data().unwrap().is_empty());
    }

    #[tokio::test]
    async fn forward_batch_returns_one_action_per_observation() {
        let reloadable = HotReloadableModel::new_from_module(stub_module(vec![1]), DeviceType::Cpu)
            .await
            .unwrap();
        let actor_id = Uuid::new_v4();
        let observations = vec![float_any_tensor(&[1.0, 2.0]), float_any_tensor(&[3.0, 4.0])];
        let masks = vec![None, None];
        let rewards = vec![1.5, 2.5];

        let actions = reloadable
            .forward_batch::<1, 1>(&observations, &masks, &rewards, actor_id)
            .unwrap();

        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].get_rew(), 1.5);
        assert_eq!(actions[1].get_rew(), 2.5);
        assert_eq!(actions[0].get_agent_id(), Some(&actor_id));
        assert_eq!(actions[1].get_agent_id(), Some(&actor_id));
        assert!(actions.iter().all(|action| action.get_act().is_some()));
    }

    #[test]
    #[cfg(all(feature = "ndarray-backend", not(feature = "tch-backend")))]
    fn default_dtype_prefers_ndarray_when_tch_is_unavailable() {
        assert_eq!(default_dtype(), DType::NdArray(NdArrayDType::F32));
    }
}
