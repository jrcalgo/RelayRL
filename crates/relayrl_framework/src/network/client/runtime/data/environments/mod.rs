use crate::network::client::runtime::data::environments::vec_env::{
    BatchVecEnv, EnvResetRecord, EnvStepRecord, IntoAnyTensorKind, ScalarVecEnv, VecEnvError,
    VecEnvTrait,
};

use relayrl_env_trait::*;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher, DType, DeviceType};
use relayrl_types::prelude::tensor::burn::{TensorKind, backend::Backend};

pub(crate) mod vec_env;

use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EnvironmentInterfaceError {
    #[error("Environment not set: {0}")]
    EnvironmentNotSetError(String),
    #[error("Unsupported environment dtype: {0}")]
    UnsupportedEnvDType(String),
    #[error(transparent)]
    VecEnvError(#[from] VecEnvError),
}

fn map_env_dtype(dtype: EnvDType) -> Result<DType, EnvironmentInterfaceError> {
    match dtype {
        EnvDType::NdArray(dtype) => {
            let mapped = match dtype {
                EnvNdArrayDType::F16 => relayrl_types::data::tensor::NdArrayDType::F16,
                EnvNdArrayDType::F32 => relayrl_types::data::tensor::NdArrayDType::F32,
                EnvNdArrayDType::F64 => relayrl_types::data::tensor::NdArrayDType::F64,
                EnvNdArrayDType::I8 => relayrl_types::data::tensor::NdArrayDType::I8,
                EnvNdArrayDType::I16 => relayrl_types::data::tensor::NdArrayDType::I16,
                EnvNdArrayDType::I32 => relayrl_types::data::tensor::NdArrayDType::I32,
                EnvNdArrayDType::I64 => relayrl_types::data::tensor::NdArrayDType::I64,
                EnvNdArrayDType::Bool => relayrl_types::data::tensor::NdArrayDType::Bool,
            };
            Ok(DType::NdArray(mapped))
        }
        EnvDType::Tch(dtype) => {
            #[cfg(feature = "tch-backend")]
            {
                let mapped = match dtype {
                    EnvTchDType::F16 => relayrl_types::data::tensor::TchDType::F16,
                    EnvTchDType::Bf16 => relayrl_types::data::tensor::TchDType::Bf16,
                    EnvTchDType::F32 => relayrl_types::data::tensor::TchDType::F32,
                    EnvTchDType::F64 => relayrl_types::data::tensor::TchDType::F64,
                    EnvTchDType::I8 => relayrl_types::data::tensor::TchDType::I8,
                    EnvTchDType::I16 => relayrl_types::data::tensor::TchDType::I16,
                    EnvTchDType::I32 => relayrl_types::data::tensor::TchDType::I32,
                    EnvTchDType::I64 => relayrl_types::data::tensor::TchDType::I64,
                    EnvTchDType::U8 => relayrl_types::data::tensor::TchDType::U8,
                    EnvTchDType::Bool => relayrl_types::data::tensor::TchDType::Bool,
                };
                Ok(DType::Tch(mapped))
            }
            #[cfg(not(feature = "tch-backend"))]
            {
                let _ = dtype;
                Err(EnvironmentInterfaceError::UnsupportedEnvDType(
                    "Tch dtype requested, but relayrl_framework was built without the tch-backend feature"
                        .to_string(),
                ))
            }
        }
    }
}

pub(crate) struct EnvironmentInterface<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    client_namespace: Arc<str>,
    device: DeviceType,
    auto_reset: bool,
    env: Option<Box<dyn VecEnvTrait<B, D_IN, D_OUT>>>,
    current_obs: HashMap<EnvironmentUuid, AnyBurnTensor<B, D_IN>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    EnvironmentInterface<B, D_IN, D_OUT>
{
    pub(crate) fn new(client_namespace: Arc<str>, device: DeviceType) -> Self {
        Self {
            client_namespace,
            device,
            auto_reset: true,
            env: None,
            current_obs: HashMap::new(),
        }
    }

    pub(crate) fn current_observations(
        &self,
    ) -> Result<Vec<(EnvironmentUuid, AnyBurnTensor<B, D_IN>)>, EnvironmentInterfaceError> {
        if self.env.is_none() {
            return Err(EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            ));
        }

        Ok(self
            .current_obs
            .iter()
            .map(|(env_id, obs)| (*env_id, obs.clone()))
            .collect())
    }

    pub(crate) fn ensure_ready(
        &mut self,
    ) -> Result<Vec<(EnvironmentUuid, AnyBurnTensor<B, D_IN>)>, EnvironmentInterfaceError> {
        if self.current_obs.is_empty() {
            let resets = self.reset_all()?;
            self.current_obs = resets
                .into_iter()
                .map(|record| (record.env_id, record.observation))
                .collect();
        }

        self.current_observations()
    }

    pub(crate) fn set_env<KindIn, KindOut>(
        &mut self,
        env: Option<Box<dyn Environment<B, D_IN, D_OUT, KindIn, KindOut>>>,
        count: usize,
    ) -> Result<(), EnvironmentInterfaceError>
    where
        KindIn: TensorKind<B>
            + burn_tensor::BasicOps<B>
            + IntoAnyTensorKind<B, D_IN>
            + Send
            + Sync
            + 'static,
        KindOut: TensorKind<B> + burn_tensor::BasicOps<B> + Send + Sync + 'static,
    {
        self.current_obs.clear();
        self.env = match env {
            None => None,
            Some(env) => {
                let observation_dtype = map_env_dtype(env.observation_dtype())?;
                let action_dtype = map_env_dtype(env.action_dtype())?;
                let boxed_env = match env.into_handle() {
                    EnvironmentHandle::Scalar(s) => {
                        Box::new(ScalarVecEnv::<B, D_IN, D_OUT, KindIn, KindOut>::init_boxed(
                            self.client_namespace.clone(),
                            s,
                            count,
                            self.device.clone(),
                            observation_dtype.clone(),
                            action_dtype.clone(),
                        )?) as Box<dyn VecEnvTrait<B, D_IN, D_OUT>>
                    }
                    EnvironmentHandle::Vector(v) => {
                        Box::new(BatchVecEnv::<B, D_IN, D_OUT, KindIn, KindOut>::init_boxed(
                            self.client_namespace.clone(),
                            v,
                            count,
                            self.device.clone(),
                            observation_dtype,
                            action_dtype,
                        )?) as Box<dyn VecEnvTrait<B, D_IN, D_OUT>>
                    }
                };
                Some(boxed_env)
            }
        };
        Ok(())
    }

    pub(crate) fn remove_env(&mut self) -> Result<(), EnvironmentInterfaceError> {
        self.current_obs.clear();
        if let Some(env) = self.env.take() {
            drop(env);
        } else {
            return Err(EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn get_env_count(&self) -> Result<u32, EnvironmentInterfaceError> {
        if let Some(env) = self.env.as_ref() {
            Ok(env.get_env_count()? as u32)
        } else {
            Err(EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            ))
        }
    }

    pub(crate) fn increase_env_count(
        &mut self,
        count: u32,
    ) -> Result<(), EnvironmentInterfaceError> {
        if let Some(env) = &mut self.env {
            env.resize(env.get_env_count()? as usize + count as usize)
                .map_err(EnvironmentInterfaceError::from)
        } else {
            Err(EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            ))
        }
    }

    pub(crate) fn decrease_env_count(
        &mut self,
        count: u32,
    ) -> Result<(), EnvironmentInterfaceError> {
        if let Some(env) = &mut self.env {
            let current = env.get_env_count()?;
            let next = current.saturating_sub(count as usize);
            env.resize(next as usize)
                .map_err(EnvironmentInterfaceError::from)
        } else {
            Err(EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            ))
        }
    }

    pub(crate) fn reset_all(
        &mut self,
    ) -> Result<Vec<EnvResetRecord<B, D_IN>>, EnvironmentInterfaceError> {
        let env = self.env.as_mut().ok_or_else(|| {
            EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            )
        })?;
        env.reset_all().map_err(EnvironmentInterfaceError::from)
    }

    pub(crate) fn step_once(
        &mut self,
        actions: &[(EnvironmentUuid, AnyBurnTensor<B, D_OUT>)],
    ) -> Result<Vec<EnvStepRecord<B, D_IN>>, EnvironmentInterfaceError> {
        let env = self.env.as_mut().ok_or_else(|| {
            EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            )
        })?;

        let steps = env.step(actions)?;
        for step in &steps {
            self.current_obs
                .insert(step.env_id, step.observation.clone());
        }

        if self.auto_reset {
            let done_ids: Vec<_> = steps
                .iter()
                .filter(|step| step.terminated || step.truncated)
                .map(|step| step.env_id)
                .collect();

            if !done_ids.is_empty() {
                let resets = env.reset_where(&done_ids)?;
                for reset in resets {
                    self.current_obs.insert(reset.env_id, reset.observation);
                }
            }
        }

        Ok(steps)
    }
}
