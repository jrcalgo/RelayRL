use crate::network::client::runtime::data::environments::vec_env::{
    BatchVecEnv, ScalarVecEnv, VecEnvError, VecEnvTrait,
};

use relayrl_env_trait::*;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher, DType, DeviceType};
use relayrl_types::prelude::tensor::burn::{TensorKind, backend::Backend};

pub(crate) mod vec_env;

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

pub(crate) struct EnvironmentInterface {
    client_namespace: Arc<str>,
    device: DeviceType,
    env: Option<Box<dyn VecEnvTrait>>,
    obs_dtype: Option<EnvDType>,
    act_dtype: Option<EnvDType>,
}

impl EnvironmentInterface {
    pub(crate) fn new(client_namespace: Arc<str>, device: DeviceType) -> Self {
        Self {
            client_namespace,
            device,
            env: None,
            obs_dtype: None,
            act_dtype: None,
        }
    }

    pub(crate) fn ensure_ready(&mut self) -> Result<(), EnvironmentInterfaceError> {
        if self.env.is_some() {
            self.reset_all()?;
        }

        Ok(())
    }

    pub(crate) fn set_env(
        &mut self,
        env: Option<Box<dyn Environment>>,
        count: usize,
    ) -> Result<(), EnvironmentInterfaceError> {
        self.env = match env {
            Some(env) => {
                self.obs_dtype = match env.observation_dtype() {
                    EnvDType::NdArray(_) => Some(env.observation_dtype()),
                    #[cfg(feature = "tch-backend")]
                    EnvDType::Tch(_) => Some(env.observation_dtype()),
                    _ => None,
                };
                self.act_dtype = match env.action_dtype() {
                    EnvDType::NdArray(_) => Some(env.action_dtype()),
                    #[cfg(feature = "tch-backend")]
                    EnvDType::Tch(_) => Some(env.action_dtype()),
                    _ => None,
                };

                let obs_dtype = map_env_dtype(env.observation_dtype())?;
                let act_dtype = map_env_dtype(env.action_dtype())?;
                let boxed_env = match env.into_handle() {
                    EnvironmentHandle::Scalar(s) => Box::new(ScalarVecEnv::init_boxed(
                        self.client_namespace.clone(),
                        s,
                        count,
                        self.device.clone(),
                        obs_dtype.clone(),
                        act_dtype.clone(),
                    )?) as Box<dyn VecEnvTrait>,
                    EnvironmentHandle::Vector(v) => Box::new(BatchVecEnv::init_boxed(
                        self.client_namespace.clone(),
                        v,
                        count,
                        self.device.clone(),
                        obs_dtype,
                        act_dtype,
                    )?) as Box<dyn VecEnvTrait>,
                };
                Some(boxed_env)
            }
            None => None,
        };

        Ok(())
    }

    pub(crate) fn remove_env(&mut self) -> Result<(), EnvironmentInterfaceError> {
        self.obs_dtype = None;
        self.act_dtype = None;

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

    pub(crate) fn reset_all(&mut self) -> Result<(), EnvironmentInterfaceError> {
        let env = self.env.as_mut().ok_or_else(|| {
            EnvironmentInterfaceError::EnvironmentNotSetError(
                "[EnvironmentInterface] Environment not set".to_string(),
            )
        })?;
        env.reset_all().map_err(EnvironmentInterfaceError::from)
    }

    pub(crate) fn n_envs_dims(&self) -> Option<(usize, usize, usize)> {
        self.env.as_ref().and_then(|env| env.n_envs_dims())
    }

    pub(crate) fn flat_observation_bytes(&self) -> Option<Vec<u8>> {
        self.env
            .as_ref()
            .and_then(|env| env.flat_observation_bytes())
    }

    pub(crate) fn step_bytes(&mut self, actions: &[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)> {
        self.env.as_mut().and_then(|env| env.step_bytes(actions))
    }

    pub(crate) fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        self.env.as_ref().and_then(|env| env.flat_env_ids())
    }

    pub(crate) fn obs_dtype(&self) -> Option<EnvDType> {
        self.obs_dtype.clone()
    }

    pub(crate) fn act_dtype(&self) -> Option<EnvDType> {
        self.act_dtype.clone()
    }

    pub(crate) fn action_is_discrete(&self) -> Option<bool> {
        self.env.as_ref().and_then(|env| env.action_is_discrete())
    }
}
