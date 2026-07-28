use crate::network::client::runtime::control::coordinator::ClientNamespace;
use crate::network::client::runtime::data::environments::vec_env::{
    BatchVecEnv, ScalarVecEnv, VecEnvError, VecEnvTrait,
};

use relayrl_env_trait::*;
use relayrl_types::data::tensor::{DType, DeviceType};

pub(crate) mod vec_env;

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
    client_namespace: ClientNamespace,
    device: DeviceType,
    env: Option<Box<dyn VecEnvTrait>>,
    obs_dtype: Option<EnvDType>,
    act_dtype: Option<EnvDType>,
}

impl EnvironmentInterface {
    pub(crate) fn new(client_namespace: ClientNamespace, device: DeviceType) -> Self {
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
                    #[cfg(not(feature = "tch-backend"))]
                    EnvDType::Tch(_) => None,
                };
                self.act_dtype = match env.action_dtype() {
                    EnvDType::NdArray(_) => Some(env.action_dtype()),
                    #[cfg(feature = "tch-backend")]
                    EnvDType::Tch(_) => Some(env.action_dtype()),
                    #[cfg(not(feature = "tch-backend"))]
                    EnvDType::Tch(_) => None,
                };

                let obs_dtype: DType = map_env_dtype(env.observation_dtype())?;
                let act_dtype: DType = map_env_dtype(env.action_dtype())?;

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
            env.resize(env.get_env_count()? + count as usize)
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
            env.resize(next).map_err(EnvironmentInterfaceError::from)
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

    pub(crate) fn flat_mask_bytes(&self) -> Option<Vec<u8>> {
        self.env.as_ref().and_then(|env| env.flat_mask_bytes())
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn step_bytes(
        &mut self,
        actions: &[u8],
    ) -> Option<(Vec<u8>, Option<Vec<u8>>, Vec<f32>, Vec<bool>, Vec<bool>)> {
        self.env.as_mut().and_then(|env| env.step_bytes(actions))
    }

    #[allow(unused)]
    pub(crate) fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        self.env.as_ref().and_then(|env| env.flat_env_ids())
    }

    pub(crate) fn obs_dtype(&self) -> Option<EnvDType> {
        self.obs_dtype.clone()
    }

    pub(crate) fn act_dtype(&self) -> Option<EnvDType> {
        self.act_dtype.clone()
    }

    #[allow(unused)]
    pub(crate) fn obs_dim(&self) -> Option<usize> {
        self.env.as_ref().map(|env| env.obs_dim())
    }

    #[allow(unused)]
    pub(crate) fn act_dim(&self) -> Option<usize> {
        self.env.as_ref().map(|env| env.act_dim())
    }

    pub(crate) fn action_is_discrete(&self) -> Option<bool> {
        self.env.as_ref().and_then(|env| env.action_is_discrete())
    }

    #[allow(unused)]
    pub(crate) fn get_env_context(&self) -> Option<String> {
        self.env
            .as_ref()
            .map(|env| env.get_env_context().to_string())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    use std::any::Any;
    use std::sync::Mutex;

    fn owned_test_namespace(prefix: &str) -> ClientNamespace {
        let namespace_str = format!("{}-{}", prefix, EnvironmentUuid::new_v4());
        let handle = active_uuid_registry::interface::reserve_owned_namespace(&namespace_str)
            .expect("reserve owned test namespace");
        ClientNamespace::new(handle, std::sync::Arc::from(namespace_str))
    }

    #[derive(Clone)]
    struct TestScalarEnv;

    impl Environment for TestScalarEnv {
        fn run_environment(&self) -> Result<(), EnvironmentError> {
            Ok(())
        }
        fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
            Ok(Box::new(self.flat_observation_bytes()))
        }
        fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
            Ok(Box::new(()))
        }
        fn observation_dtype(&self) -> EnvDType {
            EnvDType::NdArray(EnvNdArrayDType::F32)
        }
        fn action_dtype(&self) -> EnvDType {
            EnvDType::NdArray(EnvNdArrayDType::I64)
        }
        fn observation_dim(&self) -> usize {
            2
        }
        fn action_dim(&self) -> usize {
            1
        }
        fn flat_observation_bytes(&self) -> Observation {
            vec![0u8; 8]
        }
        fn flat_mask_bytes(&self) -> Mask {
            None
        }
        fn action_is_discrete(&self) -> bool {
            true
        }
        fn kind(&self) -> EnvironmentKind {
            EnvironmentKind::Scalar
        }
        fn into_handle(self: Box<Self>) -> EnvironmentHandle {
            EnvironmentHandle::Scalar(Box::new(*self))
        }
    }

    impl ScalarEnvironment for TestScalarEnv {
        fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
            Ok(ScalarEnvReset {
                observation: self.flat_observation_bytes(),
                info: None,
            })
        }
        fn step_bytes(
            &self,
            _action: &[u8],
        ) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
            Some((self.flat_observation_bytes(), None, 0.0, false, false))
        }
    }

    /// Minimal batched double: `n_envs` reflects only environments allocated via
    /// `init_num_envs`, which is all this test needs (registry writes go through
    /// `BatchVecEnv`'s own `env_ids` bookkeeping, not this double's internal count).
    struct TestVectorEnv {
        ids: Mutex<Vec<EnvironmentUuid>>,
    }

    impl TestVectorEnv {
        fn new() -> Self {
            Self {
                ids: Mutex::new(Vec::new()),
            }
        }
    }

    impl Environment for TestVectorEnv {
        fn run_environment(&self) -> Result<(), EnvironmentError> {
            Ok(())
        }
        fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
            Ok(Box::new(()))
        }
        fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
            Ok(Box::new(()))
        }
        fn observation_dtype(&self) -> EnvDType {
            EnvDType::NdArray(EnvNdArrayDType::F32)
        }
        fn action_dtype(&self) -> EnvDType {
            EnvDType::NdArray(EnvNdArrayDType::I64)
        }
        fn observation_dim(&self) -> usize {
            2
        }
        fn action_dim(&self) -> usize {
            1
        }
        fn flat_observation_bytes(&self) -> Observation {
            Vec::new()
        }
        fn flat_mask_bytes(&self) -> Mask {
            None
        }
        fn action_is_discrete(&self) -> bool {
            true
        }
        fn kind(&self) -> EnvironmentKind {
            EnvironmentKind::Vector
        }
        fn into_handle(self: Box<Self>) -> EnvironmentHandle {
            EnvironmentHandle::Vector(self)
        }
    }

    impl VectorEnvironment for TestVectorEnv {
        fn init_num_envs(&self, num_envs: usize) -> Result<Vec<EnvironmentUuid>, EnvironmentError> {
            let mut ids = self.ids.lock().unwrap();
            let new_ids: Vec<EnvironmentUuid> =
                (0..num_envs).map(|_| EnvironmentUuid::new_v4()).collect();
            ids.extend(new_ids.iter().copied());
            Ok(new_ids)
        }
        fn reset(
            &self,
            env_ids: &[EnvironmentUuid],
        ) -> Result<Vec<VectorEnvReset>, EnvironmentError> {
            Ok(env_ids
                .iter()
                .map(|id| VectorEnvReset {
                    env_id: *id,
                    observation: Vec::new(),
                    info: None,
                })
                .collect())
        }
        fn n_envs(&self) -> usize {
            self.ids.lock().unwrap().len()
        }
        #[allow(clippy::type_complexity)]
        fn step_bytes(
            &self,
            _actions: &[u8],
        ) -> Option<(Observation, Mask, Vec<Reward>, Vec<Done>, Vec<Truncated>)> {
            None
        }
    }

    #[test]
    fn scalar_env_registers_resizes_and_removes_through_owned_namespace() {
        let client_namespace = owned_test_namespace("test-env-scalar");
        let mut interface = EnvironmentInterface::new(client_namespace, DeviceType::Cpu);

        interface
            .set_env(Some(Box::new(TestScalarEnv)), 2)
            .expect("set_env should register through the owned client namespace");
        assert_eq!(interface.get_env_count().unwrap(), 2);

        interface
            .increase_env_count(3)
            .expect("increase_env_count should reserve new ids through the owned handle");
        assert_eq!(interface.get_env_count().unwrap(), 5);

        interface
            .decrease_env_count(4)
            .expect("decrease_env_count should remove ids through the owned handle");
        assert_eq!(interface.get_env_count().unwrap(), 1);

        interface
            .remove_env()
            .expect("remove_env should succeed once an environment is set");
    }

    #[test]
    fn vector_env_registers_resizes_and_removes_through_owned_namespace() {
        let client_namespace = owned_test_namespace("test-env-vector");
        let mut interface = EnvironmentInterface::new(client_namespace, DeviceType::Cpu);

        interface
            .set_env(Some(Box::new(TestVectorEnv::new())), 2)
            .expect("set_env should register through the owned client namespace");
        assert_eq!(interface.get_env_count().unwrap(), 2);

        interface
            .increase_env_count(3)
            .expect("increase_env_count should add ids through the owned handle");
        assert_eq!(interface.get_env_count().unwrap(), 5);

        interface
            .decrease_env_count(4)
            .expect("decrease_env_count should remove ids through the owned handle");
        assert_eq!(interface.get_env_count().unwrap(), 1);

        interface
            .remove_env()
            .expect("remove_env should succeed once an environment is set");
    }
}
