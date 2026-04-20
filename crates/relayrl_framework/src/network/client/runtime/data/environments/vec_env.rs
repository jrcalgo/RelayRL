use crate::network::ENVIRONMENT_CONTEXT_PREFIX;
use crate::network::client::agent::ToAnyBurnTensor;
use active_uuid_registry::{ContextString, UuidPoolError, interface::{reserve_id, add_id}};
use relayrl_env_trait::*;
use relayrl_types::data::tensor::{AnyBurnTensor, TensorError, BackendMatcher, DeviceType};

use burn_tensor::BasicOps;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VecEnvError {
    #[error("Environment context reservation failed: {0}")]
    EnvironmentContextReservationFailed(String),
    #[error("Environment ID reservation failed: {0}")]
    EnvironmentIdReservationFailed(String),
    #[error("Invalid environment count: {0}")]
    InvalidEnvironmentCount(String),
    #[error("Invalid action collection: {0}")]
    InvalidActionCollection(String),
    #[error("Unknown environment: {0}")]
    UnknownEnv(EnvironmentUuid),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[error(transparent)]
    TensorError(#[from] TensorError),
}

pub(crate) enum TensorActList<'a, B: Backend + BackendMatcher<Backend = B>, const D_OUT: usize> {
    ScalarEnv(&'a [(EnvironmentUuid, AnyBurnTensor<B, D_OUT>)]),
    VectorEnv(&'a (EnvironmentUuid, &'a [AnyBurnTensor<B, D_OUT>])),
}

pub(crate) enum TensorObsList<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize> {
    ScalarEnv(Vec<(EnvironmentUuid, AnyBurnTensor<B, D_IN>)>),
    VectorEnv(EnvironmentUuid, Vec<AnyBurnTensor<B, D_IN>>),
}

pub(crate) enum EnvInfo {
    ScalarEnv(Option<Vec<Option<Vec<(String, String)>>>>),
    VectorEnv(Option<Vec<(String, String)>>),
}

pub(crate) trait VecEnvTrait<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B>,
    KOutput: TensorKind<B>,
>
{
    fn step(
        &self,
        actions: TensorActList<'_, B, D_OUT>,
    ) -> Result<TensorObsList<B, D_IN>, VecEnvError> where Tensor<B, D_IN, KInput>: ToAnyBurnTensor<B, D_IN>;
    fn reset(&self) -> Result<EnvInfo, VecEnvError>;
}

pub(crate) struct ScalarVecEnv<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B>,
    KOutput: TensorKind<B> + BasicOps<B>,
> {
    env_context: ContextString,
    envs: HashMap<EnvironmentUuid, Box<dyn ScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput, StepInfo = Vec<(String, String)>, ResetInfo = Vec<(String, String)>>>>,
    device: DeviceType,
    _phantom: PhantomData<(B, KInput, KOutput)>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B>,
    KOutput: TensorKind<B> + BasicOps<B>,
> ScalarVecEnv<B, D_IN, D_OUT, KInput, KOutput> {
    fn init<S>(
        client_namespace: Arc<str>,
        env: S,
        count: usize,
        device: DeviceType,
    ) -> Result<Self, VecEnvError>
    where
        S: ScalarEnvironment<
                B, D_IN, D_OUT, KInput, KOutput,
                StepInfo  = Vec<(String, String)>,
                ResetInfo = Vec<(String, String)>,
            > + Clone + 'static,
    {
        if count == 0 {
            return Err(VecEnvError::InvalidEnvironmentCount(format!(
                "Invalid environment count: `count` set to `{count}` in `VecEnv::init()`"
            )));
        }
    
        let env_context: String = format!(
            "{}:{}",
            ENVIRONMENT_CONTEXT_PREFIX,
            std::any::type_name::<S>()
        );
    
        let envs = (0..count)
            .map(|_| {
                let env_id = reserve_id(client_namespace.as_ref(), env_context.as_ref())?;
                let box_env: Box<
                    dyn ScalarEnvironment<
                        B, D_IN, D_OUT, KInput, KOutput,
                        StepInfo  = Vec<(String, String)>,
                        ResetInfo = Vec<(String, String)>,
                    >,
                > = Box::new(env.clone());
                Ok((env_id, box_env))
            })
            .collect::<Result<
                HashMap<
                    EnvironmentUuid,
                    Box<
                        dyn ScalarEnvironment<
                            B, D_IN, D_OUT, KInput, KOutput,
                            StepInfo  = Vec<(String, String)>,
                            ResetInfo = Vec<(String, String)>,
                        >,
                    >,
                >,
                VecEnvError,
            >>()?;
    
        Ok(Self {
            env_context,
            envs,
            device,
            _phantom: PhantomData,
        })
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B>,
    KOutput: TensorKind<B> + BasicOps<B>,
> VecEnvTrait<B, D_IN, D_OUT, KInput, KOutput> for ScalarVecEnv<B, D_IN, D_OUT, KInput, KOutput> {
    fn step(
        &self,
        actions: TensorActList<'_, B, D_OUT>,
    ) -> Result<TensorObsList<B, D_IN>, VecEnvError> where Tensor<B, D_IN, KInput>: ToAnyBurnTensor<B, D_IN> {
        let obs_collection: TensorObsList<B, D_IN> = match actions {
            TensorActList::ScalarEnv(acts) => {
                let device = B::get_device(&self.device)?;
                let pairs = acts
                    .iter()
                    .map(|(env_id, any_act)| -> Result<
                        (EnvironmentUuid, AnyBurnTensor<B, D_IN>),
                        VecEnvError,
                    > {
                        let env = self
                            .envs
                            .get(env_id)
                            .ok_or_else(|| VecEnvError::UnknownEnv(*env_id))?;
            
                        let data = match any_act {
                            AnyBurnTensor::Float(w) => w.tensor.to_data(),
                            AnyBurnTensor::Int(w)   => w.tensor.to_data(),
                            AnyBurnTensor::Bool(w)  => w.tensor.to_data(),
                        };
            
                        let action = Tensor::<B, D_OUT, KOutput>::from_data(data, &device);
            
                        let (obs, _step_info) = env.step(action)?;
                        Ok((*env_id, obs.to_any_burn_tensor(self.obs_dtype)))
                    })
                    .collect::<Result<Vec<_>, VecEnvError>>()?;
            
                TensorObsList::ScalarEnv(pairs)
            }
            TensorActList::VectorEnv(_) => {
                return Err(VecEnvError::InvalidActionCollection(format!(
                    "Invalid action collection: `actions` is a vector environment in `ScalarVecEnv::step()`"
                )));
            }
        };

        Ok(obs_collection)
    }

    fn reset(&self) -> Result<EnvInfo, VecEnvError> {
        let infos: Vec<Option<Vec<(String, String)>>> = self
            .envs
            .iter()
            .map(|env_ref| env_ref.1.reset().map_err(VecEnvError::from))
            .collect::<Result<_, _>>()?;
    
        Ok(EnvInfo::ScalarEnv(Some(infos)))
    }
}

pub(crate) struct BatchVecEnv<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B>,
    KOutput: TensorKind<B> + BasicOps<B>,
> {
    env_context: ContextString,
    env: Box<dyn VectorEnvironment<B, D_IN, D_OUT, KInput, KOutput, StepInfo = Vec<(String, String)>, ResetInfo = Vec<(String, String)>>>,
    device: DeviceType,
    _phantom: PhantomData<(B, KInput, KOutput)>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B>,
    KOutput: TensorKind<B> + BasicOps<B>,
> BatchVecEnv<B, D_IN, D_OUT, KInput, KOutput> {
    fn init<V>(
        client_namespace: Arc<str>,
        env: V,
        count: usize,
        device: DeviceType,
    ) -> Result<Self, VecEnvError>
    where
        V: VectorEnvironment<
                B, D_IN, D_OUT, KInput, KOutput,
                StepInfo  = Vec<(String, String)>,
                ResetInfo = Vec<(String, String)>,
            > + Clone + 'static,
    {
        if count == 0 {
            log::error!("Invalid environment count: `count` set to `{count}` in `VecEnv::init()`");
            return Err(VecEnvError::InvalidEnvironmentCount(format!(
                "Invalid environment count: `count` set to `{count}` in `VecEnv::init()`"
            )));
        }

        let env_context: String = format!(
            "{}-{}",
            ENVIRONMENT_CONTEXT_PREFIX,
            std::any::type_name::<V>()
        );

        let env: Box<V> = Box::new(env.clone());

        let env_ids: Vec<Uuid> = env.init_num_envs(count)?;

        env_ids.into_iter().map(|id| {
            add_id(client_namespace.as_ref(), env_context.as_ref(), id).map_err(VecEnvError::from)
        });

        Ok(Self {
            env_context,
            env,
            device,
            _phantom: PhantomData,
        })
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize, 
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B>,
    KOutput: TensorKind<B> + BasicOps<B>,
> VecEnvTrait<B, D_IN, D_OUT, KInput, KOutput> for BatchVecEnv<B, D_IN, D_OUT, KInput, KOutput> {
    fn step(
        &self,
        actions: TensorActList<'_, B, D_OUT>,
    ) -> Result<TensorObsList<B, D_IN>, VecEnvError> where Tensor<B, D_IN, KInput>: ToAnyBurnTensor<B, D_IN> {
        let obs_collection: TensorObsList<B, D_IN> = match actions {
            TensorActList::VectorEnv(env_acts) => {
                let device = B::get_device(&self.device).map_err(VecEnvError::from)?;

                let actions = env_acts
                    .1
                    .iter()
                    .map(|any_act| -> Result<Tensor<B, D_OUT, KOutput>, VecEnvError> {
                        let (data, dtype) = match any_act {
                            AnyBurnTensor::Float(w) => (w.tensor.to_data(), w.dtype.clone()),
                            AnyBurnTensor::Int(w)   => (w.tensor.to_data(), w.dtype.clone()),
                            AnyBurnTensor::Bool(w)  => (w.tensor.to_data(), w.dtype.clone()),
                        };
                        Ok(Tensor::<B, D_OUT, KOutput>::from_data(data, &device))
                    })
                    .collect::<Result<Vec<_>, VecEnvError>>()?;
                
                let obs_dtype = /* decide once: carry on the struct, or infer from KInput */;
                let obs = self
                    .env
                    .step(&actions)?
                    .into_iter()
                    .map(|obs| obs.to_any_burn_tensor(obs_dtype.clone()).map_err(VecEnvError::from))
                    .collect::<Result<Vec<AnyBurnTensor<B, D_IN>>, _>>()?;
                
                TensorObsList::VectorEnv(env_acts.0, obs)
            }
            TensorActList::ScalarEnv(_) => {
                return Err(VecEnvError::InvalidActionCollection(format!(
                    "Invalid action collection: `actions` is a scalar environment in `BatchVecEnv::step()`"
                )));
            }
        };

        Ok(obs_collection)
    }

    fn reset(&self) -> Result<EnvInfo, VecEnvError> {
        self.env.reset().map_err(VecEnvError::from).map(EnvInfo::VectorEnv)
    }
}
