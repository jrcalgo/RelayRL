use crate::network::ENVIRONMENT_CONTEXT_PREFIX;
use active_uuid_registry::{ContextString, UuidPoolError, interface::reserve_id};
use relayrl_env_trait::*;
use relayrl_types::data::tensor::AnyBurnTensor;
use relayrl_types::data::tensor::BackendMatcher;
use relayrl_types::data::tensor::DeviceType;

use dashmap::DashMap;
use log::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Error)]
pub enum VecEnvError {
    #[error("Environment context reservation failed: {0}")]
    EnvironmentContextReservationFailed(String),
    #[error("Environment ID reservation failed: {0}")]
    EnvironmentIdReservationFailed(String),
    #[error("Invalid environment count: {0}")]
    InvalidEnvironmentCount(String),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
}

pub(crate) enum TensorActList<'a, B: Backend + BackendMatcher<Backend = B>, const D_OUT: usize> {
    ScalarEnv(&'a [(EnvironmentUuid, AnyBurnTensor<B, D_OUT>)]),
    VectorEnv(&'a (EnvironmentUuid, &'a [AnyBurnTensor<B, D_OUT>])),
}

pub(crate) enum TensorObsList<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize> {
    ScalarEnv(Vec<(EnvironmentUuid, AnyBurnTensor<B, D_IN>)>),
    VectorEnv(EnvironmentUuid, Vec<AnyBurnTensor<B, D_IN>>),
}

pub(crate) trait VecEnvTrait<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B>,
    KOutput: TensorKind<B>,
>
{
    fn init(
        client_namespace: Arc<str>,
        env: E,
        count: usize,
        device: DeviceType,
    ) -> Result<Self, VecEnvError>;
    fn step(
        &self,
        actions: TensorActList<'_, B, D_OUT>,
    ) -> Result<TensorObsList<B, D_IN>, VecEnvError>;
    fn reset(&self) -> Result<(), VecEnvError>;
}

pub(crate) struct ScalarVecEnv<
    S: ScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>,
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B>,
    KOutput: TensorKind<B>,
> {
    env_context: ContextString,
    envs: DashMap<EnvironmentUuid, S>,
    device: DeviceType,
    _phantom: PhantomData<B>,
}

impl<
    S: ScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>,
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B>,
    KOutput: TensorKind<B>,
> VecEnvTrait<B, D_IN, D_OUT, KInput, KOutput>
    for ScalarVecEnv<S, B, D_IN, D_OUT, KInput, KOutput>
{
    fn init(
        client_namespace: Arc<str>,
        env: S,
        count: usize,
        device: DeviceType,
    ) -> Result<Self, VecEnvError> {
        if count == 0 {
            log::error!("Invalid environment count: `count` set to `{count}` in `VecEnv::init()`");
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
                Ok((env_id, env.clone()))
            })
            .collect::<Result<DashMap<EnvironmentUuid, E>, VecEnvError>>()?;

        Ok(Self {
            env_context,
            envs,
            device,
            _phantom: PhantomData,
        })
    }

    fn step(
        &self,
        actions: TensorActList<'_, B, D_OUT>,
    ) -> Result<TensorObsList<B, D_IN>, VecEnvError> {
        let obs_collection: TensorObsList<B, D_IN> = match actions {
            TensorActList::ScalarEnv(acts) => acts
                .iter()
                .map(|(env_id, any_act)| {
                    let env = self.envs.get_mut(env_id)?;
                    let (action, dtype) = match any_act {
                        AnyBurnTensor::Float(float_act) => (
                            Tensor::<B, D_OUT, KOutput>::from_floats(
                                float_act.tensor.to_data(),
                                B::get_device(&self.device)?,
                            )?,
                            float_act.dtype,
                        ),
                        AnyBurnTensor::Int(int_act) => (
                            Tensor::<B, D_OUT, KOutput>::from_ints(
                                int_act.tensor.to_data(),
                                B::get_device(&self.device)?,
                            )?,
                            int_act.dtype,
                        ),
                        AnyBurnTensor::Bool(bool_act) => (
                            Tensor::<B, D_OUT, KOutput>::from_bools(
                                bool_act.tensor.to_data(),
                                B::get_device(&self.device)?,
                            )?,
                            bool_act.dtype,
                        ),
                    };
                    let (obs, step_info) = env.value().step(action)?;
                    Ok((env_id, obs.to_any_burn_tensor(dtype)?))
                })
                .collect::<Vec<(EnvironmentUuid, AnyBurnTensor<B, D_IN>)>>()?,
            TensorActList::VectorEnv(_) => {
                return Err(VecEnvError::InvalidActionCollection(format!(
                    "Invalid action collection: `actions` is a vector environment in `ScalarVecEnv::step()`"
                )));
            }
        };

        Ok(obs_collection)
    }

    fn reset(&self) -> Result<(), VecEnvError> {
        self.envs
            .iter()
            .map(|env_ref| {
                env_ref.value().reset()?;
                Ok(())
            })
            .collect::<Result<Vec<()>, VecEnvError>>()?;
        Ok(())
    }
}

pub(crate) struct BatchVecEnv<
    V: VectorEnvironment<B, D_IN, D_OUT, KInput, KOutput>,
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B>,
    KOutput: TensorKind<B>,
> {
    env_context: ContextString,
    env: V,
    device: DeviceType,
    _phantom: PhantomData<B>,
}

impl<
    V: VectorEnvironment<B, D_IN, D_OUT, KInput, KOutput>,
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B>,
    KOutput: TensorKind<B>,
> VecEnvTrait<B, D_IN, D_OUT, KInput, KOutput> for BatchVecEnv<V, B, D_IN, D_OUT, KInput, KOutput>
{
    fn init(
        client_namespace: Arc<str>,
        env: V,
        count: usize,
        device: DeviceType,
    ) -> Result<Self, VecEnvError> {
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

        let env_ids: Vec<Uuid> = env.init_num_envs(count)?;

        env_ids.into_iter().map(|id| {
            add_id(client_namespace.as_ref(), env_context.as_ref(), id)?;
        });

        Ok(Self {
            env_context,
            env,
            device,
            _phantom: PhantomData,
        })
    }

    fn step(
        &self,
        actions: TensorActList<'_, B, D_OUT>,
    ) -> Result<TensorObsList<B, D_IN>, VecEnvError> {
        let obs_collection: TensorObsList<B, D_IN> = match actions {
            TensorActList::VectorEnv(env_acts) => {
                let actions = env_acts
                    .1
                    .iter()
                    .map(|any_act| {
                        let (action, dtype) = match any_act {
                            AnyBurnTensor::Float(float_act) => (
                                Tensor::<B, D_OUT, KOutput>::from_floats(
                                    float_act.tensor.to_data(),
                                    B::get_device(&self.device)?,
                                )?,
                                float_act.dtype,
                            ),
                            AnyBurnTensor::Int(int_act) => (
                                Tensor::<B, D_OUT, KOutput>::from_ints(
                                    int_act.tensor.to_data(),
                                    B::get_device(&self.device)?,
                                )?,
                                int_act.dtype,
                            ),
                            AnyBurnTensor::Bool(bool_act) => (
                                Tensor::<B, D_OUT, KOutput>::from_bools(
                                    bool_act.tensor.to_data(),
                                    B::get_device(&self.device)?,
                                )?,
                                bool_act.dtype,
                            ),
                        };
                        Ok(action)?
                    })
                    .collect::<Vec<Tensor<B, D_OUT, KOutput>>>();

                TensorObsList::VectorEnv(
                    env_acts.0,
                    self.env
                        .step(&actions)?
                        .iter()
                        .map(|obs| obs.to_any_burn_tensor(dtype)?)
                        .collect::<Vec<AnyBurnTensor<B, D_IN>>>(),
                )
            }
            TensorActList::ScalarEnv(_) => {
                return Err(VecEnvError::InvalidActionCollection(format!(
                    "Invalid action collection: `actions` is a scalar environment in `BatchVecEnv::step()`"
                )));
            }
        };

        Ok(obs_collection)
    }

    fn reset(&self) -> Result<(), VecEnvError> {
        self.env.reset()?;
        Ok(())
    }
}

