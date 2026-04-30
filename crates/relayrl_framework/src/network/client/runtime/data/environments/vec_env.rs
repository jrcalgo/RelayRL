use crate::network::ENVIRONMENT_CONTEXT_PREFIX;

use active_uuid_registry::{
    ContextString, UuidPoolError,
    interface::{add_id, remove_id, reserve_id},
};
use relayrl_env_trait::*;
use relayrl_types::data::tensor::NdArrayDType;
use relayrl_types::data::tensor::{DType, DeviceType};

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

const RAYON_STEP_MIN_ENVS: usize = 8;

#[derive(Debug, Error)]
pub enum VecEnvError {
    #[error("Invalid environment count: {0}")]
    InvalidEnvironmentCount(String),
    #[error("Unknown environment: {0}")]
    UnknownEnv(EnvironmentUuid),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[error("Tensor error: {0}")]
    TensorError(String),
}

/// Byte width of one element for a `relayrl_types::data::tensor::DType`.
fn dtype_bytes_per_elem_dtype(dtype: &DType) -> usize {
    match dtype {
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 | NdArrayDType::I16 => 2,
            NdArrayDType::F32 | NdArrayDType::I32 => 4,
            NdArrayDType::F64 | NdArrayDType::I64 => 8,
            NdArrayDType::I8 | NdArrayDType::Bool => 1,
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tc) => {
            use relayrl_types::data::tensor::TchDType;
            match tc {
                TchDType::F16 | TchDType::Bf16 | TchDType::I16 => 2,
                TchDType::F32 | TchDType::I32 => 4,
                TchDType::F64 | TchDType::I64 => 8,
                TchDType::I8 | TchDType::U8 | TchDType::Bool => 1,
            }
        }
    }
}

pub(crate) trait VecEnvTrait: Send + Sync {
    fn get_env_count(&self) -> Result<usize, VecEnvError>;
    fn env_ids(&self) -> Vec<EnvironmentUuid>;
    fn resize(&mut self, count: usize) -> Result<(), VecEnvError>;
    fn reset_all(&mut self) -> Result<(), VecEnvError>;
    fn reset_where(&mut self, env_ids: &[EnvironmentUuid]) -> Result<(), VecEnvError>;

    /// Returns `(n_envs, obs_dim, act_dim)` if the underlying env supports the flat path.
    fn n_envs_dims(&self) -> Option<(usize, usize, usize)> {
        None
    }
    /// Current observations as raw bytes (`[n_envs × obs_bytes_per_env]`).
    fn flat_observation_bytes(&self) -> Option<Vec<u8>> {
        None
    }
    /// Returns `(new_obs_bytes, rewards, dones)`.
    fn step_bytes(&mut self, _actions: &[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)> {
        None
    }
    /// Stable env UUIDs in flat-path order, or None if fast path unsupported.
    fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        None
    }
    /// `true` if the action space is discrete, `false` if continuous.
    fn action_is_discrete(&self) -> Option<bool> {
        None
    }
}

pub(crate) struct ScalarVecEnv {
    client_namespace: Arc<str>,
    env_context: ContextString,
    prototype: Box<dyn DynScalarEnvironment>,
    envs: Vec<Box<dyn DynScalarEnvironment>>,
    ordered_ids: Vec<EnvironmentUuid>,
    uuid_to_idx: HashMap<EnvironmentUuid, usize>,
    obs_flat: Vec<u8>,        // raw bytes; element dtype = observation_dtype
    obs_dim: usize,           // obs elements per env (for ONNX shape)
    obs_bytes_per_env: usize, // obs_flat stride per env
    act_dim: usize,           // action elements per env
    act_bytes_per_env: usize, // action bytes per env (1 for discrete, dtype-sized for continuous)
    #[allow(dead_code)]
    device: DeviceType,
    #[allow(dead_code)]
    observation_dtype: DType,
    #[allow(dead_code)]
    action_dtype: DType,
}

impl ScalarVecEnv {
    pub(crate) fn init_boxed(
        client_namespace: Arc<str>,
        env: Box<dyn DynScalarEnvironment>,
        count: usize,
        device: DeviceType,
        observation_dtype: DType,
        action_dtype: DType,
    ) -> Result<Self, VecEnvError> {
        if count == 0 {
            return Err(VecEnvError::InvalidEnvironmentCount(
                "count must be greater than zero".to_string(),
            ));
        }

        let env_context = format!("{}:scalar", ENVIRONMENT_CONTEXT_PREFIX);
        let mut envs = Vec::with_capacity(count);
        let mut ordered_ids = Vec::with_capacity(count);
        let mut uuid_to_idx = HashMap::with_capacity(count);

        for i in 0..count {
            let env_id = reserve_id(client_namespace.as_ref(), env_context.as_ref())?;
            envs.push(env.clone());
            ordered_ids.push(env_id);
            uuid_to_idx.insert(env_id, i);
        }

        // Probe fast-path support from the prototype env.
        let probe_obs = env.dyn_flat_obs();
        let act_dim = env.dyn_act_dim();
        let discrete = env.action_is_discrete();

        let obs_bytes_per_env = probe_obs.len();
        let obs_dim = if obs_bytes_per_env > 0 {
            obs_bytes_per_env / dtype_bytes_per_elem_dtype(&observation_dtype)
        } else {
            0
        };
        let act_bytes_per_env = if discrete {
            1
        } else {
            act_dim * dtype_bytes_per_elem_dtype(&action_dtype)
        };

        let obs_flat = if obs_bytes_per_env > 0 && act_dim > 0 {
            let mut buf = Vec::with_capacity(count * obs_bytes_per_env);
            for e in &envs {
                buf.extend_from_slice(&e.dyn_flat_obs());
            }
            buf
        } else {
            Vec::new()
        };

        Ok(Self {
            client_namespace,
            env_context,
            prototype: env,
            envs,
            ordered_ids,
            uuid_to_idx,
            obs_flat,
            obs_dim,
            obs_bytes_per_env,
            act_dim,
            act_bytes_per_env,
            device,
            observation_dtype,
            action_dtype,
        })
    }
}

impl VecEnvTrait for ScalarVecEnv {
    fn get_env_count(&self) -> Result<usize, VecEnvError> {
        Ok(self.envs.len())
    }

    fn env_ids(&self) -> Vec<EnvironmentUuid> {
        self.ordered_ids.clone()
    }

    fn resize(&mut self, count: usize) -> Result<(), VecEnvError> {
        let current = self.envs.len();
        if count == current {
            return Ok(());
        }
        if count > current {
            for i in 0..(count - current) {
                let env_id = reserve_id(self.client_namespace.as_ref(), self.env_context.as_ref())?;
                let new_env = self.prototype.clone();
                if self.obs_bytes_per_env > 0 {
                    self.obs_flat.extend_from_slice(&new_env.dyn_flat_obs());
                }
                self.envs.push(new_env);
                self.ordered_ids.push(env_id);
                self.uuid_to_idx.insert(env_id, i);
            }
        } else {
            let removed: Vec<_> = self.ordered_ids.drain(count..).collect();
            if self.obs_bytes_per_env > 0 {
                self.obs_flat.truncate(count * self.obs_bytes_per_env);
            }
            for env_id in removed {
                self.uuid_to_idx.remove(&env_id);
                remove_id(
                    self.client_namespace.as_ref(),
                    self.env_context.as_ref(),
                    env_id,
                )?;
            }
        }
        Ok(())
    }

    fn reset_all(&mut self) -> Result<(), VecEnvError> {
        let obs_bpe = self.obs_bytes_per_env;
        let n = self.envs.len();

        if n >= RAYON_STEP_MIN_ENVS && obs_bpe > 0 {
            let env_vec = &self.envs;
            let obs_flat = &mut self.obs_flat;
            env_vec
                .par_iter()
                .zip(obs_flat.par_chunks_mut(obs_bpe))
                .for_each(|(env, obs_chunk)| {
                    let _ = env.reset();
                    let obs = env.dyn_flat_obs();
                    obs_chunk.copy_from_slice(&obs);
                });
        } else if n >= RAYON_STEP_MIN_ENVS {
            self.envs.par_iter().for_each(|env| {
                let _ = env.reset();
            });
        } else {
            for (i, env) in self.envs.iter().enumerate() {
                env.reset()?;
                if obs_bpe > 0 {
                    let obs = env.dyn_flat_obs();
                    self.obs_flat[i * obs_bpe..(i + 1) * obs_bpe].copy_from_slice(&obs);
                }
            }
        }
        Ok(())
    }

    fn reset_where(&mut self, env_ids: &[EnvironmentUuid]) -> Result<(), VecEnvError> {
        if env_ids.is_empty() {
            return Ok(());
        }
        let obs_bpe = self.obs_bytes_per_env;

        // Validate all UUIDs upfront; fail before any reset starts.
        let indices: Vec<usize> = env_ids
            .iter()
            .map(|uuid| {
                self.uuid_to_idx
                    .get(uuid)
                    .copied()
                    .ok_or(VecEnvError::UnknownEnv(*uuid))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if indices.len() >= RAYON_STEP_MIN_ENVS {
            let env_vec = &self.envs;
            indices.par_iter().for_each(|&idx| {
                let _ = env_vec[idx].reset();
            });
        } else {
            for &idx in &indices {
                self.envs[idx].reset()?;
            }
        }

        // Sequential obs_flat sync over arbitrary indices.
        if obs_bpe > 0 {
            for &idx in &indices {
                let obs = self.envs[idx].dyn_flat_obs();
                self.obs_flat[idx * obs_bpe..(idx + 1) * obs_bpe].copy_from_slice(&obs);
            }
        }
        Ok(())
    }

    fn n_envs_dims(&self) -> Option<(usize, usize, usize)> {
        if self.obs_bytes_per_env == 0 || self.act_dim == 0 {
            return None;
        }
        let n = self.envs.len();
        if n == 0 {
            None
        } else {
            Some((n, self.obs_dim, self.act_dim))
        }
    }

    fn flat_observation_bytes(&self) -> Option<Vec<u8>> {
        if self.obs_bytes_per_env == 0 {
            return None;
        }
        Some(self.obs_flat.clone())
    }

    fn step_bytes(&mut self, actions: &[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)> {
        if self.obs_bytes_per_env == 0 {
            return None;
        }
        let n = self.envs.len();
        let obs_bpe = self.obs_bytes_per_env;
        let act_bpe = self.act_bytes_per_env;

        if n >= RAYON_STEP_MIN_ENVS {
            let mut rewards = vec![0.0f32; n];
            let mut dones = vec![false; n];

            // Disjoint field borrows across the closure boundary.
            let env_vec = &self.envs;
            let obs_flat = &mut self.obs_flat;

            let ok: bool = env_vec
                .par_iter()
                .zip(obs_flat.par_chunks_mut(obs_bpe))
                .zip(actions.par_chunks(act_bpe))
                .zip(rewards.par_iter_mut())
                .zip(dones.par_iter_mut())
                .map(|((((env, obs_chunk), env_act), reward), done)| {
                    let (obs, r, d) = env.dyn_step(env_act)?;
                    obs_chunk.copy_from_slice(&obs);
                    *reward = r;
                    *done = d;
                    Some(())
                })
                .all(|r| r.is_some());

            if ok {
                Some((self.obs_flat.clone(), rewards, dones))
            } else {
                None
            }
        } else {
            let mut rewards = Vec::with_capacity(n);
            let mut dones = Vec::with_capacity(n);

            for (i, env) in self.envs.iter().enumerate() {
                let env_act = &actions[i * act_bpe..(i + 1) * act_bpe];
                let (obs, reward, done) = env.dyn_step(env_act)?;
                self.obs_flat[i * obs_bpe..(i + 1) * obs_bpe].copy_from_slice(&obs);
                rewards.push(reward);
                dones.push(done);
            }
            Some((self.obs_flat.clone(), rewards, dones))
        }
    }

    fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        if self.obs_dim == 0 {
            return None;
        }
        Some(self.ordered_ids.clone())
    }

    fn action_is_discrete(&self) -> Option<bool> {
        if self.obs_bytes_per_env == 0 {
            return None;
        }
        Some(
            self.envs
                .first()
                .map(|env| env.action_is_discrete())
                .unwrap_or(true),
        )
    }
}

pub(crate) struct BatchVecEnv {
    client_namespace: Arc<str>,
    env_context: ContextString,
    env: Box<DynVectorEnv>,
    env_ids: Vec<EnvironmentUuid>,
    #[allow(dead_code)]
    device: DeviceType,
    #[allow(dead_code)]
    observation_dtype: DType,
    #[allow(dead_code)]
    action_dtype: DType,
}

impl BatchVecEnv {
    pub(crate) fn init_boxed(
        client_namespace: Arc<str>,
        env: Box<DynVectorEnv>,
        count: usize,
        device: DeviceType,
        observation_dtype: DType,
        action_dtype: DType,
    ) -> Result<Self, VecEnvError> {
        if count == 0 {
            return Err(VecEnvError::InvalidEnvironmentCount(
                "count must be greater than zero".to_string(),
            ));
        }

        let env_context = format!("{}:vector", ENVIRONMENT_CONTEXT_PREFIX);
        let env_ids = env.init_num_envs(count)?;
        for env_id in &env_ids {
            add_id(client_namespace.as_ref(), env_context.as_ref(), *env_id)?;
        }

        Ok(Self {
            client_namespace,
            env_context,
            env,
            env_ids,
            device,
            observation_dtype,
            action_dtype,
        })
    }
}

impl VecEnvTrait for BatchVecEnv {
    fn get_env_count(&self) -> Result<usize, VecEnvError> {
        Ok(self.env_ids.len())
    }

    fn env_ids(&self) -> Vec<EnvironmentUuid> {
        self.env_ids.clone()
    }

    fn resize(&mut self, count: usize) -> Result<(), VecEnvError> {
        let current = self.env_ids.len();
        if count == current {
            return Ok(());
        }

        if count > current {
            let new_ids = self.env.init_num_envs(count - current)?;
            for env_id in &new_ids {
                add_id(
                    self.client_namespace.as_ref(),
                    self.env_context.as_ref(),
                    *env_id,
                )?;
            }
            self.env_ids.extend(new_ids);
        } else {
            let removed = self.env_ids.split_off(count);
            for env_id in removed {
                remove_id(
                    self.client_namespace.as_ref(),
                    self.env_context.as_ref(),
                    env_id,
                )?;
            }
        }
        Ok(())
    }

    fn reset_all(&mut self) -> Result<(), VecEnvError> {
        let ids = self.env_ids.clone();
        self.reset_where(&ids)
    }

    fn reset_where(&mut self, env_ids: &[EnvironmentUuid]) -> Result<(), VecEnvError> {
        self.env.reset(env_ids)?;
        Ok(())
    }

    fn n_envs_dims(&self) -> Option<(usize, usize, usize)> {
        let n = self.env.n_envs();
        let obs = self.env.observation_dim();
        let act = self.env.action_dim();
        if n == 0 { None } else { Some((n, obs, act)) }
    }

    fn flat_observation_bytes(&self) -> Option<Vec<u8>> {
        Some(self.env.flat_observation_bytes())
    }

    fn step_bytes(&mut self, actions: &[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)> {
        self.env.step_bytes(actions)
    }

    fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        let n = self.env.n_envs();
        if n == 0 {
            None
        } else {
            Some(self.env_ids.clone())
        }
    }

    fn action_is_discrete(&self) -> Option<bool> {
        Some(self.env.action_is_discrete())
    }
}
