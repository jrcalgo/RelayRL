pub mod kernel;
pub mod replay_buffer;

pub use kernel::*;
pub use replay_buffer::*;

use crate::logging::{EpochLogger, SessionLogger};
use crate::templates::base_algorithm::{
    AlgorithmError, AlgorithmTrait, StepKernelTrait, TrajectoryData,
};
use crate::templates::base_replay_buffer::{
    Batch, BatchKey, BufferSample, GenericReplayBuffer, ReplayBufferError, SampleScalars,
};

use burn_tensor::TensorKind;
use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, TensorData};
use relayrl_types::prelude::trajectory::RelayRLTrajectory;

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

type AgentKey = String;
const DEFAULT_AGENT_KEY: &str = "__default_td3_agent__";

fn resolve_agent_key(trajectory: &RelayRLTrajectory) -> AgentKey {
    trajectory
        .get_agent_id()
        .map(|id| id.to_string())
        .or_else(|| {
            trajectory
                .actions
                .iter()
                .find_map(|a| a.get_agent_id().map(|id| id.to_string()))
        })
        .unwrap_or_else(|| DEFAULT_AGENT_KEY.to_string())
}

fn sample_buffer_blocking<RB: GenericReplayBuffer>(
    buffer: &RB,
) -> Result<Batch, ReplayBufferError> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(buffer.sample_buffer())),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReplayBufferError::BufferSamplingError(e.to_string()))?
            .block_on(buffer.sample_buffer()),
    }
}

#[derive(Default)]
struct AgentRegistry {
    indices: HashMap<AgentKey, usize>,
}

impl AgentRegistry {
    fn get(&self, key: &str) -> Option<usize> {
        self.indices.get(key).copied()
    }

    fn insert(&mut self, key: AgentKey, index: usize) {
        self.indices.insert(key, index);
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ITD3Params {
    pub gamma: f32,
    pub tau: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub batch_size: u32,
    pub buffer_size: u32,
    pub exploration_noise: f32,
    pub policy_noise: f32,
    pub noise_clip: f32,
    pub learning_starts: u32,
    pub policy_frequency: u32,
    pub train_iters: u32,
}

impl Default for ITD3Params {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            batch_size: 256,
            buffer_size: 1_000_000,
            exploration_noise: 0.1,
            policy_noise: 0.2,
            noise_clip: 0.5,
            learning_starts: 25_000,
            policy_frequency: 2,
            train_iters: 1,
        }
    }
}

pub type TD3Params = ITD3Params;

#[allow(dead_code)]
struct RuntimeArgs {
    env_dir: PathBuf,
    save_model_path: PathBuf,
    obs_dim: usize,
    act_dim: usize,
    buffer_size: usize,
}

impl Default for RuntimeArgs {
    fn default() -> Self {
        Self {
            env_dir: PathBuf::from(""),
            save_model_path: PathBuf::from(""),
            obs_dim: 1,
            act_dim: 1,
            buffer_size: 1_000_000,
        }
    }
}

struct AgentRuntimeSlot<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN> {
    #[allow(dead_code)]
    agent_key: AgentKey,
    step_count: u64,
    kernel: KN,
    replay_buffer: IndependentTD3ReplayBuffer,
    _phantom: PhantomData<(B, InK, OutK)>,
}

impl<B, InK, OutK, KN> AgentRuntimeSlot<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn new(agent_key: AgentKey, kernel: KN, replay_buffer: IndependentTD3ReplayBuffer) -> Self {
        Self {
            agent_key,
            step_count: 0,
            kernel,
            replay_buffer,
            _phantom: PhantomData,
        }
    }
}

struct RuntimeComponents<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN> {
    epoch_logger: EpochLogger,
    epoch_count: u64,
    agent_registry: AgentRegistry,
    agent_slots: Vec<AgentRuntimeSlot<B, InK, OutK, KN>>,
    seed_kernel: Option<KN>,
}

impl<B, InK, OutK, KN> Default for RuntimeComponents<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: Default,
{
    fn default() -> Self {
        Self {
            epoch_logger: EpochLogger::new(),
            epoch_count: 0,
            agent_registry: AgentRegistry::default(),
            agent_slots: Vec::new(),
            seed_kernel: Some(Default::default()),
        }
    }
}

struct RuntimeParams<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN> {
    #[allow(dead_code)]
    args: RuntimeArgs,
    components: RuntimeComponents<B, InK, OutK, KN>,
}

impl<B, InK, OutK, KN> Default for RuntimeParams<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: Default,
{
    fn default() -> Self {
        Self {
            args: Default::default(),
            components: Default::default(),
        }
    }
}

pub struct IndependentTD3Algorithm<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK>,
> {
    runtime: RuntimeParams<B, InK, OutK, KN>,
    hyperparams: ITD3Params,
}

pub type ITD3Algorithm<B, InK, OutK, KN> = IndependentTD3Algorithm<B, InK, OutK, KN>;
pub type TD3Algorithm<B, InK, OutK, KN> = IndependentTD3Algorithm<B, InK, OutK, KN>;

impl<B, InK, OutK, KN> Default for IndependentTD3Algorithm<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK> + Default,
{
    fn default() -> Self {
        Self {
            runtime: Default::default(),
            hyperparams: Default::default(),
        }
    }
}

impl<B, InK, OutK, KN> IndependentTD3Algorithm<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK> + TD3KernelTrait<B, InK, OutK> + Default,
{
    #[allow(dead_code)]
    pub(crate) fn new(
        hyperparams: Option<ITD3Params>,
        env_dir: &Path,
        save_model_path: &Path,
        obs_dim: usize,
        act_dim: usize,
        buffer_size: usize,
        kernel: KN,
    ) -> Result<Self, AlgorithmError> {
        let hyperparams = hyperparams.unwrap_or_default();

        let algorithm = IndependentTD3Algorithm {
            runtime: RuntimeParams::<B, InK, OutK, KN> {
                args: RuntimeArgs {
                    env_dir: env_dir.to_path_buf(),
                    save_model_path: save_model_path.to_path_buf(),
                    obs_dim,
                    act_dim,
                    buffer_size,
                },
                components: RuntimeComponents::<B, InK, OutK, KN> {
                    epoch_logger: EpochLogger::new(),
                    epoch_count: 0,
                    agent_registry: AgentRegistry::default(),
                    agent_slots: Vec::new(),
                    seed_kernel: Some(kernel),
                },
            },
            hyperparams,
        };

        let session_logger = SessionLogger::new();
        session_logger
            .log_session(&algorithm)
            .map_err(|e| AlgorithmError::BufferSamplingError(e.to_string()))?;

        Ok(algorithm)
    }

    fn register_agent_slot(&mut self, agent_key: AgentKey) -> usize {
        if let Some(index) = self.runtime.components.agent_registry.get(&agent_key) {
            return index;
        }

        let replay_buffer = IndependentTD3ReplayBuffer::new(
            self.runtime.args.buffer_size,
            self.hyperparams.batch_size as usize,
        );
        let obs_dim = self.runtime.args.obs_dim;
        let act_dim = self.runtime.args.act_dim;
        let kernel = self
            .runtime
            .components
            .seed_kernel
            .take()
            .unwrap_or_else(|| KN::new_for_actor(obs_dim, act_dim));
        let index = self.runtime.components.agent_slots.len();
        self.runtime
            .components
            .agent_slots
            .push(AgentRuntimeSlot::new(
                agent_key.clone(),
                kernel,
                replay_buffer,
            ));
        self.runtime
            .components
            .agent_registry
            .insert(agent_key, index);
        index
    }

    /// Reset step counts for external callers.
    pub fn reset_epoch(&mut self) {
        for slot in &mut self.runtime.components.agent_slots {
            slot.step_count = 0;
        }
    }
}

#[cfg(feature = "ndarray-backend")]
impl<B, InK, OutK, KN> IndependentTD3Algorithm<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher<Backend = B>,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK>
        + TD3KernelTrait<B, InK, OutK>
        + crate::templates::base_algorithm::WeightProvider
        + Default,
{
    /// Export the trained actor as an in-memory model (ONNX or TorchScript).
    pub fn acquire_model_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        use relayrl_types::data::tensor::{DType, NdArrayDType};

        let slot = self.runtime.components.agent_slots.first()?;
        let layer_specs = slot.kernel.get_pi_layer_specs()?;

        crate::acquire_model_module::<B>(
            "policy",
            layer_specs,
            DType::NdArray(NdArrayDType::F32),
            DType::NdArray(NdArrayDType::F32),
            vec![1, self.runtime.args.obs_dim],
            vec![1, self.runtime.args.act_dim],
            None,
        )
    }
}

impl<B, InK, OutK, KN, T> AlgorithmTrait<T> for IndependentTD3Algorithm<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher<Backend = B>,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK>
        + TD3KernelTrait<B, InK, OutK>
        + crate::templates::base_algorithm::WeightProvider
        + Default,
    T: TrajectoryData,
{
    fn save(&self, _filename: &str) {}

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        let extracted_traj: RelayRLTrajectory = trajectory.into_relayrl().ok_or_else(|| {
            AlgorithmError::TrajectoryInsertionError("Missing RelayRL trajectory".to_string())
        })?;

        let episode_len = extracted_traj.actions.len() as u64;
        let agent_key = resolve_agent_key(&extracted_traj);
        let agent_index = self.register_agent_slot(agent_key);

        let result: Box<dyn Any> = self.runtime.components.agent_slots[agent_index]
            .replay_buffer
            .insert_trajectory(extracted_traj)
            .await
            .map_err(|e| AlgorithmError::TrajectoryInsertionError(format!("{e}")))?;

        let (episode_return, episode_length) = match result.downcast::<(f32, i32)>() {
            Ok(payload) => *payload,
            Err(_) => {
                return Err(AlgorithmError::TrajectoryInsertionError(
                    "Unexpected replay buffer return payload".to_string(),
                ));
            }
        };

        self.runtime
            .components
            .epoch_logger
            .store("EpRet", episode_return);
        self.runtime
            .components
            .epoch_logger
            .store("EpLen", episode_length as f32);

        self.runtime.components.agent_slots[agent_index].step_count += episode_len;

        let ready = self.runtime.components.agent_slots[agent_index].step_count
            >= self.hyperparams.learning_starts as u64;

        if ready {
            self.runtime.components.epoch_count += 1;
            <Self as AlgorithmTrait<T>>::train_model(self);
            <Self as AlgorithmTrait<T>>::log_epoch(self);
            return Ok(true);
        }

        Ok(false)
    }

    fn train_model(&mut self) {
        let gamma = self.hyperparams.gamma;
        let tau = self.hyperparams.tau;
        let policy_noise = self.hyperparams.policy_noise;
        let noise_clip = self.hyperparams.noise_clip;
        let policy_frequency = self.hyperparams.policy_frequency;
        let train_iters = self.hyperparams.train_iters;

        for slot in &mut self.runtime.components.agent_slots {
            for _ in 0..train_iters {
                let batch = match sample_buffer_blocking(&slot.replay_buffer) {
                    Ok(b) => b,
                    Err(_) => break,
                };

                let obs: &[TensorData] = match batch.get(&BatchKey::Obs) {
                    Some(BufferSample::Tensors(t)) => t.as_ref(),
                    _ => continue,
                };
                let act: &[TensorData] = match batch.get(&BatchKey::Act) {
                    Some(BufferSample::Tensors(t)) => t.as_ref(),
                    _ => continue,
                };
                let next_obs: &[TensorData] =
                    match batch.get(&BatchKey::Custom("NextObs".to_string())) {
                        Some(BufferSample::Tensors(t)) => t.as_ref(),
                        _ => continue,
                    };
                let rew: &[f32] = match batch.get(&BatchKey::Custom("Rew".to_string())) {
                    Some(BufferSample::Scalars(SampleScalars::F32(v))) => v.as_ref(),
                    _ => continue,
                };
                let done: &[f32] = match batch.get(&BatchKey::Custom("Done".to_string())) {
                    Some(BufferSample::Scalars(SampleScalars::F32(v))) => v.as_ref(),
                    _ => continue,
                };

                let metrics = slot.kernel.td3_train_step(
                    obs,
                    act,
                    next_obs,
                    rew,
                    done,
                    gamma,
                    tau,
                    policy_noise,
                    noise_clip,
                    policy_frequency,
                );

                self.runtime
                    .components
                    .epoch_logger
                    .store("ActorLoss", metrics.actor_loss);
                self.runtime
                    .components
                    .epoch_logger
                    .store("CriticLoss", metrics.critic_loss);
            }
        }
    }

    fn log_epoch(&mut self) {
        self.runtime
            .components
            .epoch_logger
            .log_tabular("Epoch", Some(self.runtime.components.epoch_count as f32));
        self.runtime
            .components
            .epoch_logger
            .log_tabular("EpRet", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("EpLen", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("ActorLoss", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("CriticLoss", None);
        self.runtime.components.epoch_logger.dump_tabular();
    }

    #[cfg(all(
        any(feature = "tch-model", feature = "onnx-model"),
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    fn acquire_model<B2: Backend + BackendMatcher<Backend = B2>>(
        &self,
    ) -> Option<relayrl_types::model::ModelModule<B2>>
    where
        B: 'static,
        B2: 'static,
    {
        use std::any::TypeId;

        // Return None if B and B2 don't match
        if TypeId::of::<B>() != TypeId::of::<B2>() {
            return None;
        }

        // Call the existing acquire_model_module which returns ModelModule<B>
        let module_b = self.acquire_model_module()?;

        // SAFETY: TypeId check ensures B == B2
        // transmute from ModelModule<B> to ModelModule<B2>
        unsafe {
            let module_b2: relayrl_types::model::ModelModule<B2> =
                std::mem::transmute_copy(&module_b);
            std::mem::forget(module_b);
            Some(module_b2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentRegistry, DEFAULT_AGENT_KEY, ITD3Params, resolve_agent_key};
    use relayrl_types::prelude::trajectory::RelayRLTrajectory;

    #[test]
    fn resolve_agent_key_uses_default_for_missing_ids() {
        let trajectory = RelayRLTrajectory::default();
        assert_eq!(resolve_agent_key(&trajectory), DEFAULT_AGENT_KEY);
    }

    #[test]
    fn agent_registry_tracks_distinct_agents() {
        let mut registry = AgentRegistry::default();
        registry.insert("a".to_string(), 0);
        registry.insert("b".to_string(), 1);
        assert_eq!(registry.get("a"), Some(0));
        assert_eq!(registry.get("b"), Some(1));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn td3_params_default_delayed_updates() {
        let params = ITD3Params::default();
        assert_eq!(params.policy_frequency, 2);
        assert!(params.policy_noise > 0.0);
        assert!(params.noise_clip > params.policy_noise);
    }
}
