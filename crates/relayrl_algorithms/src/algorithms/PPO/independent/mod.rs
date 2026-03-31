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
const DEFAULT_AGENT_KEY: &str = "__default_ppo_agent__";

fn resolve_agent_key(trajectory: &RelayRLTrajectory) -> AgentKey {
    trajectory
        .get_agent_id()
        .map(|agent_id| agent_id.to_string())
        .or_else(|| {
            trajectory
                .actions
                .iter()
                .find_map(|action| action.get_agent_id().map(|agent_id| agent_id.to_string()))
        })
        .unwrap_or_else(|| DEFAULT_AGENT_KEY.to_string())
}

fn sample_buffer_blocking<RB: GenericReplayBuffer>(
    buffer: &RB,
) -> Result<Batch, ReplayBufferError> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(buffer.sample_buffer()),
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
    fn get(&self, agent_key: &str) -> Option<usize> {
        self.indices.get(agent_key).copied()
    }

    fn insert(&mut self, agent_key: AgentKey, index: usize) {
        self.indices.insert(agent_key, index);
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

#[allow(dead_code)]
pub struct IPPOParams {
    pub discrete: bool,
    pub gamma: f32,
    pub lam: f32,
    pub clip_ratio: f32,
    pub pi_lr: f32,
    pub vf_lr: f32,
    pub train_pi_iters: u64,
    pub train_vf_iters: u64,
    pub target_kl: f32,
    pub traj_per_epoch: u64,
}

impl Default for IPPOParams {
    fn default() -> Self {
        Self {
            discrete: true,
            gamma: 0.99,
            lam: 0.97,
            clip_ratio: 0.2,
            pi_lr: 3e-4,
            vf_lr: 1e-3,
            train_pi_iters: 80,
            train_vf_iters: 80,
            target_kl: 0.01,
            traj_per_epoch: 8,
        }
    }
}

pub type PPOParams = IPPOParams;

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
    trajectory_count: u64,
    kernel: KN,
    replay_buffer: IndependentPPOReplayBuffer,
    _phantom: PhantomData<(B, InK, OutK)>,
}

impl<B, InK, OutK, KN> AgentRuntimeSlot<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn new(agent_key: AgentKey, kernel: KN, replay_buffer: IndependentPPOReplayBuffer) -> Self {
        Self {
            agent_key,
            trajectory_count: 0,
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

pub struct IndependentPPOAlgorithm<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK>,
> {
    runtime: RuntimeParams<B, InK, OutK, KN>,
    hyperparams: IPPOParams,
}

pub type IPPOAlgorithm<B, InK, OutK, KN> = IndependentPPOAlgorithm<B, InK, OutK, KN>;
pub type PPOAlgorithm<B, InK, OutK, KN> = IndependentPPOAlgorithm<B, InK, OutK, KN>;

impl<B, InK, OutK, KN> Default for IndependentPPOAlgorithm<B, InK, OutK, KN>
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

impl<B, InK, OutK, KN> IndependentPPOAlgorithm<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK> + Default,
{
    #[allow(dead_code)]
    pub(crate) fn new(
        hyperparams: Option<IPPOParams>,
        env_dir: &Path,
        save_model_path: &Path,
        obs_dim: usize,
        act_dim: usize,
        buffer_size: usize,
        kernel: KN,
    ) -> Result<Self, AlgorithmError> {
        let hyperparams = hyperparams.unwrap_or_default();

        let algorithm = IndependentPPOAlgorithm {
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

        let replay_buffer = IndependentPPOReplayBuffer::new(
            self.runtime.args.buffer_size,
            self.hyperparams.gamma,
            self.hyperparams.lam,
        );
        let kernel = self
            .runtime
            .components
            .seed_kernel
            .take()
            .unwrap_or_default();
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

    fn all_agents_ready(&self) -> bool {
        self.runtime.components.agent_registry.len() > 0
            && self
                .runtime
                .components
                .agent_slots
                .iter()
                .all(|slot| slot.trajectory_count >= self.hyperparams.traj_per_epoch)
    }

    fn reset_agent_counts(&mut self) {
        for slot in &mut self.runtime.components.agent_slots {
            slot.trajectory_count = 0;
        }
    }
}

impl<B, InK, OutK, KN, T> AlgorithmTrait<T> for IndependentPPOAlgorithm<B, InK, OutK, KN>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    KN: StepKernelTrait<B, InK, OutK> + self::kernel::PPOKernelTrait<B, InK, OutK> + Default,
    T: TrajectoryData,
{
    fn save(&self, _filename: &str) {}

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        let extracted_traj: RelayRLTrajectory = trajectory.into_relayrl().ok_or_else(|| {
            AlgorithmError::TrajectoryInsertionError("Missing RelayRL trajectory".to_string())
        })?;

        let agent_key = resolve_agent_key(&extracted_traj);
        let agent_index = self.register_agent_slot(agent_key);
        let slot = &mut self.runtime.components.agent_slots[agent_index];
        slot.trajectory_count += 1;

        let result: Box<dyn Any> = slot
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

        if self.all_agents_ready() {
            self.runtime.components.epoch_count += 1;
            <Self as AlgorithmTrait<T>>::train_model(self);
            <Self as AlgorithmTrait<T>>::log_epoch(self);
            self.reset_agent_counts();
            return Ok(true);
        }

        Ok(false)
    }

    fn train_model(&mut self) {
        for slot in &mut self.runtime.components.agent_slots {
            let batch = match sample_buffer_blocking(&slot.replay_buffer) {
                Ok(batch) => batch,
                Err(_) => continue,
            };

            let obs: &[TensorData] = match batch.get(&BatchKey::Obs) {
                Some(BufferSample::Tensors(tensors)) => tensors.as_ref(),
                _ => continue,
            };
            let act: &[TensorData] = match batch.get(&BatchKey::Act) {
                Some(BufferSample::Tensors(tensors)) => tensors.as_ref(),
                _ => continue,
            };
            let mask: &[TensorData] = match batch.get(&BatchKey::Mask) {
                Some(BufferSample::Tensors(tensors)) => tensors.as_ref(),
                _ => continue,
            };
            let adv: &[f32] = match batch.get(&BatchKey::Custom("Adv".to_string())) {
                Some(BufferSample::Scalars(SampleScalars::F32(values))) => values.as_ref(),
                _ => continue,
            };
            let ret: &[f32] = match batch.get(&BatchKey::Custom("Ret".to_string())) {
                Some(BufferSample::Scalars(SampleScalars::F32(values))) => values.as_ref(),
                _ => continue,
            };
            let logp_old: &[TensorData] = match batch.get(&BatchKey::Custom("LogP".to_string())) {
                Some(BufferSample::Tensors(tensors)) => tensors.as_ref(),
                _ => continue,
            };

            let mut first_pi_loss: Option<f32> = None;
            let mut final_pi_loss = 0.0f32;
            let mut final_kl = 0.0f32;
            let mut final_entropy = 0.0f32;
            let mut final_clipfrac = 0.0f32;
            let mut stop_iter = 0u64;

            for i in 0..self.hyperparams.train_pi_iters {
                let (loss, info) = slot.kernel.ppo_pi_loss(
                    obs,
                    act,
                    mask,
                    adv,
                    logp_old,
                    self.hyperparams.clip_ratio,
                );

                first_pi_loss.get_or_insert(loss);
                final_pi_loss = loss;
                final_kl = *info.get("kl").unwrap_or(&0.0);
                final_entropy = *info.get("entropy").unwrap_or(&0.0);
                final_clipfrac = *info.get("clipfrac").unwrap_or(&0.0);
                stop_iter = i + 1;

                if final_kl > 1.5 * self.hyperparams.target_kl {
                    break;
                }
            }

            let mut first_vf_loss: Option<f32> = None;
            let mut final_vf_loss = 0.0f32;
            for _ in 0..self.hyperparams.train_vf_iters {
                let loss = slot.kernel.ppo_vf_loss(obs, mask, ret);
                first_vf_loss.get_or_insert(loss);
                final_vf_loss = loss;
            }

            let first_pi_loss = first_pi_loss.unwrap_or(final_pi_loss);
            let first_vf_loss = first_vf_loss.unwrap_or(final_vf_loss);

            self.runtime
                .components
                .epoch_logger
                .store("LossPi", final_pi_loss);
            self.runtime
                .components
                .epoch_logger
                .store("DeltaLossPi", final_pi_loss - first_pi_loss);
            self.runtime
                .components
                .epoch_logger
                .store("LossV", final_vf_loss);
            self.runtime
                .components
                .epoch_logger
                .store("DeltaLossV", final_vf_loss - first_vf_loss);
            self.runtime.components.epoch_logger.store("KL", final_kl);
            self.runtime
                .components
                .epoch_logger
                .store("Entropy", final_entropy);
            self.runtime
                .components
                .epoch_logger
                .store("ClipFrac", final_clipfrac);
            self.runtime
                .components
                .epoch_logger
                .store("StopIter", stop_iter as f32);
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
            .log_tabular("LossPi", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("DeltaLossPi", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("LossV", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("DeltaLossV", None);
        self.runtime.components.epoch_logger.log_tabular("KL", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("Entropy", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("ClipFrac", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("StopIter", None);
        self.runtime.components.epoch_logger.dump_tabular();
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentRegistry, DEFAULT_AGENT_KEY, IPPOParams, resolve_agent_key};
    use relayrl_types::prelude::trajectory::RelayRLTrajectory;

    #[test]
    fn resolve_agent_key_uses_default_for_missing_agent_ids() {
        let trajectory = RelayRLTrajectory::default();
        assert_eq!(resolve_agent_key(&trajectory), DEFAULT_AGENT_KEY);
    }

    #[test]
    fn agent_registry_keeps_stable_insertion_order() {
        let mut registry = AgentRegistry::default();
        registry.insert("agent-a".to_string(), 0);
        registry.insert("agent-b".to_string(), 1);

        assert_eq!(registry.get("agent-a"), Some(0));
        assert_eq!(registry.get("agent-b"), Some(1));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn ppo_params_preserve_clip_settings() {
        let params = IPPOParams::default();

        assert!(params.clip_ratio > 0.0);
        assert!(params.target_kl > 0.0);
    }
}
