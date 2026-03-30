pub mod kernel;
pub mod replay_buffer;

pub use kernel::*;
pub use replay_buffer::*;

use crate::logging::{EpochLogger, SessionLogger};
use crate::templates::base_algorithm::{AlgorithmError, AlgorithmTrait, TrajectoryData};
use crate::templates::base_replay_buffer::{Batch, GenericReplayBuffer, ReplayBufferError};

use burn_tensor::TensorKind;
use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

type AgentKey = String;
const DEFAULT_AGENT_KEY: &str = "__default_reinforce_agent__";

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
pub struct MAREINFORCEParams {
    pub discrete: bool,
    pub gamma: f32,
    pub lambda: f32,
    pub traj_per_epoch: u64,
    pub seed: u64,
    pub pi_lr: f32,
    pub vf_lr: f32,
}

impl Default for MAREINFORCEParams {
    fn default() -> Self {
        Self {
            discrete: true,
            gamma: 0.98,
            lambda: 0.97,
            traj_per_epoch: 8,
            seed: 1,
            pi_lr: 3e-4,
            vf_lr: 1e-3,
        }
    }
}

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

struct AgentRuntimeSlot {
    #[allow(dead_code)]
    agent_key: AgentKey,
    trajectory_count: u64,
    replay_buffer: MultiagentReinforceReplayBuffer,
}

impl AgentRuntimeSlot {
    fn new(agent_key: AgentKey, replay_buffer: MultiagentReinforceReplayBuffer) -> Self {
        Self {
            agent_key,
            trajectory_count: 0,
            replay_buffer,
        }
    }
}

struct RuntimeComponents<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    epoch_logger: EpochLogger,
    epoch_count: u64,
    agent_registry: AgentRegistry,
    agent_slots: Vec<AgentRuntimeSlot>,
    kernel: MultiagentReinforceKernel,
    _phantom: PhantomData<(B, InK, OutK)>,
}

impl<B, InK, OutK> Default for RuntimeComponents<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn default() -> Self {
        Self {
            epoch_logger: EpochLogger::new(),
            epoch_count: 0,
            agent_registry: AgentRegistry::default(),
            agent_slots: Vec::new(),
            kernel: MultiagentReinforceKernel::default(),
            _phantom: PhantomData,
        }
    }
}

struct RuntimeParams<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    #[allow(dead_code)]
    args: RuntimeArgs,
    components: RuntimeComponents<B, InK, OutK>,
}

impl<B, InK, OutK> Default for RuntimeParams<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn default() -> Self {
        Self {
            args: Default::default(),
            components: Default::default(),
        }
    }
}

pub struct MultiagentReinforceAlgorithm<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
> {
    runtime: RuntimeParams<B, InK, OutK>,
    hyperparams: MAREINFORCEParams,
}

pub type MAREINFORCEAlgorithm<B, InK, OutK> = MultiagentReinforceAlgorithm<B, InK, OutK>;

impl<B, InK, OutK> Default for MultiagentReinforceAlgorithm<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn default() -> Self {
        Self {
            runtime: Default::default(),
            hyperparams: Default::default(),
        }
    }
}

impl<B, InK, OutK> MultiagentReinforceAlgorithm<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    #[allow(dead_code)]
    pub(crate) fn new(
        hyperparams: Option<MAREINFORCEParams>,
        env_dir: &Path,
        save_model_path: &Path,
        obs_dim: usize,
        act_dim: usize,
        buffer_size: usize,
    ) -> Result<Self, AlgorithmError> {
        let hyperparams = hyperparams.unwrap_or_default();

        let algorithm = MultiagentReinforceAlgorithm {
            runtime: RuntimeParams::<B, InK, OutK> {
                args: RuntimeArgs {
                    env_dir: env_dir.to_path_buf(),
                    save_model_path: save_model_path.to_path_buf(),
                    obs_dim,
                    act_dim,
                    buffer_size,
                },
                components: RuntimeComponents::<B, InK, OutK> {
                    epoch_logger: EpochLogger::new(),
                    epoch_count: 0,
                    agent_registry: AgentRegistry::default(),
                    agent_slots: Vec::new(),
                    kernel: MultiagentReinforceKernel::new(
                        obs_dim,
                        act_dim,
                        hyperparams.discrete,
                        hyperparams.pi_lr,
                        hyperparams.vf_lr,
                    ),
                    _phantom: PhantomData,
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

        let replay_buffer = MultiagentReinforceReplayBuffer::new(
            self.runtime.args.buffer_size,
            self.hyperparams.gamma,
            self.hyperparams.lambda,
        );
        let index = self.runtime.components.agent_slots.len();
        self.runtime.components.agent_slots.push(AgentRuntimeSlot::new(
            agent_key.clone(),
            replay_buffer,
        ));
        self.runtime.components.agent_registry.insert(agent_key, index);
        self.runtime.components.kernel.register_agent();
        index
    }

    fn all_agents_ready(&self) -> bool {
        self.runtime.components.agent_registry.len() > 0
            && self.runtime.components.agent_slots.iter().all(|slot| {
                slot.trajectory_count >= self.hyperparams.traj_per_epoch
            })
    }

    fn reset_agent_counts(&mut self) {
        for slot in &mut self.runtime.components.agent_slots {
            slot.trajectory_count = 0;
        }
    }
}

impl<B, InK, OutK, T> AlgorithmTrait<T> for MultiagentReinforceAlgorithm<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
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
        let mut agent_batches = Vec::with_capacity(self.runtime.components.agent_slots.len());

        for slot in &self.runtime.components.agent_slots {
            let batch = match sample_buffer_blocking(&slot.replay_buffer) {
                Ok(batch) => batch,
                Err(_) => continue,
            };
            if let Some(agent_batch) = AgentBatch::from_batch(batch) {
                agent_batches.push(agent_batch);
            }
        }

        if agent_batches.is_empty() {
            return;
        }

        let metrics = self.runtime.components.kernel.train_epoch(&agent_batches);
        self.runtime
            .components
            .epoch_logger
            .store("LossPi", metrics.loss_pi);
        self.runtime
            .components
            .epoch_logger
            .store("LossV", metrics.loss_v);
        self.runtime.components.epoch_logger.store("KL", metrics.kl);
        self.runtime
            .components
            .epoch_logger
            .store("Entropy", metrics.entropy);
    }

    fn log_epoch(&mut self) {
        self.runtime
            .components
            .epoch_logger
            .log_tabular("Epoch", Some(self.runtime.components.epoch_count as f32));
        self.runtime.components.epoch_logger.log_tabular("EpRet", None);
        self.runtime.components.epoch_logger.log_tabular("EpLen", None);
        self.runtime.components.epoch_logger.log_tabular("LossPi", None);
        self.runtime.components.epoch_logger.log_tabular("LossV", None);
        self.runtime.components.epoch_logger.log_tabular("KL", None);
        self.runtime.components.epoch_logger.log_tabular("Entropy", None);
        self.runtime.components.epoch_logger.dump_tabular();
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentRegistry, MAREINFORCEParams};

    #[test]
    fn agent_registry_tracks_distinct_agents() {
        let mut registry = AgentRegistry::default();
        registry.insert("agent-a".to_string(), 0);
        registry.insert("agent-b".to_string(), 1);

        assert_eq!(registry.get("agent-a"), Some(0));
        assert_eq!(registry.get("agent-b"), Some(1));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn multiagent_params_default_to_shared_baseline_training() {
        let params = MAREINFORCEParams::default();

        assert!(params.discrete);
        assert!(params.traj_per_epoch > 0);
    }
}
