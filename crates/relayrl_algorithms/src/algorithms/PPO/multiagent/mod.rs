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
    // `block_in_place` yields the current thread to the Tokio scheduler while
    // blocking, which is safe to call from within an async multi-thread runtime
    // (unlike `Handle::block_on` which panics when called from async context).
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(buffer.sample_buffer())
        })
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| ReplayBufferError::BufferSamplingError(e.to_string()))?
            .block_on(buffer.sample_buffer())
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
pub struct MAPPOParams {
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

impl Default for MAPPOParams {
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
    replay_buffer: MultiagentPPOReplayBuffer,
}

impl AgentRuntimeSlot {
    fn new(agent_key: AgentKey, replay_buffer: MultiagentPPOReplayBuffer) -> Self {
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
    kernel: MultiagentPPOKernel,
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
            kernel: MultiagentPPOKernel::default(),
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

pub struct MultiagentPPOAlgorithm<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
> {
    runtime: RuntimeParams<B, InK, OutK>,
    hyperparams: MAPPOParams,
}

pub type MAPPOAlgorithm<B, InK, OutK> = MultiagentPPOAlgorithm<B, InK, OutK>;

impl<B, InK, OutK> Default for MultiagentPPOAlgorithm<B, InK, OutK>
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

impl<B, InK, OutK> MultiagentPPOAlgorithm<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    #[allow(dead_code)]
    pub(crate) fn new(
        hyperparams: Option<MAPPOParams>,
        env_dir: &Path,
        save_model_path: &Path,
        obs_dim: usize,
        act_dim: usize,
        buffer_size: usize,
    ) -> Result<Self, AlgorithmError> {
        let hyperparams = hyperparams.unwrap_or_default();

        let algorithm = MultiagentPPOAlgorithm {
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
                    kernel: MultiagentPPOKernel::new(
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

        let replay_buffer = MultiagentPPOReplayBuffer::new(
            self.runtime.args.buffer_size,
            self.hyperparams.gamma,
            self.hyperparams.lam,
        );
        let index = self.runtime.components.agent_slots.len();
        self.runtime
            .components
            .agent_slots
            .push(AgentRuntimeSlot::new(agent_key.clone(), replay_buffer));
        self.runtime
            .components
            .agent_registry
            .insert(agent_key, index);
        self.runtime.components.kernel.register_agent();
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

    /// No-op for multi-agent PPO; trajectory counts are managed internally.
    pub fn reset_epoch(&mut self) {}
}

#[cfg(feature = "ndarray-backend")]
impl<B, InK, OutK> MultiagentPPOAlgorithm<B, InK, OutK>
where
    B: Backend + BackendMatcher<Backend = B>,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    /// Export the shared policy as an in-memory ONNX model.
    ///
    /// Reads from the first actor in the shared `MultiagentPPOKernel`. Returns `None`
    /// if no training has occurred yet.
    pub fn acquire_model_module(
        &self,
    ) -> Option<relayrl_types::model::ModelModule<B>> {
        use crate::algorithms::onnx_builder::build_onnx_mlp_bytes;
        use relayrl_types::data::tensor::{DType, NdArrayDType};
        use relayrl_types::model::{ModelFileType, ModelMetadata, ModelModule};

        let layer_specs = self.runtime.components.kernel.get_pi_layer_specs()?;
        if layer_specs.is_empty() {
            return None;
        }

        let obs_dim = self.runtime.args.obs_dim;
        let act_dim = self.runtime.args.act_dim;

        let onnx_bytes = build_onnx_mlp_bytes(&layer_specs);
        let metadata = ModelMetadata {
            model_file: "model.onnx".to_string(),
            model_type: ModelFileType::Onnx,
            input_dtype: DType::NdArray(NdArrayDType::F32),
            output_dtype: DType::NdArray(NdArrayDType::F32),
            input_shape: vec![1, obs_dim],
            output_shape: vec![1, act_dim],
            default_device: None,
        };
        ModelModule::from_onnx_bytes(onnx_bytes, metadata).ok()
    }
}

impl<B, InK, OutK, T> AlgorithmTrait<T> for MultiagentPPOAlgorithm<B, InK, OutK>
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

        let metrics = self.runtime.components.kernel.train_epoch(
            &agent_batches,
            self.hyperparams.clip_ratio,
            self.hyperparams.target_kl,
            self.hyperparams.train_pi_iters,
            self.hyperparams.train_vf_iters,
        );
        self.runtime
            .components
            .epoch_logger
            .store("LossPi", metrics.loss_pi);
        self.runtime
            .components
            .epoch_logger
            .store("DeltaLossPi", metrics.delta_loss_pi);
        self.runtime
            .components
            .epoch_logger
            .store("LossV", metrics.loss_v);
        self.runtime
            .components
            .epoch_logger
            .store("DeltaLossV", metrics.delta_loss_v);
        self.runtime.components.epoch_logger.store("KL", metrics.kl);
        self.runtime
            .components
            .epoch_logger
            .store("Entropy", metrics.entropy);
        self.runtime
            .components
            .epoch_logger
            .store("ClipFrac", metrics.clipfrac);
        self.runtime
            .components
            .epoch_logger
            .store("StopIter", metrics.stop_iter as f32);
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
    use super::{AgentRegistry, MAPPOParams};

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
    fn mappo_params_default_to_shared_training() {
        let params = MAPPOParams::default();

        assert!(params.discrete);
        assert!(params.clip_ratio > 0.0);
    }
}
