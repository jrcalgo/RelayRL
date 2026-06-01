use crate::algorithms::PPO::kernel::{
    PPOKernel, PPOKernelFactory, PPOKernelOps, PPOKernelTraining, PPOKernelTrainingArgs,
    PPOPolicyHead,
};
use crate::algorithms::PPO::replay_buffer::PPOReplayBuffer;
use crate::algorithms::{GenericMlp, NeuralNetwork};
use crate::logging::{EpochLogger, SessionLogger};
use crate::templates::base_algorithm::{AlgorithmError, AlgorithmTrait, TrajectoryData};
use crate::templates::base_replay_buffer::GenericReplayBuffer;

use burn_tensor::BasicOps;
use burn_tensor::backend::Backend;
use burn_tensor::{Float, TensorKind};
use relayrl_types::data::tensor::TensorData;
use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use super::replay_buffer::PPOBatch;

type AgentKey = String;
const DEFAULT_AGENT_KEY: &str = "__default_ppo_agent__";

pub struct SlotTrainResult<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub kernel: PPOKernel<B, KindIn, KindOut, Pi>,
    pub pi_loss: f32,
    pub delta_pi_loss: f32,
    pub vf_loss: f32,
    pub delta_vf_loss: f32,
    pub kl: f32,
    pub entropy: f32,
    pub clipfrac: f32,
    pub stop_iter: f32,
}

pub struct EpochTrainOutput<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub slot_results: Vec<SlotTrainResult<B, KindIn, KindOut, Pi>>,
}

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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    pub ent_coef: f32,
    pub vf_coef: f32,
    pub max_version_lag: i64,
    pub normalize_obs: bool,
    pub normalize_returns: bool,
    pub max_episode_steps: Option<usize>,
    pub minibatch: Option<usize>,
    pub min_steps_per_epoch: Option<u64>,
    pub max_buffered_episodes: Option<u64>,
    pub rollout_len: Option<usize>,
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
            ent_coef: 0.0,
            vf_coef: 0.5,
            max_version_lag: 1,
            normalize_obs: false,
            normalize_returns: false,
            max_episode_steps: None,
            minibatch: None,
            min_steps_per_epoch: None,
            max_buffered_episodes: None,
            rollout_len: None,
        }
    }
}

pub type PPOParams = IPPOParams;

#[allow(dead_code)]
struct RuntimeArgs {
    env_dir: PathBuf,
    save_model_path: PathBuf,
    obs_dim: usize,
    obs_dtype: DType,
    act_dim: usize,
    act_dtype: DType,
    buffer_size: usize,
}

impl Default for RuntimeArgs {
    fn default() -> Self {
        Self {
            env_dir: PathBuf::from(""),
            save_model_path: PathBuf::from(""),
            obs_dim: 1,
            obs_dtype: DType::NdArray(NdArrayDType::F32),
            act_dim: 1,
            act_dtype: DType::NdArray(NdArrayDType::F32),
            buffer_size: 1_000,
        }
    }
}

struct AgentRuntimeSlot<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    #[allow(dead_code)]
    agent_key: AgentKey,
    trajectory_count: u64,
    kernel: Option<PPOKernel<B, KindIn, KindOut, Pi>>,
    replay_buffer: PPOReplayBuffer,
    _phantom: PhantomData<(B, KindIn, KindOut)>,
}

impl<B, KindIn, KindOut, Pi> AgentRuntimeSlot<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    fn new(
        agent_key: AgentKey,
        kernel: PPOKernel<B, KindIn, KindOut, Pi>,
        replay_buffer: PPOReplayBuffer,
    ) -> Self {
        Self {
            agent_key,
            trajectory_count: 0,
            kernel: Some(kernel),
            replay_buffer,
            _phantom: PhantomData,
        }
    }
}

struct RuntimeComponents<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    epoch_logger: EpochLogger,
    epoch_count: u64,
    model_version: i64,
    agent_registry: AgentRegistry,
    agent_slots: Vec<AgentRuntimeSlot<B, KindIn, KindOut, Pi>>,
    seed_kernel: Option<PPOKernel<B, KindIn, KindOut, Pi>>,
}

impl<B, KindIn, KindOut, Pi> Default for RuntimeComponents<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    fn default() -> Self {
        Self {
            epoch_logger: EpochLogger::new(),
            epoch_count: 0,
            model_version: 0,
            agent_registry: AgentRegistry::default(),
            agent_slots: Vec::new(),
            seed_kernel: None,
        }
    }
}

struct RuntimeParams<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    #[allow(dead_code)]
    args: RuntimeArgs,
    components: RuntimeComponents<B, KindIn, KindOut, Pi>,
}

impl<B, KindIn, KindOut, Pi> Default for RuntimeParams<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    fn default() -> Self {
        Self {
            args: Default::default(),
            components: Default::default(),
        }
    }
}

pub struct IndependentPPOAlgorithm<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    runtime: RuntimeParams<B, KindIn, KindOut, Pi>,
    hyperparams: IPPOParams,
}

impl<B, KindIn, KindOut, Pi> Default for IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    fn default() -> Self {
        Self {
            runtime: Default::default(),
            hyperparams: Default::default(),
        }
    }
}

impl<B, KindIn, KindOut, Pi> IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        hyperparams: Option<IPPOParams>,
        env_dir: &Path,
        save_model_path: &Path,
        obs_dim: &usize,
        obs_dtype: &DType,
        act_dim: &usize,
        act_dtype: &DType,
        buffer_size: &usize,
        pi_head: PPOPolicyHead<B, KindIn, KindOut, Pi>,
        vf_mlp: GenericMlp<B, KindIn, Float>,
    ) -> Result<Self, AlgorithmError> {
        let hyperparams = hyperparams.unwrap_or_default();

        let training_args = PPOKernelTrainingArgs {
            pi_lr: hyperparams.pi_lr as f64,
            vf_coef: hyperparams.vf_coef,
            lr_schedule_steps: None,
        };
        let kernel: PPOKernel<B, KindIn, KindOut, Pi> =
            PPOKernelFactory::new(pi_head, vf_mlp, training_args)?;

        let algorithm = IndependentPPOAlgorithm {
            runtime: RuntimeParams::<B, KindIn, KindOut, Pi> {
                args: RuntimeArgs {
                    env_dir: env_dir.to_path_buf(),
                    save_model_path: save_model_path.to_path_buf(),
                    obs_dim: *obs_dim,
                    obs_dtype: obs_dtype.clone(),
                    act_dim: *act_dim,
                    act_dtype: act_dtype.clone(),
                    buffer_size: *buffer_size,
                },
                components: RuntimeComponents::<B, KindIn, KindOut, Pi> {
                    epoch_logger: EpochLogger::new(),
                    epoch_count: 0,
                    model_version: 0,
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

    pub fn get_ppo_actor_kernel(
        &self,
    ) -> Result<&PPOKernel<B, KindIn, KindOut, Pi>, AlgorithmError> {
        if let Some(kernel) = self
            .runtime
            .components
            .agent_slots
            .first()
            .and_then(|slot| slot.kernel.as_ref())
        {
            return Ok(kernel);
        }
        Err(AlgorithmError::InitializationError(
            "No kernel found".to_string(),
        ))
    }

    pub fn get_ippo_actor_kernel(
        &self,
        agent_key: AgentKey,
    ) -> Result<&PPOKernel<B, KindIn, KindOut, Pi>, AlgorithmError> {
        if let Some(kernel) = self
            .runtime
            .components
            .agent_slots
            .iter()
            .find(|slot| slot.agent_key == agent_key)
            .and_then(|slot| slot.kernel.as_ref())
        {
            return Ok(kernel);
        }
        Err(AlgorithmError::InitializationError(format!(
            "No kernel found for agent key: {}",
            agent_key
        )))
    }

    fn register_agent_slot(&mut self, agent_key: AgentKey) -> Result<usize, AlgorithmError> {
        if let Some(index) = self.runtime.components.agent_registry.get(&agent_key) {
            return Ok(index);
        }

        let index = {
            let replay_buffer = PPOReplayBuffer::new(
                self.runtime.args.buffer_size,
                self.hyperparams.gamma,
                self.hyperparams.lam,
                self.hyperparams.max_buffered_episodes.map(|v| v as usize),
            );
            let kernel = self.runtime.components.seed_kernel.take().ok_or_else(|| {
                AlgorithmError::InitializationError("No seed kernel found".to_string())
            })?;
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
        };

        Ok(index)
    }

    fn all_agents_ready(&self) -> bool {
        let has_agents = self.runtime.components.agent_registry.len() > 0;
        if !has_agents {
            return false;
        }
        if let Some(min_steps) = self.hyperparams.min_steps_per_epoch {
            // Ready when buffer has enough complete steps for at least one drain
            self.runtime.components.agent_slots.iter().all(|slot| {
                slot.replay_buffer
                    .episodes_needed_for_steps(min_steps as usize)
                    > 0
            })
        } else {
            self.runtime
                .components
                .agent_slots
                .iter()
                .all(|slot| slot.trajectory_count >= self.hyperparams.traj_per_epoch)
        }
    }

    fn reset_agent_counts(&mut self) {
        for slot in &mut self.runtime.components.agent_slots {
            slot.trajectory_count = 0;
        }
    }

    /// Reset per-actor trajectory counts.
    ///
    /// Call this to prevent `receive_trajectory` from auto-triggering `train_model`
    /// when resuming from an async context mid-epoch.
    pub fn reset_epoch(&mut self) {
        self.reset_agent_counts();
    }

    /// Pre-register the first agent slot so the kernel is available for inference
    /// before any trajectory has been received.
    pub fn register_first_slot_with_key(
        &mut self,
        agent_key: String,
    ) -> Result<(), AlgorithmError> {
        if self
            .runtime
            .components
            .agent_registry
            .get(&agent_key)
            .is_none()
        {
            self.register_agent_slot(agent_key)
                .map_err(|e| AlgorithmError::InitializationError(e.to_string()))?;
        }
        Ok(())
    }

    /// Extract epoch data + kernel from all slots and launch SGD in a background thread.
    /// Returns immediately after draining buffers; collection can fill the next epoch in parallel.
    pub fn start_epoch_training(
        &mut self,
    ) -> Option<tokio::task::JoinHandle<EpochTrainOutput<B, KindIn, KindOut, Pi>>>
    where
        B: Send + 'static,
        KindIn: Send + 'static,
        KindOut: Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    {
        let traj_n_default = self.hyperparams.traj_per_epoch as usize;
        let min_steps_opt = self.hyperparams.min_steps_per_epoch;
        // model_version increments only in apply_epoch_result (one real training completion = +1),
        // so it never inflates from wasted all_agents_ready() triggers during background SGD.
        // Use model_version directly (not +1) so episodes from the preceding model push
        // (lag=1) are accepted as fresh — necessary because perform_refresh_model runs
        // asynchronously and version-0 episodes keep arriving until it completes.
        let current_version = self.runtime.components.model_version;
        let max_version_lag = self.hyperparams.max_version_lag;
        let mut jobs: Vec<(PPOKernel<B, KindIn, KindOut, Pi>, PPOBatch)> = Vec::new();
        for slot in &mut self.runtime.components.agent_slots {
            let n = if let Some(min_steps) = min_steps_opt {
                // Drain the minimum episodes covering min_steps; leave excess for next epoch
                slot.replay_buffer
                    .episodes_needed_for_steps(min_steps as usize)
            } else {
                traj_n_default
            };
            if n == 0 {
                continue;
            }
            // Take kernel first so we can recompute V(s_t) from the current Burn model,
            // eliminating stale-value bias in GAE (stored values came from the old TorchScript model).
            let kernel = slot.kernel.take()?;
            let (obs_flat, obs_dim_peek) = slot.replay_buffer.get_obs_for_first_n_episodes(n);
            let fresh_values = if !obs_flat.is_empty() {
                kernel.value_forward(&obs_flat, obs_dim_peek)
            } else {
                Vec::new()
            };
            // finalize_and_drain drains all n episodes but only includes fresh ones in the batch.
            // If all n were stale, it returns None — restore kernel so the next epoch can use it.
            match slot.replay_buffer.finalize_and_drain_first_n_blocking(
                fresh_values,
                current_version,
                max_version_lag,
                n,
                self.hyperparams.normalize_returns,
            ) {
                Some(mut batch) => {
                    // Recompute logp_old from the current burn model — eliminates both the
                    // ORT/burn numerical mismatch and same-epoch staleness. Values are already
                    // refreshed above (fresh_values); this completes the picture for log-probs.
                    // Cost: one extra CPU forward pass per epoch (no backward).
                    let fresh_logp = kernel.get_pi_logprobs(&batch.obs, batch.obs_dim, &batch.act);
                    if fresh_logp.len() == batch.logp.len() {
                        batch.logp = fresh_logp;
                    }
                    jobs.push((kernel, batch))
                }
                None => {
                    slot.kernel = Some(kernel);
                    continue;
                }
            }
        }
        if jobs.is_empty() {
            return None;
        }

        let clip_ratio = self.hyperparams.clip_ratio;
        let ent_coef = self.hyperparams.ent_coef;
        let target_kl = self.hyperparams.target_kl;
        let train_pi_iters = self.hyperparams.train_pi_iters;
        let mb_size_opt = self.hyperparams.minibatch;

        Some(tokio::task::spawn_blocking(move || {
            let slot_results = jobs
                .into_iter()
                .map(|(kernel, batch)| {
                    run_ppo_sgd_flat::<B, KindIn, KindOut, Pi>(
                        kernel,
                        batch,
                        clip_ratio,
                        ent_coef,
                        target_kl,
                        train_pi_iters,
                        mb_size_opt,
                    )
                })
                .collect();
            EpochTrainOutput { slot_results }
        }))
    }

    /// Restore kernels from a completed background training run and record training stats.
    pub fn apply_epoch_result(&mut self, output: EpochTrainOutput<B, KindIn, KindOut, Pi>) {
        // Advance the model version — this is the only place it increments, ensuring
        // it reflects real training completions and never inflates from wasted triggers.
        self.runtime.components.model_version += 1;
        for (slot, result) in self
            .runtime
            .components
            .agent_slots
            .iter_mut()
            .zip(output.slot_results)
        {
            slot.kernel = Some(result.kernel);
            self.runtime
                .components
                .epoch_logger
                .store("LossPi", result.pi_loss);
            self.runtime
                .components
                .epoch_logger
                .store("DeltaLossPi", result.delta_pi_loss);
            self.runtime
                .components
                .epoch_logger
                .store("LossV", result.vf_loss);
            self.runtime
                .components
                .epoch_logger
                .store("DeltaLossV", result.delta_vf_loss);
            self.runtime.components.epoch_logger.store("KL", result.kl);
            self.runtime
                .components
                .epoch_logger
                .store("Entropy", result.entropy);
            self.runtime
                .components
                .epoch_logger
                .store("ClipFrac", result.clipfrac);
            self.runtime
                .components
                .epoch_logger
                .store("StopIter", result.stop_iter);
        }
    }

    fn backend_f32_dtype() -> relayrl_types::data::tensor::DType {
        match B::get_supported_backend() {
            #[cfg(feature = "tch-backend")]
            relayrl_types::data::tensor::SupportedTensorBackend::Tch => {
                relayrl_types::data::tensor::DType::Tch(relayrl_types::data::tensor::TchDType::F32)
            }
            _ => DType::NdArray(relayrl_types::data::tensor::NdArrayDType::F32),
        }
    }

    pub fn acquire_pi_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        let slot = self.runtime.components.agent_slots.first()?;
        let layer_specs = slot.kernel.as_ref()?.get_pi_layer_specs()?;
        let input_dtype = self.runtime.args.obs_dtype.clone();
        let output_dtype = self.runtime.args.act_dtype.clone();
        crate::algorithms::acquire_model_module::<B>(
            "ppo_pi",
            layer_specs,
            input_dtype,
            output_dtype,
            vec![1, self.runtime.args.obs_dim],
            vec![1, self.runtime.args.act_dim],
            None,
        )
    }

    ///  Output shape is [batch, 1].
    pub fn acquire_vf_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        let slot = self.runtime.components.agent_slots.first()?;
        let layer_specs = slot.kernel.as_ref()?.get_vf_layer_specs()?;
        let input_dtype = self.runtime.args.obs_dtype.clone();
        crate::algorithms::acquire_model_module::<B>(
            "ppo_vf",
            layer_specs,
            input_dtype,
            Self::backend_f32_dtype(),
            vec![1, self.runtime.args.obs_dim],
            vec![1, 1],
            None,
        )
    }
}

fn run_ppo_sgd_flat<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
>(
    mut kernel: PPOKernel<B, KindIn, KindOut, Pi>,
    batch: PPOBatch,
    clip_ratio: f32,
    ent_coef: f32,
    target_kl: f32,
    train_iters: u64,
    mb_size_opt: Option<usize>,
) -> SlotTrainResult<B, KindIn, KindOut, Pi> {
    let n = batch.act.len();
    let obs_dim = batch.obs_dim;

    if n == 0 || obs_dim == 0 {
        return SlotTrainResult {
            kernel,
            pi_loss: 0.0,
            delta_pi_loss: 0.0,
            vf_loss: 0.0,
            delta_vf_loss: 0.0,
            kl: 0.0,
            entropy: 0.0,
            clipfrac: 0.0,
            stop_iter: 0.0,
        };
    }

    let mb_size = mb_size_opt.unwrap_or(n).clamp(1, n);
    let full_batch = mb_size >= n;

    // Persistent return normalization (SF-aligned): update running stats on full batch,
    // then use normalized returns for all mini-batch iterations this epoch.
    let ret_normalized = kernel.normalize_persistent_returns(&batch.ret);

    let mut first_pi_loss: Option<f32> = None;
    let mut first_vf_loss: Option<f32> = None;
    let mut final_pi_loss = 0.0f32;
    let mut final_vf_loss = 0.0f32;
    let mut final_kl = 0.0f32;
    let mut final_entropy = 0.0f32;
    let mut final_clipfrac = 0.0f32;
    let mut stop_iter = 0u64;

    'outer: for i in 0..train_iters {
        // Disabled shuffling to match SF's shuffle_minibatches=False
        // Keeping sequential order preserves value-bootstrap correlation within trajectories
        // idx.shuffle(&mut rng);
        let mut epoch_pi_loss = 0.0f32;
        let mut epoch_vf_loss = 0.0f32;
        let mut epoch_kl = 0.0f32;
        let mut epoch_entropy = 0.0f32;
        let mut epoch_clipfrac = 0.0f32;
        let mut mb_count = 0usize;
        let mut early_stop = false;
        let is_last_mb = i == train_iters - 1;

        for start in (0..n).step_by(mb_size) {
            let end = (start + mb_size).min(n);
            // Use sequential indices instead of shuffled
            let mb: Vec<usize> = (start..end).collect();
            let compute_stats = is_last_mb || mb_count == 0;
            let (pi_loss, vf_loss, info) = if full_batch {
                kernel.train_step(
                    &batch.obs,
                    obs_dim,
                    &batch.act,
                    &batch.adv_norm,
                    &batch.logp,
                    &ret_normalized,
                    clip_ratio,
                    ent_coef,
                    compute_stats,
                )
            } else {
                let obs_mb: Vec<TensorData> = mb.iter().map(|&j| batch.obs[j].clone()).collect();
                let act_mb: Vec<TensorData> = mb.iter().map(|&j| batch.act[j].clone()).collect();
                let adv_mb: Vec<f32> = mb.iter().map(|&j| batch.adv_norm[j]).collect();
                let logp_mb: Vec<f32> = mb.iter().map(|&j| batch.logp[j]).collect();
                let ret_mb: Vec<f32> = mb.iter().map(|&j| ret_normalized[j]).collect();
                kernel.train_step(
                    &obs_mb,
                    obs_dim,
                    &act_mb,
                    &adv_mb,
                    &logp_mb,
                    &ret_mb,
                    clip_ratio,
                    ent_coef,
                    compute_stats,
                )
            };
            epoch_pi_loss += pi_loss;
            epoch_vf_loss += vf_loss;
            let mb_kl = info.get("kl").copied().unwrap_or(0.0);
            epoch_kl = mb_kl;
            epoch_entropy = info.get("entropy").copied().unwrap_or(epoch_entropy);
            epoch_clipfrac = info.get("clipfrac").copied().unwrap_or(epoch_clipfrac);
            mb_count += 1;
            if mb_kl > 1.5 * target_kl {
                early_stop = true;
                break;
            }
        }
        if mb_count > 0 {
            epoch_pi_loss /= mb_count as f32;
            epoch_vf_loss /= mb_count as f32;
        }
        first_pi_loss.get_or_insert(epoch_pi_loss);
        first_vf_loss.get_or_insert(epoch_vf_loss);
        final_pi_loss = epoch_pi_loss;
        final_vf_loss = epoch_vf_loss;
        final_kl = epoch_kl;
        final_entropy = epoch_entropy;
        final_clipfrac = epoch_clipfrac;
        stop_iter = i + 1;
        if early_stop || final_kl > 1.5 * target_kl {
            break 'outer;
        }
    }

    SlotTrainResult {
        kernel,
        pi_loss: final_pi_loss,
        delta_pi_loss: final_pi_loss - first_pi_loss.unwrap_or(final_pi_loss),
        vf_loss: final_vf_loss,
        delta_vf_loss: final_vf_loss - first_vf_loss.unwrap_or(final_vf_loss),
        kl: final_kl,
        entropy: final_entropy,
        clipfrac: final_clipfrac,
        stop_iter: stop_iter as f32,
    }
}

impl<B, KindIn, KindOut, Pi, T> AlgorithmTrait<T>
    for IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
    T: TrajectoryData,
{
    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        let mut extracted_traj: RelayRLTrajectory = trajectory.into_relayrl().ok_or_else(|| {
            AlgorithmError::TrajectoryInsertionError("Missing RelayRL trajectory".to_string())
        })?;

        let agent_key = resolve_agent_key(&extracted_traj);
        let agent_index = self.register_agent_slot(agent_key)?;
        let slot = &mut self.runtime.components.agent_slots[agent_index];

        if slot.replay_buffer.is_full() {
            return Ok(false);
        }

        slot.trajectory_count += 1;

        // IndependentPPO runs without distributed actors (no flag_last_action path), so
        // actor-side policy_version stamping never fires. Stamp here with model_version,
        // which increments once per completed training epoch in apply_epoch_result.
        // Episodes received during epoch N's training have model_version=N; at epoch N+1
        // drain (current_version=N+1), lag = 1 ≤ max_version_lag → always fresh.
        extracted_traj.policy_version = self.runtime.components.model_version;

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
            self.reset_agent_counts();
            return Ok(true);
        }

        Ok(false)
    }

    fn train_model(&mut self) {
        // Training runs asynchronously via start_epoch_training; this stub satisfies the trait.
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

    fn save_model(&self, _filename: &str) {}

    fn acquire_model<B2: Backend + BackendMatcher<Backend = B2> + 'static>(
        &self,
    ) -> Option<relayrl_types::model::ModelModule<B2>>
    where
        B: 'static,
    {
        use std::any::TypeId;

        // Return None if B and B2 don't match
        if TypeId::of::<B>() != TypeId::of::<B2>() {
            return None;
        }

        // acquire_pi_module returns ModelModule<B> with the current pi weights
        let module_b = self.acquire_pi_module()?;

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
