pub mod kernel;
pub mod replay_buffer;

pub use kernel::*;
pub use replay_buffer::*;

use crate::templates::base_replay_buffer::{Batch, BatchKey, GenericReplayBuffer};
use crate::templates::base_algorithm::{AlgorithmError, AlgorithmTrait, StepKernelTrait, TrajectoryData};
use crate::logging::{EpochLogger, SessionLogger};

use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use burn_tensor::backend::Backend;
use burn_tensor::TensorKind;
use crate::templates::base_replay_buffer::ReplayBufferError;
use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
pub struct REINFORCEParams {
    discrete: bool,
    with_vf_baseline: bool,
    gamma: f32,
    lambda: f32,
    traj_per_epoch: u64,
    seed: u64,
    pi_lr: f32,
    vf_lr: f32,
    train_vf_iters: u64,
}

impl Default for REINFORCEParams {
    fn default() -> Self {
        Self {
            discrete: true,
            with_vf_baseline: false,
            gamma: 0.98,
            lambda: 0.97,
            traj_per_epoch: 8,
            seed: 1,
            pi_lr: 3e-4,
            vf_lr: 1e-3,
            train_vf_iters: 80,
        }
    }
}

#[allow(dead_code)]
struct RuntimeArgs {
    env_dir: PathBuf,
    save_model_path: PathBuf,
    obs_dim: usize,
    act_dim: usize,
    buffer_size: usize
}

struct RuntimeComponents<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> {
    epoch_logger: EpochLogger,
    trajectory_count: u64,
    epoch_count: u64,
    #[allow(dead_code)]
    kernel: KN,
    replay_buffer: ReinforceReplayBuffer,
    _phantom: PhantomData<(B, InK, OutK)>,
}

struct RuntimeParams<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> {
    #[allow(dead_code)]
    args: RuntimeArgs,
    components: RuntimeComponents<B, InK, OutK, KN>
}

pub struct ReinforceAlgorithm<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> {
    runtime: RuntimeParams<B, InK, OutK, KN>,
    hyperparams: REINFORCEParams,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> ReinforceAlgorithm<B, InK, OutK, KN> {
    #[allow(dead_code)]
    pub(crate) fn new(
        hyperparams: Option<REINFORCEParams>,
        env_dir: &Path,
        save_model_path: &Path,
        obs_dim: usize,
        act_dim: usize,
        buffer_size: usize,
        kernel: KN,
    ) -> Result<Self, AlgorithmError> {
        let hyperparams = hyperparams.unwrap_or_default();

        let trajectory_count: u64 = 0;
        let epoch_count: u64 = 0;

        let replay_buffer = ReinforceReplayBuffer::new(buffer_size, hyperparams.gamma, hyperparams.lambda, hyperparams.with_vf_baseline);

        let epoch_logger = EpochLogger::new();

        let algorithm = ReinforceAlgorithm {
            runtime: RuntimeParams::<B, InK, OutK, KN> {
                args: RuntimeArgs {
                    env_dir: env_dir.to_path_buf(),
                    save_model_path: save_model_path.to_path_buf(),
                    obs_dim,
                    act_dim,
                    buffer_size
                },
                components: RuntimeComponents::<B, InK, OutK, KN> {
                    epoch_logger,
                    trajectory_count,
                    epoch_count,
                    kernel,
                    replay_buffer,
                    _phantom: PhantomData,
                }
            },
            hyperparams
        };

        let session_logger = SessionLogger::new();
        session_logger
            .log_session(&algorithm)
            .map_err(|e| AlgorithmError::BufferSamplingError(e.to_string()))?;

        Ok(algorithm)
    }

    fn compute_policy_loss(&self, batch: &Batch) -> (f32, HashMap<String, f32>) {
        // Placeholder scalar path until policy forward-loss wiring is finalized in kernel API.
        let _obs = batch.get(&BatchKey::Obs);
        let _act = batch.get(&BatchKey::Act);
        let _mask = batch.get(&BatchKey::Mask);
        let _adv = batch.get(&BatchKey::Custom("Adv".to_string()));
        let _old_logp = batch.get(&BatchKey::Custom("LogP".to_string()));

        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        (0.0, info)
    }

    fn compute_value_loss(&self, batch: &Batch) -> f32 {
        let _obs = batch.get(&BatchKey::Obs);
        let _mask = batch.get(&BatchKey::Mask);
        let _ret = batch.get(&BatchKey::Custom("Ret".to_string()));
        0.0
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>, T: TrajectoryData> AlgorithmTrait<T> for ReinforceAlgorithm<B, InK, OutK, KN> {
    fn save(&self, _filename: &str) {}

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        self.runtime.components.trajectory_count += 1;

        let extracted_traj: RelayRLTrajectory = trajectory
            .into_relayrl()
            .ok_or_else(|| AlgorithmError::TrajectoryInsertionError("Missing RelayRL trajectory".to_string()))?;

        let result: Box<dyn Any> = self
            .runtime
            .components
            .replay_buffer
            .insert_trajectory(extracted_traj)
            .await
            .map_err(|e: ReplayBufferError| AlgorithmError::TrajectoryInsertionError(format!("{e}")))?;
        let (episode_return, episode_length) = match result.downcast::<(f32, i32)>() {
            Ok(v) => *v,
            Err(_) => {
                return Err(AlgorithmError::TrajectoryInsertionError(
                    "Unexpected replay buffer return payload".to_string(),
                ))
            }
        };

        self.runtime.components.epoch_logger.store("EpRet", episode_return);
        self.runtime.components.epoch_logger.store("EpLen", episode_length as f32);

        if self.runtime.components.trajectory_count > 0
            && self
                .runtime
                .components
                .trajectory_count
                .is_multiple_of(self.hyperparams.traj_per_epoch)
        {
            self.runtime.components.epoch_count += 1;
            <Self as AlgorithmTrait<T>>::train_model(self);
            <Self as AlgorithmTrait<T>>::log_epoch(self);
            return Ok(true);
        }

        Ok(false)
    }

    fn train_model(&mut self) {
        let batch: Batch = match tokio::runtime::Handle::current()
            .block_on(self.runtime.components.replay_buffer.sample_buffer())
        {
            Ok(b) => b,
            Err(_) => return,
        };

        let (old_policy_loss, _) = self.compute_policy_loss(&batch);

        let old_value_loss = if self.hyperparams.with_vf_baseline {
            Some(self.compute_value_loss(&batch))
        } else {
            None
        };

        let (policy_loss, policy_info) = self.compute_policy_loss(&batch);

        let value_loss = if self.hyperparams.with_vf_baseline {
            let mut loss = 0.0f32;
            for _ in 0..self.hyperparams.train_vf_iters {
                loss = self.compute_value_loss(&batch);
            }
            Some(loss)
        } else {
            None
        };

        let kl_divergence = *policy_info.get("kl").unwrap_or(&0.0);
        let entropy = *policy_info.get("entropy").unwrap_or(&0.0);

        let policy_loss_delta = policy_loss - old_policy_loss;
        let value_loss_delta = if self.hyperparams.with_vf_baseline {
            value_loss.unwrap_or(0.0) - old_value_loss.unwrap_or(0.0)
        } else {
            0.0
        };

        self.runtime.components.epoch_logger.store("LossPi", policy_loss);
        self.runtime.components.epoch_logger.store("DeltaLossPi", policy_loss_delta);
        self.runtime.components.epoch_logger.store("KL", kl_divergence);
        self.runtime.components.epoch_logger.store("Entropy", entropy);
        if self.hyperparams.with_vf_baseline {
            self.runtime.components.epoch_logger.store("LossV", value_loss.unwrap_or(0.0));
            self.runtime.components.epoch_logger.store("DeltaLossV", value_loss_delta);
        }
    }

    fn log_epoch(&mut self) {
        self.runtime
            .components
            .epoch_logger
            .log_tabular("Epoch", Some(self.runtime.components.epoch_count as f32));
        self.runtime.components.epoch_logger.log_tabular("EpRet", None);
        self.runtime.components.epoch_logger.log_tabular("EpLen", None);
        self.runtime.components.epoch_logger.log_tabular("LossPi", None);
        self.runtime
            .components
            .epoch_logger
            .log_tabular("DeltaLossPi", None);
        if self.hyperparams.with_vf_baseline {
            self.runtime.components.epoch_logger.log_tabular("VVals", None);
            self.runtime.components.epoch_logger.log_tabular("LossV", None);
            self.runtime
                .components
                .epoch_logger
                .log_tabular("DeltaLossV", None);
        }
        self.runtime.components.epoch_logger.log_tabular("KL", None);
        self.runtime.components.epoch_logger.log_tabular("Entropy", None);
        self.runtime.components.epoch_logger.dump_tabular();
    }
}
