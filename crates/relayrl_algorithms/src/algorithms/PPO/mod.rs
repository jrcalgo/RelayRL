use burn_tensor::backend::Backend;
use burn_tensor::TensorKind;
use crate::templates::base_algorithm::StepKernelTrait;

use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use std::path::PathBuf;
use std::marker::PhantomData;

#[allow(dead_code)]
pub struct PPOParams {
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

impl Default for PPOParams {
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

struct RuntimeComponents<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> {
    epoch_logger: EpochLogger,
    trajectory_count: u64,
    epoch_count: u64,
    #[allow(dead_code)]
    kernel: KN,
    replay_buffer: PPOReplayBuffer,
    _phantom: PhantomData<(B, InK, OutK)>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> Default for RuntimeComponents<B, InK, OutK, KN> {
    fn default() -> Self {
        Self {
            epoch_logger: EpochLogger::new(),
            trajectory_count: 0,
            epoch_count: 0,
            kernel: Default::default(),
            replay_buffer: Default::default(),
            _phantom: PhantomData,
        }
    }
}

struct RuntimeParams<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> {
    #[allow(dead_code)]
    args: RuntimeArgs,
    components: RuntimeComponents<B, InK, OutK, KN>
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> Default for RuntimeParams<B, InK, OutK, KN> {
    fn default() -> Self {
        Self {
            args: Default::default(),
            components: Default::default(),
        }
    }
}
pub struct PPOAlgorithm<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> {
    runtime: RuntimeParams<B, InK, OutK, KN>,
    hyperparams: PPOParams,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>> Default for ReinforceAlgorithm<B, InK, OutK, KN> {
    fn default() -> Self {
        Self {
            runtime: Default::default(),
            hyperparams: Default::default(),
        }
    }
}