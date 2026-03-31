// CUSTOM algorithm skeleton — allows users to plug in arbitrary implementations
// via function maps or a full AlgorithmTrait impl.
// This module is not yet registered in algorithms/mod.rs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::any::Any;

use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use burn_tensor::backend::Backend;
use burn_tensor::TensorKind;
use crate::templates::base_algorithm::{AlgorithmTrait, StepKernelTrait, TrajectoryData};

pub struct CustomParams {
    pub hyperparameters: HashMap<String, String>,
}

impl Default for CustomParams {
    fn default() -> Self {
        Self {
            hyperparameters: HashMap::new(),
        }
    }
}

struct CustomRuntimeArgs {
    env_dir: PathBuf,
    save_model_path: PathBuf,
    obs_dim: usize,
    act_dim: usize,
    buffer_size: usize,
}

impl Default for CustomRuntimeArgs {
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

struct CustomOps<T: TrajectoryData> {
    fn_map: Option<HashMap<String, Box<dyn Any + Send + Sync>>>,
    algorithm_impl: Option<Box<dyn AlgorithmTrait<T> + Send>>,
}

impl<T: TrajectoryData> CustomOps<T> {
    pub fn init() -> Self {
        Self {
            fn_map: None,
            algorithm_impl: None,
        }
    }
}

pub struct CustomAlgorithm<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK>, T: TrajectoryData> {
    #[allow(dead_code)]
    args: Arc<CustomRuntimeArgs>,
    #[allow(dead_code)]
    kernel: KN,
    #[allow(dead_code)]
    ops: CustomOps<T>,
    pub hyperparams: CustomParams,
    _phantom: std::marker::PhantomData<(B, InK, OutK)>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, KN: StepKernelTrait<B, InK, OutK> + Default, T: TrajectoryData> CustomAlgorithm<B, InK, OutK, KN, T> {
    pub fn new(
        hyperparams: CustomParams,
        env_dir: &std::path::Path,
        save_model_path: &std::path::Path,
        obs_dim: usize,
        act_dim: usize,
        buffer_size: usize,
        kernel: KN,
    ) -> Self {
        Self {
            args: Arc::new(CustomRuntimeArgs {
                env_dir: env_dir.to_path_buf(),
                save_model_path: save_model_path.to_path_buf(),
                obs_dim,
                act_dim,
                buffer_size,
            }),
            kernel,
            ops: CustomOps::init(),
            hyperparams,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_algorithm_impl(mut self, implementation: impl AlgorithmTrait<T> + Send + 'static) -> Self {
        self.ops.algorithm_impl = Some(Box::new(implementation));
        self
    }
}
