//! # RelayRL Algorithms
//!
//! Reinforcement learning algorithm implementations for the RelayRL stack. This crate sits
//! between [`relayrl_types`] (data model and tensors) and `relayrl_framework` (the runtime
//! that drives actors and collects trajectories): it owns the *learning* logic — policy and
//! value networks, rollout buffering, and the PPO family of trainers.
//!
//! All tensor math goes through [Burn](https://burn.dev), so algorithms are generic over a
//! backend `B` and run on either the CPU `ndarray` backend (default) or the `tch`
//! (LibTorch/GPU) backend via feature flags.
//!
//! ## System layout
//!
//! - [`algorithms`]: the algorithm implementations and neural-network building blocks.
//!   - [`algorithms::PPO`]: the PPO family. [`PPOTrainerSpec`](algorithms::PPO::PPOTrainerSpec)
//!     selects a variant (`PPO`, `IPPO`, or `MAPPO`) and bundles
//!     [`PPONetworkArgs`](algorithms::PPO::PPONetworkArgs) with [`TrainerArgs`];
//!     [`PPOTrainer`](algorithms::PPO::PPOTrainer) is the constructed, runnable trainer.
//!     [`IndependentPPOAlgorithm`](algorithms::PPO::IndependentPPOAlgorithm) backs single-agent
//!     PPO and independent multi-agent PPO (IPPO), while
//!     [`MultiAgentPPOAlgorithm`](algorithms::PPO::MultiAgentPPOAlgorithm) backs centralized
//!     MAPPO. Hyperparameters live in [`IPPOParams`](algorithms::PPO::IPPOParams) (aliased as
//!     [`PPOParams`](algorithms::PPO::PPOParams)) and
//!     [`MAPPOParams`](algorithms::PPO::MAPPOParams).
//!   - The PPO kernel (`algorithms::PPO::kernel`) holds the policy/value heads
//!     (`PPOPolicyHead`, discrete and continuous) and the inner training step.
//!   - `algorithms::PPO::replay_buffer`: the per-agent PPO rollout buffer.
//!   - Network primitives: [`GenericMlp`](algorithms::GenericMlp),
//!     [`ValueFunction`](algorithms::ValueFunction), the
//!     [`NeuralNetwork`](algorithms::NeuralNetwork) trait family, and a convolutional policy
//!     in `algorithms::nn::conv_policy`.
//!   - Model export helpers: `algorithms::onnx_builder` and (under `tch-model`)
//!     `algorithms::torch_builder`.
//! - [`templates`]: backend-agnostic contracts —
//!   [`AlgorithmTrait`](templates::base_algorithm::AlgorithmTrait) plus the replay-buffer
//!   traits — that new algorithms implement.
//! - [`logging`]: lightweight epoch/session loggers (`EpochLogger`, `SessionLogger`) used to
//!   surface training metrics.
//! - [`prelude`]: grouped re-exports (`ppo::algorithm`, `ppo::trainer`, `nn`, `templates`).
//! - [`TrainerArgs`]: the shared, backend-independent trainer configuration (directories,
//!   dims, dtypes, buffer size, device) consumed by `PPOTrainerSpec::default`.
//!
//! ## Design notes
//!
//! - **Generic over a Burn backend.** Public algorithm types carry the parameters
//!   `<B, KindIn, KindOut, Pi>`: the backend `B`, the input/output tensor kinds, and the
//!   policy network type `Pi`. Pick the backend (e.g. `burn_ndarray::NdArray`) and the
//!   tensor kinds (e.g. `burn_tensor::Float`) at the call site.
//! - **Spec-then-build.** Construction is a two-step flow: assemble a
//!   [`PPOTrainerSpec`](algorithms::PPO::PPOTrainerSpec) (often via its `default`
//!   constructor, which builds the networks for you), then hand it to
//!   [`PPOTrainer::new`](algorithms::PPO::PPOTrainer) to validate and instantiate.
//! - **Single-agent and multi-agent share machinery.** IPPO and single-agent PPO are the same
//!   implementation with one vs. many agent slots; MAPPO reuses the same spec shape via the
//!   [`MAPPOTrainerSpec`](algorithms::PPO::MAPPOTrainerSpec) alias.
//!
//! ## Feature flags
//!
//! - `tch-backend`: enable the `tch` (LibTorch) Burn backend.
//! - `tch-model`: enable LibTorch model export/import (`torch_builder`).
//!
//! With no features enabled the crate uses the CPU `ndarray` backend pulled in through
//! [`relayrl_types`].
//!
//! ## Quick start
//!
//! Build a single-agent PPO trainer on the CPU backend. The example is `ignore`d because the
//! full generic signature and on-disk paths are environment-specific:
//!
//! ```ignore
//! use relayrl_algorithms::prelude::ppo::trainer::{PPOTrainer, PPOTrainerSpec};
//! use relayrl_algorithms::prelude::nn::GenericMlp;
//! use relayrl_types::prelude::tensor::relayrl::{DType, NdArrayDType, DeviceType};
//! use burn_ndarray::NdArray;
//! use burn_tensor::Float;
//! use std::path::PathBuf;
//!
//! // `default` builds the policy/value networks and wraps them in a `PPO` spec.
//! let spec = PPOTrainerSpec::<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>::default(
//!     PathBuf::from("env_dir"),
//!     PathBuf::from("model.mpk"),
//!     8,  DType::NdArray(NdArrayDType::F32),  // observation dim + dtype
//!     4,  DType::NdArray(NdArrayDType::F32),  // action dim + dtype
//!     1_000,                                   // rollout buffer size
//!     DeviceType::Cpu,
//! )?;
//!
//! let trainer = PPOTrainer::new(spec)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod algorithms;
pub mod logging;
pub mod templates;

use relayrl_types::data::tensor::DType;
use relayrl_types::data::tensor::DeviceType;
use std::path::PathBuf;

pub mod prelude {
    pub mod ppo {
        pub mod algorithm {
            pub use crate::algorithms::PPO::kernel::{
                ContinuousPPOPolicyHead, DiscretePPOPolicyHead, PPOKernel, PPOKernelFactory,
                PPOKernelOps, PPOKernelSnapshot, PPOKernelTraining, PPOKernelTrainingArgs,
                PPOPolicyHead,
            };
            pub use crate::algorithms::PPO::{
                EpochTrainOutput, IPPOParams, IndependentPPOAlgorithm, MAPPOParams,
                MultiAgentPPOAlgorithm, PPOParams,
            };
        }
        pub mod trainer {
            pub use crate::algorithms::PPO::{PPONetworkArgs, PPOTrainer, PPOTrainerSpec};
        }
    }

    pub mod nn {
        pub use crate::algorithms::{
            GenericMlp, NeuralNetwork, NeuralNetworkError, NeuralNetworkForward, NeuralNetworkSpec,
            ValueFunction, WeightProvider,
        };
    }

    pub mod templates {
        pub use crate::templates::base_algorithm::{
            AlgorithmError, AlgorithmTrait, TrajectoryData,
        };
    }
}

/// Common trainer arguments (directories, dimensions, dtypes, buffer size, device) consumed by `PPOTrainerSpec::default`.
#[derive(Clone, Debug)]
pub struct TrainerArgs {
    pub env_dir: PathBuf,
    pub save_model_path: PathBuf,
    pub obs_dim: usize,
    pub obs_dtype: DType,
    pub act_dim: usize,
    pub act_dtype: DType,
    pub buffer_size: usize,
    pub device: DeviceType,
}
