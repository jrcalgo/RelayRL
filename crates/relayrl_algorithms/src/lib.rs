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
