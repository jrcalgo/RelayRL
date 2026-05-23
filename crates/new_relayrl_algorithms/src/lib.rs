pub mod algorithms;
pub mod logging;
pub mod templates;

use relayrl_types::data::tensor::DType;
use relayrl_types::data::tensor::DeviceType;
use std::path::PathBuf;

pub use templates::base_algorithm::{
    AlgorithmError, AlgorithmTrait, MultiagentKernelTrait, TrajectoryData, WeightProvider,
};

pub mod prelude {
    pub mod ppo {
        pub mod algorithm {
            pub use crate::algorithms::PPO::{
                IPPOParams, IndependentPPOAlgorithm, MAPPOParams, MultiAgentPPOAlgorithm, PPOParams,
            };
        }
        pub mod trainer {
            pub use crate::algorithms::PPO::{
                PPONetworkArgs, PPONetworkBuilder, PPOTrainer, PPOTrainerSpec,
            };
        }
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
