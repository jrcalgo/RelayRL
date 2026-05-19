pub mod algorithms;
pub mod logging;
pub mod templates;

use crate::algorithms::NeuralNetworkSpec;
use crate::algorithms::ValueFunction;
use burn_tensor::TensorKind;
use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use std::path::PathBuf;

pub use templates::base_algorithm::{
    AlgorithmError, AlgorithmTrait, MultiagentKernelTrait, StepKernelTrait, TrajectoryData,
    WeightProvider,
};

pub use algorithms::PPO::{
    IPPOAlgorithm, IPPOParams, MAPPOAlgorithm, MAPPOParams, PPOAlgorithm, PPOParams,
};

#[derive(Clone, Debug)]
pub struct TrainerArgs {
    pub env_dir: PathBuf,
    pub save_model_path: PathBuf,
    pub obs_dim: usize,
    pub act_dim: usize,
    pub buffer_size: usize,
}

// ---- PPO-related inference & algorithm interfaces ----

pub struct PpoNetworkArgs<B, KindIn, KindOut>
where
    B: Backend + BackendMatcher + Default + ?Sized,
    KindIn: TensorKind<B> + Default + ?Sized,
    KindOut: TensorKind<B> + Default + ?Sized,
{
    pi_network: Option<Box<dyn NeuralNetworkSpec<B, KindIn, KindOut>>>,
    vf_network: Option<ValueFunction<B, KindIn>>,
}

impl<B, KindIn, KindOut> Default for PpoNetworkArgs<B, KindIn, KindOut>
where
    B: Backend + BackendMatcher,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
{
    fn default() -> Self {
        Self {
            pi_network: None,
            vf_network: None,
        }
    }
}

pub enum PpoTrainerSpec<B, KindIn, KindOut>
where
    B: Backend + BackendMatcher,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
{
    PPO {
        args: TrainerArgs,
        hyperparams: Option<PPOParams>,
        networks: PpoNetworkArgs<B, KindIn, KindOut>,
    },
    IPPO {
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        networks: PpoNetworkArgs<B, KindIn, KindOut>,
    },
    MAPPO {
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
        networks: PpoNetworkArgs<B, KindIn, KindOut>,
    },
}

impl<B, KindIn, KindOut> PpoTrainerSpec<B, KindIn, KindOut>
where
    B: Backend + BackendMatcher,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
{
    pub fn ppo(
        args: TrainerArgs,
        hyperparams: Option<PPOParams>,
        networks: PpoNetworkArgs<B, KindIn, KindOut>,
    ) -> Self {
        Self::PPO {
            args,
            hyperparams,
            networks,
        }
    }

    pub fn ippo(
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        networks: PpoNetworkArgs<B, KindIn, KindOut>,
    ) -> Self {
        Self::IPPO {
            args,
            hyperparams,
            networks,
        }
    }

    pub fn mappo(
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
        networks: PpoNetworkArgs<B, KindIn, KindOut>,
    ) -> Self {
        Self::MAPPO {
            args,
            hyperparams,
            networks,
        }
    }
}

pub enum PpoTrainer<B, KindIn, KindOut>
where
    B: Backend + BackendMatcher,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
{
    PPO(PPOAlgorithm<B, KindIn, KindOut>),
    IPPO(IPPOAlgorithm<B, KindIn, KindOut>),
    MAPPO(MAPPOAlgorithm<B, KindIn, KindOut>),
}

impl<B, KindIn, KindOut> PpoTrainer<B, KindIn, KindOut>
where
    B: Backend + BackendMatcher,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
{
    pub fn new(spec: PpoTrainerSpec<B, KindIn, KindOut>) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            PpoTrainerSpec::PPO {
                args,
                hyperparams,
                networks,
            } => Self::PPO(PPOAlgorithm::new(
                hyperparams,
                &args.env_dir,
                &args.save_model_path,
                &args.obs_dim,
                &args.act_dim,
                &args.buffer_size,
                networks.pi_network,
                networks.vf_network,
            )),
            PpoTrainerSpec::IPPO {
                args,
                hyperparams,
                networks,
            } => Self::IPPO(IPPOAlgorithm::new(
                hyperparams,
                &args.env_dir,
                &args.save_model_path,
                &args.obs_dim,
                &args.act_dim,
                &args.buffer_size,
                networks.pi_network,
                networks.vf_network,
            )),
            PpoTrainerSpec::MAPPO {
                args,
                hyperparams,
                networks,
            } => Self::MAPPO(MAPPOAlgorithm::new(
                hyperparams,
                &args.env_dir,
                &args.save_model_path,
                &args.obs_dim,
                &args.act_dim,
                &args.buffer_size,
                networks.pi_network,
                networks.vf_network,
            )),
        };

        Ok(trainer)
    }
}
