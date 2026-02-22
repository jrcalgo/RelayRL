pub mod algorithms;
pub mod templates;
pub mod logging;

use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use algorithms::{DDPGAlgorithm, PPOAlgorithm, ReinforceAlgorithm, TD3Algorithm};

use burn_tensor::backend::Backend;
use burn_tensor::TensorKind;
use templates::base_algorithm::StepKernelTrait;

pub enum RelayRLAlgorithm<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, Kn: StepKernelTrait<B, InK, OutK>> {
    DDPG(DDPGAlgorithm<B, InK, OutK, Kn>),
    PPO(PPOAlgorithm<B, InK, OutK, Kn>),
    REINFORCE(ReinforceAlgorithm<B, InK, OutK, Kn>),
    TD3(TD3Algorithm<B, InK, OutK, Kn>),
    CUSTOM(String)
}

pub struct RelayRLTrainer<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, Kn: StepKernelTrait<B, InK, OutK>>(RelayRLAlgorithm<B, InK, OutK, Kn>);

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, Kn: StepKernelTrait<B, InK, OutK>> Default for RelayRLTrainer<B, InK, OutK, Kn> {
    fn default() -> Self {
        Self(RelayRLAlgorithm::REINFORCE(ReinforceAlgorithm::default()))
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>, Kn: StepKernelTrait<B, InK, OutK>> RelayRLTrainer<B, InK, OutK, Kn> {
    pub fn new(algorithm: RelayRLAlgorithm<B, InK, OutK, Kn>) -> Self {

    }
}