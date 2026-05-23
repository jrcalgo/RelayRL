pub mod kernel;
pub mod replay_buffer;

pub mod independent;
pub mod multiagent;

pub use independent::{
    EpochTrainOutput, IPPOParams, IndependentPPOAlgorithm, PPOParams, SlotTrainResult,
};
pub use multiagent::{MAPPOParams, MultiAgentPPOAlgorithm};

use crate::TrainerArgs;

use crate::algorithms::PPO::kernel::PPOPolicyHead;
use crate::algorithms::{GenericMlp, NeuralNetwork, NeuralNetworkSpec};

use crate::templates::base_algorithm::AlgorithmError;

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, TensorKind};
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::data::tensor::{DType, NdArrayDType, SupportedTensorBackend};
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

// ---- PPO-related inference & algorithm interfaces ----

pub struct PPONetworkBuilder<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    pi_head: Option<PPOPolicyHead<B, KindIn, KindOut, Pi>>,
    vf_mlp: Option<GenericMlp<B, KindIn, Float>>,
}

impl<B, KindIn, KindOut, Pi> Default for PPONetworkBuilder<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    fn default() -> Self {
        Self {
            pi_head: None,
            vf_mlp: None,
        }
    }
}

pub struct PPONetworkArgs<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    pub pi_head: PPOPolicyHead<B, KindIn, KindOut, Pi>,
    pub vf_mlp: GenericMlp<B, KindIn, Float>,
}

pub enum PPOTrainerSpec<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    PPO {
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        networks: PPONetworkArgs<B, KindIn, KindOut, Pi>,
    },
    IPPO {
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        networks: PPONetworkArgs<B, KindIn, KindOut, Pi>,
    },
    MAPPO {
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
        networks: PPONetworkArgs<B, KindIn, KindOut, Pi>,
    },
}

impl<B, KindIn, KindOut, Pi> PPOTrainerSpec<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    pub fn ppo(
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        networks: PPONetworkArgs<B, KindIn, KindOut, Pi>,
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
        networks: PPONetworkArgs<B, KindIn, KindOut, Pi>,
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
        networks: PPONetworkArgs<B, KindIn, KindOut, Pi>,
    ) -> Self {
        Self::MAPPO {
            args,
            hyperparams,
            networks,
        }
    }
}

pub enum PPOTrainer<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    PPO(IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>),
    IPPO(IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>),
    MAPPO(MultiAgentPPOAlgorithm<B, KindIn, KindOut, Pi>),
}

impl<B, KindIn, KindOut, Pi> PPOTrainer<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    pub fn new(spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            PPOTrainerSpec::PPO {
                args,
                hyperparams,
                networks,
            } => {
                validate_ppo_spec(&args, &networks)?;
                Self::PPO(IndependentPPOAlgorithm::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    &args.obs_dim,
                    &args.obs_dtype,
                    &args.act_dim,
                    &args.act_dtype,
                    &args.buffer_size,
                    networks.pi_head,
                    networks.vf_mlp,
                )?)
            }
            PPOTrainerSpec::IPPO {
                args,
                hyperparams,
                networks,
            } => {
                validate_ppo_spec(&args, &networks)?;
                Self::IPPO(IndependentPPOAlgorithm::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    &args.obs_dim,
                    &args.obs_dtype,
                    &args.act_dim,
                    &args.act_dtype,
                    &args.buffer_size,
                    networks.pi_head,
                    networks.vf_mlp,
                )?)
            }
            PPOTrainerSpec::MAPPO {
                args,
                hyperparams,
                networks,
            } => {
                validate_ppo_spec(&args, &networks)?;
                Self::MAPPO(MultiAgentPPOAlgorithm::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    &args.obs_dim,
                    &args.obs_dtype,
                    &args.act_dim,
                    &args.act_dtype,
                    &args.buffer_size,
                    networks.pi_head,
                    networks.vf_mlp,
                )?)
            }
        };

        Ok(trainer)
    }
}

fn validate_ppo_spec<
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
>(
    args: &TrainerArgs,
    networks: &PPONetworkArgs<B, KindIn, KindOut, Pi>,
) -> Result<(), AlgorithmError> {
    let pi_head = &networks.pi_head;
    let vf_mlp = &networks.vf_mlp;

    match pi_head {
        PPOPolicyHead::Discrete(pi) => {
            if *pi.pi.input_dim() != args.obs_dim
                || *pi.pi.output_dim() != args.act_dim
                || *pi.pi.input_dtype() != args.obs_dtype
                || *pi.pi.output_dtype() != args.act_dtype
            {
                return Err(AlgorithmError::InvalidSpec("PPO policy head input/output dimensions or dtypes do not match the trainer arguments".to_string()));
            }

            match B::get_supported_backend() {
                SupportedTensorBackend::NdArray => {
                    match *pi.pi.input_dtype() {
                        DType::NdArray(_) => {}
                        _ => {
                            return Err(AlgorithmError::InvalidSpec(
                                "PPO policy head input dtype does not match the trainer arguments"
                                    .to_string(),
                            ));
                        }
                    }
                    match *pi.pi.output_dtype() {
                        DType::NdArray(_) => {}
                        _ => {
                            return Err(AlgorithmError::InvalidSpec(
                                "PPO policy head output dtype does not match the trainer arguments"
                                    .to_string(),
                            ));
                        }
                    }
                }
                #[cfg(feature = "tch-backend")]
                SupportedTensorBackend::Tch => {
                    match *pi.pi.input_dtype() {
                        DType::Tch(_) => {}
                        _ => {
                            return Err(AlgorithmError::InvalidSpec(
                                "PPO policy head input dtype does not match the trainer arguments"
                                    .to_string(),
                            ));
                        }
                    }
                    match *pi.pi.output_dtype() {
                        DType::Tch(_) => {}
                        _ => {
                            return Err(AlgorithmError::InvalidSpec(
                                "PPO policy head output dtype does not match the trainer arguments"
                                    .to_string(),
                            ));
                        }
                    }
                }
                _ => {
                    return Err(AlgorithmError::InvalidSpec(
                        "Unsupported backend".to_string(),
                    ));
                }
            }
        }
        PPOPolicyHead::Continuous(pi) => {
            if *pi.pi.input_dim() != args.obs_dim
                || *pi.pi.output_dim() != args.act_dim
                || *pi.pi.input_dtype() != args.obs_dtype
                || *pi.pi.output_dtype() != args.act_dtype
            {
                return Err(AlgorithmError::InvalidSpec("PPO policy head input/output dimensions or dtypes do not match the trainer arguments".to_string()));
            }
        }
    }

    if *vf_mlp.input_dim() != args.obs_dim
        || *vf_mlp.output_dim() != 1
        || *vf_mlp.input_dtype() != args.obs_dtype
    {
        return Err(AlgorithmError::InvalidSpec("PPO value function MLP input/output dimensions or input dtype do not match the trainer arguments".to_string()));
    }

    match B::get_supported_backend() {
        SupportedTensorBackend::NdArray => {
            match *vf_mlp.input_dtype() {
                DType::NdArray(_) => {}
                _ => {
                    return Err(AlgorithmError::InvalidSpec(
                        "PPO value function MLP input dtype does not match the trainer arguments"
                            .to_string(),
                    ));
                }
            }
            match *vf_mlp.output_dtype() {
                DType::NdArray(NdArrayDType::F32) => {}
                _ => {
                    return Err(AlgorithmError::InvalidSpec(
                        "PPO value function MLP output dtype is not f32".to_string(),
                    ));
                }
            }
        }
        #[cfg(feature = "tch-backend")]
        SupportedTensorBackend::Tch => {
            match *vf_mlp.input_dtype() {
                DType::Tch(_) => {}
                _ => {
                    return Err(AlgorithmError::InvalidSpec(
                        "PPO value function MLP input dtype does not match the trainer arguments"
                            .to_string(),
                    ));
                }
            }
            match *vf_mlp.output_dtype() {
                DType::Tch(TchDType::F32) => {}
                _ => {
                    return Err(AlgorithmError::InvalidSpec(
                        "PPO value function MLP output dtype is not f32".to_string(),
                    ));
                }
            }
        }
        _ => {
            return Err(AlgorithmError::InvalidSpec(
                "Unsupported backend".to_string(),
            ));
        }
    }

    Ok(())
}
