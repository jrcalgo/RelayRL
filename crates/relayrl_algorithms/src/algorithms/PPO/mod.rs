pub mod kernel;
pub mod replay_buffer;

pub mod independent;
pub mod multiagent;

pub use independent::{
    EpochTrainOutput, IPPOParams, IndependentPPOAlgorithm, PPOParams, SlotTrainResult,
};
pub use multiagent::{MAPPOParams, MultiAgentPPOAlgorithm};

use crate::TrainerArgs;

use crate::algorithms::PPO::kernel::{DiscretePPOPolicyHead, PPOKernel, PPOPolicyHead};
use crate::algorithms::{GenericMlp, NeuralNetwork, NeuralNetworkError, NeuralNetworkSpec};

use crate::templates::base_algorithm::AlgorithmError;

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, TensorKind};
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::data::tensor::{DType, NdArrayDType, SupportedTensorBackend};
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, DeviceType};

use std::path::PathBuf;

/// Convenience alias: MAPPO and PPO share the same spec structure.
pub type MAPPOTrainerSpec<B, KindIn, KindOut, Pi> = PPOTrainerSpec<B, KindIn, KindOut, Pi>;

// ---- PPO-related inference & algorithm interfaces ----

#[derive(Debug)]
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

impl<B, KindIn, KindOut, Pi> PPONetworkArgs<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B> + Default,
    KindOut: TensorKind<B> + BasicOps<B> + Default,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default,
{
    pub fn default(
        obs_dim: usize,
        obs_dtype: DType,
        act_dim: usize,
        act_dtype: DType,
        device: B::Device,
    ) -> Result<Self, NeuralNetworkError> {
        Ok(Self {
            pi_head: PPOPolicyHead::Discrete(DiscretePPOPolicyHead::new(<Pi as NeuralNetwork<
                B,
                KindIn,
                KindOut,
            >>::default(
                obs_dim,
                obs_dtype.clone(),
                act_dim,
                act_dtype.clone(),
                &device,
            ))?),
            vf_mlp: GenericMlp::default(obs_dim, obs_dtype, act_dim, act_dtype, &device),
        })
    }
}

#[derive(Debug)]
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
    #[allow(clippy::too_many_arguments)]
    pub fn default(
        env_dir: PathBuf,
        save_model_path: PathBuf,
        obs_dim: usize,
        obs_dtype: DType,
        act_dim: usize,
        act_dtype: DType,
        buffer_size: usize,
        device: DeviceType,
    ) -> Result<Self, NeuralNetworkError> {
        let networks = {
            let burn_device = B::get_device(&device)
                .map_err(|e| NeuralNetworkError::UnsupportedDevice(e.to_string()))?;
            PPONetworkArgs::default(
                obs_dim,
                obs_dtype.clone(),
                act_dim,
                act_dtype.clone(),
                burn_device,
            )?
        };

        Ok(Self::PPO {
            args: TrainerArgs {
                env_dir,
                save_model_path,
                obs_dim,
                obs_dtype,
                act_dim,
                act_dtype,
                buffer_size,
                device,
            },
            hyperparams: None,
            networks,
        })
    }
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
                        #[cfg(feature = "tch-backend")]
                        _ => {
                            return Err(AlgorithmError::InvalidSpec(
                                "PPO policy head input dtype does not match the trainer arguments"
                                    .to_string(),
                            ));
                        }
                    }
                    match *pi.pi.output_dtype() {
                        DType::NdArray(_) => {}
                        #[cfg(feature = "tch-backend")]
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
                #[cfg(feature = "tch-backend")]
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

// ---- PPOTrainer delegation methods ----
// These minimize the amount of pattern matching required by the caller.

impl<B, KindIn, KindOut, Pi> PPOTrainer<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default + Send + 'static,
    KindIn: TensorKind<B> + BasicOps<B> + Default + Send + 'static,
    KindOut: TensorKind<B> + BasicOps<B> + Default + Send + 'static,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Default + Send + 'static,
{
    pub fn register_first_slot_with_key(
        &mut self,
        agent_key: String,
    ) -> Result<(), AlgorithmError> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => {
                inner
                    .register_first_slot_with_key(agent_key)
                    .map_err(|e| AlgorithmError::InitializationError(e.to_string()))?;
            }
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }

        Ok(())
    }

    pub fn start_epoch_training(
        &mut self,
    ) -> Option<tokio::task::JoinHandle<EpochTrainOutput<B, KindIn, KindOut, Pi>>> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.start_epoch_training(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    pub fn apply_epoch_result(&mut self, output: EpochTrainOutput<B, KindIn, KindOut, Pi>) {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.apply_epoch_result(output),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    pub fn acquire_pi_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.acquire_pi_module(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    pub fn acquire_vf_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.acquire_vf_module(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    pub async fn receive_trajectory(
        &mut self,
        trajectory: relayrl_types::data::trajectory::RelayRLTrajectory,
    ) -> Result<bool, AlgorithmError> {
        use crate::templates::base_algorithm::AlgorithmTrait;
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => AlgorithmTrait::<
                relayrl_types::data::trajectory::RelayRLTrajectory,
            >::receive_trajectory(
                inner, trajectory
            )
            .await,
            PPOTrainer::MAPPO(_) => Err(AlgorithmError::InvalidSpec(
                "MAPPO receive_trajectory not yet implemented".to_string(),
            )),
        }
    }

    pub fn log_epoch(&mut self) {
        use crate::templates::base_algorithm::AlgorithmTrait;
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => {
                AlgorithmTrait::<relayrl_types::data::trajectory::RelayRLTrajectory>::log_epoch(
                    inner,
                );
            }
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    pub fn get_ppo_actor_kernel(
        &self,
    ) -> Result<&PPOKernel<B, KindIn, KindOut, Pi>, AlgorithmError> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.get_ppo_actor_kernel(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    pub fn get_ippo_actor_kernel(
        &self,
        agent_key: String,
    ) -> Result<&PPOKernel<B, KindIn, KindOut, Pi>, AlgorithmError> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => {
                inner.get_ippo_actor_kernel(agent_key)
            }
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }
}
