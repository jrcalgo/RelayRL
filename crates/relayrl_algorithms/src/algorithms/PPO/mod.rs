pub mod kernel;
pub mod replay_buffer;

pub mod independent;
pub mod multiagent;

pub use independent::{
    EpochTrainOutput, IPPOParams, IndependentPPOAlgorithm, PPOParams, SlotTrainResult,
};
pub use multiagent::{MAPPOParams, MultiAgentPPOAlgorithm};

use crate::TrainerArgs;

use crate::algorithms::PPO::kernel::{
    ContinuousPPOPolicyHead, DiscretePPOPolicyHead, PPOKernel, PPOPolicyHead,
};
use crate::algorithms::{
    GenericMlp, NeuralNetwork, NeuralNetworkError, NeuralNetworkSpec, dtype_is_float,
};

use crate::templates::base_algorithm::AlgorithmError;

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, TensorKind};
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::data::tensor::{DType, NdArrayDType, SupportedTensorBackend};
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, DeviceType};

use std::path::PathBuf;

/// Convenience alias: MAPPO uses the same spec structure as PPO.
pub type MAPPOTrainerSpec<B, KindIn, KindOut, Pi> = PPOTrainerSpec<B, KindIn, KindOut, Pi>;

// ---- PPO-related inference & algorithm interfaces ----

/// Policy head and value function paired together for `PPOTrainerSpec` construction.
#[derive(Debug)]
pub struct PPONetworkArgs<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    pub pi_head: PPOPolicyHead<B, KindIn, KindOut, Pi>,
    pub vf_mlp: GenericMlp<B, KindIn, Float>,
}

impl<B, KindIn, KindOut, Pi> PPONetworkArgs<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    fn backend_f32_dtype() -> DType {
        match B::get_supported_backend() {
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
            _ => DType::NdArray(NdArrayDType::F32),
        }
    }

    /// Builds default discrete (categorical) policy and value networks.
    ///
    /// The value MLP always has output dim `1` and an f32 backend dtype, independent of
    /// `act_dim` / `act_dtype`.
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
                act_dtype,
                &device,
            ))?),
            vf_mlp: GenericMlp::default(obs_dim, obs_dtype, 1, Self::backend_f32_dtype(), &device),
        })
    }

    /// Builds default continuous (diagonal-Gaussian) policy and value networks.
    ///
    /// The policy network emits `2 * act_dim` floats laid out as
    /// `[mean_0..mean_{A-1}, log_std_0..log_std_{A-1}]`. `act_dtype` must be floating.
    pub fn default_continuous(
        obs_dim: usize,
        obs_dtype: DType,
        act_dim: usize,
        act_dtype: DType,
        device: B::Device,
    ) -> Result<Self, NeuralNetworkError> {
        if !dtype_is_float(&act_dtype) {
            return Err(NeuralNetworkError::InvalidContinuousActionDType(
                act_dtype.to_string(),
            ));
        }
        let policy_out =
            act_dim
                .checked_mul(2)
                .ok_or(NeuralNetworkError::InvalidContinuousOutputDim {
                    output_dim: act_dim,
                })?;
        Ok(Self {
            pi_head: PPOPolicyHead::Continuous(ContinuousPPOPolicyHead::new(
                <Pi as NeuralNetwork<B, KindIn, KindOut>>::default(
                    obs_dim,
                    obs_dtype.clone(),
                    policy_out,
                    act_dtype,
                    &device,
                ),
            )?),
            vf_mlp: GenericMlp::default(obs_dim, obs_dtype, 1, Self::backend_f32_dtype(), &device),
        })
    }
}

/// Selects the PPO algorithm variant (`PPO`, `IPPO`, or `MAPPO`) and bundles network architecture with training arguments.
///
/// ```ignore
/// use relayrl::algorithms::{GenericMlp, PPONetworkArgs, PPOTrainerSpec, PPOTrainer};
/// use burn_ndarray::NdArray;
/// use burn_tensor::Float;
/// use relayrl::types::tensor::relayrl::{DType, NdArrayDType, DeviceType};
///
/// let spec = PPOTrainerSpec::<NdArray, Float, Float, GenericMlp<_, _, _>>::default(
///     env_dir,
///     save_model_path,
///     8, DType::NdArray(NdArrayDType::F32),
///     4, DType::NdArray(NdArrayDType::F32),
///     1000, DeviceType::Cpu,
/// )?;
/// let trainer = PPOTrainer::new(spec)?;
/// ```
#[derive(Debug)]
pub enum PPOTrainerSpec<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
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
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    /// Builds a `PPO` spec with default networks from directories, dimensions, dtypes, buffer size, and device.
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

    /// Builds a `PPO` spec with default continuous (diagonal-Gaussian) networks.
    ///
    /// `act_dim` is the environment action dimension; the policy network width is `2 * act_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn default_continuous(
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
            PPONetworkArgs::default_continuous(
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
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    /// Constructs a single-agent `PPO` spec from explicit args, hyperparameters, and networks.
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

    /// Constructs an independent multi-agent `IPPO` spec.
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

    /// Constructs a centralized multi-agent `MAPPO` spec.
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

/// Dispatches training to the selected PPO algorithm variant and provides a uniform interface for training loops.
pub enum PPOTrainer<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    PPO(IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>),
    IPPO(IndependentPPOAlgorithm<B, KindIn, KindOut, Pi>),
    MAPPO(MultiAgentPPOAlgorithm<B, KindIn, KindOut, Pi>),
}

impl<B, KindIn, KindOut, Pi> PPOTrainer<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    /// Validates the spec and instantiates the corresponding runnable trainer.
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

fn dtype_matches_backend_family<B: Backend + BackendMatcher<Backend = B>>(dtype: &DType) -> bool {
    match B::get_supported_backend() {
        SupportedTensorBackend::NdArray => matches!(dtype, DType::NdArray(_)),
        #[cfg(feature = "tch-backend")]
        SupportedTensorBackend::Tch => matches!(dtype, DType::Tch(_)),
        _ => false,
    }
}

fn validate_ppo_spec<
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
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
                return Err(AlgorithmError::InvalidSpec(
                    "PPO policy head input/output dimensions or dtypes do not match the trainer arguments"
                        .to_string(),
                ));
            }
            if !dtype_matches_backend_family::<B>(pi.pi.input_dtype())
                || !dtype_matches_backend_family::<B>(pi.pi.output_dtype())
            {
                return Err(AlgorithmError::InvalidSpec(
                    "PPO policy head dtype does not match the trainer backend".to_string(),
                ));
            }
        }
        PPOPolicyHead::Continuous(pi) => {
            let expected_policy_output_dim = args.act_dim.checked_mul(2).ok_or_else(|| {
                AlgorithmError::InvalidSpec(
                    "Continuous PPO act_dim overflowed when computing 2 * act_dim".to_string(),
                )
            })?;
            if expected_policy_output_dim == 0 || expected_policy_output_dim % 2 != 0 {
                return Err(AlgorithmError::InvalidSpec(format!(
                    "Continuous PPO policy output dim must be a positive even width; expected {}",
                    expected_policy_output_dim
                )));
            }
            if !dtype_is_float(&args.act_dtype) {
                return Err(AlgorithmError::InvalidSpec(format!(
                    "Continuous PPO action dtype must be floating; got {}",
                    args.act_dtype
                )));
            }
            if *pi.pi.input_dim() != args.obs_dim
                || *pi.pi.output_dim() != expected_policy_output_dim
                || *pi.pi.input_dtype() != args.obs_dtype
                || *pi.pi.output_dtype() != args.act_dtype
            {
                return Err(AlgorithmError::InvalidSpec(format!(
                    "Continuous PPO policy head must have input_dim={}, output_dim={} (2*act_dim), and matching dtypes; got input_dim={}, output_dim={}",
                    args.obs_dim,
                    expected_policy_output_dim,
                    pi.pi.input_dim(),
                    pi.pi.output_dim()
                )));
            }
            if !dtype_matches_backend_family::<B>(pi.pi.input_dtype())
                || !dtype_matches_backend_family::<B>(pi.pi.output_dtype())
            {
                return Err(AlgorithmError::InvalidSpec(
                    "PPO policy head dtype does not match the trainer backend".to_string(),
                ));
            }
        }
    }

    if *vf_mlp.input_dim() != args.obs_dim
        || *vf_mlp.output_dim() != 1
        || *vf_mlp.input_dtype() != args.obs_dtype
    {
        return Err(AlgorithmError::InvalidSpec(
            "PPO value function MLP input/output dimensions or input dtype do not match the trainer arguments"
                .to_string(),
        ));
    }

    if !dtype_matches_backend_family::<B>(vf_mlp.input_dtype()) {
        return Err(AlgorithmError::InvalidSpec(
            "PPO value function MLP input dtype does not match the trainer backend".to_string(),
        ));
    }

    let vf_out_ok = match vf_mlp.output_dtype() {
        DType::NdArray(NdArrayDType::F32) => true,
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F32) => true,
        _ => false,
    };
    if !vf_out_ok {
        return Err(AlgorithmError::InvalidSpec(
            "PPO value function MLP output dtype is not f32".to_string(),
        ));
    }

    Ok(())
}

// ---- PPOTrainer delegation methods ----
// These minimize the amount of pattern matching required by the caller.

impl<B, KindIn, KindOut, Pi> PPOTrainer<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default + Send + 'static,
    KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
    KindOut: TensorKind<B> + BasicOps<B> + Send + 'static,
    Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
{
    /// Registers the first agent slot under `agent_key` for IPPO key-based routing.
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

    /// Spawns the epoch's training computation, returning a join handle to its `EpochTrainOutput`.
    pub fn start_epoch_training(
        &mut self,
    ) -> Option<tokio::task::JoinHandle<EpochTrainOutput<B, KindIn, KindOut, Pi>>> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.start_epoch_training(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    /// Applies a completed epoch's trained kernels back into the trainer.
    pub fn apply_epoch_result(&mut self, output: EpochTrainOutput<B, KindIn, KindOut, Pi>) {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.apply_epoch_result(output),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    /// Exports the current policy network as a `ModelModule` for inference or hot-swap.
    pub fn acquire_pi_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.acquire_pi_module(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    /// Exports the current value network as a `ModelModule`.
    pub fn acquire_vf_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.acquire_vf_module(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    /// Ingests a trajectory into the rollout buffer, returning `true` when an epoch is ready to train.
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

    /// Emits the current epoch's accumulated training metrics.
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

    /// Saves the current policy model into `output_dir`.
    ///
    /// The directory will contain `metadata.json` and the backend-specific model artifact.
    /// MAPPO is not yet supported and returns [`AlgorithmError::InvalidSpec`].
    pub fn save_model(&self, output_dir: &str) -> Result<(), AlgorithmError> {
        use crate::templates::base_algorithm::AlgorithmTrait;
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => {
                AlgorithmTrait::<relayrl_types::data::trajectory::RelayRLTrajectory>::save_model(
                    inner, output_dir,
                )
            }
            PPOTrainer::MAPPO(_) => Err(AlgorithmError::InvalidSpec(
                "MAPPO save_model not yet implemented".to_string(),
            )),
        }
    }

    /// Returns the single-agent PPO inference kernel.
    pub fn get_ppo_actor_kernel(
        &self,
    ) -> Result<&PPOKernel<B, KindIn, KindOut, Pi>, AlgorithmError> {
        match self {
            PPOTrainer::PPO(inner) | PPOTrainer::IPPO(inner) => inner.get_ppo_actor_kernel(),
            PPOTrainer::MAPPO(_) => unimplemented!(),
        }
    }

    /// Returns the IPPO inference kernel for the agent registered under `agent_key`.
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

#[cfg(test)]
mod continuous_spec_tests {
    use super::*;
    use crate::algorithms::{ActivationKind, GenericMlp};
    use burn_ndarray::NdArray;
    use burn_nn::activation::Relu;
    use burn_tensor::Float;
    use std::path::PathBuf;

    type B = NdArray;
    type Pi = GenericMlp<B, Float, Float>;

    fn f32() -> DType {
        DType::NdArray(NdArrayDType::F32)
    }

    fn i64_dtype() -> DType {
        DType::NdArray(NdArrayDType::I64)
    }

    fn trainer_args(act_dim: usize, act_dtype: DType) -> TrainerArgs {
        TrainerArgs {
            env_dir: PathBuf::from("env"),
            save_model_path: PathBuf::from("model.mpk"),
            obs_dim: 3,
            obs_dtype: f32(),
            act_dim,
            act_dtype,
            buffer_size: 64,
            device: DeviceType::Cpu,
        }
    }

    fn mlp(out_dim: usize, out_dtype: DType) -> Pi {
        let device = <B as Backend>::Device::default();
        GenericMlp::new(
            3,
            f32(),
            &[8],
            out_dim,
            out_dtype,
            ActivationKind::ReLU(Relu::new()),
            &device,
        )
    }

    fn vf() -> GenericMlp<B, Float, Float> {
        let device = <B as Backend>::Device::default();
        GenericMlp::new(
            3,
            f32(),
            &[8],
            1,
            f32(),
            ActivationKind::ReLU(Relu::new()),
            &device,
        )
    }

    #[test]
    fn default_spec_stays_discrete() {
        let spec = PPOTrainerSpec::<B, Float, Float, Pi>::default(
            PathBuf::from("env"),
            PathBuf::from("model.mpk"),
            3,
            f32(),
            2,
            f32(),
            64,
            DeviceType::Cpu,
        )
        .expect("default discrete");
        let PPOTrainerSpec::PPO { networks, .. } = spec else {
            panic!("expected PPO variant");
        };
        assert!(matches!(networks.pi_head, PPOPolicyHead::Discrete(_)));
        assert_eq!(*networks.vf_mlp.output_dim(), 1);
        assert_eq!(*networks.vf_mlp.output_dtype(), f32());
    }

    #[test]
    fn default_continuous_spec_builds_width_two_times_action_dim() {
        let spec = PPOTrainerSpec::<B, Float, Float, Pi>::default_continuous(
            PathBuf::from("env"),
            PathBuf::from("model.mpk"),
            3,
            f32(),
            2,
            f32(),
            64,
            DeviceType::Cpu,
        )
        .expect("default continuous");
        let PPOTrainerSpec::PPO { args, networks, .. } = spec else {
            panic!("expected PPO variant");
        };
        assert_eq!(args.act_dim, 2);
        match networks.pi_head {
            PPOPolicyHead::Continuous(head) => assert_eq!(*head.pi.output_dim(), 4),
            _ => panic!("expected continuous head"),
        }
        assert_eq!(*networks.vf_mlp.output_dim(), 1);
    }

    #[test]
    fn continuous_spec_accepts_policy_width_two_times_action_dim() {
        let networks = PPONetworkArgs {
            pi_head: PPOPolicyHead::Continuous(
                ContinuousPPOPolicyHead::new(mlp(4, f32())).expect("head"),
            ),
            vf_mlp: vf(),
        };
        let trainer = PPOTrainer::<B, Float, Float, Pi>::new(PPOTrainerSpec::ppo(
            trainer_args(2, f32()),
            None,
            networks,
        ));
        assert!(trainer.is_ok());
    }

    #[test]
    fn continuous_spec_rejects_policy_width_equal_action_dim() {
        let networks = PPONetworkArgs {
            pi_head: PPOPolicyHead::Continuous(
                ContinuousPPOPolicyHead::new(mlp(2, f32())).expect("even width head"),
            ),
            vf_mlp: vf(),
        };
        match PPOTrainer::<B, Float, Float, Pi>::new(PPOTrainerSpec::ppo(
            trainer_args(2, f32()),
            None,
            networks,
        )) {
            Err(AlgorithmError::InvalidSpec(_)) => {}
            Ok(_) => panic!("expected InvalidSpec for output_dim == act_dim"),
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn continuous_spec_rejects_odd_policy_width() {
        let device = <B as Backend>::Device::default();
        let pi: Pi = GenericMlp::new(
            3,
            f32(),
            &[8],
            3,
            f32(),
            ActivationKind::ReLU(Relu::new()),
            &device,
        );
        match ContinuousPPOPolicyHead::new(pi) {
            Err(NeuralNetworkError::InvalidContinuousOutputDim { output_dim: 3 }) => {}
            other => panic!("expected InvalidContinuousOutputDim, got {other:?}"),
        }
    }

    #[test]
    fn continuous_spec_rejects_integer_action_dtype() {
        let err = PPONetworkArgs::<B, Float, Float, Pi>::default_continuous(
            3,
            f32(),
            2,
            i64_dtype(),
            <B as Backend>::Device::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            NeuralNetworkError::InvalidContinuousActionDType(_)
        ));
    }

    #[test]
    fn continuous_spec_rejects_bool_action_dtype() {
        let err = PPONetworkArgs::<B, Float, Float, Pi>::default_continuous(
            3,
            f32(),
            2,
            DType::NdArray(NdArrayDType::Bool),
            <B as Backend>::Device::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            NeuralNetworkError::InvalidContinuousActionDType(_)
        ));
    }

    #[test]
    fn vf_default_output_is_one_f32() {
        let networks = PPONetworkArgs::<B, Float, Float, Pi>::default(
            3,
            f32(),
            4,
            f32(),
            <B as Backend>::Device::default(),
        )
        .expect("default networks");
        assert_eq!(*networks.vf_mlp.output_dim(), 1);
        assert_eq!(*networks.vf_mlp.output_dtype(), f32());
    }
}
