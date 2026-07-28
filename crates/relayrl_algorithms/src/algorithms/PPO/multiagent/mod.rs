use crate::algorithms::PPO::kernel::PPOPolicyHead;
use crate::algorithms::{GenericMlp, NeuralNetwork};
use crate::templates::base_algorithm::AlgorithmError;
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, TensorKind};
use relayrl_types::data::tensor::DType;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use std::marker::PhantomData;
use std::path::Path;

/// Alias for `IPPOParams`; MAPPO uses the same hyperparameter set as IPPO.
pub type MAPPOParams = crate::algorithms::PPO::independent::IPPOParams;

/// Multi-agent PPO algorithm stub; full implementation is coming soon.
pub struct MultiAgentPPOAlgorithm<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    _phantom: PhantomData<(B, KindIn, KindOut, Pi)>,
}

impl<B, KindIn, KindOut, Pi> MultiAgentPPOAlgorithm<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B> + Default,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    /// Constructs the MAPPO algorithm stub (full training implementation is coming soon).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn new(
        _hyperparams: Option<MAPPOParams>,
        _env_dir: &Path,
        _save_model_path: &Path,
        _obs_dim: &usize,
        _obs_dtype: &DType,
        _act_dim: &usize,
        _act_dtype: &DType,
        _buffer_size: &usize,
        _pi_head: PPOPolicyHead<B, KindIn, KindOut, Pi>,
        _vf_mlp: GenericMlp<B, KindIn, Float>,
    ) -> Result<Self, AlgorithmError> {
        let _hyperparams = _hyperparams.unwrap_or_default();

        let algorithm = MultiAgentPPOAlgorithm {
            _phantom: PhantomData,
        };

        Ok(algorithm)
    }
}
