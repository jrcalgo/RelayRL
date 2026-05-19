use crate::WeightProvider;
use crate::algorithms::{
    NeuralNetworkError, NeuralNetworkForward, NeuralNetworkSpec, ValueFunction,
};
use crate::algorithms::{convert_byte_dtype_to_f32, dtype_to_byte_count};

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Bool, Float, Int, Tensor, TensorKind};
use rand::Rng;
use rand_distr::Distribution;
use rayon::prelude::*;
#[cfg(feature = "ndarray-backend")]
use relayrl_types::data::tensor::NdArrayDType;
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::data::tensor::{DType, TensorData};
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use std::collections::HashMap;

#[allow(clippy::large_enum_variant)]
enum PolicyHead<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
    KindOut: BasicOps<B>,
{
    Discrete(DiscretePPOPolicyNetwork<B, KindIn, KindOut, Pi>),
    Continuous(ContinuousPPOPolicyNetwork<B, KindIn, KindOut, Pi>),
}

pub struct DiscretePPOPolicyNetwork<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub pi: Pi,
    pub input_dim: usize,
    pub input_dtype: DType,
    pub output_dim: usize,
    pub output_dtype: DType,
}

impl<B, KindIn, KindOut, Pi> DiscretePPOPolicyNetwork<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub fn new(pi: Pi) -> Result<Self, NeuralNetworkError> {
        Ok(Self {
            pi,
            input_dim: pi.input_dim()?,
            input_dtype: pi.input_dtype()?,
            output_dim: pi.output_dim()?,
            output_dtype: pi.output_dtype()?,
        })
    }

    pub fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut> {
        self.pi.forward(obs)
    }
}

pub struct ContinuousPPOPolicyNetwork<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub pi: Pi,
    pub input_dim: usize,
    pub input_dtype: DType,
    pub output_dim: usize,
    pub output_dtype: DType,
}

impl<B, KindIn, KindOut, Pi> ContinuousPPOPolicyNetwork<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub fn new(pi: Pi) -> Result<Self, NeuralNetworkError> {
        Ok(Self {
            pi,
            input_dim: pi.input_dim()?,
            input_dtype: pi.input_dtype()?,
            output_dim: pi.output_dim()?,
            output_dtype: pi.output_dtype()?,
        })
    }

    pub fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut> {
        self.pi.forward(obs)
    }
}

pub trait PPOKernelTrait<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
>
{
    fn pi_loss(
        &mut self,
        obs: &[TensorData],
        act: &&[TensorData],
        mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
        clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>);
    fn vf_loss(&mut self, obs: &[TensorData], mask: &[TensorData], ret: &[f32]) -> f32;
}

pub struct PPOKernelFactory<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>;

pub struct DiscretePPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub pi: DiscretePPOPolicyNetwork<B, KindIn, KindOut, Pi>,
    pub vf: ValueFunction<B, KindIn>,
}

pub struct ContinuousPPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub pi: ContinuousPPOPolicyNetwork<B, KindIn, KindOut, Pi>,
    pub vf: ValueFunction<B, KindIn>,
}

pub enum PPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B>,
    KindOut: TensorKind<B>,
    Pi,
> where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    Discrete(DiscretePPOKernel<B, KindIn, KindOut, Pi>),
    Continuous(ContinuousPPOKernel<B, KindIn, KindOut, Pi>),
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>, Pi>
    PPOKernelFactory<B, KindIn, KindOut, Pi>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
    Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>,
{
    pub fn new(
        pi_head: PolicyHead<B, KindIn, KindOut, Pi>,
        vf_mlp: GenericMlp<B, KindIn, Float>,
    ) -> Result<PPOKernel<B, KindIn, KindOut, Pi>, NeuralNetworkError> {
        #[inline]
        fn check_input_dim<
            B2: Backend + BackendMatcher<Backend = B2>,
            KindIn2: TensorKind<B2>,
            KindOut2: TensorKind<B2>,
            Pi2: NeuralNetworkSpec<B2, KindIn2, KindOut2> + NeuralNetworkForward<B2, KindIn2, KindOut2>,
        >(
            pi_nn: &Pi2,
            vf_nn: &ValueFunction<B2, KindIn2>,
        ) -> Result<(), NeuralNetworkError> {
            if pi_nn.input_dim() != vf_nn.input_dim() {
                return Err(NeuralNetworkError::InputDimMismatch(
                    pi_nn.input_dim().unwrap(),
                    vf_nn.input_dim().unwrap(),
                ));
            }
            Ok(())
        }

        let vf = ValueFunction::new(vf_mlp)?;

        match pi_head {
            PolicyHead::Discrete(discrete_pi) => {
                check_input_dim::<B, KindIn, KindOut, Pi>(&discrete_pi.pi, &vf)?;
                Ok(PPOKernel::<B, KindIn, KindOut, Pi>::Discrete(
                    DiscretePPOKernel { pi: pi_nn, vf },
                ))
            }
            PolicyHead::Continuous(continuous_pi) => {
                check_input_dim::<B, KindIn, KindOut, Pi>(&continuous_pi.pi, &vf)?;
                Ok(PPOKernel::<B, KindIn, KindOut, Pi>::Continuous(
                    ContinuousPPOKernel { pi: pi_nn, vf },
                ))
            }
        }
    }
}

const MIN_RAYON_PARALLEL_ENVS: usize = 8;

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    PPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    pub fn get_action_bytes(
        &self,
        raw_model_output: &TensorData,
        mask_bytes: Option<&[u8]>,
        n_envs: usize,
    ) -> Result<Vec<u8>, NeuralNetworkError> {
        let (logits, act_byte_count) = (
            convert_byte_dtype_to_f32(
                raw_model_output.data.clone(),
                raw_model_output.dtype.clone(),
            )?,
            dtype_to_byte_count(raw_model_output.dtype.clone())?,
        );

        let act_dim = match self {
            PPOKernel::Discrete(kernel) => kernel.pi.output_dim,
            PPOKernel::Continuous(kernel) => kernel.pi.output_dim,
        };

        let logp_bytes = Vec::<u8>::with_capacity(n_envs * 4); // 4 bytes per f32
        let mut rng = rand::rng();

        let action_bytes = match self {
            PPOKernel::Discrete(kernel) => {
                let action_bytes = Vec::<u8>::with_capacity(n_envs * act_byte_count);

                let discrete_action_fn = |i: usize| {
                    kernel.get_env_byte_action(i, &logits, mask_bytes, act_dim, &mut rng)
                };

                let (act_idx, logps) = match n_envs {
                    _ if n_envs < MIN_RAYON_PARALLEL_ENVS => {
                        let pairs = (0..n_envs).into_iter().map(discrete_action_fn).collect::<Vec<(i64, f32)>>();
                        (pairs.iter().map(|(idx, _)| idx).collect(), pairs.iter().map(|(_, logp)| logp).collect())
                    }
                    _ => {
                        let pairs = (0..n_envs).into_par_iter().map(discrete_action_fn).collect::<Vec<(i64, f32)>>();
                        (pairs.iter().map(|(idx, _)| idx).collect(), pairs.iter().map(|(_, logp)| logp).collect())
                    }
                };

                append_discrete_action_for_env(&mut action_bytes, act_idx, &raw_model_output.dtype)?;

                action_bytes
            }
            PPOKernel::Continuous(kernel) => {
                let action_bytes = Vec::<u8>::with_capacity(n_envs * act_dim * act_byte_count);

                let continuous_action_fn = |i: usize| {
                    kernel.get_env_byte_action(i, &logits, mask_bytes, act_dim, &mut rng)
                };

                let (act_idx, logps) = match n_envs {
                    _ if n_envs < MIN_RAYON_PARALLEL_ENVS => {
                        let pairs = (0..n_envs).into_iter().map(continuous_action_fn).collect::<Vec<(Vec<f32>, f32)>>();
                        (pairs.iter().map(|(act, _)| act).collect(), pairs.iter().map(|(_, logp)| logp).collect())
                    }
                    _ => (0..n_envs)
                        .into_par_iter()
                        .map(continuous_action_fn)
                        .unzip(),
                };

                append_continuous_action_for_env(&mut action_bytes, act_idx, &raw_model_output.dtype)?;

                action_bytes
            }
        };

        logp_bytes.extend_from_slice(&logps.to_le_bytes());

        Ok((action_bytes, logp_bytes))
    }

    pub fn pi_train_step() {}

    pub fn vf_train_step() {}

    fn append_discrete_action_for_env(
        action_bytes: &mut Vec<u8>,
        act_idx: i64,
        act_dtype: &DType,
    ) -> Result<(), NeuralNetworkError> {
        match act_dtype {
            #[cfg(feature = "ndarray-backend")]
            DType::NdArray(nd) => match nd {},
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    DiscretePPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    pub(crate) fn get_env_byte_action(
        env_id: usize,
        logits: &[f32],
        mask_bytes: Option<&[u8]>,
        act_dim: usize,
        rng: &mut impl Rng,
    ) -> (i64, f32) {
        let start = env_id * act_dim;
        let env_logits = &logits[start..start + act_dim];

        let mut masked_logits = env_logits.to_vec();
        if let Some(mask) = mask_bytes {
            for j in 0..act_dim {
                if mask[env_id * act_dim + j] == 0 {
                    masked_logits[j] = f32::NEG_INFINITY
                }
            }
        }

        let max_length = masked_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exponentials = masked_logits
            .iter()
            .map(|&x| ((x - max_length) as f64).exp())
            .collect::<Vec<f64>>();
        let exp_sum = exponentials.iter().sum::<f64>();
        let probabilities = exponentials
            .iter()
            .map(|&x| x / exp_sum)
            .collect::<Vec<f64>>();

        let act_idx = match rand::distr::weighted::WeightedIndex::new(probabilities) {
            Ok(dist) => {
                use rand::distr::Distribution;
                dist.sample(rng) as i64
            }
            Err(_) => return Err(NeuralNetworkError::InvalidDistribution),
        };
        let log_prob = (probabilities[act_idx as usize] as f32).ln();

        Ok((act_idx, log_prob))
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    PPOKernelTrait<B, KindIn, KindOut> for DiscretePPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    fn pi_loss(
        &mut self,
        obs: &[TensorData],
        act: &&[TensorData],
        mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
        clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>) {
    }

    fn vf_loss(&mut self, obs: &[TensorData], mask: &[TensorData], ret: &[f32]) -> f32 {}
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    WeightProvider for DiscretePPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {}
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    ContinuousPPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    pub(crate) fn get_env_byte_action(
        env_id: usize,
        logits: &[f32],
        act_dim: usize,
        rng: &mut impl Rng,
    ) -> Result<(Vec<f32>, f32), NeuralNetworkError> {
        use rand_distr::Normal;

        let stride = act_dim.saturating_mul(2);
        let start = env_id * stride;
        let env_logits = &logits[start..start + stride];

        let mean = &env_logits[..act_dim];
        let log_std = &env_logits[act_dim..stride];

        let mut act_vec = Vec::<f32>::with_capacity(act_dim);
        let mut total_log_prob = 0.0f32;

        for j in 0..act_dim {
            let sum = log_std[j].exp();
            let distribution = match Normal::new(mean[j], sum) {
                Ok(dist) => dist,
                Err(e) => return Err(NeuralNetworkError::InvalidDistribution),
            };

            let action = distribution.sample(rng);

            total_log_prob += -0.5 * (((action - mean[j]) / sum).powi(2))
                - log_std[j]
                - (0.5 * (2.0 * std::f32::consts::PI).ln());

            act_vec.push(action);
        }

        Ok((act_vec, total_log_prob))
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    PPOKernelTrait<B, KindIn, KindOut> for ContinuousPPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    fn pi_loss(
        &mut self,
        obs: &[TensorData],
        act: &&[TensorData],
        mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
        clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>) {
    }

    fn vf_loss(&mut self, obs: &[TensorData], mask: &[TensorData], ret: &[f32]) -> f32 {}
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    WeightProvider for ContinuousPPOKernel<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {}
}
