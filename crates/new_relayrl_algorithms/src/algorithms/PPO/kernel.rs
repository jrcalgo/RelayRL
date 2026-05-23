use crate::WeightProvider;
use crate::algorithms::{
    GenericMlp, NeuralNetwork, NeuralNetworkError, NeuralNetworkForward, NeuralNetworkSpec,
    ValueFunction,
};
use crate::algorithms::{convert_byte_dtype_to_f32, dtype_to_byte_count};

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Bool, Float, Int, Tensor, TensorKind};
use rand::Rng;
use rand::RngExt;
use rand_distr::Distribution;
use rayon::prelude::*;
use relayrl_types::data::tensor::NdArrayDType;
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::data::tensor::{DType, TensorData, BurnTensorData};
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use std::collections::HashMap;

// ---- policy network head definitions ----

#[allow(clippy::large_enum_variant)]
pub enum PPOPolicyHead<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    Discrete(DiscretePPOPolicyHead<B, KindIn, KindOut, Pi>),
    Continuous(ContinuousPPOPolicyHead<B, KindIn, KindOut, Pi>),
}

#[derive(Clone, Debug)]
pub struct DiscretePPOPolicyHead<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub pi: Pi,
    _phantom: std::marker::PhantomData<(B, KindIn, KindOut)>,
}

impl<B, KindIn, KindOut, Pi> DiscretePPOPolicyHead<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    pub fn new(pi: Pi) -> Result<Self, NeuralNetworkError> {
        Ok(Self {
            pi,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut> {
        self.pi.forward(obs)
    }

    pub fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        self.pi.get_layer_specs()
    }
}

#[derive(Clone, Debug)]
pub struct ContinuousPPOPolicyHead<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub pi: Pi,
    _phantom: std::marker::PhantomData<(B, KindIn, KindOut)>,
}

impl<B, KindIn, KindOut, Pi> ContinuousPPOPolicyHead<B, KindIn, KindOut, Pi>
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
{
    pub fn new(pi: Pi) -> Result<Self, NeuralNetworkError> {
        Ok(Self {
            pi,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut> {
        self.pi.forward(obs)
    }

    pub fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        self.pi.get_layer_specs()
    }
}

// ---- continuous and discrete kernel interfaces ----

pub type PiLoss = f32;
pub type VfLoss = f32;
pub type Info = HashMap<String, f32>;

pub trait PPOKernelTraining<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
>
{
    fn train_step(
        &mut self,
        obs: &[TensorData],
        obs_dim: usize,
        act: &[TensorData],
        adv: &[f32],
        logp_old: &[f32],
        ret: &[f32],
        clip_ratio: f32,
        ent_coef: f32,
        compute_stats: bool,
    ) -> (PiLoss, VfLoss, Info);
}

pub type ActBytes = Vec<u8>;
pub type LogpBytes = Vec<u8>;

pub trait PPOKernelOps<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
>
{
    fn policy_forward_bytes(
        &self,
        raw_model_output: &TensorData,
        mask_bytes: Option<&[u8]>,
        n_envs: usize,
    ) -> Result<(ActBytes, LogpBytes), NeuralNetworkError>;
    fn get_pi_logprobs(&self, obs: &[TensorData], obs_dim: usize, act: &[TensorData]) -> Vec<f32>;
    fn value_forward(&self, obs: &[TensorData], obs_dim: usize) -> Vec<f32>;
    fn normalize_persistent_returns(&self, ret: &[f32]) -> Vec<f32>;
}

/// Factory for constructing continuous or discrete PPO kernels.
pub struct PPOKernelFactory<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    _phantom: std::marker::PhantomData<(B, KindIn, KindOut, Pi)>,
}

pub struct DiscretePPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub pi: DiscretePPOPolicyHead<B, KindIn, KindOut, Pi>,
    pub vf: ValueFunction<B, KindIn>,
    pub trainer: PPOActorCriticTrainer<B, KindIn, KindOut, Pi>,
}

pub struct ContinuousPPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub pi: ContinuousPPOPolicyHead<B, KindIn, KindOut, Pi>,
    pub vf: ValueFunction<B, KindIn>,
    pub trainer: PPOActorCriticTrainer<B, KindIn, KindOut, Pi>,
}

pub enum PPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    Discrete(DiscretePPOKernel<B, KindIn, KindOut, Pi>),
    Continuous(ContinuousPPOKernel<B, KindIn, KindOut, Pi>),
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernelFactory<B, KindIn, KindOut, Pi>
{
    pub fn new(
        pi_head: PPOPolicyHead<B, KindIn, KindOut, Pi>,
        vf_mlp: GenericMlp<B, KindIn, Float>,
    ) -> Result<PPOKernel<B, KindIn, KindOut, Pi>, NeuralNetworkError> {
        #[inline]
        fn check_input_dim<
            B2: Backend + BackendMatcher<Backend = B2>,
            KindIn2: TensorKind<B2> + BasicOps<B2>,
            KindOut2: TensorKind<B2> + BasicOps<B2>,
            Pi2: NeuralNetwork<B2, KindIn2, KindOut2>,
        >(
            pi_nn: &Pi2,
            vf_nn: &ValueFunction<B2, KindIn2>,
        ) -> Result<(), NeuralNetworkError> {
            if *pi_nn.input_dim() != *<ValueFunction<B2, KindIn2> as NeuralNetworkSpec<B2, KindIn2, KindOut2>>::input_dim(vf_nn) {
                return Err(NeuralNetworkError::InputDimMismatch(
                    *pi_nn.input_dim(),
                    *<ValueFunction<B2, KindIn2> as NeuralNetworkSpec<B2, KindIn2, KindOut2>>::input_dim(vf_nn),
                ));
            }
            Ok(())
        }

        let vf: ValueFunction<B, KindIn> = ValueFunction::new(vf_mlp)?;

        match pi_head {
            PPOPolicyHead::Discrete(discrete_pi) => {
                check_input_dim::<B, KindIn, KindOut, Pi>(&discrete_pi.pi, &vf)?;
                Ok(PPOKernel::<B, KindIn, KindOut, Pi>::Discrete(
                    DiscretePPOKernel {
                        pi: discrete_pi,
                        vf,
                    },
                ))
            }
            PPOPolicyHead::Continuous(continuous_pi) => {
                check_input_dim::<B, KindIn, KindOut, Pi>(&continuous_pi.pi, &vf)?;
                Ok(PPOKernel::<B, KindIn, KindOut, Pi>::Continuous(
                    ContinuousPPOKernel {
                        pi: continuous_pi,
                        vf,
                    },
                ))
            }
        }
    }
}

const MIN_RAYON_PARALLEL_ENVS: usize = 8;

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernel<B, KindIn, KindOut, Pi>
{
    pub fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        match self {
            PPOKernel::Discrete(kernel) => kernel.pi.get_pi_layer_specs(),
            PPOKernel::Continuous(kernel) => kernel.pi.get_pi_layer_specs(),
        }
    }

    pub fn get_vf_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        match self {
            PPOKernel::Discrete(kernel) => kernel.vf.get_vf_layer_specs(),
            PPOKernel::Continuous(kernel) => kernel.vf.get_vf_layer_specs(),
        }
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernelOps<B, KindIn, KindOut, Pi> for PPOKernel<B, KindIn, KindOut, Pi>
{
    fn policy_forward_bytes(
        &self,
        raw_model_output: &TensorData,
        mask_bytes: Option<&[u8]>,
        n_envs: usize,
    ) -> Result<(ActBytes, LogpBytes), NeuralNetworkError> {
        fn append_discrete_action_for_env() {}

        fn append_continuous_action_for_env() {}

        let (logits, act_byte_count) = (
            convert_byte_dtype_to_f32(
                raw_model_output.data.clone(),
                raw_model_output.dtype.clone(),
            )?,
            dtype_to_byte_count(raw_model_output.dtype.clone()),
        );

        let act_dim = match self {
            PPOKernel::Discrete(kernel) => kernel.pi.pi.output_dim(),
            PPOKernel::Continuous(kernel) => kernel.pi.pi.output_dim(),
        };

        let logp_bytes = Vec::<u8>::with_capacity(n_envs * 4); // 4 bytes per f32
        let mut rng = rand::rng();

        let (action_bytes, logps) = match self {
            PPOKernel::Discrete(kernel) => {
                let action_bytes = Vec::<u8>::with_capacity(n_envs * act_byte_count);

                let discrete_action_fn = |i: usize| {
                    DiscretePPOKernel::<B, KindIn, KindOut, Pi>::get_env_byte_action(
                        i, &logits, mask_bytes, *act_dim, &mut rng,
                    )
                };

                let (act_idx, logps) = match n_envs {
                    _ if n_envs < MIN_RAYON_PARALLEL_ENVS => {
                        let pairs = (0..n_envs)
                            .into_iter()
                            .map(discrete_action_fn)
                            .collect::<Vec<(i64, f32)>>();
                        (
                            pairs.iter().map(|(idx, _)| idx).collect(),
                            pairs.iter().map(|(_, logp)| logp).collect(),
                        )
                    }
                    _ => {
                        let pairs = (0..n_envs)
                            .into_par_iter()
                            .map(discrete_action_fn)
                            .collect::<Vec<(i64, f32)>>();
                        (
                            pairs.iter().map(|(idx, _)| idx).collect(),
                            pairs.iter().map(|(_, logp)| logp).collect(),
                        )
                    }
                };

                append_discrete_action_for_env(
                    &mut action_bytes,
                    act_idx,
                    &raw_model_output.dtype,
                )?;

                (action_bytes, logps)
            }
            PPOKernel::Continuous(kernel) => {
                let action_bytes = Vec::<u8>::with_capacity(n_envs * act_dim * act_byte_count);

                let continuous_action_fn = |i: usize| {
                    ContinuousPPOKernel::<B, KindIn, KindOut, Pi>::get_env_byte_action(
                        i, &logits, *act_dim, &mut rng,
                    )
                };

                let (act_idx, logps) = match n_envs {
                    _ if n_envs < MIN_RAYON_PARALLEL_ENVS => {
                        let pairs = (0..n_envs)
                            .into_iter()
                            .map(continuous_action_fn)
                            .collect::<Vec<(Vec<f32>, f32)>>();
                        (
                            pairs.iter().map(|(act, _)| act).collect(),
                            pairs.iter().map(|(_, logp)| logp).collect(),
                        )
                    }
                    _ => (0..n_envs)
                        .into_par_iter()
                        .map(continuous_action_fn)
                        .unzip(),
                };

                append_continuous_action_for_env(
                    &mut action_bytes,
                    act_idx,
                    &raw_model_output.dtype,
                )?;

                (action_bytes, logps)
            }
        };

        logp_bytes.extend_from_slice(&logps.to_le_bytes());

        Ok((action_bytes, logp_bytes))
    }

    fn get_pi_logprobs(&self, obs: &[TensorData], obs_dim: usize, act: &[TensorData]) -> Vec<f32> {
        if let Some(trainer) = &self.actor_critic_trainer {
            return trainer.logprobs_flat(obs_flat, obs_dim, act_flat);
        }
        vec![0.0; act_flat.len()]
    }

    fn value_forward(&self, obs: &[TensorData], obs_dim: usize) -> Vec<f32> {
        if obs.is_empty() {
            return Vec::new();
        }
        let n = obs.len();
        let obs_dim = obs[0].shape[0];
        let device = <B as burn_tensor::backend::Backend>::Device::default();
        let flat: Vec<f32> = obs.iter()
            .flat_map(|td| td.data.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])))
            .collect();
        let obs_t = burn_tensor::Tensor::<B, 2, InK>::from_data(
            burn_tensor::TensorData::new(flat, [n, obs_dim]),
            &device,
        );
        let mask_t = burn_tensor::Tensor::<B, 2, OutK>::ones([n, self.output_dim], &device);
        let v = self.baseline.forward(obs_t, mask_t);
        v.into_data().to_vec::<f32>().unwrap_or_default()
    }

    fn normalize_persistent_returns(&self, ret: &[f32]) -> Vec<f32> {
        for &r in ret {
            self.returns_count += 1;
            let delta = r - self.returns_mean;
            self.returns_mean += delta / self.returns_count as f32;
            let delta2 = r - self.returns_mean;
            self.returns_variance += delta * delta2;
        }
        let std = if self.returns_count > 1 {
            (self.returns_variance / (self.returns_count - 1) as f32).sqrt().max(1e-8)
        } else {
            1.0
        };
        ret.iter()
            .map(|&r| ((r - self.returns_mean) / std).clamp(-5.0, 5.0))
            .collect()
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernelTraining<B, KindIn, KindOut, Pi> for PPOKernel<B, KindIn, KindOut, Pi>
{
    fn train_step(
        &mut self,
        obs: &[TensorData],
        obs_dim: usize,
        act: &[TensorData],
        adv: &[f32],
        logp_old: &[f32],
        ret: &[f32],
        clip_ratio: f32,
        ent_coef: f32,
        compute_stats: bool,
    ) -> (PiLoss, VfLoss, Info) {
        let n = (obs.len() / obs_dim.max(1))
            .min(act.len())
            .min(adv.len())
            .min(logp_old.len())
            .min(ret.len());
        if n == 0 {
            return (0.0, 0.0, zero_pi_info().1);
        }

        match self {
            PPOKernel::Discrete(kernel) => {

            }
            PPOKernel::Continuous(kernel) => {

            }
        }

        let net = match self.network.take() {
            Some(net) => net,
            None => return (0.0, 0.0, zero_pi_info().1),
        };
        let device = B::Device::default();

        let obs = Tensor::<B, 2, KindIn>::from_data(
            BurnTensorData::new(obs[..n * obs_dim].to_vec(), [n, obs_dim]),
            &device,
        );

        // ── Policy head ───────────────────────────────────────────────
        let logits = net.pi_forward(obs.clone());
        let log_probs_full = log_softmax(logits, 1);
        let act = Tensor::<B, 2, KindOut>::from_data(
            BurnTensorData::new(act[..n].to_vec(), [n, 1]),
            &device,
        );
        let logp = log_probs_full.clone().gather(1, act).reshape([n]);
        let adv_tensor =
            Tensor::<B, 1, Float>::from_data(BurnTensorData::new(adv[..n].to_vec(), [n]), &device);
        let logp_old_tensor = Tensor::<B, 1, Float>::from_data(
            BurnTensorData::new(logp_old[..n].to_vec(), [n]),
            &device,
        );
        let ratio = (logp.clone() - logp_old_tensor).exp();
        let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);
        let clip_obj = (ratio.clone() * adv_tensor.clone())
            .min_pair(clipped_ratio * adv_tensor)
            .mean();
        let entropy_t = (log_probs_full.clone().exp() * log_probs_full)
            .neg()
            .sum_dim(1)
            .reshape([n])
            .mean();
        let pi_loss_t = -(clip_obj + ent_coef * entropy_t.clone());

        // ── Value head ────────────────────────────────────────────────
        let v_pred = net.vf_forward(obs).reshape([n]);
        let ret_tensor =
            Tensor::<B, 1, Float>::from_data(BurnTensorData::new(ret[..n].to_vec(), [n]), &device);
        let vf_loss_t = (v_pred - ret_tensor).powf_scalar(2.0).mean();

        // ── Combined loss → single backward pass ──────────────────────
        let vf_coef_t = self.vf_coef;
        let total_loss = pi_loss_t.clone() + vf_loss_t.clone() * vf_coef_t;

        let pi_loss_val = scalar_from_tensor(&pi_loss_t);
        let vf_loss_val = scalar_from_tensor(&vf_loss_t);

        let grads = total_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &net);
        let lr = self.effective_lr();
        let net = self.optimizer.step(lr, net, grads_params);
        self.network = Some(net);
        self.grad_step_count += 1;

        if !compute_stats {
            return (pi_loss_val, vf_loss_val, HashMap::new());
        }

        let entropy_val = entropy_t.into_scalar();
        let approx_kl = ((ratio.clone() - 1.0) - ratio.clone().log())
            .mean()
            .into_scalar();
        let ratio_values = ratio
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![1.0; n]);
        let clipfrac = ratio_values
            .iter()
            .filter(|r| (**r - 1.0).abs() > clip_ratio)
            .count() as f32
            / n as f32;

        let mut info = HashMap::new();
        info.insert("kl".to_string(), approx_kl);
        info.insert("entropy".to_string(), entropy_val);
        info.insert("clipfrac".to_string(), clipfrac);
        (pi_loss_val, vf_loss_val, info)
    }
}

// ---- discrete kernel implementation ----

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> DiscretePPOKernel<B, KindIn, KindOut, Pi>
{
    #[inline(always)]
    pub(super) fn get_env_byte_action(
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

        // compute softmax probabilities
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

        // categorical sampling
        let rand_selected_prob = rng.random::<f64>();
        let mut cumulative_prob = 0.0;
        let act_idx = probabilities
            .iter()
            .enumerate()
            .find(|(_, p)| {
                cumulative_prob += *p;
                cumulative_prob >= rand_selected_prob
            })
            .map(|(idx, _)| idx as i64)
            .unwrap_or((act_dim - 1) as i64);

        let log_prob = (probabilities[act_idx as usize] as f32).ln();

        (act_idx, log_prob)
    }
}

// ---- continuous kernel implementation ----

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> ContinuousPPOKernel<B, KindIn, KindOut, Pi>
{
    #[inline(always)]
    pub(super) fn get_env_byte_action(
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

mod trainer {
    use super::*;

    pub struct PPOActorCriticTrainer<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B> + BasicOps<B>, KindOut: TensorKind<B> + BasicOps<B>, Pi: NeuralNetwork<B, KindIn, KindOut>> {
    }
}