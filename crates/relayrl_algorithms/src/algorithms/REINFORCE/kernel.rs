use crate::templates::base_algorithm::{ForwardOutput, StepAction, ForwardKernelTrait, StepKernelTrait};
use std::collections::HashMap;
use std::sync::Arc;

use burn_core::module::Param;
use burn_nn::{Linear, LinearConfig, Relu, Tanh};
use burn_tensor::activation::{log_softmax, softmax};
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Int, Tensor};
use rand::distr::Distribution as RandDistribution;
use rand::distr::weighted::WeightedIndex;

use relayrl_types::types::data::tensor::{
    BackendMatcher, ConversionBurnTensor, DType, SupportedTensorBackend, TensorData, TensorError,
};

#[derive(Clone, Copy, Debug)]
pub enum ActivationKind {
    ReLU,
    Tanh,
}

impl Default for ActivationKind {
    fn default() -> Self {
        Self::ReLU
    }
}

#[derive(Debug)]
pub struct Mlp<B: Backend + BackendMatcher> {
    layers: Vec<Linear<B>>,
    relu: Relu,
    tanh: Tanh,
    activation: ActivationKind,
}

impl<B: Backend + BackendMatcher> Mlp<B> {
    pub fn new(
        input_dim: usize,
        hidden_sizes: &[usize],
        output_dim: usize,
        activation: ActivationKind,
        device: &B::Device,
    ) -> Self {
        let mut dims = Vec::with_capacity(hidden_sizes.len() + 2);
        dims.push(input_dim);
        dims.extend_from_slice(hidden_sizes);
        dims.push(output_dim);

        let mut layers = Vec::with_capacity(dims.len() - 1);
        for window in dims.windows(2) {
            let layer = LinearConfig::new(window[0], window[1]).init(device);
            layers.push(layer);
        }

        Self {
            layers,
            relu: Relu::new(),
            tanh: Tanh::new(),
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if idx < self.layers.len() - 1 {
                x = match self.activation {
                    ActivationKind::ReLU => self.relu.forward(x),
                    ActivationKind::Tanh => self.tanh.forward(x),
                };
            }
        }
        x
    }
}

fn backend_f32_dtype<B: Backend + BackendMatcher>() -> Result<DType, TensorError> {
    match B::get_supported_backend() {
        #[cfg(feature = "ndarray-backend")]
        SupportedTensorBackend::NdArray => Ok(DType::NdArray(
            relayrl_types::types::data::tensor::NdArrayDType::F32,
        )),
        #[cfg(feature = "tch-backend")]
        SupportedTensorBackend::Tch => Ok(DType::Tch(
            relayrl_types::types::data::tensor::TchDType::F32,
        )),
        _ => Err(TensorError::BackendError(
            "Unsupported backend for f32 TensorData conversion".to_string(),
        )),
    }
}

fn float_tensor_to_data<B: Backend + BackendMatcher, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<TensorData, TensorError> {
    TensorData::try_from(ConversionBurnTensor {
        inner: Arc::new(tensor),
        conversion_dtype: backend_f32_dtype::<B>()?,
    })
}

pub struct DiscretePolicyNetwork<B: Backend + BackendMatcher> {
    pi_network: Mlp<B>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl<B: Backend + BackendMatcher> DiscretePolicyNetwork<B> {
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
        Self {
            pi_network: Mlp::new(obs_dim, hidden_sizes, act_dim, ActivationKind::ReLU, device),
            input_dim: obs_dim,
            output_dim: act_dim,
        }
    }

    pub fn distribution(&self, obs: Tensor<B, 2>, mask: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let logits_raw = self.pi_network.forward(obs);
        let masked_logits = logits_raw + (mask - 1.0f32) * 1e8f32;
        let probs = softmax(masked_logits.clone(), 1);
        (probs, masked_logits)
    }

    pub fn sample_for_action(&self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        let [batch_size, act_dim] = probs.dims();
        let probs_vec = probs.to_data().to_vec::<f32>().unwrap_or_default();
        let mut rng = rand::rng();
        let mut sampled = Vec::with_capacity(batch_size);

        for row in 0..batch_size {
            let start = row * act_dim;
            let end = start + act_dim;
            let row_probs = probs_vec[start..end]
                .iter()
                .map(|p| p.max(0.0) as f64)
                .collect::<Vec<_>>();
            let sum: f64 = row_probs.iter().sum();
            if sum <= f64::EPSILON {
                sampled.push(0_i64);
                continue;
            }

            match WeightedIndex::new(&row_probs) {
                Ok(dist) => sampled.push(dist.sample(&mut rng) as i64),
                Err(_) => sampled.push(0_i64),
            }
        }

        Tensor::<B, 2, Int>::from_data(
            burn_tensor::TensorData::new(sampled, [batch_size, 1]),
            &probs.device(),
        )
    }

    pub fn log_prob_from_distribution(
        &self,
        logits: Tensor<B, 2>,
        act: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let log_pmf = log_softmax(logits, 1);
        log_pmf.gather(1, act)
    }
}

impl<B: Backend + BackendMatcher> ForwardKernelTrait<B> for DiscretePolicyNetwork<B> {
    fn forward(
        &self,
        obs: Tensor<B, 2>,
        mask: Tensor<B, 2>,
        act: Option<Tensor<B, 2>>,
    ) -> ForwardOutput<B> {
        let (probs, logits) = self.distribution(obs, mask);
        let logp_a = act.map(|a| self.log_prob_from_distribution(logits.clone(), a.int()));
        ForwardOutput::Discrete {
            probs,
            logits,
            logp_a,
        }
    }
}

pub struct ContinuousPolicyNetwork<B: Backend + BackendMatcher> {
    pi_network: Mlp<B>,
    log_std: Param<Tensor<B, 1>>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl<B: Backend + BackendMatcher> ContinuousPolicyNetwork<B> {
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
        let log_std_tensor = Tensor::<B, 1>::from_data(
            burn_tensor::TensorData::new(vec![-0.5f32; act_dim], [act_dim]),
            device,
        );
        Self {
            pi_network: Mlp::new(obs_dim, hidden_sizes, act_dim, ActivationKind::ReLU, device),
            log_std: Param::from_tensor(log_std_tensor),
            input_dim: obs_dim,
            output_dim: act_dim,
        }
    }

    pub fn distribution(&self, obs: Tensor<B, 2>, mask: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mean_raw = self.pi_network.forward(obs);
        let mean = mean_raw + (mask - 1.0f32) * 1e8f32;
        let std = self.log_std.val().exp().unsqueeze_dim::<2>(0);
        (mean, std)
    }

    pub fn sample_for_action(&self, mean: Tensor<B, 2>, std: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, act_dim] = mean.dims();
        let eps = Tensor::<B, 2>::random(
            [batch_size, act_dim],
            Distribution::Normal(0.0, 1.0),
            &mean.device(),
        );
        mean + std * eps
    }

    pub fn log_prob_from_distribution(
        &self,
        mean: Tensor<B, 2>,
        std: Tensor<B, 2>,
        act: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let var = std.clone().powf_scalar(2.0f32);
        let squared_err = (act - mean).powf_scalar(2.0f32);
        let log_prob = -0.5f32
            * (squared_err / var + 2.0f32 * std.log() + (2.0f32 * core::f32::consts::PI).ln());
        log_prob.sum_dim(1)
    }
}

impl<B: Backend + BackendMatcher> ForwardKernelTrait<B> for ContinuousPolicyNetwork<B> {
    fn forward(
        &self,
        obs: Tensor<B, 2>,
        mask: Tensor<B, 2>,
        act: Option<Tensor<B, 2>>,
    ) -> ForwardOutput<B> {
        let (mean, std) = self.distribution(obs, mask);
        let logp_a = act.map(|a| self.log_prob_from_distribution(mean.clone(), std.clone(), a));
        ForwardOutput::Continuous { mean, std, logp_a }
    }
}

pub struct BaselineValueNetwork<B: Backend + BackendMatcher> {
    v_network: Mlp<B>,
}

impl<B: Backend + BackendMatcher> BaselineValueNetwork<B> {
    pub fn new(
        obs_dim: usize,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        device: &B::Device,
    ) -> Self {
        Self {
            v_network: Mlp::new(obs_dim, hidden_sizes, 1, activation, device),
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>, _mask: Tensor<B, 2>) -> Tensor<B, 2> {
        self.v_network.forward(obs)
    }
}

enum PolicyHead<B: Backend + BackendMatcher> {
    Discrete(DiscretePolicyNetwork<B>),
    Continuous(ContinuousPolicyNetwork<B>),
}

pub struct PolicyWithBaseline<B: Backend + BackendMatcher> {
    policy: PolicyHead<B>,
    baseline: BaselineValueNetwork<B>,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend + BackendMatcher> PolicyWithBaseline<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        device: &B::Device,
    ) -> Self {
        let policy = if discrete {
            PolicyHead::Discrete(DiscretePolicyNetwork::new(obs_dim, hidden_sizes, act_dim, device))
        } else {
            PolicyHead::Continuous(ContinuousPolicyNetwork::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                device,
            ))
        };
        let baseline = BaselineValueNetwork::new(obs_dim, hidden_sizes, activation, device);
        Self {
            policy,
            baseline,
            input_dim: obs_dim,
            output_dim: act_dim,
        }
    }
}

impl<B: Backend + BackendMatcher> StepKernelTrait<B> for PolicyWithBaseline<B> {
    fn step(
        &self,
        obs: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        let mut data = HashMap::new();
        match &self.policy {
            PolicyHead::Discrete(policy) => {
                let (probs, logits) = policy.distribution(obs.clone(), mask.clone());
                let act = policy.sample_for_action(probs);
                let logp_a = policy.log_prob_from_distribution(logits, act.clone());
                let v = self.baseline.forward(obs, mask);

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("v".to_string(), float_tensor_to_data(v.clone())?);
                data.insert("val".to_string(), float_tensor_to_data(v)?);
                data.insert("act_tensor".to_string(), float_tensor_to_data(act.clone().float())?);
                Ok((StepAction::Discrete(act), data))
            }
            PolicyHead::Continuous(policy) => {
                let (mean, std) = policy.distribution(obs.clone(), mask.clone());
                let act = policy.sample_for_action(mean.clone(), std.clone());
                let logp_a = policy.log_prob_from_distribution(mean, std, act.clone());
                let v = self.baseline.forward(obs, mask);

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("v".to_string(), float_tensor_to_data(v.clone())?);
                data.insert("val".to_string(), float_tensor_to_data(v)?);
                data.insert("act_tensor".to_string(), float_tensor_to_data(act.clone())?);
                Ok((StepAction::Continuous(act), data))
            }
        }
    }

    fn get_input_dim(&self) -> usize {
        self.input_dim
    }

    fn get_output_dim(&self) -> usize {
        self.output_dim
    }
}

pub struct PolicyWithoutBaseline<B: Backend + BackendMatcher> {
    policy: PolicyHead<B>,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend + BackendMatcher> PolicyWithoutBaseline<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        device: &B::Device,
    ) -> Self {
        let policy = if discrete {
            PolicyHead::Discrete(DiscretePolicyNetwork::new(obs_dim, hidden_sizes, act_dim, device))
        } else {
            PolicyHead::Continuous(ContinuousPolicyNetwork::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                device,
            ))
        };
        Self {
            policy,
            input_dim: obs_dim,
            output_dim: act_dim,
        }
    }
}

impl<B: Backend + BackendMatcher> StepKernelTrait<B> for PolicyWithoutBaseline<B> {
    fn step(
        &self,
        obs: Tensor<B, 2>,
        mask: Tensor<B, 2>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        let mut data = HashMap::new();
        match &self.policy {
            PolicyHead::Discrete(policy) => {
                let (probs, logits) = policy.distribution(obs, mask);
                let act = policy.sample_for_action(probs);
                let logp_a = policy.log_prob_from_distribution(logits, act.clone());

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("act_tensor".to_string(), float_tensor_to_data(act.clone().float())?);
                Ok((StepAction::Discrete(act), data))
            }
            PolicyHead::Continuous(policy) => {
                let (mean, std) = policy.distribution(obs, mask);
                let act = policy.sample_for_action(mean.clone(), std.clone());
                let logp_a = policy.log_prob_from_distribution(mean, std, act.clone());

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("act_tensor".to_string(), float_tensor_to_data(act.clone())?);
                Ok((StepAction::Continuous(act), data))
            }
        }
    }

    fn get_input_dim(&self) -> usize {
        self.input_dim
    }

    fn get_output_dim(&self) -> usize {
        self.output_dim
    }
}