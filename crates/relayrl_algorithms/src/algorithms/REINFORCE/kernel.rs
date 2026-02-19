use crate::templates::base_algorithm::{
    ForwardKernelTrait, ForwardOutput, StepAction, StepKernelTrait,
};
use std::collections::HashMap;
use std::sync::Arc;

use burn_core::module::Param;
use burn_nn::{Linear, LinearConfig, Relu, Tanh};
use burn_tensor::activation::{log_softmax, softmax};
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Distribution, Float, Int, Tensor, TensorKind};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution as RandDistribution;
use std::marker::PhantomData;

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
pub struct Mlp<B: Backend + BackendMatcher, InK: TensorKind<B>> {
    layers: Vec<Linear<B>>,
    relu: Relu,
    tanh: Tanh,
    activation: ActivationKind,
    _in_k: PhantomData<InK>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>> Mlp<B, InK> {
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
            _in_k: PhantomData::<InK>::default(),
        }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D, InK>) -> Tensor<B, D, Float> where InK: BasicOps<B> {
        let device = input.device();
        let mut x: Tensor<B, D, Float> = Tensor::from_data(input.into_data().convert::<f32>(), &device);

        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward::<B, D>(x);
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
    tensor: Tensor<B, D, Float>,
) -> Result<TensorData, TensorError> {
    TensorData::try_from(ConversionBurnTensor {
        inner: Arc::new(tensor),
        conversion_dtype: backend_f32_dtype::<B>()?,
    })
}

pub struct DiscretePolicyNetwork<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    pi_network: Mlp<B, InK>,
    pub input_dim: usize,
    pub output_dim: usize,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> DiscretePolicyNetwork<B, InK, OutK> {
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
        Self {
            pi_network: Mlp::new(obs_dim, hidden_sizes, act_dim, ActivationKind::ReLU, device),
            input_dim: obs_dim,
            output_dim: act_dim,
            _out_k: PhantomData::<OutK>::default(),
        }
    }

    pub fn distribution<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
    ) -> (Tensor<B, OutD, Float>, Tensor<B, OutD, Float>) {
        let logits_raw = self.pi_network.forward(obs).reshape(mask.dims());
        let masked_logits = logits_raw + (mask - 1.0f32) * 1e8f32;
        let probs = softmax(masked_logits.clone(), 1);
        (probs, masked_logits)
    }

    pub fn sample_for_action(&self, probs: Tensor<B, 2, Float>) -> Tensor<B, 2, Int> {
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

    pub fn log_prob_from_distribution<const OutD: usize>(
        &self,
        logits: Tensor<B, OutD, Float>,
        act: Tensor<B, OutD, Int>,
    ) -> Tensor<B, OutD, Float> {
        let log_pmf = log_softmax(logits, 1);
        log_pmf.gather(1, act)
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> ForwardKernelTrait<B, InK, OutK> for DiscretePolicyNetwork<B, InK, OutK> {
    fn forward<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
        act: Option<Tensor<B, OutD, OutK>>,
    ) -> ForwardOutput<B, OutD> {
        let (probs, logits) = self.distribution(obs, mask);
        let logp_a = act.map(|a| self.log_prob_from_distribution(logits.clone(), a.int()));
        ForwardOutput::Discrete {
            probs,
            logits,
            logp_a,
        }
    }
}

pub struct ContinuousPolicyNetwork<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> where OutK: BasicOps<B> {
    pi_network: Mlp<B, InK>,
    log_std: Param<Tensor<B, 1, Float>>,
    pub input_dim: usize,
    pub output_dim: usize,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> ContinuousPolicyNetwork<B, InK, OutK> where OutK: BasicOps<B> {
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
        let log_std_tensor = Tensor::<B, 1, Float>::from_data(
            burn_tensor::TensorData::new(vec![-0.5f32; act_dim], [act_dim]),
            device,
        );
        Self {
            pi_network: Mlp::new(obs_dim, hidden_sizes, act_dim, ActivationKind::ReLU, device),
            log_std: Param::from_tensor(log_std_tensor),
            input_dim: obs_dim,
            output_dim: act_dim,
            _out_k: PhantomData::<OutK>::default(),
        }
    }

    pub fn distribution<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
    ) -> (Tensor<B, OutD, Float>, Tensor<B, 2, Float>) {
        let mean_raw = self.pi_network.forward(obs).reshape(mask.dims());
        let mean = mean_raw + (mask - 1.0f32) * 1e8f32;
        let std = self.log_std.val().exp().unsqueeze_dim::<2>(0);
        (mean, std)
    }

    pub fn sample_for_action<const OutD: usize>(
        &self,
        mean: Tensor<B, OutD, Float>,
        std: Tensor<B, 2, Float>,
    ) -> Tensor<B, OutD, Float> {
        let eps = Tensor::<B, OutD, Float>::random(
            mean.shape(),
            Distribution::Normal(0.0, 1.0),
            &mean.device(),
        );
        mean + std * eps
    }

    pub fn log_prob_from_distribution<const OutD: usize>(
        &self,
        mean: Tensor<B, OutD, Float>,
        std: Tensor<B, 2, Float>,
        act: Tensor<B, OutD, Float>,
    ) -> Tensor<B, OutD, Float> {
        let var = std.clone().powf_scalar(2.0f32);
        let squared_err = (act - mean).powf_scalar(2.0f32);
        let log_prob = -0.5f32
            * (squared_err / var + 2.0f32 * std.log() + (2.0f32 * core::f32::consts::PI).ln());
        log_prob.sum_dim(OutD - 1)
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> ForwardKernelTrait<B, InK, OutK> for ContinuousPolicyNetwork<B, InK, OutK> where OutK: BasicOps<B> {
    fn forward<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
        act: Option<Tensor<B, OutD, OutK>>,
    ) -> ForwardOutput<B, OutD> {
        let (mean, std) = self.distribution(obs, mask);
        let logp_a = act.map(|a| {
            let a_float: Tensor<B, OutD, Float> = Tensor::from_data(a.into_data().convert::<f32>(), &mean.device());
            self.log_prob_from_distribution(mean.clone(), std.clone(), a_float)
        });
        ForwardOutput::Continuous { mean, std, logp_a }
    }
}

pub struct BaselineValueNetwork<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    v_network: Mlp<B, InK>,
    _in_k: PhantomData<InK>,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> BaselineValueNetwork<B, InK, OutK> {
    pub fn new(
        obs_dim: usize,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        device: &B::Device,
    ) -> Self {
        Self {
            v_network: Mlp::new(obs_dim, hidden_sizes, 1, activation, device),
            _in_k: PhantomData::<InK>::default(),
            _out_k: PhantomData::<OutK>::default(),
        }
    }

    pub fn forward<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        _mask: Tensor<B, OutD, OutK>,
    ) -> Tensor<B, OutD, OutK> {
        let v = self.v_network.forward(obs);
        
        let device = v.device();
        let out_k_tensor: Tensor<B, OutD, OutK> = match OutK::name() {
            "Float" => Tensor::from_data(v.into_data().convert::<f32>(), &device),
            "Int" => Tensor::from_data(v.into_data().convert::<i32>(), &device),
            "Bool" => Tensor::from_data(v.into_data().convert::<bool>(), &device),
        };
        out_k_tensor
    }
}

enum PolicyHead<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> where OutK: BasicOps<B> {
    Discrete(DiscretePolicyNetwork<B, InK, OutK>),
    Continuous(ContinuousPolicyNetwork<B, InK, OutK>),
}

pub struct PolicyWithBaseline<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> where OutK: BasicOps<B> {
    pub policy: PolicyHead<B, InK, OutK>,
    pub baseline: BaselineValueNetwork<B, InK, OutK>,
    input_dim: usize,
    output_dim: usize,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> PolicyWithBaseline<B, InK, OutK> where OutK: BasicOps<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        device: &B::Device,
    ) -> Self {
        let policy = if discrete {
            PolicyHead::Discrete(DiscretePolicyNetwork::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                device,
            ))
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

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> StepKernelTrait<B, InK, OutK> for PolicyWithBaseline<B, InK, OutK> where OutK: BasicOps<B> {
    fn step<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        if obs.dims()[InD - 1] != self.input_dim || mask.dims()[OutD - 1] != self.output_dim {
            return Err(TensorError::ShapeError(format!(
                "Expected obs/mask trailing dims ({}, {}), got ({}, {})",
                self.input_dim,
                self.output_dim,
                obs.dims()[InD - 1],
                mask.dims()[OutD - 1]
            )));
        }

        let mut data = HashMap::new();
        match &self.policy {
            PolicyHead::Discrete(policy) => {
                let (probs, logits) = policy.distribution(obs.clone(), mask.clone());
                let probs_rank2 = probs.clone().reshape([probs.dims()[0], probs.dims()[OutD - 1]]);
                let act = policy.sample_for_action(probs_rank2);
                let act_for_log_prob = act.clone().reshape(logits.dims());
                let logp_a = policy.log_prob_from_distribution(logits, act_for_log_prob);
                let v = self.baseline.forward(obs.clone(), mask.clone());

                let device = v.device();
                let v_float: Tensor<B, OutD, Float> = Tensor::from_data(v.into_data().convert::<f32>(), &device);

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("val".to_string(), float_tensor_to_data(v_float)?);

                Ok((StepAction::Discrete(act), data))
            }
            PolicyHead::Continuous(policy) => {
                let (mean, std) = policy.distribution(obs.clone(), mask.clone());
                let act = policy.sample_for_action(mean.clone(), std.clone());
                let logp_a = policy.log_prob_from_distribution(mean, std, act.clone());
                let v = self.baseline.forward(obs.clone(), mask.clone());
                let act_for_step = act.clone().reshape([act.dims()[0], act.dims()[OutD - 1]]);

                let device = v.device();
                let v_float: Tensor<B, OutD, Float> = Tensor::from_data(v.into_data().convert::<f32>(), &device);

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("val".to_string(), float_tensor_to_data(v_float)?);

                Ok((StepAction::Continuous(act_for_step), data))
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

pub struct PolicyWithoutBaseline<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> where OutK: BasicOps<B> {
    pub policy: PolicyHead<B, InK, OutK>,
    input_dim: usize,
    output_dim: usize,
    _in_k: PhantomData<InK>,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> PolicyWithoutBaseline<B, InK, OutK> where OutK: BasicOps<B> {
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        device: &B::Device,
    ) -> Self {
        let policy = if discrete {
            PolicyHead::Discrete(DiscretePolicyNetwork::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                device,
            ))
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
            _in_k: PhantomData::<InK>::default(),
            _out_k: PhantomData::<OutK>::default(),
        }
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> StepKernelTrait<B, InK, OutK> for PolicyWithoutBaseline<B, InK, OutK> where OutK: BasicOps<B> {
    fn step<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        if obs.dims()[InD - 1] != self.input_dim || mask.dims()[OutD - 1] != self.output_dim {
            return Err(TensorError::ShapeError(format!(
                "Expected obs/mask trailing dims ({}, {}), got ({}, {})",
                self.input_dim,
                self.output_dim,
                obs.dims()[InD - 1],
                mask.dims()[OutD - 1]
            )));
        }

        let mut data = HashMap::new();
        match &self.policy {
            PolicyHead::Discrete(policy) => {
                let (probs, logits) = policy.distribution(obs, mask);
                let probs_rank2 = probs.clone().reshape([probs.dims()[0], probs.dims()[OutD - 1]]);
                let act = policy.sample_for_action(probs_rank2);
                let act_for_log_prob = act.clone().reshape(logits.dims());
                let logp_a = policy.log_prob_from_distribution(logits, act_for_log_prob);

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);

                Ok((StepAction::Discrete(act), data))
            }
            PolicyHead::Continuous(policy) => {
                let (mean, std) = policy.distribution(obs, mask);
                let act = policy.sample_for_action(mean.clone(), std.clone());
                let logp_a = policy.log_prob_from_distribution(mean, std, act.clone());
                let act_for_step = act.clone().reshape([act.dims()[0], act.dims()[OutD - 1]]);

                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);

                Ok((StepAction::Continuous(act_for_step), data))
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
