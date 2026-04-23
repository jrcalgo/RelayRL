#![allow(non_upper_case_globals)]

use crate::algorithms::REINFORCE::{
    ActivationKind, BaselineValueNetwork, ContinuousPolicyNetwork, DiscretePolicyNetwork,
};
use crate::templates::base_algorithm::{StepAction, StepKernelTrait};

use std::collections::HashMap;
use std::sync::Arc;

use burn_tensor::activation::log_softmax;
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Int, Tensor, TensorData as BurnTensorData, TensorKind};

use relayrl_types::data::tensor::{
    BackendMatcher, ConversionBurnTensor, DType, SupportedTensorBackend, TensorData, TensorError,
};

fn backend_f32_dtype<B: Backend + BackendMatcher>() -> Result<DType, TensorError> {
    match B::get_supported_backend() {
        #[cfg(feature = "ndarray-backend")]
        SupportedTensorBackend::NdArray => Ok(DType::NdArray(
            relayrl_types::data::tensor::NdArrayDType::F32,
        )),
        #[cfg(feature = "tch-backend")]
        SupportedTensorBackend::Tch => Ok(DType::Tch(relayrl_types::data::tensor::TchDType::F32)),
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

#[allow(clippy::large_enum_variant)]
enum PolicyHead<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>>
where
    OutK: BasicOps<B>,
{
    Discrete(DiscretePolicyNetwork<B, InK, OutK>),
    Continuous(ContinuousPolicyNetwork<B, InK, OutK>),
}

pub trait PPOKernelTrait<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>:
    StepKernelTrait<B, InK, OutK>
{
    /// Construct a correctly-shaped kernel for a new actor slot.
    ///
    /// Used instead of `Default::default()` when registering agents beyond the first,
    /// so each actor is initialised with the right `obs_dim` / `act_dim` rather than
    /// the placeholder dimensions that `Default` produces.
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self;

    fn ppo_pi_loss(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
        clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>);

    fn ppo_vf_loss(&mut self, obs: &[TensorData], mask: &[TensorData], ret: &[f32]) -> f32;
}

#[cfg(feature = "ndarray-backend")]
mod training {
    use super::*;

    extern crate burn_core as burn;

    use burn_autodiff::Autodiff;
    use burn_core::module::Module;
    use burn_ndarray::NdArray;
    use burn_nn::{Linear, LinearConfig, Relu};
    use burn_optim::adaptor::OptimizerAdaptor;
    use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer};

    pub type TB = Autodiff<NdArray>;

    #[derive(Module, Debug)]
    pub struct TrainMlp<B: burn_tensor::backend::Backend> {
        pub layers: Vec<Linear<B>>,
        pub relu: Relu,
        pub obs_dim: usize,
        pub act_dim: usize,
    }

    impl<B: burn_tensor::backend::Backend> TrainMlp<B> {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            device: &B::Device,
        ) -> Self {
            let mut dims = vec![obs_dim];
            dims.extend_from_slice(hidden_sizes);
            dims.push(act_dim);
            let layers = dims
                .windows(2)
                .map(|window| LinearConfig::new(window[0], window[1]).init(device))
                .collect();

            Self {
                layers,
                relu: Relu::new(),
                obs_dim,
                act_dim,
            }
        }

        pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = input;
            for (index, layer) in self.layers.iter().enumerate() {
                x = layer.forward(x);
                if index < self.layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }
    }

    pub struct DiscretePpoTrainer {
        pub network: Option<TrainMlp<TB>>,
        pub optimizer: OptimizerAdaptor<Adam, TrainMlp<TB>, TB>,
        pub pi_lr: f64,
    }

    impl DiscretePpoTrainer {
        pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, pi_lr: f64) -> Self {
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let network = TrainMlp::new(obs_dim, hidden_sizes, act_dim, &device);
            let optimizer = AdamConfig::new().init::<TB, TrainMlp<TB>>();

            Self {
                network: Some(network),
                optimizer,
                pi_lr,
            }
        }

        pub fn train_step(
            &mut self,
            obs_tensors: &[TensorData],
            act_tensors: &[TensorData],
            adv: &[f32],
            logp_old_tensors: &[TensorData],
            clip_ratio: f32,
        ) -> (f32, HashMap<String, f32>) {
            let n = obs_tensors
                .len()
                .min(act_tensors.len())
                .min(adv.len())
                .min(logp_old_tensors.len());
            if n == 0 {
                return zero_pi_info();
            }

            let net = match self.network.take() {
                Some(network) => network,
                None => return zero_pi_info(),
            };

            let obs_dim = net.obs_dim;
            let device = <TB as burn_tensor::backend::Backend>::Device::default();

            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat(&obs_tensors[..n]), [n, obs_dim]),
                &device,
            );
            let logits = net.forward(obs);
            let log_probs = log_softmax(logits, 1);
            let act = Tensor::<TB, 2, Int>::from_data(
                BurnTensorData::new(action_indices(&act_tensors[..n]), [n, 1]),
                &device,
            );
            let logp = log_probs.gather(1, act).reshape([n]);
            let adv_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(adv[..n].to_vec(), [n]),
                &device,
            );
            let logp_old_values = scalar_tensor_values(&logp_old_tensors[..n]);
            let logp_old_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(logp_old_values.clone(), [n]),
                &device,
            );
            let ratio = (logp.clone() - logp_old_tensor).exp();
            let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);
            let unclipped = ratio.clone() * adv_tensor.clone();
            let clipped = clipped_ratio * adv_tensor;
            let loss = unclipped.min_pair(clipped).mean().neg();

            let logp_values = logp
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; n]);
            let ratio_values = ratio
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![1.0; n]);

            let entropy = -logp_values.iter().sum::<f32>() / n as f32;
            let approx_kl = logp_old_values
                .iter()
                .zip(logp_values.iter())
                .map(|(old, new)| old - new)
                .sum::<f32>()
                / n as f32;
            let clipfrac = ratio_values
                .iter()
                .filter(|ratio| (**ratio - 1.0).abs() > clip_ratio)
                .count() as f32
                / n as f32;
            let loss_value = scalar_from_loss(&loss);

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &net);
            let net = self.optimizer.step(self.pi_lr, net, grads_params);
            self.network = Some(net);

            let mut info = HashMap::new();
            info.insert("kl".to_string(), approx_kl);
            info.insert("entropy".to_string(), entropy);
            info.insert("clipfrac".to_string(), clipfrac);
            (loss_value, info)
        }
    }

    pub struct VfTrainer {
        pub network: Option<TrainMlp<TB>>,
        pub optimizer: OptimizerAdaptor<Adam, TrainMlp<TB>, TB>,
        pub vf_lr: f64,
    }

    impl VfTrainer {
        pub fn new(obs_dim: usize, hidden_sizes: &[usize], vf_lr: f64) -> Self {
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let network = TrainMlp::new(obs_dim, hidden_sizes, 1, &device);
            let optimizer = AdamConfig::new().init::<TB, TrainMlp<TB>>();

            Self {
                network: Some(network),
                optimizer,
                vf_lr,
            }
        }

        pub fn train_step(&mut self, obs_tensors: &[TensorData], ret: &[f32]) -> f32 {
            let n = obs_tensors.len().min(ret.len());
            if n == 0 {
                return 0.0;
            }

            let net = match self.network.take() {
                Some(network) => network,
                None => return 0.0,
            };

            let obs_dim = net.obs_dim;
            let device = <TB as burn_tensor::backend::Backend>::Device::default();

            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat(&obs_tensors[..n]), [n, obs_dim]),
                &device,
            );
            let v_pred = net.forward(obs).reshape([n]);
            let ret_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(ret[..n].to_vec(), [n]),
                &device,
            );
            let loss = (v_pred - ret_tensor).powf_scalar(2.0).mean();
            let loss_value = scalar_from_loss(&loss);

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &net);
            let net = self.optimizer.step(self.vf_lr, net, grads_params);
            self.network = Some(net);

            loss_value
        }
    }

    fn zero_pi_info() -> (f32, HashMap<String, f32>) {
        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        info.insert("clipfrac".to_string(), 0.0);
        (0.0, info)
    }

    fn scalar_from_loss(loss: &Tensor<TB, 1, Float>) -> f32 {
        loss.clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0])[0]
    }

    fn obs_flat(obs_tensors: &[TensorData]) -> Vec<f32> {
        obs_tensors
            .iter()
            .flat_map(|tensor| bytemuck::cast_slice::<u8, f32>(&tensor.data).to_vec())
            .collect()
    }

    fn action_indices(act_tensors: &[TensorData]) -> Vec<i64> {
        act_tensors
            .iter()
            .map(|tensor| {
                if tensor.data.len() >= 8 {
                    bytemuck::cast_slice::<u8, i64>(&tensor.data[..8])
                        .first()
                        .copied()
                        .unwrap_or(0)
                } else if tensor.data.len() >= 4 {
                    bytemuck::cast_slice::<u8, f32>(&tensor.data[..4])
                        .first()
                        .copied()
                        .unwrap_or(0.0) as i64
                } else {
                    0
                }
            })
            .collect()
    }

    fn scalar_tensor_values(tensors: &[TensorData]) -> Vec<f32> {
        tensors
            .iter()
            .map(|tensor| {
                bytemuck::cast_slice::<u8, f32>(&tensor.data)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

pub struct PPOPolicyWithBaseline<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
> where
    OutK: BasicOps<B>,
    InK: BasicOps<B>,
{
    policy: PolicyHead<B, InK, OutK>,
    pub baseline: BaselineValueNetwork<B, InK, OutK>,
    input_dim: usize,
    output_dim: usize,
    #[cfg(feature = "ndarray-backend")]
    pi_trainer: Option<training::DiscretePpoTrainer>,
    #[cfg(feature = "ndarray-backend")]
    vf_trainer: Option<training::VfTrainer>,
}

pub type DefaultPPOKernel<B, InK, OutK> = PPOPolicyWithBaseline<B, InK, OutK>;
pub type PPOKernel<B, InK, OutK> = PPOPolicyWithBaseline<B, InK, OutK>;
pub type PlaceholderPPOKernel<B, InK, OutK> = PPOPolicyWithBaseline<B, InK, OutK>;

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        pi_lr: f64,
        vf_lr: f64,
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

        #[cfg(feature = "ndarray-backend")]
        let pi_trainer = if discrete {
            Some(training::DiscretePpoTrainer::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                pi_lr,
            ))
        } else {
            None
        };

        #[cfg(feature = "ndarray-backend")]
        let vf_trainer = Some(training::VfTrainer::new(obs_dim, hidden_sizes, vf_lr));

        Self {
            policy,
            baseline,
            input_dim: obs_dim,
            output_dim: act_dim,
            #[cfg(feature = "ndarray-backend")]
            pi_trainer,
            #[cfg(feature = "ndarray-backend")]
            vf_trainer,
        }
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> Default
    for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: TensorKind<B> + BasicOps<B>,
    OutK: TensorKind<B> + BasicOps<B>,
{
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(
            1,
            1,
            true,
            &[64, 64],
            ActivationKind::ReLU,
            3e-4,
            1e-3,
            &device,
        )
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn step<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, InK>,
        mask: Tensor<B, OUT_D, OutK>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        if obs.dims()[IN_D - 1] != self.input_dim || mask.dims()[OUT_D - 1] != self.output_dim {
            return Err(TensorError::ShapeError(format!(
                "Expected obs/mask trailing dims ({}, {}), got ({}, {})",
                self.input_dim,
                self.output_dim,
                obs.dims()[IN_D - 1],
                mask.dims()[OUT_D - 1]
            )));
        }

        let mut data = HashMap::new();
        match &self.policy {
            PolicyHead::Discrete(policy) => {
                let (probs, logits) = policy.distribution(obs.clone(), mask.clone());
                let probs_rank2 = probs
                    .clone()
                    .reshape([probs.dims()[0], probs.dims()[OUT_D - 1]]);
                let act = policy.sample_for_action(probs_rank2);
                let act_for_log_prob = act.clone().reshape(logits.dims());
                let logp_a = policy.log_prob_from_distribution(logits, act_for_log_prob);
                let v = self.baseline.forward(obs.clone(), mask.clone());
                let device = v.device();
                let v_float: Tensor<B, OUT_D, Float> =
                    Tensor::from_data(v.into_data().convert::<f32>(), &device);
                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("val".to_string(), float_tensor_to_data(v_float)?);
                Ok((StepAction::Discrete(act), data))
            }
            PolicyHead::Continuous(policy) => {
                let (mean, std) = policy.distribution(obs.clone(), mask.clone());
                let act = policy.sample_for_action(mean.clone(), std.clone());
                let logp_a = policy.log_prob_from_distribution(mean, std, act.clone());
                let v = self.baseline.forward(obs.clone(), mask.clone());
                let act_for_step = act.clone().reshape([act.dims()[0], act.dims()[OUT_D - 1]]);
                let device = v.device();
                let v_float: Tensor<B, OUT_D, Float> =
                    Tensor::from_data(v.into_data().convert::<f32>(), &device);
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

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PPOKernelTrait<B, InK, OutK> for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self {
        let device = B::Device::default();
        Self::new(
            obs_dim,
            act_dim,
            true,
            &[64, 64],
            ActivationKind::ReLU,
            3e-4,
            1e-3,
            &device,
        )
    }

    fn ppo_pi_loss(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        _mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
        clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>) {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.pi_trainer {
            return trainer.train_step(obs, act, adv, logp_old, clip_ratio);
        }

        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        info.insert("clipfrac".to_string(), 0.0);
        (0.0, info)
    }

    fn ppo_vf_loss(&mut self, obs: &[TensorData], _mask: &[TensorData], ret: &[f32]) -> f32 {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.vf_trainer {
            return trainer.train_step(obs, ret);
        }

        0.0
    }
}

#[cfg(feature = "ndarray-backend")]
impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    crate::templates::base_algorithm::WeightProvider for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    /// Extract per-layer weight specs from the discrete policy trainer network.
    ///
    /// Returns `None` if no training has happened yet (the trainer or its network is
    /// absent) or if this kernel uses a continuous policy (not yet supported).
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        let trainer = self.pi_trainer.as_ref()?;
        let network = trainer.network.as_ref()?;

        let mut specs = Vec::new();
        for layer in &network.layers {
            let w = layer.weight.val();
            let dims = w.dims();
            let in_dim = dims[0];
            let out_dim = dims[1];
            let weights: Vec<f32> = w.into_data().to_vec::<f32>().unwrap_or_default();
            let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                bias_param
                    .val()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
            } else {
                vec![0.0; out_dim]
            };
            specs.push((in_dim, out_dim, weights, biases));
        }
        Some(specs)
    }
}
