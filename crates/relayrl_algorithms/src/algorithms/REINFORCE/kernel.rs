#![allow(non_upper_case_globals)]

use crate::templates::base_algorithm::{
    ForwardKernelTrait, ForwardOutput, StepAction, StepKernelTrait, TrainableKernelTrait,
};
use std::collections::HashMap;
use std::sync::Arc;

use burn_nn::{Linear, LinearConfig, Relu, Tanh};
use burn_tensor::Distribution;
use burn_tensor::activation::{log_softmax, softmax};
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Int, Tensor, TensorData as BurnTensorData, TensorKind};
use rand::distr::Distribution as RandDistribution;
use rand::distr::weighted::WeightedIndex;
use std::marker::PhantomData;

use relayrl_types::data::tensor::{
    BackendMatcher, ConversionBurnTensor, DType, SupportedTensorBackend, TensorData, TensorError,
};

#[derive(Clone, Copy, Debug, Default)]
pub enum ActivationKind {
    #[default]
    ReLU,
    Tanh,
}

/// Multi-layer perceptron.
pub struct Mlp<B: Backend, InK: TensorKind<B>> {
    layers: Vec<Linear<B>>,
    relu: Relu,
    tanh: Tanh,
    activation: ActivationKind,
    _in_k: PhantomData<InK>,
}

impl<B: Backend, InK: TensorKind<B>> Mlp<B, InK> {
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

        let layers = dims
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();

        Self {
            layers,
            relu: Relu::new(),
            tanh: Tanh::new(),
            activation,
            _in_k: PhantomData,
        }
    }

    /// Inference forward from a generic InK tensor (converts to Float).
    pub fn forward<const D: usize>(&self, input: Tensor<B, D, InK>) -> Tensor<B, D, Float>
    where
        InK: BasicOps<B>,
    {
        let device = input.device();
        let x: Tensor<B, D, Float> = Tensor::from_data(input.into_data().convert::<f32>(), &device);
        self.forward_float(x)
    }

    /// Float forward — preserves autodiff graph when B: AutodiffBackend.
    pub fn forward_float<const D: usize>(&self, input: Tensor<B, D, Float>) -> Tensor<B, D, Float> {
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

pub struct DiscretePolicyNetwork<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>> {
    pi_network: Mlp<B, InK>,
    pub input_dim: usize,
    pub output_dim: usize,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>> DiscretePolicyNetwork<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
        Self {
            pi_network: Mlp::new(obs_dim, hidden_sizes, act_dim, ActivationKind::ReLU, device),
            input_dim: obs_dim,
            output_dim: act_dim,
            _out_k: PhantomData,
        }
    }

    pub fn distribution<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
    ) -> (Tensor<B, OutD, Float>, Tensor<B, OutD, Float>) {
        let logits_raw = self.pi_network.forward(obs).reshape(mask.dims());
        let mask_f: Tensor<B, OutD, Float> =
            Tensor::from_data(mask.into_data().convert::<f32>(), &logits_raw.device());
        let masked_logits = logits_raw + (mask_f - 1.0f32) * 1e8f32;
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
            BurnTensorData::new(sampled, [batch_size, 1]),
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

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    ForwardKernelTrait<B, InK, OutK> for DiscretePolicyNetwork<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn forward<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
        act: Option<Tensor<B, OutD, OutK>>,
    ) -> ForwardOutput<B, OutD> {
        let (probs, logits) = self.distribution(obs, mask);
        let logp_a = act.map(|a| {
            let a_int: Tensor<B, OutD, Int> =
                Tensor::from_data(a.into_data().convert::<i32>(), &logits.device());
            self.log_prob_from_distribution(logits.clone(), a_int)
        });
        ForwardOutput::Discrete {
            probs,
            logits,
            logp_a,
        }
    }
}

pub struct ContinuousPolicyNetwork<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>>
where
    OutK: BasicOps<B>,
{
    pi_network: Mlp<B, InK>,
    log_std: Tensor<B, 1, Float>,
    pub input_dim: usize,
    pub output_dim: usize,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>> ContinuousPolicyNetwork<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
        // SpinningUp core.py: log_std = -0.5 * np.ones(act_dim)
        let log_std = Tensor::<B, 1, Float>::from_data(
            BurnTensorData::new(vec![-0.5f32; act_dim], [act_dim]),
            device,
        );
        Self {
            pi_network: Mlp::new(obs_dim, hidden_sizes, act_dim, ActivationKind::ReLU, device),
            log_std,
            input_dim: obs_dim,
            output_dim: act_dim,
            _out_k: PhantomData,
        }
    }

    pub fn distribution<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
    ) -> (Tensor<B, OutD, Float>, Tensor<B, 2, Float>) {
        let mean_raw = self.pi_network.forward(obs).reshape(mask.dims());
        let mask_f: Tensor<B, OutD, Float> =
            Tensor::from_data(mask.into_data().convert::<f32>(), &mean_raw.device());
        let mean = mean_raw + (mask_f - 1.0f32) * 1e8f32;
        let std = self.log_std.clone().exp().unsqueeze_dim::<2>(0);
        (mean, std)
    }

    pub fn sample_for_action<const OutD: usize>(
        &self,
        mean: Tensor<B, OutD, Float>,
        std: Tensor<B, 2, Float>,
    ) -> Tensor<B, OutD, Float> {
        let mean_dims = mean.dims();
        let batch_size = mean_dims[0];
        let act_dim = mean_dims[OutD - 1];

        let epsilon = Tensor::<B, OutD, Float>::random(
            mean.shape(),
            Distribution::Normal(0.0, 1.0),
            &mean.device(),
        );

        let std_rank2 = std.reshape([batch_size, act_dim]);
        let std_broadcast =
            Tensor::from_data(std_rank2.into_data().convert::<f32>(), &mean.device());

        // Reparameterization: a = mean + std * ε
        mean.add(std_broadcast.mul(epsilon))
    }

    pub fn log_prob_from_distribution<const OutD: usize>(
        &self,
        mean: Tensor<B, OutD, Float>,
        std: Tensor<B, 2, Float>,
        act: Tensor<B, OutD, Float>,
    ) -> Tensor<B, OutD, Float> {
        let mean_dims = mean.dims();
        let batch_size = mean_dims[0];
        let act_dim = mean_dims[OutD - 1];

        let std_rank2 = std.reshape([batch_size, act_dim]);
        let std_broadcast: Tensor<B, OutD, Float> =
            Tensor::from_data(std_rank2.into_data().convert::<f32>(), &mean.device());

        let variance = std_broadcast.clone().powf_scalar(2.0);
        let squared_error = (act - mean).powf_scalar(2.0);

        let log_prob = -0.5f32
            * (squared_error / variance
                + 2.0f32 * std_broadcast.log()
                + (2.0f32 * core::f32::consts::PI).ln());

        log_prob.sum_dim(OutD - 1).unsqueeze_dim::<OutD>(OutD - 1)
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    ForwardKernelTrait<B, InK, OutK> for ContinuousPolicyNetwork<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn forward<const InD: usize, const OutD: usize>(
        &self,
        obs: Tensor<B, InD, InK>,
        mask: Tensor<B, OutD, OutK>,
        act: Option<Tensor<B, OutD, OutK>>,
    ) -> ForwardOutput<B, OutD> {
        let (mean, std) = self.distribution(obs, mask);
        let logp_a = act.map(|a| {
            let a_float: Tensor<B, OutD, Float> =
                Tensor::from_data(a.into_data().convert::<f32>(), &mean.device());
            self.log_prob_from_distribution(mean.clone(), std.clone(), a_float)
        });
        ForwardOutput::Continuous { mean, std, logp_a }
    }
}

pub struct BaselineValueNetwork<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>> {
    v_network: Mlp<B, InK>,
    _in_k: PhantomData<InK>,
    _out_k: PhantomData<OutK>,
}

impl<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>> BaselineValueNetwork<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(
        obs_dim: usize,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        device: &B::Device,
    ) -> Self {
        Self {
            v_network: Mlp::new(obs_dim, hidden_sizes, 1, activation, device),
            _in_k: PhantomData,
            _out_k: PhantomData,
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
            _ => Tensor::from_data(v.into_data().convert::<f32>(), &device),
        };
        out_k_tensor
    }
}

#[allow(clippy::large_enum_variant)]
enum PolicyHead<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>>
where
    OutK: BasicOps<B>,
{
    Discrete(DiscretePolicyNetwork<B, InK, OutK>),
    Continuous(ContinuousPolicyNetwork<B, InK, OutK>),
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
                .map(|w| LinearConfig::new(w[0], w[1]).init(device))
                .collect();
            Self {
                layers,
                relu: Relu::new(),
                obs_dim,
                act_dim,
            }
        }

        /// Float forward pass (used for both inference and training).
        pub fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = x;
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }
    }

    /// Discrete policy trainer.
    /// Owns a `TrainMlp<TB>` (stored in Option for the ownership-transfer optimizer pattern)
    /// and an Adam optimizer adaptor.
    pub struct DiscretePiTrainer {
        pub network: Option<TrainMlp<TB>>,
        pub optimizer: OptimizerAdaptor<Adam, TrainMlp<TB>, TB>,
        pub pi_lr: f64,
    }

    impl DiscretePiTrainer {
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

        /// REINFORCE policy gradient step: loss_pi = -mean(logp * adv)
        pub fn train_step(
            &mut self,
            obs_tensors: &[TensorData],
            act_tensors: &[TensorData],
            adv: &[f32],
            logp_old_tensors: &[TensorData],
        ) -> (f32, HashMap<String, f32>) {
            let n = obs_tensors.len().min(act_tensors.len()).min(adv.len());
            if n == 0 {
                return (
                    0.0,
                    HashMap::from([("kl".to_string(), 0.0), ("entropy".to_string(), 0.0)]),
                );
            }

            let net = match self.network.take() {
                Some(n) => n,
                None => {
                    return (
                        0.0,
                        HashMap::from([("kl".to_string(), 0.0), ("entropy".to_string(), 0.0)]),
                    );
                }
            };

            let obs_dim = net.obs_dim;
            let act_dim = net.act_dim;
            let device = <TB as burn_tensor::backend::Backend>::Device::default();

            // Build obs [n, obs_dim]
            let obs_flat: Vec<f32> = obs_tensors[..n]
                .iter()
                .flat_map(|td| bytemuck::cast_slice::<u8, f32>(&td.data).to_vec())
                .collect();
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat, [n, obs_dim]),
                &device,
            );

            // Forward: logits [n, act_dim]
            let logits = net.forward(obs);

            // log_softmax → log probs [n, act_dim]
            let log_probs = log_softmax(logits, 1);

            // Build action indices [n, 1] (i64)
            let act_flat: Vec<i64> = act_tensors[..n]
                .iter()
                .map(|td| {
                    if td.data.len() >= 8 {
                        bytemuck::cast_slice::<u8, i64>(&td.data[..8])
                            .first()
                            .copied()
                            .unwrap_or(0)
                    } else if td.data.len() >= 4 {
                        bytemuck::cast_slice::<u8, f32>(&td.data[..4])
                            .first()
                            .copied()
                            .unwrap_or(0.0) as i64
                    } else {
                        0
                    }
                })
                .collect();
            let act =
                Tensor::<TB, 2, Int>::from_data(BurnTensorData::new(act_flat, [n, 1]), &device);

            // Gather log probs at chosen actions [n, 1]
            let logp = log_probs.gather(1, act); // [n, 1]
            let logp_1d = logp.reshape([n]); // [n]

            let adv_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(adv[..n].to_vec(), [n]),
                &device,
            );

            // REINFORCE loss: -(logp * adv).mean()
            let loss = (logp_1d.clone() * adv_tensor).mean().neg();

            // Scalar metrics (before backward — clone data out)
            let logp_data: Vec<f32> = logp_1d
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; n]);
            let entropy = -logp_data.iter().sum::<f32>() / n as f32;

            let approx_kl = if !logp_old_tensors.is_empty() {
                let logp_old_vals: Vec<f32> = logp_old_tensors[..n]
                    .iter()
                    .map(|td| {
                        bytemuck::cast_slice::<u8, f32>(&td.data)
                            .first()
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .collect();
                logp_old_vals
                    .iter()
                    .zip(logp_data.iter())
                    .map(|(lp_old, lp_new)| lp_old - lp_new)
                    .sum::<f32>()
                    / n as f32
            } else {
                0.0
            };

            let loss_val = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or(vec![0.0])[0];

            // Backward + optimizer step
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &net);
            let net = self.optimizer.step(self.pi_lr, net, grads_params);
            self.network = Some(net);

            let mut info = HashMap::new();
            info.insert("kl".to_string(), approx_kl);
            info.insert("entropy".to_string(), entropy);
            (loss_val, info)
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
            let network = TrainMlp::new(obs_dim, hidden_sizes, 1, &device); // output_dim=1 for V(s)
            let optimizer = AdamConfig::new().init::<TB, TrainMlp<TB>>();
            Self {
                network: Some(network),
                optimizer,
                vf_lr,
            }
        }

        /// Value function MSE loss: mean((V(obs) - ret)^2)
        pub fn train_step(&mut self, obs_tensors: &[TensorData], ret: &[f32]) -> f32 {
            let n = obs_tensors.len().min(ret.len());
            if n == 0 {
                return 0.0;
            }

            let net = match self.network.take() {
                Some(n) => n,
                None => return 0.0,
            };

            let obs_dim = net.obs_dim;
            let device = <TB as burn_tensor::backend::Backend>::Device::default();

            let obs_flat: Vec<f32> = obs_tensors[..n]
                .iter()
                .flat_map(|td| bytemuck::cast_slice::<u8, f32>(&td.data).to_vec())
                .collect();
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat, [n, obs_dim]),
                &device,
            );

            let v_pred = net.forward(obs); // [n, 1]
            let v_pred_1d = v_pred.reshape([n]);

            let ret_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(ret[..n].to_vec(), [n]),
                &device,
            );

            let loss = (v_pred_1d - ret_tensor).powf_scalar(2.0).mean();
            let loss_val = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or(vec![0.0])[0];

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &net);
            let net = self.optimizer.step(self.vf_lr, net, grads_params);
            self.network = Some(net);

            loss_val
        }
    }
}

pub struct PolicyWithBaseline<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
where
    OutK: BasicOps<B>,
    InK: BasicOps<B>,
{
    policy: PolicyHead<B, InK, OutK>,
    pub baseline: BaselineValueNetwork<B, InK, OutK>,
    input_dim: usize,
    output_dim: usize,
    with_vf_baseline: bool,
    #[cfg(feature = "ndarray-backend")]
    pi_trainer: Option<training::DiscretePiTrainer>,
    #[cfg(feature = "ndarray-backend")]
    vf_trainer: Option<training::VfTrainer>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PolicyWithBaseline<B, InK, OutK>
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
        with_vf_baseline: bool,
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
            Some(training::DiscretePiTrainer::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                pi_lr,
            ))
        } else {
            None // Continuous trainer uses a separate path (not yet implemented)
        };

        #[cfg(feature = "ndarray-backend")]
        let vf_trainer = if with_vf_baseline {
            Some(training::VfTrainer::new(obs_dim, hidden_sizes, vf_lr))
        } else {
            None
        };

        Self {
            policy,
            baseline,
            input_dim: obs_dim,
            output_dim: act_dim,
            with_vf_baseline,
            #[cfg(feature = "ndarray-backend")]
            pi_trainer,
            #[cfg(feature = "ndarray-backend")]
            vf_trainer,
        }
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for PolicyWithBaseline<B, InK, OutK>
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

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> TrainableKernelTrait
    for PolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn train_pi_step(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        _mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
    ) -> (f32, HashMap<String, f32>) {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.pi_trainer {
            return trainer.train_step(obs, act, adv, logp_old);
        }

        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        (0.0, info)
    }

    fn train_vf_step(&mut self, obs: &[TensorData], _mask: &[TensorData], ret: &[f32]) -> f32 {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.vf_trainer {
            return trainer.train_step(obs, ret);
        }
        0.0
    }
}

pub struct PolicyWithoutBaseline<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
> where
    OutK: BasicOps<B>,
{
    policy: PolicyHead<B, InK, OutK>,
    input_dim: usize,
    output_dim: usize,
    _in_k: PhantomData<InK>,
    _out_k: PhantomData<OutK>,
    #[cfg(feature = "ndarray-backend")]
    pi_trainer: Option<training::DiscretePiTrainer>,
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PolicyWithoutBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        pi_lr: f64,
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

        #[cfg(feature = "ndarray-backend")]
        let pi_trainer = if discrete {
            Some(training::DiscretePiTrainer::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                pi_lr,
            ))
        } else {
            None
        };

        Self {
            policy,
            input_dim: obs_dim,
            output_dim: act_dim,
            _in_k: PhantomData,
            _out_k: PhantomData,
            #[cfg(feature = "ndarray-backend")]
            pi_trainer,
        }
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for PolicyWithoutBaseline<B, InK, OutK>
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
                let (probs, logits) = policy.distribution(obs, mask);
                let probs_rank2 = probs
                    .clone()
                    .reshape([probs.dims()[0], probs.dims()[OUT_D - 1]]);
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
                let act_for_step = act.clone().reshape([act.dims()[0], act.dims()[OUT_D - 1]]);
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

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> TrainableKernelTrait
    for PolicyWithoutBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn train_pi_step(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        _mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
    ) -> (f32, HashMap<String, f32>) {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.pi_trainer {
            return trainer.train_step(obs, act, adv, logp_old);
        }
        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        (0.0, info)
    }

    fn train_vf_step(&mut self, _obs: &[TensorData], _mask: &[TensorData], _ret: &[f32]) -> f32 {
        0.0 // no value function
    }
}
