use crate::algorithms::{
    GenericMlp, LayerSpecs, NeuralNetwork, NeuralNetworkError, NeuralNetworkSpec, ValueFunction,
};
use crate::algorithms::{convert_byte_dtype_to_f32, convert_byte_dtype_to_i64};

use burn_tensor::TensorData as BurnTensorData;
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Int, Tensor, TensorKind};
use rand::RngExt;
use rand_distr::Distribution;
use rayon::prelude::*;
use relayrl_types::data::tensor::NdArrayDType;
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::data::tensor::{DType, TensorData};
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use std::{collections::HashMap, sync::Arc};

// ---- training module  ----

pub(crate) mod training {
    use super::*;

    extern crate burn_core as burn;

    use burn_autodiff::Autodiff;
    use burn_core::module::Initializer;
    use burn_core::module::Module;
    use burn_nn::{Linear, LinearConfig, Relu};
    use burn_optim::adaptor::OptimizerAdaptor;
    use burn_optim::grad_clipping::GradientClipping;
    use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer};
    use burn_tensor::activation::log_softmax;

    #[cfg(feature = "tch-backend")]
    use burn_tch::LibTorch;
    #[cfg(feature = "tch-backend")]
    pub type TB = Autodiff<LibTorch>;

    #[cfg(not(feature = "tch-backend"))]
    use burn_ndarray::NdArray;
    #[cfg(not(feature = "tch-backend"))]
    pub type TB = Autodiff<NdArray>;

    /// Separate pi and vf layer stacks. Trained with one shared Adam optimizer.
    #[derive(Module, Debug)]
    pub struct ActorCriticMlp<B: burn_tensor::backend::Backend> {
        pub pi_layers: Vec<Linear<B>>,
        pub vf_layers: Vec<Linear<B>>,
        pub relu: Relu,
        pub obs_dim: usize,
        pub act_dim: usize,
    }

    impl<B: burn_tensor::backend::Backend> ActorCriticMlp<B> {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            device: &B::Device,
        ) -> Self {
            let mut pi_dims = vec![obs_dim];
            pi_dims.extend_from_slice(hidden_sizes);
            pi_dims.push(act_dim);
            let pi_n = pi_dims.len() - 1;
            let pi_layers = pi_dims
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let gain = if i < pi_n - 1 { 2.0f64.sqrt() } else { 0.01 };
                    let mut layer = LinearConfig::new(w[0], w[1])
                        .with_initializer(Initializer::Zeros)
                        .init(device);
                    layer.weight = Initializer::Orthogonal { gain }.init_with(
                        [w[0], w[1]],
                        Some(w[0]),
                        Some(w[1]),
                        device,
                    );
                    layer
                })
                .collect();

            let mut vf_dims = vec![obs_dim];
            vf_dims.extend_from_slice(hidden_sizes);
            vf_dims.push(1);
            let vf_n = vf_dims.len() - 1;
            let vf_layers = vf_dims
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let gain = if i < vf_n - 1 { 2.0f64.sqrt() } else { 1.0 };
                    let mut layer = LinearConfig::new(w[0], w[1])
                        .with_initializer(Initializer::Zeros)
                        .init(device);
                    layer.weight = Initializer::Orthogonal { gain }.init_with(
                        [w[0], w[1]],
                        Some(w[0]),
                        Some(w[1]),
                        device,
                    );
                    layer
                })
                .collect();

            Self {
                pi_layers,
                vf_layers,
                relu: Relu::new(),
                obs_dim,
                act_dim,
            }
        }

        pub fn pi_forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = input;
            for (i, layer) in self.pi_layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.pi_layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }

        pub fn vf_forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = input;
            for (i, layer) in self.vf_layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.vf_layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }
    }

    pub struct PPOActorCriticTrainer {
        pub network: Option<ActorCriticMlp<TB>>,
        pub optimizer: OptimizerAdaptor<Adam, ActorCriticMlp<TB>, TB>,
        pub lr: f64,
        pub vf_coef: f32,
        pub lr_schedule_steps: Option<u64>,
        pub grad_step_count: u64,
    }

    impl PPOActorCriticTrainer {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            lr: f64,
            vf_coef: f32,
            lr_schedule_steps: Option<u64>,
        ) -> Self {
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let network = ActorCriticMlp::new(obs_dim, hidden_sizes, act_dim, &device);
            let optimizer = AdamConfig::new()
                .init::<TB, ActorCriticMlp<TB>>()
                .with_grad_clipping(GradientClipping::Norm(4.0));
            Self {
                network: Some(network),
                optimizer,
                lr,
                vf_coef,
                lr_schedule_steps,
                grad_step_count: 0,
            }
        }

        pub fn effective_lr(&self) -> f64 {
            match self.lr_schedule_steps {
                Some(total) if total > 0 => {
                    let frac = 1.0 - (self.grad_step_count as f64 / total as f64).min(1.0);
                    self.lr * frac.max(0.0)
                }
                _ => self.lr,
            }
        }

        /// Combined pi+vf forward+backward. `act_flat` is i64 action indices. Returns (pi_loss, vf_loss, stats).
        #[allow(clippy::too_many_arguments)]
        pub fn train_step_discrete(
            &mut self,
            obs_flat: &[f32],
            obs_dim: usize,
            act_flat: &[i64],
            adv: &[f32],
            logp_old: &[f32],
            ret: &[f32],
            clip_ratio: f32,
            ent_coef: f32,
            compute_stats: bool,
        ) -> (f32, f32, HashMap<String, f32>) {
            let n = (obs_flat.len() / obs_dim.max(1))
                .min(act_flat.len())
                .min(adv.len())
                .min(logp_old.len())
                .min(ret.len());
            if n == 0 {
                return (0.0, 0.0, zero_pi_info().1);
            }
            let net = match self.network.take() {
                Some(net) => net,
                None => return (0.0, 0.0, zero_pi_info().1),
            };
            let device = <TB as burn_tensor::backend::Backend>::Device::default();

            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat[..n * obs_dim].to_vec(), [n, obs_dim]),
                &device,
            );

            // ── Policy head ──────────────────────────────────────────────
            let logits = net.pi_forward(obs.clone());
            let log_probs_full = log_softmax(logits, 1);
            let act = Tensor::<TB, 2, Int>::from_data(
                BurnTensorData::new(act_flat[..n].to_vec(), [n, 1]),
                &device,
            );
            let logp = log_probs_full.clone().gather(1, act).reshape([n]);
            let adv_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(adv[..n].to_vec(), [n]),
                &device,
            );
            let logp_old_tensor = Tensor::<TB, 1, Float>::from_data(
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
            let ret_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(ret[..n].to_vec(), [n]),
                &device,
            );
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

        /// Value-only forward (no grad). Used for deferred GAE.
        pub fn value_forward_flat(&self, obs_flat: &[f32], obs_dim: usize) -> Vec<f32> {
            #[allow(clippy::manual_checked_ops)]
            let n = if obs_dim > 0 {
                obs_flat.len() / obs_dim
            } else {
                0
            };
            if n == 0 {
                return Vec::new();
            }
            let net = match self.network.as_ref() {
                Some(net) => net,
                None => return vec![0.0; n],
            };
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat[..n * obs_dim].to_vec(), [n, obs_dim]),
                &device,
            );
            let v = net.vf_forward(obs);
            v.into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; n])
        }

        /// Log-probs for (obs, discrete act). Used for logp_old recompute.
        pub fn logprobs_flat(
            &self,
            obs_flat: &[f32],
            obs_dim: usize,
            act_flat: &[i64],
        ) -> Vec<f32> {
            let n = (obs_flat.len() / obs_dim.max(1)).min(act_flat.len());
            if n == 0 {
                return Vec::new();
            }
            let net = match self.network.as_ref() {
                Some(net) => net,
                None => return vec![0.0; n],
            };
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat[..n * obs_dim].to_vec(), [n, obs_dim]),
                &device,
            );
            let logits = net.pi_forward(obs);
            let log_probs = log_softmax(logits, 1);
            let act = Tensor::<TB, 2, Int>::from_data(
                BurnTensorData::new(act_flat[..n].to_vec(), [n, 1]),
                &device,
            );
            let logp = log_probs.gather(1, act).reshape([n]);
            logp.into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; n])
        }

        pub fn get_pi_layer_specs(&self) -> Option<LayerSpecs> {
            let network = self.network.as_ref()?;
            let mut specs = Vec::new();
            for layer in &network.pi_layers {
                let w = layer.weight.val();
                let dims = w.dims();
                let weights: Vec<f32> = w.into_data().to_vec::<f32>().unwrap_or_default();
                let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                    bias_param
                        .val()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                } else {
                    vec![0.0; dims[1]]
                };
                specs.push((dims[0], dims[1], weights, biases));
            }
            Some(specs)
        }

        pub fn get_vf_layer_specs(&self) -> Option<LayerSpecs> {
            let network = self.network.as_ref()?;
            let mut specs = Vec::new();
            for layer in &network.vf_layers {
                let w = layer.weight.val();
                let dims = w.dims();
                let weights: Vec<f32> = w.into_data().to_vec::<f32>().unwrap_or_default();
                let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                    bias_param
                        .val()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                } else {
                    vec![0.0; dims[1]]
                };
                specs.push((dims[0], dims[1], weights, biases));
            }
            Some(specs)
        }
    }

    pub fn zero_pi_info() -> (f32, HashMap<String, f32>) {
        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        info.insert("clipfrac".to_string(), 0.0);
        (0.0, info)
    }

    fn scalar_from_tensor(t: &Tensor<TB, 1, Float>) -> f32 {
        t.clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0])[0]
    }

    /// Convert a slice of obs TensorData to a flat Vec<f32>.
    pub fn obs_flat_from_tdata(obs: &[TensorData]) -> Result<Vec<f32>, NeuralNetworkError> {
        let mut out = Vec::new();
        for td in obs {
            let vals = convert_byte_dtype_to_f32(td.data.clone(), td.dtype.clone())?;
            out.extend_from_slice(&vals);
        }
        Ok(out)
    }

    /// Convert a slice of act TensorData to i64 action indices.
    pub fn action_indices_from_tdata(act: &[TensorData]) -> Vec<i64> {
        act.iter()
            .map(|td| {
                convert_byte_dtype_to_i64(&td.data, &td.dtype)
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ---- policy network head definitions ----

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
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

    pub fn get_pi_layer_specs(&self) -> LayerSpecs {
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

    pub fn get_pi_layer_specs(&self) -> LayerSpecs {
        self.pi.get_layer_specs()
    }
}

// ---- kernel interfaces ----

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
    #[allow(clippy::too_many_arguments)]
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
        act_dtype: &DType,
    ) -> Result<(ActBytes, LogpBytes), NeuralNetworkError>;
    fn get_pi_logprobs(&self, obs: &[TensorData], obs_dim: usize, act: &[TensorData]) -> Vec<f32>;
    fn value_forward(&self, obs: &[TensorData], obs_dim: usize) -> Vec<f32>;
    fn normalize_persistent_returns(&mut self, ret: &[f32]) -> Vec<f32>;
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
    pub trainer: Option<training::PPOActorCriticTrainer>,
    pub returns_mean: f32,
    pub returns_variance: f32,
    pub returns_count: u64,
}

pub struct ContinuousPPOKernel<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    pub pi: ContinuousPPOPolicyHead<B, KindIn, KindOut, Pi>,
    pub vf: ValueFunction<B, KindIn>,
    pub trainer: Option<training::PPOActorCriticTrainer>,
    pub returns_mean: f32,
    pub returns_variance: f32,
    pub returns_count: u64,
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

pub struct PPOKernelSnapshot<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> {
    kernel: Arc<PPOKernel<B, KindIn, KindOut, Pi>>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> Clone for PPOKernelSnapshot<B, KindIn, KindOut, Pi>
{
    fn clone(&self) -> Self {
        Self {
            kernel: Arc::clone(&self.kernel),
        }
    }
}

/// Training parameters for the actor-critic kernel.
pub struct PPOKernelTrainingArgs {
    pub pi_lr: f64,
    pub vf_coef: f32,
    pub lr_schedule_steps: Option<u64>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernelFactory<B, KindIn, KindOut, Pi>
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        pi_head: PPOPolicyHead<B, KindIn, KindOut, Pi>,
        vf_mlp: GenericMlp<B, KindIn, Float>,
        training_args: PPOKernelTrainingArgs,
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
                let obs_dim = *discrete_pi.pi.input_dim();
                let act_dim = *discrete_pi.pi.output_dim();
                // Derive hidden sizes from layer specs (all out-dims except last)
                let hidden_sizes: Vec<usize> = discrete_pi
                    .pi
                    .get_layer_specs()
                    .iter()
                    .rev()
                    .skip(1) // skip last layer (hidden → act_dim)
                    .rev()
                    .map(|(_, out, _, _)| *out)
                    .collect();
                let trainer = Some(training::PPOActorCriticTrainer::new(
                    obs_dim,
                    &hidden_sizes,
                    act_dim,
                    training_args.pi_lr,
                    training_args.vf_coef,
                    training_args.lr_schedule_steps,
                ));
                Ok(PPOKernel::<B, KindIn, KindOut, Pi>::Discrete(
                    DiscretePPOKernel {
                        pi: discrete_pi,
                        vf,
                        trainer,
                        returns_mean: 0.0,
                        returns_variance: 1.0,
                        returns_count: 0,
                    },
                ))
            }
            PPOPolicyHead::Continuous(continuous_pi) => {
                check_input_dim::<B, KindIn, KindOut, Pi>(&continuous_pi.pi, &vf)?;
                let obs_dim = *continuous_pi.pi.input_dim();
                let act_dim = *continuous_pi.pi.output_dim();
                let hidden_sizes: Vec<usize> = continuous_pi
                    .pi
                    .get_layer_specs()
                    .iter()
                    .rev()
                    .skip(1)
                    .rev()
                    .map(|(_, out, _, _)| *out)
                    .collect();
                let trainer = Some(training::PPOActorCriticTrainer::new(
                    obs_dim,
                    &hidden_sizes,
                    act_dim,
                    training_args.pi_lr,
                    training_args.vf_coef,
                    training_args.lr_schedule_steps,
                ));
                Ok(PPOKernel::<B, KindIn, KindOut, Pi>::Continuous(
                    ContinuousPPOKernel {
                        pi: continuous_pi,
                        vf,
                        trainer,
                        returns_mean: 0.0,
                        returns_variance: 1.0,
                        returns_count: 0,
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
    Pi: NeuralNetwork<B, KindIn, KindOut> + Clone,
> PPOKernel<B, KindIn, KindOut, Pi>
{
    pub fn clone_for_inference(&self) -> Self {
        match self {
            PPOKernel::Discrete(kernel) => PPOKernel::Discrete(DiscretePPOKernel {
                pi: kernel.pi.clone(),
                vf: kernel.vf.clone(),
                trainer: None,
                returns_mean: kernel.returns_mean,
                returns_variance: kernel.returns_variance,
                returns_count: kernel.returns_count,
            }),
            PPOKernel::Continuous(kernel) => PPOKernel::Continuous(ContinuousPPOKernel {
                pi: kernel.pi.clone(),
                vf: kernel.vf.clone(),
                trainer: None,
                returns_mean: kernel.returns_mean,
                returns_variance: kernel.returns_variance,
                returns_count: kernel.returns_count,
            }),
        }
    }

    pub fn to_arc_snapshot(&self) -> PPOKernelSnapshot<B, KindIn, KindOut, Pi> {
        PPOKernelSnapshot {
            kernel: Arc::new(self.clone_for_inference()),
        }
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernelSnapshot<B, KindIn, KindOut, Pi>
{
    pub fn policy_forward_bytes(
        &self,
        raw_model_output: &TensorData,
        mask_bytes: Option<&[u8]>,
        n_envs: usize,
        act_dtype: &DType,
    ) -> Result<(ActBytes, LogpBytes), NeuralNetworkError> {
        self.kernel
            .policy_forward_bytes(raw_model_output, mask_bytes, n_envs, act_dtype)
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
    Pi: NeuralNetwork<B, KindIn, KindOut>,
> PPOKernel<B, KindIn, KindOut, Pi>
{
    /// Extract pi layer specs from the trainer (after training); falls back to inference pi.
    pub fn get_pi_layer_specs(&self) -> Option<LayerSpecs> {
        match self {
            PPOKernel::Discrete(kernel) => {
                if let Some(t) = &kernel.trainer
                    && let Some(specs) = t.get_pi_layer_specs()
                {
                    return Some(specs);
                }

                Some(kernel.pi.get_pi_layer_specs())
            }
            PPOKernel::Continuous(kernel) => {
                if let Some(t) = &kernel.trainer
                    && let Some(specs) = t.get_pi_layer_specs()
                {
                    return Some(specs);
                }

                Some(kernel.pi.get_pi_layer_specs())
            }
        }
    }

    pub fn get_vf_layer_specs(&self) -> Option<LayerSpecs> {
        match self {
            PPOKernel::Discrete(kernel) => {
                if let Some(t) = &kernel.trainer
                    && let Some(specs) = t.get_vf_layer_specs()
                {
                    return Some(specs);
                }

                Some(kernel.vf.get_vf_layer_specs())
            }
            PPOKernel::Continuous(kernel) => {
                if let Some(t) = &kernel.trainer
                    && let Some(specs) = t.get_vf_layer_specs()
                {
                    return Some(specs);
                }

                Some(kernel.vf.get_vf_layer_specs())
            }
        }
    }
}

// ---- action byte encoding helpers ----

fn encode_action_i64_as_dtype(act: i64, dtype: &DType) -> Vec<u8> {
    match dtype {
        DType::NdArray(nd) => match nd {
            NdArrayDType::I8 => (act as i8).to_le_bytes().to_vec(),
            NdArrayDType::I16 => (act as i16).to_le_bytes().to_vec(),
            NdArrayDType::I32 => (act as i32).to_le_bytes().to_vec(),
            NdArrayDType::I64 => act.to_le_bytes().to_vec(),
            NdArrayDType::F16 => half::f16::from_f32(act as f32).to_le_bytes().to_vec(),
            NdArrayDType::F32 => (act as f32).to_le_bytes().to_vec(),
            NdArrayDType::F64 => (act as f64).to_le_bytes().to_vec(),
            NdArrayDType::Bool => vec![if act != 0 { 1u8 } else { 0u8 }],
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::I8 => (act as i8).to_le_bytes().to_vec(),
            TchDType::I16 => (act as i16).to_le_bytes().to_vec(),
            TchDType::I32 => (act as i32).to_le_bytes().to_vec(),
            TchDType::I64 => act.to_le_bytes().to_vec(),
            TchDType::F16 => half::f16::from_f32(act as f32).to_le_bytes().to_vec(),
            TchDType::Bf16 => half::bf16::from_f32(act as f32).to_le_bytes().to_vec(),
            TchDType::F32 => (act as f32).to_le_bytes().to_vec(),
            TchDType::F64 => (act as f64).to_le_bytes().to_vec(),
            TchDType::U8 => (act as u8).to_le_bytes().to_vec(),
            TchDType::Bool => vec![if act != 0 { 1u8 } else { 0u8 }],
        },
    }
}

fn encode_action_f32_as_dtype(act: f32, dtype: &DType) -> Vec<u8> {
    match dtype {
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => half::f16::from_f32(act).to_le_bytes().to_vec(),
            NdArrayDType::F32 => act.to_le_bytes().to_vec(),
            NdArrayDType::F64 => (act as f64).to_le_bytes().to_vec(),
            NdArrayDType::I8 => (act as i8).to_le_bytes().to_vec(),
            NdArrayDType::I16 => (act as i16).to_le_bytes().to_vec(),
            NdArrayDType::I32 => (act as i32).to_le_bytes().to_vec(),
            NdArrayDType::I64 => (act as i64).to_le_bytes().to_vec(),
            NdArrayDType::Bool => vec![if act != 0.0 { 1u8 } else { 0u8 }],
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => half::f16::from_f32(act).to_le_bytes().to_vec(),
            TchDType::Bf16 => half::bf16::from_f32(act).to_le_bytes().to_vec(),
            TchDType::F32 => act.to_le_bytes().to_vec(),
            TchDType::F64 => (act as f64).to_le_bytes().to_vec(),
            TchDType::I8 => (act as i8).to_le_bytes().to_vec(),
            TchDType::I16 => (act as i16).to_le_bytes().to_vec(),
            TchDType::I32 => (act as i32).to_le_bytes().to_vec(),
            TchDType::I64 => (act as i64).to_le_bytes().to_vec(),
            TchDType::U8 => (act as u8).to_le_bytes().to_vec(),
            TchDType::Bool => vec![if act != 0.0 { 1u8 } else { 0u8 }],
        },
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
        act_dtype: &DType,
    ) -> Result<(ActBytes, LogpBytes), NeuralNetworkError> {
        let logits = convert_byte_dtype_to_f32(
            raw_model_output.data.clone(),
            raw_model_output.dtype.clone(),
        )?;

        let act_dim = match self {
            PPOKernel::Discrete(kernel) => *kernel.pi.pi.output_dim(),
            PPOKernel::Continuous(kernel) => *kernel.pi.pi.output_dim(),
        };

        let mut action_bytes = Vec::<u8>::new();
        let mut logp_bytes = Vec::<u8>::with_capacity(n_envs * 4);

        match self {
            PPOKernel::Discrete(_) => {
                let pairs: Vec<(i64, f32)> = if n_envs < MIN_RAYON_PARALLEL_ENVS {
                    (0..n_envs)
                        .map(|i| {
                            DiscretePPOKernel::<B, KindIn, KindOut, Pi>::get_env_byte_action(
                                i, &logits, mask_bytes, act_dim,
                            )
                        })
                        .collect()
                } else {
                    (0..n_envs)
                        .into_par_iter()
                        .map(|i| {
                            DiscretePPOKernel::<B, KindIn, KindOut, Pi>::get_env_byte_action(
                                i, &logits, mask_bytes, act_dim,
                            )
                        })
                        .collect()
                };
                for (act_idx, logp) in &pairs {
                    action_bytes
                        .extend_from_slice(&encode_action_i64_as_dtype(*act_idx, act_dtype));
                    logp_bytes.extend_from_slice(&logp.to_le_bytes());
                }
            }
            PPOKernel::Continuous(_) => {
                let results: Vec<Result<(Vec<f32>, f32), NeuralNetworkError>> =
                    if n_envs < MIN_RAYON_PARALLEL_ENVS {
                        (0..n_envs)
                            .map(|i| {
                                ContinuousPPOKernel::<B, KindIn, KindOut, Pi>::get_env_byte_action(
                                    i, &logits, act_dim,
                                )
                            })
                            .collect()
                    } else {
                        (0..n_envs)
                            .into_par_iter()
                            .map(|i| {
                                ContinuousPPOKernel::<B, KindIn, KindOut, Pi>::get_env_byte_action(
                                    i, &logits, act_dim,
                                )
                            })
                            .collect()
                    };
                for res in results {
                    let (act_vec, logp) = res?;
                    for &a in &act_vec {
                        action_bytes.extend_from_slice(&encode_action_f32_as_dtype(a, act_dtype));
                    }
                    logp_bytes.extend_from_slice(&logp.to_le_bytes());
                }
            }
        }

        Ok((action_bytes, logp_bytes))
    }

    fn get_pi_logprobs(&self, obs: &[TensorData], obs_dim: usize, act: &[TensorData]) -> Vec<f32> {
        {
            let trainer = match self {
                PPOKernel::Discrete(k) => k.trainer.as_ref(),
                PPOKernel::Continuous(k) => k.trainer.as_ref(),
            };
            if let Some(t) = trainer {
                let obs_flat = match training::obs_flat_from_tdata(obs) {
                    Ok(f) => f,
                    Err(_) => return vec![0.0; act.len()],
                };
                let act_flat = training::action_indices_from_tdata(act);
                return t.logprobs_flat(&obs_flat, obs_dim, &act_flat);
            }
        }
        vec![0.0; act.len()]
    }

    fn value_forward(&self, obs: &[TensorData], obs_dim: usize) -> Vec<f32> {
        if obs.is_empty() {
            return Vec::new();
        }

        let trainer = match self {
            PPOKernel::Discrete(k) => k.trainer.as_ref(),
            PPOKernel::Continuous(k) => k.trainer.as_ref(),
        };
        if let Some(t) = trainer {
            let obs_flat = match training::obs_flat_from_tdata(obs) {
                Ok(f) => f,
                Err(_) => return vec![0.0; obs.len()],
            };
            return t.value_forward_flat(&obs_flat, obs_dim);
        }
        vec![0.0; obs.len()]
    }

    fn normalize_persistent_returns(&mut self, ret: &[f32]) -> Vec<f32> {
        let (mean, variance, count) = match self {
            PPOKernel::Discrete(k) => (
                &mut k.returns_mean,
                &mut k.returns_variance,
                &mut k.returns_count,
            ),
            PPOKernel::Continuous(k) => (
                &mut k.returns_mean,
                &mut k.returns_variance,
                &mut k.returns_count,
            ),
        };
        for &r in ret {
            *count += 1;
            let delta = r - *mean;
            *mean += delta / *count as f32;
            let delta2 = r - *mean;
            *variance += delta * delta2;
        }
        let std = if *count > 1 {
            (*variance / (*count - 1) as f32).sqrt().max(1e-8)
        } else {
            1.0
        };
        ret.iter()
            .map(|&r| ((r - *mean) / std).clamp(-5.0, 5.0))
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
        {
            match self {
                PPOKernel::Discrete(kernel) => {
                    if let Some(trainer) = kernel.trainer.as_mut() {
                        let obs_flat = match training::obs_flat_from_tdata(obs) {
                            Ok(f) => f,
                            Err(_) => return (0.0, 0.0, HashMap::new()),
                        };
                        let act_flat = training::action_indices_from_tdata(act);
                        return trainer.train_step_discrete(
                            &obs_flat,
                            obs_dim,
                            &act_flat,
                            adv,
                            logp_old,
                            ret,
                            clip_ratio,
                            ent_coef,
                            compute_stats,
                        );
                    }
                }
                PPOKernel::Continuous(_kernel) => {
                    // Continuous training deferred; return zeros
                }
            }
        }
        (0.0, 0.0, HashMap::new())
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
    ) -> (i64, f32) {
        let mut rng = rand::rng();
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
    ) -> Result<(Vec<f32>, f32), NeuralNetworkError> {
        use rand_distr::Normal;
        let mut rng = rand::rng();

        let stride = act_dim.saturating_mul(2);
        let start = env_id * stride;
        let env_logits = &logits[start..start + stride];

        let mean = &env_logits[..act_dim];
        let log_std = &env_logits[act_dim..stride];

        let mut act_vec = Vec::<f32>::with_capacity(act_dim);
        let mut total_log_prob = 0.0f32;

        for j in 0..act_dim {
            let std = log_std[j].exp();
            let distribution =
                Normal::new(mean[j], std).map_err(|_| NeuralNetworkError::InvalidDistribution)?;

            let action = distribution.sample(&mut rng);

            total_log_prob += -0.5 * (((action - mean[j]) / std).powi(2))
                - log_std[j]
                - (0.5 * (2.0 * std::f32::consts::PI).ln());

            act_vec.push(action);
        }

        Ok((act_vec, total_log_prob))
    }
}
