use crate::templates::base_algorithm::{StepAction, StepKernelTrait, WeightProvider};

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Tensor, TensorKind};

use relayrl_types::data::tensor::{BackendMatcher, TensorData, TensorError};

use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Default)]
pub struct DDPGTrainMetrics {
    pub actor_loss: f32,
    pub critic_loss: f32,
}

pub trait DDPGKernelTrait<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>:
    StepKernelTrait<B, InK, OutK>
{
    /// Construct a correctly-shaped kernel for a new actor slot.
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self;

    /// Perform one DDPG gradient update step given a sampled batch from the replay buffer.
    fn ddpg_train_step(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        next_obs: &[TensorData],
        rew: &[f32],
        done: &[f32],
        gamma: f32,
        tau: f32,
    ) -> DDPGTrainMetrics;
}

// ── Inference actor (B backend) ────────────────────────────────────────────────

use burn_nn::{Linear, LinearConfig, Relu};

pub struct DeterministicActorNet<B: Backend + BackendMatcher> {
    pub layers: Vec<Linear<B>>,
    pub relu: Relu,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl<B: Backend + BackendMatcher> DeterministicActorNet<B> {
    pub fn new(obs_dim: usize, hidden_sizes: &[usize], act_dim: usize, device: &B::Device) -> Self {
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
            input_dim: obs_dim,
            output_dim: act_dim,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < self.layers.len() - 1 {
                x = self.relu.forward(x);
            } else {
                x = x.tanh();
            }
        }
        x
    }
}

// ── DefaultDDPGKernel ──────────────────────────────────────────────────────────

pub struct DefaultDDPGKernel<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    pub inference_actor: DeterministicActorNet<B>,
    pub obs_dim: usize,
    pub act_dim: usize,
    #[cfg(feature = "ndarray-backend")]
    pub trainer: Option<training::DDPGTrainer>,
    _phantom: PhantomData<(InK, OutK)>,
}

pub type PlaceholderDDPGKernel<B, InK, OutK> = DefaultDDPGKernel<B, InK, OutK>;

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    DefaultDDPGKernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(obs_dim: usize, act_dim: usize, actor_lr: f64, critic_lr: f64) -> Self {
        let device = B::Device::default();
        let hidden_sizes = vec![256usize, 256];
        Self {
            inference_actor: DeterministicActorNet::new(obs_dim, &hidden_sizes, act_dim, &device),
            obs_dim,
            act_dim,
            #[cfg(feature = "ndarray-backend")]
            trainer: Some(training::DDPGTrainer::new(
                obs_dim,
                &hidden_sizes,
                act_dim,
                actor_lr,
                critic_lr,
            )),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> Default
    for DefaultDDPGKernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn default() -> Self {
        Self::new(1, 1, 3e-4, 3e-4)
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for DefaultDDPGKernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn step<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, InK>,
        _mask: Tensor<B, OUT_D, OutK>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        let batch = obs.dims()[0];
        let device = B::Device::default();
        let obs_data = obs.into_data();
        let obs_flat: Tensor<B, IN_D, Float> =
            Tensor::from_data(obs_data.convert::<f32>(), &device);
        let obs_2d: Tensor<B, 2, Float> = obs_flat.reshape([batch, self.obs_dim]);
        let action = self.inference_actor.forward(obs_2d);
        Ok((StepAction::Continuous(action), HashMap::new()))
    }

    fn get_input_dim(&self) -> usize {
        self.obs_dim
    }

    fn get_output_dim(&self) -> usize {
        self.act_dim
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    DDPGKernelTrait<B, InK, OutK> for DefaultDDPGKernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self {
        Self::new(obs_dim, act_dim, 3e-4, 3e-4)
    }

    fn ddpg_train_step(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        next_obs: &[TensorData],
        rew: &[f32],
        done: &[f32],
        gamma: f32,
        tau: f32,
    ) -> DDPGTrainMetrics {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.trainer {
            return trainer.train_step(obs, act, next_obs, rew, done, gamma, tau);
        }
        DDPGTrainMetrics::default()
    }
}

#[cfg(feature = "ndarray-backend")]
impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> WeightProvider
    for DefaultDDPGKernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        let trainer = self.trainer.as_ref()?;
        let actor = trainer.actor.as_ref()?;
        let mut specs = Vec::new();
        for layer in &actor.layers {
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

// ── Training backend (ndarray + autodiff) ─────────────────────────────────────

#[cfg(feature = "ndarray-backend")]
pub mod training {
    use super::DDPGTrainMetrics;

    extern crate burn_core as burn;

    use burn_autodiff::Autodiff;
    use burn_core::module::{Module, Param, ParamId};
    use burn_ndarray::NdArray;
    use burn_nn::{Linear, LinearConfig, Relu};
    use burn_optim::adaptor::OptimizerAdaptor;
    use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer};
    use burn_tensor::backend::Backend;
    use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};
    use relayrl_types::prelude::tensor::relayrl::TensorData;

    pub type TB = Autodiff<NdArray>;

    // ── Actor MLP ────────────────────────────────────────────────────────────

    #[derive(Module, Debug)]
    pub struct ActorMlp<B: burn_tensor::backend::Backend> {
        pub layers: Vec<Linear<B>>,
        pub relu: Relu,
        pub obs_dim: usize,
        pub act_dim: usize,
    }

    impl<B: burn_tensor::backend::Backend> ActorMlp<B> {
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

        pub fn forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = input;
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.layers.len() - 1 {
                    x = self.relu.forward(x);
                } else {
                    x = x.tanh();
                }
            }
            x
        }
    }

    // ── Critic MLP ────────────────────────────────────────────────────────────

    #[derive(Module, Debug)]
    pub struct CriticMlp<B: burn_tensor::backend::Backend> {
        pub layers: Vec<Linear<B>>,
        pub relu: Relu,
        pub input_dim: usize,
    }

    impl<B: burn_tensor::backend::Backend> CriticMlp<B> {
        pub fn new(
            obs_dim: usize,
            act_dim: usize,
            hidden_sizes: &[usize],
            device: &B::Device,
        ) -> Self {
            let input_dim = obs_dim + act_dim;
            let mut dims = vec![input_dim];
            dims.extend_from_slice(hidden_sizes);
            dims.push(1);
            let layers = dims
                .windows(2)
                .map(|w| LinearConfig::new(w[0], w[1]).init(device))
                .collect();
            Self {
                layers,
                relu: Relu::new(),
                input_dim,
            }
        }

        pub fn forward(
            &self,
            obs: Tensor<B, 2, Float>,
            act: Tensor<B, 2, Float>,
        ) -> Tensor<B, 2, Float> {
            let x = Tensor::cat(vec![obs, act], 1);
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

    // ── DDPGTrainer ───────────────────────────────────────────────────────────

    pub struct DDPGTrainer {
        pub actor: Option<ActorMlp<TB>>,
        pub actor_target: Option<ActorMlp<NdArray>>,
        pub critic: Option<CriticMlp<TB>>,
        pub critic_target: Option<CriticMlp<NdArray>>,
        pub actor_optimizer: OptimizerAdaptor<Adam, ActorMlp<TB>, TB>,
        pub critic_optimizer: OptimizerAdaptor<Adam, CriticMlp<TB>, TB>,
        pub actor_lr: f64,
        pub critic_lr: f64,
        pub obs_dim: usize,
        pub act_dim: usize,
        hidden_sizes: Vec<usize>,
    }

    impl DDPGTrainer {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            actor_lr: f64,
            critic_lr: f64,
        ) -> Self {
            let device_tb = <TB as Backend>::Device::default();
            let device_nd = <NdArray as Backend>::Device::default();

            let actor = ActorMlp::new(obs_dim, hidden_sizes, act_dim, &device_tb);
            let actor_target = ActorMlp::new(obs_dim, hidden_sizes, act_dim, &device_nd);
            let critic = CriticMlp::new(obs_dim, act_dim, hidden_sizes, &device_tb);
            let critic_target = CriticMlp::new(obs_dim, act_dim, hidden_sizes, &device_nd);

            Self {
                actor: Some(actor),
                actor_target: Some(actor_target),
                critic: Some(critic),
                critic_target: Some(critic_target),
                actor_optimizer: AdamConfig::new().init::<TB, ActorMlp<TB>>(),
                critic_optimizer: AdamConfig::new().init::<TB, CriticMlp<TB>>(),
                actor_lr,
                critic_lr,
                obs_dim,
                act_dim,
                hidden_sizes: hidden_sizes.to_vec(),
            }
        }

        pub fn train_step(
            &mut self,
            obs_tensors: &[TensorData],
            act_tensors: &[TensorData],
            next_obs_tensors: &[TensorData],
            rewards: &[f32],
            dones: &[f32],
            gamma: f32,
            tau: f32,
        ) -> DDPGTrainMetrics {
            let n = obs_tensors
                .len()
                .min(act_tensors.len())
                .min(next_obs_tensors.len())
                .min(rewards.len())
                .min(dones.len());
            if n == 0 {
                return DDPGTrainMetrics::default();
            }

            let actor = match self.actor.take() {
                Some(a) => a,
                None => return DDPGTrainMetrics::default(),
            };
            let critic = match self.critic.take() {
                Some(c) => c,
                None => {
                    self.actor = Some(actor);
                    return DDPGTrainMetrics::default();
                }
            };
            let actor_target = match self.actor_target.take() {
                Some(a) => a,
                None => {
                    self.actor = Some(actor);
                    self.critic = Some(critic);
                    return DDPGTrainMetrics::default();
                }
            };
            let critic_target = match self.critic_target.take() {
                Some(c) => c,
                None => {
                    self.actor = Some(actor);
                    self.critic = Some(critic);
                    self.actor_target = Some(actor_target);
                    return DDPGTrainMetrics::default();
                }
            };

            let device_tb = <TB as Backend>::Device::default();
            let device_nd = <NdArray as Backend>::Device::default();

            // Convert inputs to tensors
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(tensor_flat(obs_tensors), [n, self.obs_dim]),
                &device_tb,
            );
            let act = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(tensor_flat(act_tensors), [n, self.act_dim]),
                &device_tb,
            );
            let next_obs_nd = Tensor::<NdArray, 2, Float>::from_data(
                BurnTensorData::new(tensor_flat(next_obs_tensors), [n, self.obs_dim]),
                &device_nd,
            );
            let rew = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(rewards[..n].to_vec(), [n]),
                &device_tb,
            );
            let done = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(dones[..n].to_vec(), [n]),
                &device_tb,
            );

            // Compute target Q using target networks (NdArray, no grad)
            let next_act_nd = actor_target.forward(next_obs_nd.clone());
            let target_q_nd = critic_target.forward(next_obs_nd, next_act_nd).reshape([n]);
            let target_q_vals: Vec<f32> = target_q_nd
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; n]);
            let target_q_tb = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(target_q_vals, [n]),
                &device_tb,
            );

            // target = r + gamma * (1 - done) * Q_target
            let not_done = done.neg().add_scalar(1.0f32);
            let target = rew + not_done * target_q_tb * gamma;

            // ── Critic update ────────────────────────────────────────────────
            let current_q = critic.forward(obs.clone(), act.clone()).reshape([n]);
            let critic_loss = (current_q - target).powf_scalar(2.0).mean();
            let critic_loss_val = scalar_f32(&critic_loss);
            let grads_c = critic_loss.backward();
            let critic_grads = GradientsParams::from_grads::<TB, CriticMlp<TB>>(grads_c, &critic);
            let critic = self
                .critic_optimizer
                .step(self.critic_lr, critic, critic_grads);

            // ── Actor update ─────────────────────────────────────────────────
            let actor_actions = actor.forward(obs.clone());
            let actor_q = critic
                .forward(obs.clone(), actor_actions)
                .reshape([n])
                .mean();
            let actor_loss = actor_q.neg();
            let actor_loss_val = scalar_f32(&actor_loss);
            let grads_a = actor_loss.backward();
            let actor_grads = GradientsParams::from_grads::<TB, ActorMlp<TB>>(grads_a, &actor);
            let actor = self.actor_optimizer.step(self.actor_lr, actor, actor_grads);

            // ── Soft update targets ──────────────────────────────────────────
            let mut actor_target = actor_target;
            let mut critic_target = critic_target;
            soft_update_actor(&actor, &mut actor_target, tau);
            soft_update_critic(&critic, &mut critic_target, tau);

            self.actor = Some(actor);
            self.critic = Some(critic);
            self.actor_target = Some(actor_target);
            self.critic_target = Some(critic_target);

            DDPGTrainMetrics {
                actor_loss: actor_loss_val,
                critic_loss: critic_loss_val,
            }
        }
    }

    fn soft_update_actor(actor: &ActorMlp<TB>, target: &mut ActorMlp<NdArray>, tau: f32) {
        for (a_layer, t_layer) in actor.layers.iter().zip(target.layers.iter_mut()) {
            let cur_w = a_layer.weight.val().inner();
            let tgt_w = t_layer.weight.val();
            let new_w = cur_w * tau + tgt_w * (1.0 - tau);
            t_layer.weight = Param::initialized(ParamId::new(), new_w);

            if let (Some(cur_b), Some(tgt_b)) = (&a_layer.bias, &mut t_layer.bias) {
                let cur_bv = cur_b.val().inner();
                let tgt_bv = tgt_b.val();
                let new_b = cur_bv * tau + tgt_bv * (1.0 - tau);
                *tgt_b = Param::initialized(ParamId::new(), new_b);
            }
        }
    }

    fn soft_update_critic(critic: &CriticMlp<TB>, target: &mut CriticMlp<NdArray>, tau: f32) {
        for (c_layer, t_layer) in critic.layers.iter().zip(target.layers.iter_mut()) {
            let cur_w = c_layer.weight.val().inner();
            let tgt_w = t_layer.weight.val();
            let new_w = cur_w * tau + tgt_w * (1.0 - tau);
            t_layer.weight = Param::initialized(ParamId::new(), new_w);

            if let (Some(cur_b), Some(tgt_b)) = (&c_layer.bias, &mut t_layer.bias) {
                let cur_bv = cur_b.val().inner();
                let tgt_bv = tgt_b.val();
                let new_b = cur_bv * tau + tgt_bv * (1.0 - tau);
                *tgt_b = Param::initialized(ParamId::new(), new_b);
            }
        }
    }

    fn scalar_f32(t: &Tensor<TB, 1, Float>) -> f32 {
        t.clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0])[0]
    }

    pub fn tensor_flat(tensors: &[TensorData]) -> Vec<f32> {
        tensors
            .iter()
            .flat_map(|t| bytemuck::cast_slice::<u8, f32>(&t.data).to_vec())
            .collect()
    }
}
