use crate::templates::base_algorithm::{StepAction, StepKernelTrait, WeightProvider};

use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Tensor, TensorKind};

use relayrl_types::data::tensor::{BackendMatcher, TensorData, TensorError};

use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Default)]
pub struct TD3TrainMetrics {
    pub actor_loss: f32,
    pub critic_loss: f32,
    pub policy_updated: bool,
}

pub trait TD3KernelTrait<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>:
    StepKernelTrait<B, InK, OutK>
{
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self;

    /// One TD3 gradient step: update twin critics, delay actor update by policy_frequency.
    fn td3_train_step(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        next_obs: &[TensorData],
        rew: &[f32],
        done: &[f32],
        gamma: f32,
        tau: f32,
        policy_noise: f32,
        noise_clip: f32,
        policy_frequency: u32,
    ) -> TD3TrainMetrics;
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

// ── DefaultTD3Kernel ──────────────────────────────────────────────────────────

pub struct DefaultTD3Kernel<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> {
    pub inference_actor: DeterministicActorNet<B>,
    pub obs_dim: usize,
    pub act_dim: usize,
    #[cfg(feature = "ndarray-backend")]
    pub trainer: Option<training::TD3Trainer>,
    _phantom: PhantomData<(InK, OutK)>,
}

pub type PlaceholderTD3Kernel<B, InK, OutK> = DefaultTD3Kernel<B, InK, OutK>;

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    DefaultTD3Kernel<B, InK, OutK>
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
            trainer: Some(training::TD3Trainer::new(
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
    for DefaultTD3Kernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn default() -> Self {
        Self::new(1, 1, 3e-4, 3e-4)
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for DefaultTD3Kernel<B, InK, OutK>
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
    TD3KernelTrait<B, InK, OutK> for DefaultTD3Kernel<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self {
        Self::new(obs_dim, act_dim, 3e-4, 3e-4)
    }

    fn td3_train_step(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        next_obs: &[TensorData],
        rew: &[f32],
        done: &[f32],
        gamma: f32,
        tau: f32,
        policy_noise: f32,
        noise_clip: f32,
        policy_frequency: u32,
    ) -> TD3TrainMetrics {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.trainer {
            return trainer.train_step(
                obs,
                act,
                next_obs,
                rew,
                done,
                gamma,
                tau,
                policy_noise,
                noise_clip,
                policy_frequency,
            );
        }
        TD3TrainMetrics::default()
    }
}

#[cfg(feature = "ndarray-backend")]
impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> WeightProvider
    for DefaultTD3Kernel<B, InK, OutK>
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
            let biases: Vec<f32> = if let Some(bp) = &layer.bias {
                bp.val().into_data().to_vec::<f32>().unwrap_or_default()
            } else {
                vec![0.0; out_dim]
            };
            specs.push((in_dim, out_dim, weights, biases));
        }
        Some(specs)
    }
}

// ── Training backend ──────────────────────────────────────────────────────────

#[cfg(feature = "ndarray-backend")]
pub mod training {
    use super::TD3TrainMetrics;

    extern crate burn_core as burn;

    use burn_autodiff::Autodiff;
    use burn_core::module::{Module, Param, ParamId};
    use burn_ndarray::NdArray;
    use burn_nn::{Linear, LinearConfig, Relu};
    use burn_optim::adaptor::OptimizerAdaptor;
    use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer};
    use burn_tensor::backend::Backend;
    use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};
    use rand::RngExt;
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

    // ── Twin Critic MLP (Q1 + Q2) ────────────────────────────────────────────

    #[derive(Module, Debug)]
    pub struct TwinCriticMlp<B: burn_tensor::backend::Backend> {
        pub q1_layers: Vec<Linear<B>>,
        pub q2_layers: Vec<Linear<B>>,
        pub relu: Relu,
        pub input_dim: usize,
    }

    impl<B: burn_tensor::backend::Backend> TwinCriticMlp<B> {
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
            let q1_layers = dims
                .windows(2)
                .map(|w| LinearConfig::new(w[0], w[1]).init(device))
                .collect();
            let q2_layers = dims
                .windows(2)
                .map(|w| LinearConfig::new(w[0], w[1]).init(device))
                .collect();
            Self {
                q1_layers,
                q2_layers,
                relu: Relu::new(),
                input_dim,
            }
        }

        fn forward_q(
            layers: &[Linear<B>],
            relu: &Relu,
            obs: Tensor<B, 2, Float>,
            act: Tensor<B, 2, Float>,
        ) -> Tensor<B, 2, Float> {
            let mut x = Tensor::cat(vec![obs, act], 1);
            for (i, layer) in layers.iter().enumerate() {
                x = layer.forward(x);
                if i < layers.len() - 1 {
                    x = relu.forward(x);
                }
            }
            x
        }

        pub fn forward_both(
            &self,
            obs: Tensor<B, 2, Float>,
            act: Tensor<B, 2, Float>,
        ) -> (Tensor<B, 2, Float>, Tensor<B, 2, Float>) {
            let q1 = Self::forward_q(&self.q1_layers, &self.relu, obs.clone(), act.clone());
            let q2 = Self::forward_q(&self.q2_layers, &self.relu, obs, act);
            (q1, q2)
        }

        pub fn forward_q1(
            &self,
            obs: Tensor<B, 2, Float>,
            act: Tensor<B, 2, Float>,
        ) -> Tensor<B, 2, Float> {
            Self::forward_q(&self.q1_layers, &self.relu, obs, act)
        }
    }

    // ── Single Critic for targets (NdArray) ───────────────────────────────────

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
            let mut x = Tensor::cat(vec![obs, act], 1);
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }
    }

    // ── TD3Trainer ────────────────────────────────────────────────────────────

    pub struct TD3Trainer {
        pub actor: Option<ActorMlp<TB>>,
        pub actor_target: Option<ActorMlp<NdArray>>,
        pub critic: Option<TwinCriticMlp<TB>>,
        pub critic_target: Option<CriticMlp<NdArray>>,
        pub actor_optimizer: OptimizerAdaptor<Adam, ActorMlp<TB>, TB>,
        pub critic_optimizer: OptimizerAdaptor<Adam, TwinCriticMlp<TB>, TB>,
        pub actor_lr: f64,
        pub critic_lr: f64,
        pub obs_dim: usize,
        pub act_dim: usize,
        hidden_sizes: Vec<usize>,
        total_it: u64,
    }

    impl TD3Trainer {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            actor_lr: f64,
            critic_lr: f64,
        ) -> Self {
            let device_tb = <TB as Backend>::Device::default();
            let device_nd = <NdArray as Backend>::Device::default();
            Self {
                actor: Some(ActorMlp::new(obs_dim, hidden_sizes, act_dim, &device_tb)),
                actor_target: Some(ActorMlp::new(obs_dim, hidden_sizes, act_dim, &device_nd)),
                critic: Some(TwinCriticMlp::new(
                    obs_dim,
                    act_dim,
                    hidden_sizes,
                    &device_tb,
                )),
                critic_target: Some(CriticMlp::new(obs_dim, act_dim, hidden_sizes, &device_nd)),
                actor_optimizer: AdamConfig::new().init::<TB, ActorMlp<TB>>(),
                critic_optimizer: AdamConfig::new().init::<TB, TwinCriticMlp<TB>>(),
                actor_lr,
                critic_lr,
                obs_dim,
                act_dim,
                hidden_sizes: hidden_sizes.to_vec(),
                total_it: 0,
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
            policy_noise: f32,
            noise_clip: f32,
            policy_frequency: u32,
        ) -> TD3TrainMetrics {
            self.total_it += 1;
            let n = obs_tensors
                .len()
                .min(act_tensors.len())
                .min(next_obs_tensors.len())
                .min(rewards.len())
                .min(dones.len());
            if n == 0 {
                return TD3TrainMetrics::default();
            }

            let actor = match self.actor.take() {
                Some(a) => a,
                None => return TD3TrainMetrics::default(),
            };
            let critic = match self.critic.take() {
                Some(c) => c,
                None => {
                    self.actor = Some(actor);
                    return TD3TrainMetrics::default();
                }
            };
            let actor_target = match self.actor_target.take() {
                Some(a) => a,
                None => {
                    self.actor = Some(actor);
                    self.critic = Some(critic);
                    return TD3TrainMetrics::default();
                }
            };
            let critic_target = match self.critic_target.take() {
                Some(c) => c,
                None => {
                    self.actor = Some(actor);
                    self.critic = Some(critic);
                    self.actor_target = Some(actor_target);
                    return TD3TrainMetrics::default();
                }
            };

            let device_tb = <TB as Backend>::Device::default();
            let device_nd = <NdArray as Backend>::Device::default();

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

            // Target policy smoothing: next_act = clamp(actor_target(s') + noise, -1, 1)
            let next_act_raw_nd = actor_target.forward(next_obs_nd.clone());
            let noise_nd: Vec<f32> = (0..n * self.act_dim)
                .map(|_| {
                    use rand::Rng;
                    let v: f32 = rand::rng().random::<f32>() * 2.0 - 1.0;
                    v.clamp(-noise_clip, noise_clip) * policy_noise
                })
                .collect();
            let noise_tensor_nd = Tensor::<NdArray, 2, Float>::from_data(
                BurnTensorData::new(noise_nd, [n, self.act_dim]),
                &device_nd,
            );
            let next_act_nd = (next_act_raw_nd + noise_tensor_nd).clamp(-1.0, 1.0);

            // Target Q: min(Q1_target, Q2_target) — approximate with single critic target
            let tgt_q1_nd = critic_target
                .forward(next_obs_nd.clone(), next_act_nd.clone())
                .reshape([n]);
            // Use same critic_target for Q2 approximation (single target critic for simplicity)
            let tgt_q2_nd = critic_target.forward(next_obs_nd, next_act_nd).reshape([n]);
            let tgt_q_vals: Vec<f32> = tgt_q1_nd
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|_| vec![0.0; n])
                .iter()
                .zip(
                    tgt_q2_nd
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_else(|_| vec![0.0; n])
                        .iter(),
                )
                .map(|(a, b)| a.min(*b))
                .collect();
            let tgt_q_tb =
                Tensor::<TB, 1, Float>::from_data(BurnTensorData::new(tgt_q_vals, [n]), &device_tb);
            let not_done = done.neg().add_scalar(1.0f32);
            let target = rew + not_done * tgt_q_tb * gamma;

            // ── Twin critic update ────────────────────────────────────────────
            let (q1, q2) = critic.forward_both(obs.clone(), act.clone());
            let critic_loss = (q1.reshape([n]) - target.clone()).powf_scalar(2.0).mean()
                + (q2.reshape([n]) - target).powf_scalar(2.0).mean();
            let critic_loss_val = scalar_f32(&critic_loss);
            let grads_c = critic_loss.backward();
            let critic_grads =
                GradientsParams::from_grads::<TB, TwinCriticMlp<TB>>(grads_c, &critic);
            let critic = self
                .critic_optimizer
                .step(self.critic_lr, critic, critic_grads);

            // ── Delayed actor update ──────────────────────────────────────────
            let policy_updated = self.total_it % policy_frequency as u64 == 0;
            let actor_loss_val;
            let actor = if policy_updated {
                let actor_actions = actor.forward(obs.clone());
                let actor_q = critic.forward_q1(obs, actor_actions).reshape([n]).mean();
                let actor_loss = actor_q.neg();
                actor_loss_val = scalar_f32(&actor_loss);
                let grads_a = actor_loss.backward();
                let actor_grads = GradientsParams::from_grads::<TB, ActorMlp<TB>>(grads_a, &actor);
                self.actor_optimizer.step(self.actor_lr, actor, actor_grads)
            } else {
                actor_loss_val = 0.0;
                actor
            };

            // ── Soft update targets ───────────────────────────────────────────
            if policy_updated {
                let mut actor_target = actor_target;
                soft_update_actor(&actor, &mut actor_target, tau);
                self.actor_target = Some(actor_target);
            } else {
                self.actor_target = Some(actor_target);
            }

            let mut critic_target = critic_target;
            soft_update_critic_target(&critic, &mut critic_target, tau);

            self.actor = Some(actor);
            self.critic = Some(critic);
            self.critic_target = Some(critic_target);

            TD3TrainMetrics {
                actor_loss: actor_loss_val,
                critic_loss: critic_loss_val,
                policy_updated,
            }
        }
    }

    fn soft_update_actor(actor: &ActorMlp<TB>, target: &mut ActorMlp<NdArray>, tau: f32) {
        for (a_layer, t_layer) in actor.layers.iter().zip(target.layers.iter_mut()) {
            let cur_w = a_layer.weight.val().inner();
            let tgt_w = t_layer.weight.val();
            t_layer.weight = Param::initialized(ParamId::new(), cur_w * tau + tgt_w * (1.0 - tau));
            if let (Some(cb), Some(tb)) = (&a_layer.bias, &mut t_layer.bias) {
                let cw = cb.val().inner();
                let tw = tb.val();
                *tb = Param::initialized(ParamId::new(), cw * tau + tw * (1.0 - tau));
            }
        }
    }

    fn soft_update_critic_target(
        critic: &TwinCriticMlp<TB>,
        target: &mut CriticMlp<NdArray>,
        tau: f32,
    ) {
        // Update target using Q1 layers from twin critic
        for (c_layer, t_layer) in critic.q1_layers.iter().zip(target.layers.iter_mut()) {
            let cur_w = c_layer.weight.val().inner();
            let tgt_w = t_layer.weight.val();
            t_layer.weight = Param::initialized(ParamId::new(), cur_w * tau + tgt_w * (1.0 - tau));
            if let (Some(cb), Some(tb)) = (&c_layer.bias, &mut t_layer.bias) {
                let cw = cb.val().inner();
                let tw = tb.val();
                *tb = Param::initialized(ParamId::new(), cw * tau + tw * (1.0 - tau));
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
