use crate::templates::base_replay_buffer::{Batch, BatchKey, BufferSample, SampleScalars};
use relayrl_types::prelude::tensor::relayrl::TensorData;

#[derive(Clone, Debug, Default)]
pub struct AgentBatch {
    pub obs: Vec<TensorData>,
    pub act: Vec<TensorData>,
    pub next_obs: Vec<TensorData>,
    pub rew: Vec<f32>,
    pub done: Vec<f32>,
}

impl AgentBatch {
    pub fn from_batch(mut batch: Batch) -> Option<Self> {
        let obs = match batch.remove(&BatchKey::Obs) {
            Some(BufferSample::Tensors(t)) => Vec::from(t),
            _ => return None,
        };
        let act = match batch.remove(&BatchKey::Act) {
            Some(BufferSample::Tensors(t)) => Vec::from(t),
            _ => return None,
        };
        let next_obs = match batch.remove(&BatchKey::Custom("NextObs".to_string())) {
            Some(BufferSample::Tensors(t)) => Vec::from(t),
            _ => return None,
        };
        let rew = match batch.remove(&BatchKey::Custom("Rew".to_string())) {
            Some(BufferSample::Scalars(SampleScalars::F32(v))) => Vec::from(v),
            _ => return None,
        };
        let done = match batch.remove(&BatchKey::Custom("Done".to_string())) {
            Some(BufferSample::Scalars(SampleScalars::F32(v))) => Vec::from(v),
            _ => return None,
        };

        Some(Self {
            obs,
            act,
            next_obs,
            rew,
            done,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MultiagentDDPGTrainMetrics {
    pub actor_loss: f32,
    pub critic_loss: f32,
}

pub struct MultiagentDDPGKernel {
    obs_dim: usize,
    act_dim: usize,
    hidden_sizes: Vec<usize>,
    #[cfg(feature = "ndarray-backend")]
    trainer: Option<training::SharedDDPGTrainer>,
}

impl Default for MultiagentDDPGKernel {
    fn default() -> Self {
        Self::new(1, 1, 3e-4, 3e-4, 0.005)
    }
}

impl MultiagentDDPGKernel {
    pub fn new(obs_dim: usize, act_dim: usize, actor_lr: f32, critic_lr: f32, tau: f32) -> Self {
        let hidden_sizes = vec![256usize, 256];
        Self {
            obs_dim,
            act_dim,
            hidden_sizes: hidden_sizes.clone(),
            #[cfg(feature = "ndarray-backend")]
            trainer: Some(training::SharedDDPGTrainer::new(
                obs_dim,
                &hidden_sizes,
                act_dim,
                actor_lr as f64,
                critic_lr as f64,
                tau,
            )),
        }
    }

    pub fn register_agent(&mut self) {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.trainer {
            trainer.add_agent();
        }
    }

    pub fn agent_count(&self) -> usize {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &self.trainer {
            return trainer.agent_count();
        }
        0
    }

    pub fn train_epoch(
        &mut self,
        agent_batches: &[AgentBatch],
        gamma: f32,
        tau: f32,
        policy_frequency: usize,
    ) -> MultiagentDDPGTrainMetrics {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.trainer {
            return trainer.train_epoch(agent_batches, gamma, tau, policy_frequency);
        }
        MultiagentDDPGTrainMetrics::default()
    }

    #[allow(dead_code)]
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    #[allow(dead_code)]
    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    #[allow(dead_code)]
    pub fn hidden_sizes(&self) -> &[usize] {
        &self.hidden_sizes
    }

    /// Extract per-layer weight specs from the first actor in the shared policy.
    #[cfg(feature = "ndarray-backend")]
    pub fn get_actor_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        let trainer = self.trainer.as_ref()?;
        let module = trainer.module.as_ref()?;
        let actor = module.actors.first()?;

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

#[cfg(feature = "ndarray-backend")]
impl crate::templates::base_algorithm::WeightProvider for MultiagentDDPGKernel {
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        self.get_actor_layer_specs()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel traits
// ─────────────────────────────────────────────────────────────────────────────

use crate::templates::base_algorithm::{MultiagentKernelTrait, StepAction, StepKernelTrait};
use burn_tensor::TensorKind;
use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::burn::Tensor;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::tensor::relayrl::TensorError;
use std::collections::HashMap;

/// Kernel trait for multi-agent DDPG algorithms.
///
/// Extends [`MultiagentKernelTrait`] with the DDPG-specific batched training
/// method used by [`MultiagentDDPGAlgorithm`].
pub trait MultiagentDDPGKernelTrait<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
>: MultiagentKernelTrait<B, InK, OutK>
{
    fn train_epoch(
        &mut self,
        agent_batches: &[AgentBatch],
        gamma: f32,
        tau: f32,
        policy_frequency: usize,
    ) -> MultiagentDDPGTrainMetrics;
}

impl<B, InK, OutK> StepKernelTrait<B, InK, OutK> for MultiagentDDPGKernel
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn step<const IN_D: usize, const OUT_D: usize>(
        &self,
        _obs: Tensor<B, IN_D, InK>,
        _mask: Tensor<B, OUT_D, OutK>,
    ) -> Result<
        (
            StepAction<B>,
            HashMap<String, relayrl_types::prelude::tensor::relayrl::TensorData>,
        ),
        TensorError,
    > {
        Err(TensorError::BackendError(
            "MultiagentDDPGKernel inference should be performed through the framework actor, not directly".to_string(),
        ))
    }

    fn get_input_dim(&self) -> usize {
        self.obs_dim
    }

    fn get_output_dim(&self) -> usize {
        self.act_dim
    }
}

impl<B, InK, OutK> MultiagentKernelTrait<B, InK, OutK> for MultiagentDDPGKernel
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn register_agent(&mut self) {
        MultiagentDDPGKernel::register_agent(self);
    }
}

impl<B, InK, OutK> MultiagentDDPGKernelTrait<B, InK, OutK> for MultiagentDDPGKernel
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    fn train_epoch(
        &mut self,
        agent_batches: &[AgentBatch],
        gamma: f32,
        tau: f32,
        policy_frequency: usize,
    ) -> MultiagentDDPGTrainMetrics {
        MultiagentDDPGKernel::train_epoch(self, agent_batches, gamma, tau, policy_frequency)
    }
}

#[cfg(feature = "ndarray-backend")]
pub mod training {
    use super::{AgentBatch, MultiagentDDPGTrainMetrics};

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

    #[derive(Module, Debug, Clone)]
    pub struct SharedDDPGModule {
        pub actors: Vec<ActorMlp<TB>>,
        pub critic: CriticMlp<TB>,
        pub obs_dim: usize,
        pub act_dim: usize,
    }

    impl SharedDDPGModule {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            agent_count: usize,
            device: &<TB as Backend>::Device,
        ) -> Self {
            let actors = (0..agent_count)
                .map(|_| ActorMlp::new(obs_dim, hidden_sizes, act_dim, device))
                .collect();
            Self {
                actors,
                critic: CriticMlp::new(obs_dim, act_dim, hidden_sizes, device),
                obs_dim,
                act_dim,
            }
        }

        pub fn add_agent(&mut self, hidden_sizes: &[usize], device: &<TB as Backend>::Device) {
            self.actors.push(ActorMlp::new(
                self.obs_dim,
                hidden_sizes,
                self.act_dim,
                device,
            ));
        }
    }

    pub struct SharedDDPGTrainer {
        pub module: Option<SharedDDPGModule>,
        pub actor_targets: Vec<ActorMlp<NdArray>>,
        pub critic_target: Option<CriticMlp<NdArray>>,
        pub actor_optimizer: OptimizerAdaptor<Adam, SharedDDPGModule, TB>,
        pub tau: f32,
        pub actor_lr: f64,
        pub critic_lr: f64,
        hidden_sizes: Vec<usize>,
        total_it: u64,
    }

    impl SharedDDPGTrainer {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            actor_lr: f64,
            critic_lr: f64,
            tau: f32,
        ) -> Self {
            let device_tb = <TB as Backend>::Device::default();
            let device_nd = <NdArray as Backend>::Device::default();
            let module = SharedDDPGModule::new(obs_dim, hidden_sizes, act_dim, 0, &device_tb);
            let critic_target = CriticMlp::new(obs_dim, act_dim, hidden_sizes, &device_nd);
            Self {
                module: Some(module),
                actor_targets: Vec::new(),
                critic_target: Some(critic_target),
                actor_optimizer: AdamConfig::new().init::<TB, SharedDDPGModule>(),
                tau,
                actor_lr,
                critic_lr,
                hidden_sizes: hidden_sizes.to_vec(),
                total_it: 0,
            }
        }

        pub fn add_agent(&mut self) {
            if let Some(module) = &mut self.module {
                let device_tb = <TB as Backend>::Device::default();
                let device_nd = <NdArray as Backend>::Device::default();
                module.add_agent(&self.hidden_sizes, &device_tb);
                self.actor_targets.push(ActorMlp::new(
                    module.obs_dim,
                    &self.hidden_sizes,
                    module.act_dim,
                    &device_nd,
                ));
            }
        }

        pub fn agent_count(&self) -> usize {
            self.module.as_ref().map(|m| m.actors.len()).unwrap_or(0)
        }

        pub fn train_epoch(
            &mut self,
            agent_batches: &[AgentBatch],
            gamma: f32,
            tau: f32,
            _policy_frequency: usize,
        ) -> MultiagentDDPGTrainMetrics {
            if agent_batches.is_empty() {
                return MultiagentDDPGTrainMetrics::default();
            }

            let module = match self.module.take() {
                Some(m) => m,
                None => return MultiagentDDPGTrainMetrics::default(),
            };

            let device_tb = <TB as Backend>::Device::default();
            let device_nd = <NdArray as Backend>::Device::default();

            let mut total_actor_loss = 0.0f32;
            let mut total_critic_loss = 0.0f32;
            let mut terms = 0usize;

            let mut total_critic_loss_tensor: Option<Tensor<TB, 1, Float>> = None;

            for (agent_idx, batch) in agent_batches.iter().enumerate() {
                if agent_idx >= module.actors.len() {
                    continue;
                }
                let n = batch
                    .obs
                    .len()
                    .min(batch.act.len())
                    .min(batch.next_obs.len())
                    .min(batch.rew.len())
                    .min(batch.done.len());
                if n == 0 {
                    continue;
                }

                let obs = Tensor::<TB, 2, Float>::from_data(
                    BurnTensorData::new(flat_f32(&batch.obs[..n]), [n, module.obs_dim]),
                    &device_tb,
                );
                let act = Tensor::<TB, 2, Float>::from_data(
                    BurnTensorData::new(flat_f32(&batch.act[..n]), [n, module.act_dim]),
                    &device_tb,
                );
                let next_obs_nd = Tensor::<NdArray, 2, Float>::from_data(
                    BurnTensorData::new(flat_f32(&batch.next_obs[..n]), [n, module.obs_dim]),
                    &device_nd,
                );
                let rew = Tensor::<TB, 1, Float>::from_data(
                    BurnTensorData::new(batch.rew[..n].to_vec(), [n]),
                    &device_tb,
                );
                let done = Tensor::<TB, 1, Float>::from_data(
                    BurnTensorData::new(batch.done[..n].to_vec(), [n]),
                    &device_tb,
                );

                // Target Q using target actor and target critic (NdArray, no grad)
                if let Some(actor_tgt) = self.actor_targets.get(agent_idx) {
                    let next_act_nd = actor_tgt.forward(next_obs_nd.clone());
                    if let Some(critic_tgt) = &self.critic_target {
                        let tgt_q_nd = critic_tgt.forward(next_obs_nd, next_act_nd).reshape([n]);
                        let tgt_q_vals: Vec<f32> = tgt_q_nd
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap_or_else(|_| vec![0.0; n]);
                        let tgt_q_tb = Tensor::<TB, 1, Float>::from_data(
                            BurnTensorData::new(tgt_q_vals, [n]),
                            &device_tb,
                        );
                        let not_done = done.neg().add_scalar(1.0f32);
                        let target = rew + not_done * tgt_q_tb * gamma;

                        let cur_q = module.critic.forward(obs.clone(), act).reshape([n]);
                        let critic_loss = (cur_q - target).powf_scalar(2.0).mean();
                        total_critic_loss += scalar_f32(&critic_loss);
                        total_critic_loss_tensor = Some(match total_critic_loss_tensor {
                            Some(acc) => acc + critic_loss,
                            None => critic_loss,
                        });
                        terms += 1;
                    }
                }
                drop(obs);
            }

            if terms == 0 {
                self.module = Some(module);
                return MultiagentDDPGTrainMetrics::default();
            }

            // Critic gradient step
            let critic_loss_combined = match total_critic_loss_tensor {
                Some(l) => l,
                None => {
                    self.module = Some(module);
                    return MultiagentDDPGTrainMetrics::default();
                }
            };
            let grads_c = critic_loss_combined.backward();
            let critic_grads =
                GradientsParams::from_grads::<TB, SharedDDPGModule>(grads_c, &module);
            let module = self
                .actor_optimizer
                .step(self.critic_lr, module, critic_grads);

            // Actor loss across all agents
            let mut total_actor_loss_tensor: Option<Tensor<TB, 1, Float>> = None;
            for (agent_idx, batch) in agent_batches.iter().enumerate() {
                if agent_idx >= module.actors.len() {
                    continue;
                }
                let n = batch.obs.len().min(batch.act.len());
                if n == 0 {
                    continue;
                }
                let obs = Tensor::<TB, 2, Float>::from_data(
                    BurnTensorData::new(flat_f32(&batch.obs[..n]), [n, module.obs_dim]),
                    &device_tb,
                );
                let actor_act = module.actors[agent_idx].forward(obs.clone());
                let actor_q = module.critic.forward(obs, actor_act).reshape([n]).mean();
                let actor_loss = actor_q.neg();
                total_actor_loss += scalar_f32(&actor_loss);
                total_actor_loss_tensor = Some(match total_actor_loss_tensor {
                    Some(acc) => acc + actor_loss,
                    None => actor_loss,
                });
            }

            let module = if let Some(actor_loss_combined) = total_actor_loss_tensor {
                let grads_a = actor_loss_combined.backward();
                let actor_grads =
                    GradientsParams::from_grads::<TB, SharedDDPGModule>(grads_a, &module);
                self.actor_optimizer
                    .step(self.actor_lr, module, actor_grads)
            } else {
                module
            };

            // Soft update targets
            for (i, actor) in module.actors.iter().enumerate() {
                if let Some(tgt) = self.actor_targets.get_mut(i) {
                    soft_update_actor(actor, tgt, tau);
                }
            }
            if let Some(critic_tgt) = &mut self.critic_target {
                soft_update_critic(&module.critic, critic_tgt, tau);
            }

            self.total_it += 1;
            self.module = Some(module);

            let denom = terms.max(1) as f32;
            MultiagentDDPGTrainMetrics {
                actor_loss: total_actor_loss / denom,
                critic_loss: total_critic_loss / denom,
            }
        }
    }

    fn soft_update_actor(actor: &ActorMlp<TB>, target: &mut ActorMlp<NdArray>, tau: f32) {
        for (a_layer, t_layer) in actor.layers.iter().zip(target.layers.iter_mut()) {
            let cur_w = a_layer.weight.val().inner();
            let tgt_w = t_layer.weight.val();
            let new_w = cur_w * tau + tgt_w * (1.0 - tau);
            t_layer.weight = Param::initialized(ParamId::new(), new_w);
            if let (Some(cb), Some(tb)) = (&a_layer.bias, &mut t_layer.bias) {
                let cw = cb.val().inner();
                let tw = tb.val();
                let nw = cw * tau + tw * (1.0 - tau);
                *tb = Param::initialized(ParamId::new(), nw);
            }
        }
    }

    fn soft_update_critic(critic: &CriticMlp<TB>, target: &mut CriticMlp<NdArray>, tau: f32) {
        for (c_layer, t_layer) in critic.layers.iter().zip(target.layers.iter_mut()) {
            let cur_w = c_layer.weight.val().inner();
            let tgt_w = t_layer.weight.val();
            let new_w = cur_w * tau + tgt_w * (1.0 - tau);
            t_layer.weight = Param::initialized(ParamId::new(), new_w);
            if let (Some(cb), Some(tb)) = (&c_layer.bias, &mut t_layer.bias) {
                let cw = cb.val().inner();
                let tw = tb.val();
                let nw = cw * tau + tw * (1.0 - tau);
                *tb = Param::initialized(ParamId::new(), nw);
            }
        }
    }

    fn scalar_f32(t: &Tensor<TB, 1, Float>) -> f32 {
        t.clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0])[0]
    }

    pub fn flat_f32(tensors: &[TensorData]) -> Vec<f32> {
        tensors
            .iter()
            .flat_map(|t| bytemuck::cast_slice::<u8, f32>(&t.data).to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentBatch, MultiagentDDPGKernel};

    #[test]
    fn register_agent_tracks_count() {
        let mut kernel = MultiagentDDPGKernel::default();
        assert_eq!(kernel.agent_count(), 0);
        kernel.register_agent();
        kernel.register_agent();
        assert_eq!(kernel.agent_count(), 2);
    }

    #[test]
    fn agent_batch_returns_none_on_missing_keys() {
        use std::collections::HashMap;
        let empty = HashMap::new();
        assert!(AgentBatch::from_batch(empty).is_none());
    }
}
