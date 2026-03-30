use crate::templates::base_replay_buffer::{Batch, BatchKey, BufferSample, SampleScalars};

use relayrl_types::prelude::tensor::relayrl::TensorData;

#[derive(Clone, Debug, Default)]
pub struct AgentBatch {
    pub obs: Vec<TensorData>,
    pub act: Vec<TensorData>,
    pub adv: Vec<f32>,
    pub ret: Vec<f32>,
    pub logp_old: Vec<TensorData>,
}

impl AgentBatch {
    pub fn from_batch(mut batch: Batch) -> Option<Self> {
        let obs = match batch.remove(&BatchKey::Obs) {
            Some(BufferSample::Tensors(tensors)) => Vec::from(tensors),
            _ => return None,
        };
        let act = match batch.remove(&BatchKey::Act) {
            Some(BufferSample::Tensors(tensors)) => Vec::from(tensors),
            _ => return None,
        };
        let adv = match batch.remove(&BatchKey::Custom("Adv".to_string())) {
            Some(BufferSample::Scalars(SampleScalars::F32(values))) => Vec::from(values),
            _ => return None,
        };
        let ret = match batch.remove(&BatchKey::Custom("Ret".to_string())) {
            Some(BufferSample::Scalars(SampleScalars::F32(values))) => Vec::from(values),
            _ => return None,
        };
        let logp_old = match batch.remove(&BatchKey::Custom("LogP".to_string())) {
            Some(BufferSample::Tensors(tensors)) => Vec::from(tensors),
            _ => Vec::new(),
        };

        Some(Self {
            obs,
            act,
            adv,
            ret,
            logp_old,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MultiagentTrainMetrics {
    pub loss_pi: f32,
    pub loss_v: f32,
    pub kl: f32,
    pub entropy: f32,
}

pub struct MultiagentReinforceKernel {
    discrete: bool,
    obs_dim: usize,
    act_dim: usize,
    hidden_sizes: Vec<usize>,
    #[cfg(feature = "ndarray-backend")]
    trainer: Option<training::SharedTrainer>,
}

impl Default for MultiagentReinforceKernel {
    fn default() -> Self {
        Self::new(1, 1, true, 3e-4, 1e-3)
    }
}

impl MultiagentReinforceKernel {
    pub fn new(obs_dim: usize, act_dim: usize, discrete: bool, pi_lr: f32, vf_lr: f32) -> Self {
        let hidden_sizes = vec![64, 64];

        Self {
            discrete,
            obs_dim,
            act_dim,
            hidden_sizes: hidden_sizes.clone(),
            #[cfg(feature = "ndarray-backend")]
            trainer: Some(training::SharedTrainer::new(
                obs_dim,
                &hidden_sizes,
                act_dim,
                pi_lr as f64,
                vf_lr as f64,
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

    pub fn train_epoch(&mut self, agent_batches: &[AgentBatch]) -> MultiagentTrainMetrics {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.trainer {
            return trainer.train_epoch(agent_batches, self.discrete);
        }

        MultiagentTrainMetrics::default()
    }

    pub fn concat_sample_count(agent_batches: &[AgentBatch]) -> usize {
        agent_batches.iter().map(|batch| batch.ret.len()).sum()
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
}

#[cfg(feature = "ndarray-backend")]
mod training {
    use super::{AgentBatch, MultiagentTrainMetrics};

    extern crate burn_core as burn;

    use burn_autodiff::Autodiff;
    use burn_core::module::Module;
    use burn_ndarray::NdArray;
    use burn_nn::{Linear, LinearConfig, Relu};
    use burn_optim::adaptor::OptimizerAdaptor;
    use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer};
    use burn_tensor::activation::log_softmax;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Float, Int, Tensor, TensorData as BurnTensorData};
    use relayrl_types::prelude::tensor::relayrl::TensorData;

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

    #[derive(Module, Debug, Clone)]
    pub struct SharedTrainModule {
        pub actors: Vec<TrainMlp<TB>>,
        pub critic: TrainMlp<TB>,
        pub obs_dim: usize,
        pub act_dim: usize,
    }

    impl SharedTrainModule {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            agent_count: usize,
            device: &<TB as Backend>::Device,
        ) -> Self {
            let actors = (0..agent_count)
                .map(|_| TrainMlp::new(obs_dim, hidden_sizes, act_dim, device))
                .collect();

            Self {
                actors,
                critic: TrainMlp::new(obs_dim, hidden_sizes, 1, device),
                obs_dim,
                act_dim,
            }
        }

        pub fn add_agent(
            &mut self,
            hidden_sizes: &[usize],
            device: &<TB as Backend>::Device,
        ) {
            self.actors
                .push(TrainMlp::new(self.obs_dim, hidden_sizes, self.act_dim, device));
        }
    }

    pub struct SharedTrainer {
        pub module: Option<SharedTrainModule>,
        pub optimizer: OptimizerAdaptor<Adam, SharedTrainModule, TB>,
        pub pi_lr: f64,
        pub vf_lr: f64,
        hidden_sizes: Vec<usize>,
    }

    impl SharedTrainer {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            pi_lr: f64,
            vf_lr: f64,
        ) -> Self {
            let device = <TB as Backend>::Device::default();

            Self {
                module: Some(SharedTrainModule::new(
                    obs_dim,
                    hidden_sizes,
                    act_dim,
                    0,
                    &device,
                )),
                optimizer: AdamConfig::new().init::<TB, SharedTrainModule>(),
                pi_lr,
                vf_lr,
                hidden_sizes: hidden_sizes.to_vec(),
            }
        }

        pub fn add_agent(&mut self) {
            if let Some(module) = &mut self.module {
                let device = <TB as Backend>::Device::default();
                module.add_agent(&self.hidden_sizes, &device);
            }
        }

        pub fn agent_count(&self) -> usize {
            self.module
                .as_ref()
                .map(|module| module.actors.len())
                .unwrap_or(0)
        }

        pub fn train_epoch(
            &mut self,
            agent_batches: &[AgentBatch],
            discrete: bool,
        ) -> MultiagentTrainMetrics {
            if !discrete || agent_batches.is_empty() {
                return MultiagentTrainMetrics::default();
            }

            let module = match self.module.take() {
                Some(module) => module,
                None => return MultiagentTrainMetrics::default(),
            };
            let device = <TB as Backend>::Device::default();
            let mut total_policy_loss: Option<Tensor<TB, 1, Float>> = None;
            let mut policy_loss_sum = 0.0f32;
            let mut kl_sum = 0.0f32;
            let mut entropy_sum = 0.0f32;
            let mut policy_terms = 0usize;
            let mut obs_concat = Vec::new();
            let mut ret_concat = Vec::new();

            for (agent_index, batch) in agent_batches.iter().enumerate() {
                if agent_index >= module.actors.len() {
                    continue;
                }

                let n = batch
                    .obs
                    .len()
                    .min(batch.act.len())
                    .min(batch.adv.len())
                    .min(batch.ret.len());
                if n == 0 {
                    continue;
                }

                let obs_flat = obs_flat(&batch.obs[..n]);
                let obs = Tensor::<TB, 2, Float>::from_data(
                    BurnTensorData::new(obs_flat.clone(), [n, module.obs_dim]),
                    &device,
                );
                let logits = module.actors[agent_index].forward(obs);
                let log_probs = log_softmax(logits, 1);
                let act_indices = action_indices(&batch.act[..n]);
                let act = Tensor::<TB, 2, Int>::from_data(
                    BurnTensorData::new(act_indices, [n, 1]),
                    &device,
                );
                let logp = log_probs.gather(1, act).reshape([n]);
                let adv = Tensor::<TB, 1, Float>::from_data(
                    BurnTensorData::new(batch.adv[..n].to_vec(), [n]),
                    &device,
                );
                let loss_pi = (logp.clone() * adv).mean().neg();
                let loss_pi_value = scalar_from_loss(&loss_pi);
                let logp_values = logp
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_else(|_| vec![0.0; n]);
                let entropy = -logp_values.iter().sum::<f32>() / n as f32;
                let approx_kl = if batch.logp_old.len() >= n {
                    let old_values = scalar_tensor_values(&batch.logp_old[..n]);
                    old_values
                        .iter()
                        .zip(logp_values.iter())
                        .map(|(old, new)| old - new)
                        .sum::<f32>()
                        / n as f32
                } else {
                    0.0
                };

                total_policy_loss = Some(match total_policy_loss {
                    Some(accumulated) => accumulated + loss_pi,
                    None => loss_pi,
                });
                policy_loss_sum += loss_pi_value;
                kl_sum += approx_kl;
                entropy_sum += entropy;
                policy_terms += 1;
                obs_concat.extend(obs_flat);
                ret_concat.extend_from_slice(&batch.ret[..n]);
            }

            if ret_concat.is_empty() {
                self.module = Some(module);
                return MultiagentTrainMetrics::default();
            }

            let total_samples = ret_concat.len();
            let obs_all = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_concat, [total_samples, module.obs_dim]),
                &device,
            );
            let ret_all = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(ret_concat, [total_samples]),
                &device,
            );
            let vf_prediction = module.critic.forward(obs_all).reshape([total_samples]);
            let loss_vf = (vf_prediction - ret_all).powf_scalar(2.0).mean();
            let loss_v_value = scalar_from_loss(&loss_vf);
            let vf_scale = if self.pi_lr.abs() > f64::EPSILON {
                (self.vf_lr / self.pi_lr) as f32
            } else {
                1.0
            };
            let total_loss = match total_policy_loss {
                Some(policy_loss) => policy_loss + loss_vf.clone() * vf_scale,
                None => loss_vf.clone() * vf_scale,
            };
            let grads = total_loss.backward();
            let grads_params =
                GradientsParams::from_grads::<TB, SharedTrainModule>(grads, &module);
            let module = self.optimizer.step(self.pi_lr, module, grads_params);
            self.module = Some(module);

            let denom = policy_terms.max(1) as f32;
            MultiagentTrainMetrics {
                loss_pi: policy_loss_sum / denom,
                loss_v: loss_v_value,
                kl: kl_sum / denom,
                entropy: entropy_sum / denom,
            }
        }
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

#[cfg(test)]
mod tests {
    use super::{AgentBatch, MultiagentReinforceKernel};

    #[test]
    fn register_agent_tracks_actor_count() {
        let mut kernel = MultiagentReinforceKernel::default();
        assert_eq!(kernel.agent_count(), 0);

        kernel.register_agent();
        kernel.register_agent();

        assert_eq!(kernel.agent_count(), 2);
    }

    #[test]
    fn concat_sample_count_sums_across_agents() {
        let batches = vec![
            AgentBatch {
                ret: vec![1.0, 2.0],
                ..Default::default()
            },
            AgentBatch {
                ret: vec![3.0],
                ..Default::default()
            },
        ];

        assert_eq!(MultiagentReinforceKernel::concat_sample_count(&batches), 3);
    }
}
