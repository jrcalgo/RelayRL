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
pub struct MultiagentPPOTrainMetrics {
    pub loss_pi: f32,
    pub delta_loss_pi: f32,
    pub loss_v: f32,
    pub delta_loss_v: f32,
    pub kl: f32,
    pub entropy: f32,
    pub clipfrac: f32,
    pub stop_iter: u64,
}

pub struct MultiagentPPOKernel {
    discrete: bool,
    obs_dim: usize,
    act_dim: usize,
    hidden_sizes: Vec<usize>,
    #[cfg(feature = "ndarray-backend")]
    trainer: Option<training::SharedTrainer>,
}

impl Default for MultiagentPPOKernel {
    fn default() -> Self {
        Self::new(1, 1, true, 3e-4, 1e-3)
    }
}

impl MultiagentPPOKernel {
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

    pub fn train_epoch(
        &mut self,
        agent_batches: &[AgentBatch],
        clip_ratio: f32,
        target_kl: f32,
        train_pi_iters: u64,
        train_vf_iters: u64,
    ) -> MultiagentPPOTrainMetrics {
        #[cfg(feature = "ndarray-backend")]
        if let Some(trainer) = &mut self.trainer {
            return trainer.train_epoch(
                agent_batches,
                self.discrete,
                clip_ratio,
                target_kl,
                train_pi_iters,
                train_vf_iters,
            );
        }

        MultiagentPPOTrainMetrics::default()
    }

    pub fn concat_sample_count(agent_batches: &[AgentBatch]) -> usize {
        agent_batches.iter().map(sample_count_for_batch).sum()
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
    ///
    /// Returns `None` if no training has occurred yet (trainer or module absent) or if
    /// no actors have been registered.
    #[cfg(feature = "ndarray-backend")]
    pub fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
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
            let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                bias_param.val().into_data().to_vec::<f32>().unwrap_or_default()
            } else {
                vec![0.0; out_dim]
            };
            specs.push((in_dim, out_dim, weights, biases));
        }
        Some(specs)
    }
}

fn sample_count_for_batch(batch: &AgentBatch) -> usize {
    batch
        .obs
        .len()
        .min(batch.act.len())
        .min(batch.adv.len())
        .min(batch.ret.len())
        .min(batch.logp_old.len())
}

#[cfg(feature = "ndarray-backend")]
mod training {
    use super::{AgentBatch, MultiagentPPOTrainMetrics, sample_count_for_batch};

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

        pub fn add_agent(&mut self, hidden_sizes: &[usize], device: &<TB as Backend>::Device) {
            self.actors.push(TrainMlp::new(
                self.obs_dim,
                hidden_sizes,
                self.act_dim,
                device,
            ));
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
            clip_ratio: f32,
            target_kl: f32,
            train_pi_iters: u64,
            train_vf_iters: u64,
        ) -> MultiagentPPOTrainMetrics {
            if !discrete || agent_batches.is_empty() {
                return MultiagentPPOTrainMetrics::default();
            }

            let mut module = match self.module.take() {
                Some(module) => module,
                None => return MultiagentPPOTrainMetrics::default(),
            };

            let device = <TB as Backend>::Device::default();
            let mut obs_concat = Vec::new();
            let mut ret_concat = Vec::new();

            for batch in agent_batches {
                let n = sample_count_for_batch(batch);
                if n == 0 {
                    continue;
                }
                obs_concat.extend(obs_flat(&batch.obs[..n]));
                ret_concat.extend_from_slice(&batch.ret[..n]);
            }

            if ret_concat.is_empty() {
                self.module = Some(module);
                return MultiagentPPOTrainMetrics::default();
            }

            let total_samples = ret_concat.len();
            let vf_scale = if self.pi_lr.abs() > f64::EPSILON {
                (self.vf_lr / self.pi_lr) as f32
            } else {
                1.0
            };

            let mut first_pi_loss: Option<f32> = None;
            let mut final_pi_loss = 0.0f32;
            let mut final_kl = 0.0f32;
            let mut final_entropy = 0.0f32;
            let mut final_clipfrac = 0.0f32;
            let mut stop_iter = 0u64;
            let mut policy_stopped = false;

            let mut first_vf_loss: Option<f32> = None;
            let mut final_vf_loss = 0.0f32;

            let combined_iters = train_pi_iters.max(train_vf_iters);
            for iter in 0..combined_iters {
                let mut total_policy_loss: Option<Tensor<TB, 1, Float>> = None;
                let mut pi_loss_sum = 0.0f32;
                let mut kl_sum = 0.0f32;
                let mut entropy_sum = 0.0f32;
                let mut clipfrac_sum = 0.0f32;
                let mut policy_terms = 0usize;

                if !policy_stopped && iter < train_pi_iters {
                    for (agent_index, batch) in agent_batches.iter().enumerate() {
                        if agent_index >= module.actors.len() {
                            continue;
                        }

                        let n = sample_count_for_batch(batch);
                        if n == 0 {
                            continue;
                        }

                        let obs = Tensor::<TB, 2, Float>::from_data(
                            BurnTensorData::new(obs_flat(&batch.obs[..n]), [n, module.obs_dim]),
                            &device,
                        );
                        let logits = module.actors[agent_index].forward(obs);
                        let log_probs = log_softmax(logits, 1);
                        let act = Tensor::<TB, 2, Int>::from_data(
                            BurnTensorData::new(action_indices(&batch.act[..n]), [n, 1]),
                            &device,
                        );
                        let logp = log_probs.gather(1, act).reshape([n]);
                        let adv = Tensor::<TB, 1, Float>::from_data(
                            BurnTensorData::new(batch.adv[..n].to_vec(), [n]),
                            &device,
                        );
                        let logp_old_values = scalar_tensor_values(&batch.logp_old[..n]);
                        let logp_old = Tensor::<TB, 1, Float>::from_data(
                            BurnTensorData::new(logp_old_values.clone(), [n]),
                            &device,
                        );
                        let ratio = (logp.clone() - logp_old).exp();
                        let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);
                        let unclipped = ratio.clone() * adv.clone();
                        let clipped = clipped_ratio * adv;
                        let loss_pi = unclipped.min_pair(clipped).mean().neg();

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

                        total_policy_loss = Some(match total_policy_loss {
                            Some(accumulated) => accumulated + loss_pi.clone(),
                            None => loss_pi.clone(),
                        });
                        pi_loss_sum += scalar_from_loss(&loss_pi);
                        kl_sum += approx_kl;
                        entropy_sum += entropy;
                        clipfrac_sum += clipfrac;
                        policy_terms += 1;
                    }
                }

                let mut value_loss_tensor: Option<Tensor<TB, 1, Float>> = None;
                if iter < train_vf_iters {
                    let obs_all = Tensor::<TB, 2, Float>::from_data(
                        BurnTensorData::new(obs_concat.clone(), [total_samples, module.obs_dim]),
                        &device,
                    );
                    let ret_all = Tensor::<TB, 1, Float>::from_data(
                        BurnTensorData::new(ret_concat.clone(), [total_samples]),
                        &device,
                    );
                    let vf_prediction = module.critic.forward(obs_all).reshape([total_samples]);
                    let loss_vf = (vf_prediction - ret_all).powf_scalar(2.0).mean();
                    let loss_v_value = scalar_from_loss(&loss_vf);
                    first_vf_loss.get_or_insert(loss_v_value);
                    final_vf_loss = loss_v_value;
                    value_loss_tensor = Some(loss_vf);
                }

                if policy_terms > 0 {
                    let denom = policy_terms as f32;
                    let loss_pi_value = pi_loss_sum / denom;
                    first_pi_loss.get_or_insert(loss_pi_value);
                    final_pi_loss = loss_pi_value;
                    final_kl = kl_sum / denom;
                    final_entropy = entropy_sum / denom;
                    final_clipfrac = clipfrac_sum / denom;
                    stop_iter = iter + 1;
                    if final_kl > 1.5 * target_kl {
                        policy_stopped = true;
                    }
                }

                if total_policy_loss.is_none() && value_loss_tensor.is_none() {
                    break;
                }

                let total_loss = match (total_policy_loss, value_loss_tensor) {
                    (Some(policy_loss), Some(loss_vf)) => policy_loss + loss_vf * vf_scale,
                    (Some(policy_loss), None) => policy_loss,
                    (None, Some(loss_vf)) => loss_vf * vf_scale,
                    (None, None) => break,
                };

                let grads = total_loss.backward();
                let grads_params =
                    GradientsParams::from_grads::<TB, SharedTrainModule>(grads, &module);
                module = self.optimizer.step(self.pi_lr, module, grads_params);
            }

            self.module = Some(module);

            let first_pi_loss = first_pi_loss.unwrap_or(final_pi_loss);
            let first_vf_loss = first_vf_loss.unwrap_or(final_vf_loss);
            MultiagentPPOTrainMetrics {
                loss_pi: final_pi_loss,
                delta_loss_pi: final_pi_loss - first_pi_loss,
                loss_v: final_vf_loss,
                delta_loss_v: final_vf_loss - first_vf_loss,
                kl: final_kl,
                entropy: final_entropy,
                clipfrac: final_clipfrac,
                stop_iter,
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
    use super::{AgentBatch, MultiagentPPOKernel};
    use relayrl_types::data::tensor::{DType, SupportedTensorBackend, TensorData};

    #[cfg(feature = "ndarray-backend")]
    use relayrl_types::data::tensor::NdArrayDType;
    #[cfg(all(feature = "tch-backend", not(feature = "ndarray-backend")))]
    use relayrl_types::data::tensor::TchDType;

    fn dummy_tensor_data() -> TensorData {
        #[cfg(feature = "ndarray-backend")]
        {
            return TensorData::new(
                vec![1],
                DType::NdArray(NdArrayDType::F32),
                0.0f32.to_le_bytes().to_vec(),
                SupportedTensorBackend::NdArray,
            );
        }

        #[cfg(all(feature = "tch-backend", not(feature = "ndarray-backend")))]
        {
            return TensorData::new(
                vec![1],
                DType::Tch(TchDType::F32),
                0.0f32.to_le_bytes().to_vec(),
                SupportedTensorBackend::Tch,
            );
        }
    }

    #[test]
    fn register_agent_tracks_actor_count() {
        let mut kernel = MultiagentPPOKernel::default();
        assert_eq!(kernel.agent_count(), 0);

        kernel.register_agent();
        kernel.register_agent();

        assert_eq!(kernel.agent_count(), 2);
    }

    #[test]
    fn concat_sample_count_sums_across_agents() {
        let batches = vec![
            AgentBatch {
                obs: vec![dummy_tensor_data(), dummy_tensor_data()],
                act: vec![dummy_tensor_data(), dummy_tensor_data()],
                adv: vec![1.0, 2.0],
                ret: vec![1.0, 2.0],
                logp_old: vec![dummy_tensor_data(), dummy_tensor_data()],
            },
            AgentBatch {
                obs: vec![dummy_tensor_data()],
                act: vec![dummy_tensor_data()],
                adv: vec![3.0],
                ret: vec![3.0],
                logp_old: vec![dummy_tensor_data()],
            },
        ];

        assert_eq!(MultiagentPPOKernel::concat_sample_count(&batches), 3);
    }
}
