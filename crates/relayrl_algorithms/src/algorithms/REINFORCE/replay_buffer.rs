use crate::algorithms::{compute_normed_advantages, discounted_cumsum, scalar_stats};
use crate::templates::base_replay_buffer::{
    Batch, BatchKey, BufferSample, BufferTensors, GenericReplayBuffer, ReplayBufferError,
    SampleScalars,
};
use async_trait::async_trait;
use relayrl_types::prelude::action::RelayRLData;
use relayrl_types::prelude::tensor::relayrl::TensorData;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

struct Buffers {
    observations: BufferTensors,
    actions: BufferTensors,
    masks: BufferTensors,
    rewards: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    logprobs: BufferTensors,
    values: Option<Vec<f32>>,
}

struct BufferMetadata {
    gamma: f32,
    lambda: f32,
    with_vf_baseline: bool,
    buffer_size: usize,
    buffer_pointer: AtomicUsize,
    buffer_path_start_idx: AtomicUsize,
}

pub struct ReinforceReplayBuffer {
    buffers: Arc<Mutex<Buffers>>,
    metadata: Arc<BufferMetadata>,
}

impl ReinforceReplayBuffer {
    pub fn new(buffer_size: usize, gamma: f32, lambda: f32, with_vf_baseline: bool) -> Self {
        let buffers = Buffers {
            observations: Vec::with_capacity(buffer_size),
            actions: Vec::with_capacity(buffer_size),
            masks: Vec::with_capacity(buffer_size),
            rewards: Vec::with_capacity(buffer_size),
            advantages: Vec::with_capacity(buffer_size),
            returns: Vec::with_capacity(buffer_size),
            logprobs: Vec::with_capacity(buffer_size),
            values: with_vf_baseline.then(|| Vec::with_capacity(buffer_size)),
        };

        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            metadata: Arc::new(BufferMetadata {
                gamma,
                lambda,
                with_vf_baseline,
                buffer_size,
                buffer_pointer: AtomicUsize::new(0),
                buffer_path_start_idx: AtomicUsize::new(0),
            }),
        }
    }

    fn tensor_scalar_f32(data: &TensorData) -> f32 {
        let values: &[f32] = bytemuck::cast_slice(&data.data);
        values.first().copied().unwrap_or(0.0)
    }

    fn finish_path(&self, buffers: &mut Buffers, final_value: Option<f32>) {
        let final_value = final_value.unwrap_or(0.0);
        let start = self.metadata.buffer_path_start_idx.load(Ordering::SeqCst);
        let end = self.metadata.buffer_pointer.load(Ordering::SeqCst);
        if start >= end {
            return;
        }
        let slice = start..end;

        if self.metadata.with_vf_baseline {
            let mut rewards = buffers.rewards[slice.clone()].to_vec();
            let mut values = buffers
                .values
                .as_ref()
                .map(|v| v[slice.clone()].to_vec())
                .unwrap_or_default();
            rewards.push(final_value);
            values.push(final_value);

            let deltas: Vec<f32> = (0..rewards.len() - 1)
                .map(|i| rewards[i] + self.metadata.gamma * values[i + 1] - values[i])
                .collect();
            let advantages = discounted_cumsum(&deltas, self.metadata.gamma * self.metadata.lambda);
            buffers.advantages[slice.clone()].copy_from_slice(&advantages);
            buffers.returns[slice.clone()].copy_from_slice(&discounted_cumsum(
                &buffers.rewards[slice.clone()],
                self.metadata.gamma,
            ));
        } else {
            let rewards = &buffers.rewards[slice.clone()];
            let advantages = discounted_cumsum(rewards, self.metadata.gamma);
            let returns = discounted_cumsum(rewards, self.metadata.gamma);
            buffers.advantages[slice.clone()].copy_from_slice(&advantages);
            buffers.returns[slice.clone()].copy_from_slice(&returns);
        }

        self.metadata
            .buffer_path_start_idx
            .store(end, Ordering::SeqCst);
    }
}

#[async_trait]
impl GenericReplayBuffer for ReinforceReplayBuffer {
    async fn insert_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<Box<dyn Any>, ReplayBufferError> {
        let mut buffers = self.buffers.lock().await;
        let mut episode_return = 0.0f32;
        let mut episode_length = 0i32;

        for action in &trajectory.actions {
            episode_length += 1;
            let reward = action.get_rew();
            episode_return += reward;

            buffers.observations.push(action.get_obs().cloned());
            buffers.actions.push(action.get_act().cloned());
            buffers.masks.push(action.get_mask().cloned());
            buffers.logprobs.push(None);
            buffers.rewards.push(reward);
            buffers.advantages.push(0.0);
            buffers.returns.push(0.0);

            if let Some(map) = action.get_data() {
                if let Some(RelayRLData::Tensor(logp)) = map.get("logp_a")
                    && let Some(slot) = buffers.logprobs.last_mut()
                {
                    *slot = Some(logp.clone());
                }
                if self.metadata.with_vf_baseline {
                    let value = match map.get("val") {
                        Some(RelayRLData::Tensor(val)) => Self::tensor_scalar_f32(val),
                        _ => 0.0,
                    };
                    if let Some(values) = buffers.values.as_mut() {
                        values.push(value);
                    }
                }
            } else if self.metadata.with_vf_baseline
                && let Some(values) = buffers.values.as_mut()
            {
                values.push(0.0);
            }

            let next = self.metadata.buffer_pointer.load(Ordering::SeqCst) + 1;
            self.metadata.buffer_pointer.store(next, Ordering::SeqCst);

            if action.get_done() {
                self.finish_path(&mut buffers, Some(reward));
            }
        }

        Ok(Box::new((episode_return, episode_length)))
    }

    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        let mut buffers = self.buffers.lock().await;
        let capacity = self.metadata.buffer_pointer.load(Ordering::SeqCst);
        if capacity == 0 {
            return Err(ReplayBufferError::BufferSamplingError(
                "Replay buffer is empty".to_string(),
            ));
        }
        if capacity > self.metadata.buffer_size {
            return Err(ReplayBufferError::BufferSamplingError(
                "Replay buffer capacity exceeded".to_string(),
            ));
        }

        let adv_raw = &buffers.advantages[..capacity];
        let (adv_mean, adv_std) = scalar_stats(adv_raw);
        let adv_norm = compute_normed_advantages(adv_raw, adv_mean, adv_std.max(1e-8));

        let obs: Vec<TensorData> = buffers.observations[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let act: Vec<TensorData> = buffers.actions[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let mask: Vec<TensorData> = buffers.masks[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let logp: Vec<TensorData> = buffers.logprobs[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let ret: Vec<f32> = buffers.returns[..capacity].to_vec();

        self.metadata.buffer_pointer.store(0, Ordering::SeqCst);
        self.metadata.buffer_path_start_idx.store(0, Ordering::SeqCst);
        buffers.observations.clear();
        buffers.actions.clear();
        buffers.masks.clear();
        buffers.rewards.clear();
        buffers.advantages.clear();
        buffers.returns.clear();
        buffers.logprobs.clear();
        if let Some(values) = buffers.values.as_mut() {
            values.clear();
        }

        let mut batch: HashMap<BatchKey, BufferSample> = HashMap::new();
        batch.insert(BatchKey::Obs, BufferSample::Tensors(obs.into_boxed_slice()));
        batch.insert(BatchKey::Act, BufferSample::Tensors(act.into_boxed_slice()));
        batch.insert(BatchKey::Mask, BufferSample::Tensors(mask.into_boxed_slice()));
        batch.insert(
            BatchKey::Custom("Adv".to_string()),
            BufferSample::Scalars(SampleScalars::F32(adv_norm.into_boxed_slice())),
        );
        batch.insert(
            BatchKey::Custom("Ret".to_string()),
            BufferSample::Scalars(SampleScalars::F32(ret.into_boxed_slice())),
        );
        batch.insert(
            BatchKey::Custom("LogP".to_string()),
            BufferSample::Tensors(logp.into_boxed_slice()),
        );

        Ok(batch)
    }
}
