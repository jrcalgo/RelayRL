use crate::templates::base_replay_buffer::{
    Batch, BatchKey, BufferSample, GenericReplayBuffer, ReplayBufferError, SampleScalars,
};
use async_trait::async_trait;
use relayrl_types::prelude::tensor::relayrl::TensorData;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

struct Buffers {
    observations: Vec<Option<TensorData>>,
    actions: Vec<Option<TensorData>>,
    next_observations: Vec<Option<TensorData>>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    pointer: usize,
    current_size: usize,
}

struct BufferMetadata {
    buffer_size: usize,
    batch_size: usize,
}

pub struct TD3ReplayBuffer {
    buffers: Arc<Mutex<Buffers>>,
    metadata: Arc<BufferMetadata>,
}

impl Default for TD3ReplayBuffer {
    fn default() -> Self {
        Self::new(1_000_000, 128)
    }
}

impl TD3ReplayBuffer {
    pub fn new(buffer_size: usize, batch_size: usize) -> Self {
        let capacity = buffer_size;
        let buffers = Buffers {
            observations: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            next_observations: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            pointer: 0,
            current_size: 0,
        };
        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            metadata: Arc::new(BufferMetadata {
                buffer_size,
                batch_size,
            }),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.metadata.batch_size
    }
}

#[async_trait]
impl GenericReplayBuffer for TD3ReplayBuffer {
    async fn insert_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<Box<dyn Any>, ReplayBufferError> {
        let mut buffers = self.buffers.lock().await;
        let capacity = self.metadata.buffer_size;
        let actions = &trajectory.actions;

        let mut episode_return = 0.0f32;
        let mut episode_length = 0i32;

        for (i, action) in actions.iter().enumerate() {
            episode_length += 1;
            let rew = action.get_rew();
            episode_return += rew;

            let obs = action.get_obs().cloned();
            let act = action.get_act().cloned();
            let done = if action.get_done() { 1.0f32 } else { 0.0f32 };

            let next_obs = if action.get_done() || i + 1 >= actions.len() {
                action.get_obs().cloned()
            } else {
                actions[i + 1].get_obs().cloned()
            };

            let ptr = buffers.pointer;
            if ptr < buffers.observations.len() {
                buffers.observations[ptr] = obs;
                buffers.actions[ptr] = act;
                buffers.next_observations[ptr] = next_obs;
                buffers.rewards[ptr] = rew;
                buffers.dones[ptr] = done;
            } else {
                buffers.observations.push(obs);
                buffers.actions.push(act);
                buffers.next_observations.push(next_obs);
                buffers.rewards.push(rew);
                buffers.dones.push(done);
            }

            buffers.pointer = (ptr + 1) % capacity;
            buffers.current_size = (buffers.current_size + 1).min(capacity);
        }

        Ok(Box::new((episode_return, episode_length)))
    }

    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        let buffers = self.buffers.lock().await;
        let current_size = buffers.current_size;
        let batch_size = self.metadata.batch_size;

        if current_size < batch_size {
            return Err(ReplayBufferError::BufferSamplingError(format!(
                "TD3 replay buffer has {current_size} transitions, need {batch_size}"
            )));
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..current_size).collect();
        indices.shuffle(&mut rng);
        indices.truncate(batch_size);

        let obs: Vec<TensorData> = indices
            .iter()
            .filter_map(|&i| buffers.observations[i].clone())
            .collect();
        let act: Vec<TensorData> = indices
            .iter()
            .filter_map(|&i| buffers.actions[i].clone())
            .collect();
        let next_obs: Vec<TensorData> = indices
            .iter()
            .filter_map(|&i| buffers.next_observations[i].clone())
            .collect();
        let rew: Vec<f32> = indices.iter().map(|&i| buffers.rewards[i]).collect();
        let done: Vec<f32> = indices.iter().map(|&i| buffers.dones[i]).collect();

        let mut batch: HashMap<BatchKey, BufferSample> = HashMap::new();
        batch.insert(BatchKey::Obs, BufferSample::Tensors(obs.into_boxed_slice()));
        batch.insert(BatchKey::Act, BufferSample::Tensors(act.into_boxed_slice()));
        batch.insert(
            BatchKey::Custom("NextObs".to_string()),
            BufferSample::Tensors(next_obs.into_boxed_slice()),
        );
        batch.insert(
            BatchKey::Custom("Rew".to_string()),
            BufferSample::Scalars(SampleScalars::F32(rew.into_boxed_slice())),
        );
        batch.insert(
            BatchKey::Custom("Done".to_string()),
            BufferSample::Scalars(SampleScalars::F32(done.into_boxed_slice())),
        );

        Ok(batch)
    }
}
