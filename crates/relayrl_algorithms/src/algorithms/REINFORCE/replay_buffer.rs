use crate::templates::base_replay_buffer::{BufferTensors, BatchKey, BufferSample, SampleScalars, Batch, GenericReplayBuffer, ReplayBufferError};

use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::action::RelayRLData;
use relayrl_types::prelude::tensor::relayrl::{TensorData, BackendMatcher};

use burn_tensor::backend::Backend;
use tokio::sync::Mutex;
use tokio::sync::MutexGuard;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::collections::HashMap;

struct Buffers {
    observations: BufferTensors,
    actions: BufferTensors,
    masks: BufferTensors,
    rewards: Vec<f32>,
    advantages: BufferTensors,
    returns: Vec<f32>,
    logprobs: BufferTensors,
    values: Option<BufferTensors>,
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
    metadata: Arc<BufferMetadata>
}

impl ReinforceReplayBuffer {
    pub fn new(buffer_size: usize, gamma: f32, lambda: f32, with_vf_baseline: bool) -> Self {
        let buffers = Buffers {
            observations: BufferTensors::new(),
            actions: BufferTensors::new(),
            masks: BufferTensors::new(),
            advantages: BufferTensors::new(),
            rewards: Vec::<f32>::new(),
            returns: Vec::<f32>::new(),
            logprobs: BufferTensors::new(),
            values: {
                match with_vf_baseline {
                    true => Some(BufferTensors::new()),
                    false => None,
                }
            }
        };

        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            metadata: Arc::new(BufferMetadata {
                gamma,
                lambda,
                with_vf_baseline,
                buffer_size,
                buffer_pointer: AtomicUsize::new(0),
                buffer_path_start_idx: AtomicUsize::new(0)
            }),
        }

    }

    async fn compute_path_end(&self, buffer_lock: MutexGuard<Buffers>, final_value: Option<f32>) {
        let final_value = final_value.unwrap_or(0.0);

        let start = self.metadata.buffer_path_start_idx.load(Ordering::SeqCst);
        let end = self.metadata.buffer_pointer.load(Ordering::SeqCst);

        let slice = start..end;
        let slice = start..end;

        if self.metadata.with_vf_baseline {
            let rewards = buffer_lock.rewards[slice.clone()].to_vec();
            let values = buffer_lock.values[slice.clone()].to_vec();
            rewards.push(final_value);
            values.push(final_value);

            let deltas = (0..rewards.len()-1).map(|i| rewards[i] + self.metadata.gamma * vals[i+1] - values[i]).collect();

            let advantages = discounted_cumsum(&deltas, self.metadata.gamma * self.metadata.lambda)?;
            buffer_lock.advantages[slice.clone()].copy_from_slice(&advantages);
        } else {
            let rewards = buffer_lock.rewards[slice.clone()];

            let advantages = discounted_cumsum(&rewards, self.metadata.gamma)?;
            buffer_lock.advantages[slice.clone()].copy_from_slice(&advantages);

            let returns = discounted_cumsum(&rewards, self.metadata.gamma)?;
            buffer_lock.returns[slice.clone()].copy_from_slice(&returns);
        }

        self.metadata.buffer_path_start_idx.store(end, Ordering::SeqCst);
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> GenericReplayBuffer<B> for ReinforceReplayBuffer {
    async fn insert_trajectory(&self, trajectory: RelayRLTrajectory) -> Result<(i64, u64), ReplayBufferError> {
        let buffer_lock = self.buffers.lock().await?;
        let (mut episode_return, mut episode_length) = (0, 0);

        for action in trajectory.actions.iter() {
            episode_length += 1;

            match action.get_obs() {
                Some(obs) => {
                    buffer_lock.observations.push(Some(obs));
                },
                None => buffer_lock.observations.push(None),
            };

            match action.get_act() {
                Some(act) => {
                    buffer_lock.actions.push(Some(act));
                },
                None => buffer_lock.actions.push(None),
            };

            match action.get_mask() {
                Some(mask) => {
                    buffer_lock.masks.push(Some(mask));
                },
                None => buffer_lock.masks.push(None),
            };

            let reward = action.get_rew();
            if !action.get_done() {
                buffer_lock.rewards.push(reward);
            } else {
                self.compute_path_end(reward).await
            };
            episode_return += reward;

            match action.get_data() {
                Some(data_map) => {
                    match data_map.get("logp_a") {
                        Some(data) => {
                                match data {
                                    RelayRLData::Tensor(logp_a) => {
                                        buffer_lock.logprobs.push(Some(logp_a));
                                    },
                                    _ => return Err(ReplayBufferError::TrajectoryInsertionError("`LogProb` expected to be RelayRLData::Tensor".to_string())),
                                }
                        },
                        None => buffer_lock.logprobs.push(None),
                    }

                    if self.with_vf_baseline {
                        match data_map.get("val") {
                            Some(data) => {
                                match data {
                                    RelayRLData::Tensor(val) => {
                                        buffer_lock.values.push(Some(val));
                                    },
                                    _ => return Err(ReplayBufferError::TrajectoryInsertionError("`Val` expected to be RelayRLData::Tensor".to_string())),
                                }
                            },
                            None => buffer_lock.values.push(None),
                        }
                    }
                },
                None => {
                    buffer_lock.logprobs.push(None);

                    if self.with_vf_baseline {
                        buffer_lock.values.push(None);
                    }
                }
            };

            self.metadata.buffer_pointer.store((self.metadata.buffer_lock.load(Ordering::SeqCst) + 1) as usize, Ordering::SeqCst);
        }

        Ok((episode_return, episode_length))
    }

    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        assert!(self.metadata.buffer_pointer < self.metadata.buffer_size);

        let buffer_lock = self.buffers.lock().await;

        let capacity = self.metadata.buffer_pointer.load(Ordering::SeqCst);
        self.metadata.buffer_pointer.store(0, Ordering::SeqCst);
        self.metadata.buffer_path_start_idx.store(0, Ordering::SeqCst);

        let boxed_adv: Box<[TensorData]> = {
            let adv_buffer = &buffer_lock.advantages;
            let adv_at_capacity = adv_buffer[..capacity];

            let (adv_mean, adv_std) = scalar_stats(adv_at_capacity)?;

            let advantages = compute_normed_advantages(&adv_at_capacity, adv_mean, adv_std)?;

            advantages.to_vec().to_boxed_slice()
        };

        let boxed_obs: Box<[TensorData]> = {
            let obs_buffer = &buffer_lock.observations;
            let obs_at_capacity = obs_buffer[..capacity];

            obs_at_capacity.to_vec().to_boxed_slice()
        };

        let boxed_act: Box<[TensorData]> = {
            let act_buffer = &buffer_lock.actions;
            let act_at_capacity = act_buffer[..capacity];

            act_at_capacity.to_vec().to_boxed_slice()
        };

        let boxed_mask: Box<[TensorData]> = {
            let mask_buffer = &buffer_lock.masks;
            let mask_at_capacity = mask_buffer[..capacity];

            mask_at_capacity.to_vec().to_boxed_slice()
        };

        let boxed_ret: Box<[f32]> = {
            let ret_buffer = &buffer_lock.returns;
            let ret_at_capacity = ret_buffer[..capacity];

            ret_at_capacity.to_vec().to_boxed_slice()
        };

        let boxed_logp: Box<[TensorData]> = {
            let logp_buffer = &buffer_lock.logprobs;
            let logp_at_capacity = logp_buffer[..capacity];

            logp_at_capacity.to_vec().to_boxed_slice()
        };

        let mut batch: HashMap<BatchKey, BufferSample> = HashMap::new();

        batch.insert(BatchKey::Obs, BufferSample::Tensors(boxed_obs));
        batch.insert(BatchKey::Act, BufferSample::Tensors(boxed_act));
        batch.insert(BatchKey::Mask, BufferSample::Tensors(boxed_mask));
        batch.insert(BatchKey::Custom("Adv".to_string()), BufferSample::Tensors(boxed_adv));
        batch.insert(BatchKey::Custom("Ret".to_string()), BufferSample::Scalars(SampleScalars::F32(boxed_ret)));
        batch.insert(BatchKey::Custom("LogP".to_string()), BufferSample::Tensors(boxed_logp));

        Ok(batch)
    }
}
