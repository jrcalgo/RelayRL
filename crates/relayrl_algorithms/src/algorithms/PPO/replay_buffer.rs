use crate::algorithms::{compute_normed_advantages, discounted_cumsum, scalar_stats};
use crate::templates::base_replay_buffer::{Batch, GenericReplayBuffer, ReplayBufferError};
use async_trait::async_trait;
use relayrl_types::prelude::action::RelayRLData;
use relayrl_types::prelude::tensor::relayrl::TensorData;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Buffers {
    obs: Vec<TensorData>,
    obs_dim: usize,
    act: Vec<TensorData>,
    logp: Vec<f32>,
    rewards: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    values: Vec<f32>,
    episode_boundaries: Vec<(usize, usize, bool)>,
    episode_versions: Vec<i64>,
}

struct BufferMetadata {
    gamma: f32,
    lam: f32,
    _buffer_size: usize,
    buffer_pointer: AtomicUsize,
    buffer_path_start_idx: AtomicUsize,
}

pub struct PPOBatch {
    pub obs: Vec<TensorData>,
    pub obs_dim: usize,
    pub act: Vec<TensorData>,
    pub logp: Vec<f32>,
    pub adv_norm: Vec<f32>,
    pub ret: Vec<f32>,
    pub val: Vec<f32>,
}

pub struct PPOReplayBuffer {
    buffers: Arc<Mutex<Buffers>>,
    metadata: Arc<BufferMetadata>,
    max_buffered_episodes: Option<usize>,
}

impl Default for PPOReplayBuffer {
    fn default() -> Self {
        Self::new(1_000, 0.99, 0.97, None)
    }
}

impl PPOReplayBuffer {
    pub fn new(
        buffer_size: usize,
        gamma: f32,
        lam: f32,
        max_buffered_episodes: Option<usize>,
    ) -> Self {
        let buffers = Buffers {
            obs: Vec::with_capacity(buffer_size),
            obs_dim: 0,
            act: Vec::with_capacity(buffer_size),
            logp: Vec::with_capacity(buffer_size),
            rewards: Vec::with_capacity(buffer_size),
            advantages: Vec::with_capacity(buffer_size),
            returns: Vec::with_capacity(buffer_size),
            values: Vec::with_capacity(buffer_size),
            episode_boundaries: Vec::new(),
            episode_versions: Vec::new(),
        };
        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            metadata: Arc::new(BufferMetadata {
                gamma,
                lam,
                _buffer_size: buffer_size,
                buffer_pointer: AtomicUsize::new(0),
                buffer_path_start_idx: AtomicUsize::new(0),
            }),
            max_buffered_episodes,
        }
    }

    fn compute_gae_episode(
        buffers: &mut Buffers,
        gamma: f32,
        lam: f32,
        start: usize,
        end: usize,
        bootstrap: f32,
    ) {
        if start >= end {
            return;
        }
        let mut rews = buffers.rewards[start..end].to_vec();
        let mut vals = buffers.values[start..end].to_vec();
        rews.push(bootstrap);
        vals.push(bootstrap);

        let deltas: Vec<f32> = (0..rews.len() - 1)
            .map(|i| rews[i] + gamma * vals[i + 1] - vals[i])
            .collect();
        let advantages = discounted_cumsum(&deltas, gamma * lam);
        buffers.advantages[start..end].copy_from_slice(&advantages);

        let full_returns = discounted_cumsum(&rews, gamma);
        buffers.returns[start..end].copy_from_slice(&full_returns[..full_returns.len() - 1]);
    }

    pub fn get_obs_for_gae_blocking(&self) -> (Vec<TensorData>, usize) {
        let buffers = self.buffers.lock().unwrap();
        let ptr = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        let obs_dim = buffers.obs_dim;
        if obs_dim == 0 || ptr == 0 {
            return (Vec::new(), 1);
        }
        (buffers.obs[..ptr].to_vec(), obs_dim)
    }

    /// Return obs for exactly the first `n` complete episodes.
    /// Returns empty if fewer than `n` complete episodes exist in the buffer.
    pub fn get_obs_for_first_n_episodes(&self, n: usize) -> (Vec<TensorData>, usize) {
        let buffers = self.buffers.lock().unwrap();
        if buffers.episode_boundaries.len() < n {
            return (Vec::new(), 1);
        }
        let obs_dim = buffers.obs_dim;
        if obs_dim == 0 {
            return (Vec::new(), 1);
        }
        let cut_step = buffers.episode_boundaries[n - 1].1;
        (buffers.obs[..cut_step].to_vec(), obs_dim)
    }

    pub fn finalize_gae_blocking(&self, values: Vec<f32>) {
        let mut buffers = self.buffers.lock().unwrap();
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;

        let fill_len = values.len().min(buffers.values.len());
        buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);

        let boundaries: Vec<_> = buffers.episode_boundaries.clone();
        for (start, end, is_truncated) in boundaries {
            let bootstrap = if is_truncated {
                values.get(end.saturating_sub(1)).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, start, end, bootstrap);
        }
    }

    pub fn finalize_and_drain_blocking(&self, values: Vec<f32>) -> Option<PPOBatch> {
        let mut buffers = self.buffers.lock().unwrap();
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;
        let capacity = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        if capacity == 0 {
            return None;
        }
        let obs_dim = buffers.obs_dim;

        if !values.is_empty() {
            let fill_len = values.len().min(buffers.values.len());
            buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);
        }

        let boundaries: Vec<_> = buffers.episode_boundaries.clone();
        for (start, end, is_truncated) in boundaries {
            let bootstrap = if is_truncated {
                buffers
                    .values
                    .get(end.saturating_sub(1))
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, start, end, bootstrap);
        }

        let adv_raw = &buffers.advantages[..capacity];
        let (adv_mean, adv_std) = scalar_stats(adv_raw);
        let adv_norm = compute_normed_advantages(adv_raw, adv_mean, adv_std.max(1e-8));

        let obs = buffers.obs[..capacity.min(buffers.obs.len())].to_vec();
        let act = buffers.act[..capacity.min(buffers.act.len())].to_vec();
        let logp = buffers.logp[..capacity.min(buffers.logp.len())].to_vec();
        let ret_raw = &buffers.returns[..capacity];
        let (ret_mean, ret_std) = scalar_stats(ret_raw);
        let ret = compute_normed_advantages(ret_raw, ret_mean, ret_std.max(1e-8));
        let val = buffers.values[..capacity].to_vec();

        self.metadata.buffer_pointer.store(0, Ordering::Relaxed);
        self.metadata
            .buffer_path_start_idx
            .store(0, Ordering::Relaxed);
        buffers.obs.clear();
        buffers.obs_dim = 0;
        buffers.act.clear();
        buffers.logp.clear();
        buffers.rewards.clear();
        buffers.advantages.clear();
        buffers.returns.clear();
        buffers.values.clear();
        buffers.episode_boundaries.clear();
        buffers.episode_versions.clear();

        Some(PPOBatch {
            obs,
            obs_dim,
            act,
            logp,
            adv_norm,
            ret,
            val,
        })
    }

    /// Computes GAE for all n episodes, then builds the training batch from only non-stale episodes
    pub fn finalize_and_drain_first_n_blocking(
        &self,
        values: Vec<f32>,
        current_version: i64,
        max_version_lag: i64,
        n: usize,
        normalize_returns: bool,
    ) -> Option<PPOBatch> {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.episode_boundaries.len() < n {
            return None;
        }
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;
        let obs_dim = buffers.obs_dim;
        let cut_step = buffers.episode_boundaries[n - 1].1;

        if !values.is_empty() {
            let fill_len = values.len().min(cut_step).min(buffers.values.len());
            buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);
        }

        let boundaries_n: Vec<_> = buffers.episode_boundaries[..n].to_vec();
        let versions_n: Vec<i64> = buffers.episode_versions[..n].to_vec();
        for (start, end, is_truncated) in &boundaries_n {
            let bootstrap = if *is_truncated {
                buffers
                    .values
                    .get(end.saturating_sub(1))
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, *start, *end, bootstrap);
        }

        let mut fresh_obs: Vec<TensorData> = Vec::new();
        let mut fresh_acts: Vec<TensorData> = Vec::new();
        let mut fresh_logp: Vec<f32> = Vec::new();
        let mut fresh_adv: Vec<f32> = Vec::new();
        let mut fresh_ret: Vec<f32> = Vec::new();
        let mut fresh_val: Vec<f32> = Vec::new();

        for (i, &(start, end, _)) in boundaries_n.iter().enumerate() {
            let ep_version = versions_n.get(i).copied().unwrap_or(0);
            let lag = current_version.saturating_sub(ep_version);
            if lag > max_version_lag {
                continue; // stale: drain but don't train on this episode
            }
            fresh_obs.extend_from_slice(&buffers.obs[start..end]);
            fresh_acts.extend_from_slice(&buffers.act[start..end]);
            fresh_logp.extend_from_slice(&buffers.logp[start..end]);
            fresh_adv.extend_from_slice(&buffers.advantages[start..end]);
            fresh_ret.extend_from_slice(&buffers.returns[start..end]);
            fresh_val.extend_from_slice(&buffers.values[start..end]);
        }

        let total_steps = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        let remaining = total_steps - cut_step;

        buffers.obs.drain(0..cut_step);
        buffers.act.drain(0..cut_step);
        buffers.logp.copy_within(cut_step..total_steps, 0);
        buffers.logp.truncate(remaining);
        buffers.rewards.copy_within(cut_step..total_steps, 0);
        buffers.rewards.truncate(remaining);
        buffers.advantages.copy_within(cut_step..total_steps, 0);
        buffers.advantages.truncate(remaining);
        buffers.returns.copy_within(cut_step..total_steps, 0);
        buffers.returns.truncate(remaining);
        buffers.values.copy_within(cut_step..total_steps, 0);
        buffers.values.truncate(remaining);

        let remaining_boundaries: Vec<_> = buffers.episode_boundaries[n..]
            .iter()
            .map(|&(s, e, trunc)| (s - cut_step, e - cut_step, trunc))
            .collect();
        buffers.episode_boundaries = remaining_boundaries;
        buffers.episode_versions = buffers.episode_versions[n..].to_vec();

        self.metadata
            .buffer_pointer
            .store(remaining, Ordering::Relaxed);
        let old_path_start = self.metadata.buffer_path_start_idx.load(Ordering::Relaxed);
        self.metadata
            .buffer_path_start_idx
            .store(old_path_start.saturating_sub(cut_step), Ordering::Relaxed);

        if fresh_obs.is_empty() {
            return None;
        }

        let (adv_mean, adv_std) = scalar_stats(&fresh_adv);
        let adv_norm = compute_normed_advantages(&fresh_adv, adv_mean, adv_std.max(1e-8));

        let ret_flat = if normalize_returns {
            let (ret_mean, ret_std) = scalar_stats(&fresh_ret);
            compute_normed_advantages(&fresh_ret, ret_mean, ret_std.max(1e-8))
        } else {
            fresh_ret
        };

        Some(PPOBatch {
            obs: fresh_obs,
            obs_dim,
            act: fresh_acts,
            logp: fresh_logp,
            adv_norm,
            ret: ret_flat,
            val: fresh_val,
        })
    }

    /// Remove all leading stale episodes from the buffer front.
    /// An episode is stale if (current_version - ep_version) > max_version_lag.
    pub fn purge_stale_episodes(&self, current_version: i64, max_version_lag: i64) {
        let mut buffers = self.buffers.lock().unwrap();
        let stale_count = buffers
            .episode_versions
            .iter()
            .take_while(|&&v| current_version.saturating_sub(v) > max_version_lag)
            .count();
        if stale_count == 0 {
            return;
        }
        let cut_step = buffers.episode_boundaries[stale_count - 1].1;
        let total_steps = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        let remaining = total_steps - cut_step;

        buffers.obs.drain(0..cut_step);
        buffers.act.drain(0..cut_step);
        if remaining > 0 {
            buffers.logp.copy_within(cut_step..total_steps, 0);
            buffers.rewards.copy_within(cut_step..total_steps, 0);
            buffers.advantages.copy_within(cut_step..total_steps, 0);
            buffers.returns.copy_within(cut_step..total_steps, 0);
            buffers.values.copy_within(cut_step..total_steps, 0);
        }
        buffers.obs.truncate(remaining);
        buffers.act.truncate(remaining);
        buffers.logp.truncate(remaining);
        buffers.rewards.truncate(remaining);
        buffers.advantages.truncate(remaining);
        buffers.returns.truncate(remaining);
        buffers.values.truncate(remaining);

        let remaining_boundaries: Vec<_> = buffers.episode_boundaries[stale_count..]
            .iter()
            .map(|&(s, e, trunc)| (s - cut_step, e - cut_step, trunc))
            .collect();
        buffers.episode_boundaries = remaining_boundaries;
        buffers.episode_versions = buffers.episode_versions[stale_count..].to_vec();

        self.metadata
            .buffer_pointer
            .store(remaining, Ordering::Relaxed);
        let old_path_start = self.metadata.buffer_path_start_idx.load(Ordering::Relaxed);
        self.metadata
            .buffer_path_start_idx
            .store(old_path_start.saturating_sub(cut_step), Ordering::Relaxed);
    }

    pub fn get_episode_count(&self) -> usize {
        self.buffers.lock().unwrap().episode_boundaries.len()
    }

    /// Total steps across all complete episodes (step index of last boundary end).
    pub fn get_complete_step_count(&self) -> usize {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .episode_boundaries
            .last()
            .map(|&(_, end, _)| end)
            .unwrap_or(0)
    }

    pub fn is_full(&self) -> bool {
        match self.max_buffered_episodes {
            None => false,
            Some(max) => self.buffers.lock().unwrap().episode_boundaries.len() >= max,
        }
    }

    pub fn episodes_needed_for_steps(&self, min_steps: usize) -> usize {
        let buffers = self.buffers.lock().unwrap();
        for (i, &(_, end, _)) in buffers.episode_boundaries.iter().enumerate() {
            if end >= min_steps {
                return i + 1;
            }
        }
        0 // not enough complete steps yet
    }
}

#[async_trait]
impl GenericReplayBuffer for PPOReplayBuffer {
    async fn insert_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<Box<dyn Any>, ReplayBufferError> {
        let mut buffers = self.buffers.lock().unwrap();
        let mut episode_return = 0.0f32;
        let mut episode_length = 0i32;

        for action in &trajectory.actions {
            episode_length += 1;
            let reward = action.get_rew();
            episode_return += reward;

            if let Some(obs_td) = action.get_obs() {
                if buffers.obs_dim == 0 {
                    buffers.obs_dim = obs_td.shape.iter().product::<usize>();
                }
                buffers.obs.push(obs_td.clone());
            }

            if let Some(act_td) = action.get_act() {
                buffers.act.push(act_td.clone());
            }

            let logp = if let Some(map) = action.get_data() {
                if let Some(RelayRLData::Tensor(logp_td)) = map.get("logp_a") {
                    bytemuck::cast_slice::<u8, f32>(&logp_td.data)
                        .first()
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            buffers.logp.push(logp);

            let value = if let Some(map) = action.get_data() {
                if let Some(RelayRLData::Tensor(val_td)) = map.get("value") {
                    bytemuck::cast_slice::<u8, f32>(&val_td.data)
                        .first()
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            buffers.rewards.push(reward);
            buffers.advantages.push(0.0);
            buffers.returns.push(0.0);
            buffers.values.push(value);

            let next = self.metadata.buffer_pointer.load(Ordering::Relaxed) + 1;
            self.metadata.buffer_pointer.store(next, Ordering::Relaxed);

            if action.get_done() {
                let start = self.metadata.buffer_path_start_idx.load(Ordering::Relaxed);
                let end = self.metadata.buffer_pointer.load(Ordering::Relaxed);
                buffers
                    .episode_boundaries
                    .push((start, end, trajectory.is_truncated));
                buffers.episode_versions.push(trajectory.policy_version);
                self.metadata
                    .buffer_path_start_idx
                    .store(end, Ordering::Relaxed);
            }
        }

        Ok(Box::new((episode_return, episode_length)))
    }

    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        Err(ReplayBufferError::BufferSamplingError(
            "PPOReplayBuffer: use finalize_and_drain_blocking instead of sample_buffer".to_string(),
        ))
    }
}
