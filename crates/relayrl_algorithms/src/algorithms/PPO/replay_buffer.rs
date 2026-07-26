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

/// A normalized training batch ready for PPO gradient updates: observations, actions, log-probabilities, advantages, returns, and values.
pub struct PPOBatch {
    pub obs: Vec<TensorData>,
    pub obs_dim: usize,
    pub act: Vec<TensorData>,
    pub logp: Vec<f32>,
    pub adv_norm: Vec<f32>,
    pub ret: Vec<f32>,
    pub val: Vec<f32>,
    pub ret_mean: f32,
    pub ret_std: f32,
}

/// Experience replay buffer for PPO that accumulates trajectories and computes GAE-normalized advantages on drain.
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
    /// Creates a PPO replay buffer with the given capacity, discount (`gamma`), GAE lambda (`lam`), and optional episode cap.
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

        // Value targets = GAE advantages + V(s) (SF-style lambda-return), rather than
        // a pure Monte-Carlo discounted return (lambda=1). This keeps the value
        // function's regression target consistent with the same lambda used for
        // the advantage estimator feeding the policy loss. `advantages` and
        // `vals[..vals.len() - 1]` are both exactly `end - start` elements (one
        // per real transition, excluding the appended bootstrap), so the full
        // vector must be copied into `returns[start..end]` — copying `[..len-1]`
        // (n-1 elements) into an n-slot destination panics on the first episode.
        let returns: Vec<f32> = advantages
            .iter()
            .zip(vals[..vals.len() - 1].iter())
            .map(|(a, v)| a + v)
            .collect();
        buffers.returns[start..end].copy_from_slice(&returns);
    }

    /// Returns all buffered observations and their dimension for a value-function pass before GAE.
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

    /// Stores the value estimates and computes GAE advantages/returns over all buffered episodes.
    pub fn finalize_gae_blocking(&self, values: Vec<f32>) {
        let mut buffers = self.buffers.lock().unwrap();
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;

        let fill_len = values.len().min(buffers.values.len());
        buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);

        let boundaries: Vec<_> = buffers.episode_boundaries.clone();
        for (start, end, is_truncated) in boundaries {
            // Bootstrap with V(s_end) (the state reached after the chunk's last
            // action), falling back to V(s_{end-1}) if s_end isn't buffered yet.
            let bootstrap = if is_truncated {
                values
                    .get(end)
                    .or_else(|| values.get(end.saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, start, end, bootstrap);
        }
    }

    /// Finalizes GAE and drains the buffer into a normalized `PPOBatch` ready for training.
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
            // Bootstrap with V(s_end) (the state reached after the chunk's last
            // action), falling back to V(s_{end-1}) if s_end isn't buffered yet.
            let bootstrap = if is_truncated {
                buffers
                    .values
                    .get(end)
                    .or_else(|| buffers.values.get(end.saturating_sub(1)))
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
            ret_mean,
            ret_std,
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
            // Bootstrap with V(s_end) (the state reached after the chunk's last
            // action), falling back to V(s_{end-1}) if s_end isn't buffered yet.
            let bootstrap = if *is_truncated {
                buffers
                    .values
                    .get(*end)
                    .or_else(|| buffers.values.get(end.saturating_sub(1)))
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

        // Pass raw lambda-returns through. `normalize_persistent_returns` (run
        // unconditionally in run_ppo_sgd_flat) is the SF-aligned RunningMeanStd
        // normalizer; z-scoring per-batch here first would feed it an
        // already mean=0/std=1 stream, making it a redundant no-op and
        // recalibrating the vf's target scale from scratch (with per-batch
        // sampling noise) every epoch instead of tracking a smoothly-evolving
        // running statistic.
        let _ = normalize_returns;
        let (ret, ret_mean, ret_std) = (fresh_ret, 0.0, 1.0);

        Some(PPOBatch {
            obs: fresh_obs,
            obs_dim,
            act: fresh_acts,
            logp: fresh_logp,
            adv_norm,
            ret,
            val: fresh_val,
            ret_mean,
            ret_std,
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

    /// Returns the number of complete episodes currently buffered.
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

    /// Returns `true` when the buffered episode count has reached the configured maximum.
    pub fn is_full(&self) -> bool {
        match self.max_buffered_episodes {
            None => false,
            Some(max) => self.buffers.lock().unwrap().episode_boundaries.len() >= max,
        }
    }

    /// Returns how many leading episodes are needed to accumulate at least `min_steps` steps.
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

        let last_idx = trajectory.actions.len().saturating_sub(1);
        for (idx, action) in trajectory.actions.iter().enumerate() {
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
                if let Some(RelayRLData::Tensor(val_td)) = map.get("val") {
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

            // A trajectory chunk ends a GAE segment either because the true env
            // episode terminated/truncated (action.get_done()) or because the
            // collector cut it off at rollout_len steps (trajectory.is_truncated,
            // set via set_truncated() with the episode still ongoing). Without the
            // latter, steps from rollout-length cutoffs never get an
            // episode_boundaries entry and sit dead in the buffer (no GAE/return)
            // until the underlying episode eventually ends, possibly many epochs
            // later — starving training of fresh transitions.
            if idx == last_idx && (action.get_done() || trajectory.is_truncated) {
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use relayrl_types::data::tensor::{DType, NdArrayDType, SupportedTensorBackend};
    use relayrl_types::prelude::action::RelayRLAction;

    fn scalar_tensor(value: f32) -> TensorData {
        TensorData::new(
            vec![1],
            DType::NdArray(NdArrayDType::F32),
            value.to_le_bytes().to_vec(),
            SupportedTensorBackend::NdArray,
        )
    }

    fn action_with(reward: f32, done: bool, value: f32, logp: f32) -> RelayRLAction {
        let mut data = std::collections::HashMap::new();
        data.insert("val".to_string(), RelayRLData::Tensor(scalar_tensor(value)));
        data.insert(
            "logp_a".to_string(),
            RelayRLData::Tensor(scalar_tensor(logp)),
        );
        RelayRLAction::new(
            Some(scalar_tensor(0.0)),
            Some(scalar_tensor(0.0)),
            None,
            reward,
            done,
            Some(data),
            None,
        )
    }

    #[tokio::test]
    async fn insert_trajectory_reads_val_key_for_value_estimate() {
        let buffer = PPOReplayBuffer::new(16, 0.99, 0.97, None);
        let mut traj = RelayRLTrajectory::new(4);
        traj.add_action(action_with(1.0, true, 3.5, 0.1));

        buffer.insert_trajectory(traj).await.unwrap();

        let batch = buffer
            .finalize_and_drain_blocking(vec![])
            .expect("single completed episode should drain into a batch");
        assert_eq!(
            batch.val,
            vec![3.5],
            "insert_trajectory should read the framework's \"val\" key, not \"value\""
        );
    }

    fn vector_tensor(values: &[f32]) -> TensorData {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        TensorData::new(
            vec![values.len()],
            DType::NdArray(NdArrayDType::F32),
            bytes,
            SupportedTensorBackend::NdArray,
        )
    }

    #[tokio::test]
    async fn continuous_action_tensor_shape_survives_drain() {
        let buffer = PPOReplayBuffer::new(16, 0.99, 0.97, None);
        let mut traj = RelayRLTrajectory::new(8);
        let mut data = std::collections::HashMap::new();
        data.insert("val".to_string(), RelayRLData::Tensor(scalar_tensor(0.5)));
        data.insert(
            "logp_a".to_string(),
            RelayRLData::Tensor(scalar_tensor(-1.25)),
        );
        traj.add_action(RelayRLAction::new(
            Some(vector_tensor(&[0.1, 0.2, 0.3])),
            Some(vector_tensor(&[0.5, -0.25])),
            None,
            1.0,
            true,
            Some(data),
            None,
        ));
        buffer.insert_trajectory(traj).await.unwrap();
        let batch = buffer
            .finalize_and_drain_first_n_blocking(vec![], 0, 0, 1, false)
            .expect("drain");
        assert_eq!(batch.act.len(), 1);
        assert_eq!(batch.act[0].shape, vec![2]);
        assert_eq!(batch.act[0].data.len(), 8);
        assert_eq!(batch.logp, vec![-1.25]);
        assert_eq!(batch.obs.len(), batch.logp.len());
    }

    #[tokio::test]
    async fn batch_logp_matches_rollout_time_logp_unmodified() {
        // The drained batch.logp must be exactly the rollout-time log-probs
        // (from the "logp_a" key), since IndependentPPOAlgorithm::start_epoch_training
        // now uses it directly as PPO's logp_old instead of recomputing it from
        // the epoch-start network (which previously made the importance ratio
        // ~1.0 at epoch start and disabled the clip).
        let buffer = PPOReplayBuffer::new(16, 0.99, 0.97, None);
        let mut traj = RelayRLTrajectory::new(8);
        traj.add_action(action_with(1.0, false, 0.0, -0.1));
        traj.add_action(action_with(1.0, false, 0.0, -0.2));
        traj.add_action(action_with(1.0, true, 0.0, -0.3));

        buffer.insert_trajectory(traj).await.unwrap();

        let batch = buffer
            .finalize_and_drain_first_n_blocking(vec![], 0, 0, 1, false)
            .expect("3-step episode should drain");

        assert_eq!(batch.logp, vec![-0.1, -0.2, -0.3]);
    }

    #[tokio::test]
    async fn insert_trajectory_closes_boundary_on_truncation_without_done() {
        let buffer = PPOReplayBuffer::new(16, 0.99, 0.97, None);
        let mut traj = RelayRLTrajectory::new(4);
        // done=false: the episode itself hasn't ended, only the rollout chunk has.
        traj.add_action(action_with(1.0, false, 3.5, 0.1));
        traj.set_truncated();

        buffer.insert_trajectory(traj).await.unwrap();

        assert_eq!(
            buffer.get_episode_count(),
            1,
            "a rollout-cutoff (is_truncated) chunk must still close a GAE boundary even without a done action"
        );
    }

    #[tokio::test]
    async fn compute_gae_copies_full_length_returns_without_panicking() {
        let buffer = PPOReplayBuffer::new(16, 0.99, 0.97, None);
        let mut traj = RelayRLTrajectory::new(8);
        traj.add_action(action_with(1.0, false, 1.0, 0.0));
        traj.add_action(action_with(1.0, false, 1.0, 0.0));
        traj.add_action(action_with(1.0, true, 1.0, 0.0));

        buffer.insert_trajectory(traj).await.unwrap();

        // Pre-fix, compute_gae_episode copied an (n-1)-length slice into an
        // n-length destination and panicked here on the very first episode.
        let batch = buffer
            .finalize_and_drain_blocking(vec![])
            .expect("3-step episode should drain into a batch");
        assert_eq!(batch.val.len(), 3);
        assert_eq!(batch.ret.len(), 3);
        assert_eq!(batch.adv_norm.len(), 3);
    }

    #[tokio::test]
    async fn truncated_episode_bootstraps_from_v_of_s_end_not_s_end_minus_1() {
        // gamma = lam = 1.0 keeps the expected numbers simple to hand-verify.
        let buffer = PPOReplayBuffer::new(16, 1.0, 1.0, None);

        // Episode 1: one step, truncated (not done). V(s0) = 10.0.
        let mut traj1 = RelayRLTrajectory::new(4);
        traj1.add_action(action_with(0.0, false, 10.0, 0.0));
        traj1.set_truncated();
        buffer.insert_trajectory(traj1).await.unwrap();

        // A second, still-open trajectory buffers V(s_end) = V(s1) = 20.0 —
        // the value of the state episode 1 was cut off at.
        let mut traj2 = RelayRLTrajectory::new(4);
        traj2.add_action(action_with(0.0, false, 20.0, 0.0));
        buffer.insert_trajectory(traj2).await.unwrap();

        let batch = buffer
            .finalize_and_drain_first_n_blocking(vec![], 0, 0, 1, false)
            .expect("first episode should drain");

        // delta = reward + gamma*bootstrap - V(s0) = 0 + 20.0 - 10.0 = 10.0
        // return = delta + V(s0) = 10.0 + 10.0 = 20.0 = bootstrap.
        // A V(s_end-1) bootstrap (the pre-fix behavior) would instead give
        // delta = 0 + 10.0 - 10.0 = 0.0 and return = 10.0.
        assert_eq!(
            batch.ret,
            vec![20.0],
            "truncated bootstrap should use V(s_end)=20.0, not V(s_end-1)=10.0"
        );
    }

    #[tokio::test]
    async fn finalize_and_drain_first_n_returns_raw_returns_regardless_of_normalize_flag() {
        let buffer = PPOReplayBuffer::new(16, 0.99, 0.97, None);
        let mut traj = RelayRLTrajectory::new(4);
        traj.add_action(action_with(5.0, true, 1.0, 0.0));
        buffer.insert_trajectory(traj).await.unwrap();

        // normalize_returns=true must not z-score the batch: a single-sample
        // z-score would collapse the return to 0.0, discarding the signal that
        // the persistent (running-stats) normalizer in run_ppo_sgd_flat expects
        // to see in raw, reward-scale form.
        let batch = buffer
            .finalize_and_drain_first_n_blocking(vec![], 0, 0, 1, true)
            .expect("single completed episode should drain");

        assert_eq!(batch.ret_mean, 0.0);
        assert_eq!(batch.ret_std, 1.0);
        assert_eq!(
            batch.ret,
            vec![5.0],
            "raw lambda-return should pass through untouched"
        );
    }
}
