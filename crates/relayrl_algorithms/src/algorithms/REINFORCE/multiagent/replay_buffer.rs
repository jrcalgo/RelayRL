use crate::templates::base_replay_buffer::{Batch, GenericReplayBuffer, ReplayBufferError};

use async_trait::async_trait;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;

use std::any::Any;

type SharedReplayBuffer = super::super::replay_buffer::ReinforceReplayBuffer;

pub struct MultiagentReinforceReplayBuffer {
    inner: SharedReplayBuffer,
}

pub type MAREINFORCEReplayBuffer = MultiagentReinforceReplayBuffer;

impl Default for MultiagentReinforceReplayBuffer {
    fn default() -> Self {
        Self::new(1_000_000, 0.98, 0.97)
    }
}

impl MultiagentReinforceReplayBuffer {
    pub fn new(buffer_size: usize, gamma: f32, lambda: f32) -> Self {
        Self {
            inner: SharedReplayBuffer::new(buffer_size, gamma, lambda, true),
        }
    }
}

#[async_trait]
impl GenericReplayBuffer for MultiagentReinforceReplayBuffer {
    async fn insert_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<Box<dyn Any>, ReplayBufferError> {
        self.inner.insert_trajectory(trajectory).await
    }

    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        self.inner.sample_buffer().await
    }
}
