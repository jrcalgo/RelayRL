use crate::templates::base_replay_buffer::{Batch, GenericReplayBuffer, ReplayBufferError};

use async_trait::async_trait;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;

use std::any::Any;

type SharedReplayBuffer = super::super::replay_buffer::DDPGReplayBuffer;

pub struct MultiagentDDPGReplayBuffer {
    inner: SharedReplayBuffer,
}

pub type MADDPGReplayBuffer = MultiagentDDPGReplayBuffer;

impl Default for MultiagentDDPGReplayBuffer {
    fn default() -> Self {
        Self::new(1_000_000, 128)
    }
}

impl MultiagentDDPGReplayBuffer {
    pub fn new(buffer_size: usize, batch_size: usize) -> Self {
        Self {
            inner: SharedReplayBuffer::new(buffer_size, batch_size),
        }
    }
}

#[async_trait]
impl GenericReplayBuffer for MultiagentDDPGReplayBuffer {
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
