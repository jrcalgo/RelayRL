
use relayrl_types::prelude::data::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::data::tensor::relayrl::{AnyBurnTensor, TensorData};

use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use dashmap::DashMap;
use thiserror::Error;
use async_trait::async_trait;

#[derive(Clone, Debug, Error)]
pub enum ReplayBufferError {
    #[error("Insertion of trajectory failed: {}")]
    TrajectoryInsertionError(String),
    #[error("Buffer sampling failed: {}")]
    BufferSamplingError(String),
}

pub type BufferTensors = Vec<Option<TensorData>>;

#[derive(Hash, Eq, PartialEq)]
pub enum BatchKey {
    Obs,
    Act,
    Mask,
    Custom(String),
}

pub enum BufferSample {
    Tensors(Box<[TensorData]>),
    Scalars(SampleScalars)
}

pub enum SampleScalars {
    U8(Box<[u8]>),
    I16(Box<[i16]>),
    I32(Box<[i32]>),
    I64(Box<[i64]>),
    F32(Box<[f32]>),
    F64(Box<[f64]>),
    Bool(Box<[bool]>)
}

pub type Batch = HashMap<BatchKey, BufferSample>;

#[async_trait]
pub async trait GenericReplayBuffer: Send + Sync {
    async fn insert_trajectory(&self, trajectory: RelayRLTrajectory) -> Result<(), ReplayBufferError>;
    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError>;
}

pub fn discounted_cumsum(x: &[f32], discounted: f32) {

}

pub fn scalar_stats(x: &[])

pub fn compute_normed_advantages(advantages: &[f32], mean: f32, std: f32) -> &[f32] {

}
