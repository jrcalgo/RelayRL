
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::tensor::relayrl::TensorData;

use std::boxed::Box;
use std::collections::HashMap;
use thiserror::Error;
use async_trait::async_trait;
use std::any::Any;

#[derive(Clone, Debug, Error)]
pub enum ReplayBufferError {
    #[error("Insertion of trajectory failed: {0}")]
    TrajectoryInsertionError(String),
    #[error("Buffer sampling failed: {0}")]
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
pub trait GenericReplayBuffer: Send + Sync {
    async fn insert_trajectory(&self, trajectory: RelayRLTrajectory) -> Result<Box<dyn Any>, ReplayBufferError>;
    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError>;
}