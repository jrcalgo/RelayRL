pub mod actors;
pub mod benchmark;
pub mod cache;
pub mod environment;
pub mod heuristics;
pub mod host;
pub mod metrics;
pub mod training;
pub mod workload;

pub const OBSERVATION_DIM: usize = 24;
pub const ACTION_DIM: usize = 6;
