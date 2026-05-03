mod kernel;
mod replay_buffer;

pub mod independent;
pub mod multiagent;

pub use independent::kernel::*;
pub use independent::replay_buffer::*;
pub use independent::{
    DDPGAlgorithm, DDPGParams, IDDPGAlgorithm, IDDPGParams, IndependentDDPGAlgorithm,
};
pub use multiagent::kernel::MultiagentDDPGKernel;
pub use multiagent::replay_buffer::MultiagentDDPGReplayBuffer;
pub use multiagent::{
    MADDPGAlgorithm, MADDPGParams, MultiagentDDPGAlgorithm, MultiagentDDPGKernelTrait,
};
