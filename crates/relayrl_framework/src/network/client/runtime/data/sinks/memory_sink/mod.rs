use relayrl_types::data::trajectory::RelayRLTrajectory;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemorySinkError {
    #[error("Failed to write trajectory to memory: {0}")]
    WriteMemoryTrajectoryError(#[from] TrajectoryError),
}

pub(crate) fn write_to_memory(trajectory: Arc<RelayRLTrajectory>) -> Result<(), MemorySinkError> {

}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use relayrl_types::data::trajectory::RelayRLTrajectory;

    #[test]
    fn write_to_memory_writes_trajectory_to_memory() {

    }

    #[test]
    fn retrieve_from_memory_returns_empty_buffer() {

    }

    #[test]
    fn retrieved_memory_matches_written_memory() {
        
    }
}