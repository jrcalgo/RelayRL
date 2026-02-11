use crate::data::trajectory::RelayRLTrajectoryTrait;
use arrow::{};

#[derive(thiserror::Error, Debug, Clone)]
enum ArrowTrajectoryError {
    #[error("Failed to build Arrow schema: {}")]
    SchemaBuildFailure(String),
    #[error("Failed to build Arrow record batch: {}")]
    RecordBatchBuildFailure(String),
    #[error("Failed to write Arrow record batch: {}")]
    RecordBatchWriteFailure(String),
}

pub struct ArrowTrajectory {
    pub trajectory: Option<RelayRLTrajectory>,
};

impl ArrowTrajectory {
    pub fn new(trajectory: Option<RelayRLTrajectory>) -> Self {
        Self {
            trajectory,
        }
    }

    pub fn to_arrow<P: AsRef<Path>>(mut self, path: P) -> Result<Self, ArrowTrajectoryError> {

    }

    pub fn from_arrow<P: AsRef<Path>>(mut self, path: P) -> Result<Self, ArrowTrajectoryError> {

    }

    fn build_trajectory(&self, array: ) -> Result<(), ArrowTrajectoryError> { }

    fn write_trajectory(&self, writer: &FileWriter, trajectory: &RelayRLTrajectory, start_row: usize, end_row: usize) -> Result<(), ArrowTrajectoryError> {

    }
}

impl RelayRLTrajectoryTrait for ArrowTrajectory {
    type Action = RelayRLAction;

    fn add_action(&mut self, action: Self::Action) {
        match &self.trajectory {
            Some(traj) => traj.add_action(action),
            None => {
                let traj = RelayRLTrajectory::default();
                traj.add_action(action);
                self.trajectory = Some(traj);
            }
        }
    }
}