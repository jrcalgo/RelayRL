use crate::network::client::agent::LocalTrajectoryFileType;

use relayrl_types::data::records::{
    arrow::ArrowTrajectory, arrow::ArrowTrajectoryError, csv::CsvTrajectory,
    csv::CsvTrajectoryError,
};
use relayrl_types::data::trajectory::RelayRLTrajectory;

use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum FileSinkError {
    #[error("Failed to write trajectory Arrow file: {0}")]
    WriteArrowTrajectoryFileError(#[from] ArrowTrajectoryError),
    #[error("Failed to write trajectory CSV file: {0}")]
    WriteCsvTrajectoryFileError(#[from] CsvTrajectoryError),
}

pub(crate) fn write_local_trajectory_file(
    trajectory: Arc<RelayRLTrajectory>,
    path: &Path,
    file_type: &LocalTrajectoryFileType,
) -> Result<(), FileSinkError> {
    match file_type {
        LocalTrajectoryFileType::Arrow => {
            let arrow_trajectory = ArrowTrajectory::new(Some(trajectory.deref().clone()));
            arrow_trajectory
                .to_arrow(path)
                .map_err(FileSinkError::from)?;
            Ok(())
        }
        LocalTrajectoryFileType::Csv => {
            let csv_trajectory = CsvTrajectory::new(Some(trajectory.deref().clone()));
            csv_trajectory
                .to_csv(path, 10_000_000)
                .map_err(FileSinkError::from)?;
            Ok(())
        }
    }
}

// the size of this file and purpose of this module abstraction bugs me

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::network::client::agent::LocalTrajectoryFileType;
    use relayrl_types::data::trajectory::RelayRLTrajectory;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_trajectory(num_actions: usize) -> Arc<RelayRLTrajectory> {
        use relayrl_types::data::action::RelayRLAction;
        let mut traj = RelayRLTrajectory::new(num_actions.max(1));
        for i in 0..num_actions {
            let done = i + 1 == num_actions;
            traj.add_action(RelayRLAction::minimal(i as f32 * 0.1, done));
        }
        Arc::new(traj)
    }

    #[test]
    fn write_arrow_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_traj.arrow");
        let traj = make_trajectory(3);

        let result = write_local_trajectory_file(traj, &path, &LocalTrajectoryFileType::Arrow);
        assert!(result.is_ok(), "Arrow write failed: {:?}", result.err());
        assert!(path.exists(), "Arrow file not created");
    }

    #[test]
    fn write_csv_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_traj.csv");
        let traj = make_trajectory(3);

        let result = write_local_trajectory_file(traj, &path, &LocalTrajectoryFileType::Csv);
        assert!(result.is_ok(), "CSV write failed: {:?}", result.err());
        assert!(path.exists(), "CSV file not created");
    }

    #[test]
    fn write_empty_trajectory_arrow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.arrow");
        let traj = make_trajectory(0);

        // Should not panic even with 0 actions
        let result = write_local_trajectory_file(traj, &path, &LocalTrajectoryFileType::Arrow);
        // We accept Ok or Err; what matters is no panic
        let _ = result;
    }

    #[test]
    fn write_empty_trajectory_csv() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.csv");
        let traj = make_trajectory(0);

        let result = write_local_trajectory_file(traj, &path, &LocalTrajectoryFileType::Csv);
        let _ = result;
    }

    #[test]
    fn failure_invalid_path() {
        // write should fail
        let path = std::path::Path::new("/nonexistent_relayrl_test_dir/traj.arrow");
        let traj = make_trajectory(2);

        let result = write_local_trajectory_file(traj, path, &LocalTrajectoryFileType::Arrow);
        assert!(result.is_err(), "Expected error for invalid path, got Ok");
    }

    #[test]
    fn arrow_file_is_nonempty_for_non_trivial_trajectory() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonempty.arrow");
        let traj = make_trajectory(5);

        write_local_trajectory_file(traj, &path, &LocalTrajectoryFileType::Arrow).unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0, "Arrow file should not be empty");
    }
}
