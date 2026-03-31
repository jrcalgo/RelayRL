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
