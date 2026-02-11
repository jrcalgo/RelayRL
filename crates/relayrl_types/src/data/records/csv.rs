use crate::types::data::trajectory::RelayRLTrajectoryTrait;
use csv::{ReaderBuilder, WriterBuilder, StringRecord};

#[derive(thiserror::Error, Debug, Clone)]
enum CsvTrajectoryError {
    #[error("Failed to use csv file: {}")]
    CsvFailure(csv::Error),
    #[error("Failed to build RelayRLTrajectory: {}")]
    TrajectoryBuildFailure(String),
}

struct WriterObject {
    writer: Writer,
    csv_capacity: usize,
    path: PathBuf
}

struct ReaderObject {
    reader: Reader,
    csv_capacity: usize,
    path: PathBuf
}

pub struct CsvTrajectory {
    pub trajectory: Option<RelayRLTrajectory>, 
    writer: Option<WriterObject>, 
    reader: Option<ReaderObject>
};

impl CsvTrajectory {
    pub fn new(trajectory: Option<RelayRLTrajectory>) {
        Self {
            trajectory
        }
    }

    pub fn write_csv<P: AsRef<Path>>(mut self, path: P, csv_capacity: usize) -> Result<Self, CsvTrajectoryError> {
        let mut writer: &Writer = {
            match self.writer_obj {
                Some(obj) => {
                    if obj.csv_capacity != csv_capacity || obj.path != path {
                        obj = ReaderObject {
                            reader: ReaderBuilder::new().buffer_capacity(csv_capacity),
                            csv_capacity,
                            path
                        }
                    }
                }
                None => {
                    self.writer_obj = WriterObject {
                        writer: WriterBuilder::new(),
                        csv_capacity,
                        path
                    }
                }
            }

            &self.writer_obj.writer
        }
    }

    pub fn read_csv<P: AsRef<Path>>(mut self, path: P, csv_capacity: usize) -> Result<Self, CsvTrajectoryError> {
        let reader: &Reader = {
            match self.reader_obj {
                Some(obj) => {
                    if obj.csv_capacity != csv_capacity || obj.path != path {
                        obj = ReaderObject {
                            reader: ReaderBuilder::new().buffer_capacity(csv_capacity).comment(b'#').trim(Trim::All).from_path(path),
                            csv_capacity,
                            path
                        }
                    }
                }
                None => {
                    self.reader_obj = ReaderObject {
                        reader: ReaderBuilder::new().buffer_capacity(csv_capacity).comment(b'#').trim(Trim::All).from_path(path),
                        csv_capacity,
                        path
                    }
                }
            }

            &self.reader_obj.reader
        }

        let record: Vec<StringRecord> = reader.records().collect::<Result<Vec<StringRecord>, CsvTrajectoryError>>().map_err(CsvTrajectoryError::from)?;

        self.build_trajectory(record)?
    }

    fn build_trajectory(&self, record: Vec<StringRecord>) -> Result<(), CsvTrajectoryError> {

    }
}

impl RelayRLTrajectoryTrait for StringRecord {
    type Action = RelayRLAction;

    fn add_action(&mut self, action: Self::Action) {
        match &self.trajectory {
            Some(traj) => traj.add_action(action),
            None => {
                let traj = RelayRLTrajectory::Default;
                traj.add_action(action);
                self.trajectory = Some(traj);
            }
        }
    }
}
