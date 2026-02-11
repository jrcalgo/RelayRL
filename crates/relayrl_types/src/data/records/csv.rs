use crate::data::trajectory::RelayRLTrajectoryTrait;
use csv::{ReaderBuilder, WriterBuilder, StringRecord};

#[derive(thiserror::Error, Debug, Clone)]
enum CsvTrajectoryError {
    #[error("Failed to use csv file: {}")]
    CsvFailure(csv::Error),
    #[error("Failed to build RelayRLTrajectory: {}")]
    TrajectoryBuildFailure(String),
    #[error("Trajectory not initialized")]
    TrajectoryNotInitialized(String),
    #[error("Reader cache not initialized")]
    ReaderCacheNotInitialized(String),
    #[error("Writer cache not initialized")]
    WriterCacheNotInitialized(String),
}

struct WriterCache {
    writer: Writer,
    byte_capacity: usize,
    path: PathBuf
}

struct ReaderCache {
    reader: Reader,
    byte_capacity: usize,
    path: PathBuf,
    record: Vec<StringRecord>
}

pub struct CsvTrajectory {
    pub trajectory: Option<RelayRLTrajectory>, 
    writer_cache: Option<WriterCache>, 
    reader_cache: Option<ReaderCache>
};

impl CsvTrajectory {
    pub fn new(trajectory: Option<RelayRLTrajectory>) -> Self {
        Self {
            trajectory,
            writer_cache: None,
            reader_cache: None
        }
    }

    pub fn get_record(&self) -> Result<&[StringRecord], CsvTrajectoryError> {
        &self.reader_cache.as_ref().ok_or(CsvTrajectoryError::ReaderCacheNotInitialized)?.record
    }

    pub fn to_csv<P: AsRef<Path>>(mut self, path: P, byte_capacity: usize) -> Result<Self, CsvTrajectoryError> {
        match &self.trajectory {
            Some(traj) => {
                let (writer_ref, new_file): (&Writer, bool) = {
                    let new_file = match &self.writer_obj {
                        Some(cache) if cache.path == path.as_ref() && cache.byte_capacity == byte_capacity => {
                            false
                        }
                        _ => {
                            self.writer_obj = Some(WriterCache {
                                writer: WriterBuilder::new()
                                    .has_headers(true)
                                    .flexible(true)
                                    .trim(Trim::All)
                                    .buffer_capacity(byte_capacity)
                                    .from_path(path)
                                    .map_err(CsvTrajectoryError::CsvFailure)?,  // Handle error properly
                                byte_capacity,
                                path: path.as_ref().to_path_buf(),
                            });
                            true
                        }
                    };
                
                    // Safe because we just set it above if it was None
                    let cache = self.writer_obj.as_ref().unwrap();
                    (&cache.writer, new_file)
                };
        
                let (start_row, end_row): (usize, usize) = if new_file {
                    (0, traj.len() + 1)
                } else {
                    // read csv file, get last row
                    // check if last row is less than capacity
                    // if last row is less than capacity, return total length
                    // if greater, throw error
                    // if equal, 
                    let last_row = {
                        let mut reader: Reader<File> = ReaderBuilder::new().from_path(path)?;
                        reader.
                    }
                }
        
                self.write_trajectory(writer_ref, traj, start_row, end_row)?;

                Ok(self)
            }
            None => {
                return Err(CsvTrajectoryError::TrajectoryNotInitialized);
            }
        }
    }

    pub fn from_csv<P: AsRef<Path>>(mut self, path: P, byte_capacity: usize) -> Result<Self, CsvTrajectoryError> {
        let mut (reader_ref, new_file): (&Reader, bool) = {
            let new_file = match &self.reader_cache {
                Some(cache) if cache.path == path.as_ref() cache.byte_capacity == byte_capacity => {
                    false
                }
                _ => {
                    self.reader_cache = Some(ReaderCache {
                        reader: ReaderBuilder::new()
                            .has_headers(true)
                            .flexible(true)
                            .trim(Trim::All)
                            .buffer_capacity(byte_capacity)
                            .from_path(path)
                            .map_err(CsvTrajectoryError::CsvFailure)?,  // Handle error properly
                        byte_capacity,
                        path: path.as_ref().to_path_buf(),
                        record: Vec::new(),
                    });

                    true
                }
            };
        
            // Safe because we just set it above if it was None
            let cache = self.reader_cache.as_ref().unwrap();
            (&cache.reader, new_file)
        }

        let record: &[StringRecord] = if new_file {
            // Read all records from the file and return new records
            let records = reader_ref.records().collect::<Result<Vec<StringRecord>, CsvTrajectoryError>>().map_err(CsvTrajectoryError::from)?;
            self.reader_cache.as_mut().unwrap().record = records;
            &self.reader_cache.as_ref().unwrap().record
        } else {
            // Return existing records
            &self.reader_cache.as_ref().unwrap().record
        };

        self.build_trajectory(record)?;

        Ok(self)
    }

    fn build_trajectory(&self, record: &[StringRecord]) -> Result<(), CsvTrajectoryError> {
        let mut trajectory = {
            let max_length = record.len()-1; // subtract 1 for headers
            RelayRLTrajectory::new(max_length)
        };

        for row in record.iter().skip(1) {
            let action = RelayRLAction::deserialize(row)?;
            trajectory.add_action(&action);
        }
        
        self.trajectory = Some(trajectory);

        Ok(())
    }

    fn write_trajectory(&self, writer: &Writer, trajectory: &RelayRLTrajectory, start_row: usize, end_row: usize) -> Result<(), CsvTrajectoryError> {

    }
}

impl RelayRLTrajectoryTrait for StringRecord {
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
