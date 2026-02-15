use crate::data::action::{RelayRLAction, RelayRLData};
use crate::data::tensor::{DType, NdArrayDType, TensorData};
#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
use crate::data::trajectory::RelayRLTrajectory;
use csv::{Reader, ReaderBuilder, StringRecord, Trim, Writer, WriterBuilder};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(thiserror::Error, Debug)]
pub enum CsvDataError {
    #[error("Failed to use csv file: {0}")]
    CsvFailure(csv::Error),
    #[error("Failed to build RelayRLTrajectory: {0}")]
    TrajectoryBuildFailure(String),
    #[error("Trajectory not initialized: {0}")]
    TrajectoryNotInitialized(String),
    #[error("Reader cache not initialized: {0}")]
    ReaderCacheNotInitialized(String),
    #[error("Writer cache not initialized: {0}")]
    WriterCacheNotInitialized(String),
}

pub type CsvTrajectoryError = CsvDataError;

struct WriterCache {
    writer: Writer<File>,
    byte_capacity: usize,
    path: PathBuf,
    rows_written: usize,
}

struct ReaderCache {
    reader: Reader<File>,
    byte_capacity: usize,
    path: PathBuf,
    records: Vec<StringRecord>,
}

const HEADER_FIELD_COUNT: usize = 21;

const HEADER_NAMES: [&str; HEADER_FIELD_COUNT] = [
    "backend",
    "actor_id",
    "timestamp",
    "obs_dtype",
    "obs_shape",
    "obs_f32",
    "obs_f64",
    "obs_binary",
    "act_dtype",
    "act_shape",
    "act_f32",
    "act_f64",
    "act_binary",
    "mask_dtype",
    "mask_shape",
    "mask_f32",
    "mask_f64",
    "mask_binary",
    "reward",
    "done",
    "aux_data",
];

#[derive(Debug, Clone, Copy)]
enum CsvHeaderFields {
    Backend = 0,
    ActorID = 1,
    Timestamp = 2,
    ObsDType = 3,
    ObsShape = 4,
    ObsF32 = 5,
    ObsF64 = 6,
    ObsBinary = 7,
    ActDType = 8,
    ActShape = 9,
    ActF32 = 10,
    ActF64 = 11,
    ActBinary = 12,
    MaskDType = 13,
    MaskShape = 14,
    MaskF32 = 15,
    MaskF64 = 16,
    MaskBinary = 17,
    Reward = 18,
    Done = 19,
    AuxData = 20,
}

const CSV_FIELDS: [CsvHeaderFields; HEADER_FIELD_COUNT] = [
    CsvHeaderFields::Backend,
    CsvHeaderFields::ActorID,
    CsvHeaderFields::Timestamp,
    CsvHeaderFields::ObsDType,
    CsvHeaderFields::ObsShape,
    CsvHeaderFields::ObsF32,
    CsvHeaderFields::ObsF64,
    CsvHeaderFields::ObsBinary,
    CsvHeaderFields::ActDType,
    CsvHeaderFields::ActShape,
    CsvHeaderFields::ActF32,
    CsvHeaderFields::ActF64,
    CsvHeaderFields::ActBinary,
    CsvHeaderFields::MaskDType,
    CsvHeaderFields::MaskShape,
    CsvHeaderFields::MaskF32,
    CsvHeaderFields::MaskF64,
    CsvHeaderFields::MaskBinary,
    CsvHeaderFields::Reward,
    CsvHeaderFields::Done,
    CsvHeaderFields::AuxData,
];

struct TrajectoryValidationCache {
    backend: Option<String>,
    actor_id: Option<Uuid>,
    timestamp: Option<u64>,
    episode: Option<u64>,
    training_step: Option<u64>,
}

#[derive(Default)]
struct TensorAccumulator {
    dtype: Option<DType>,
    shape: Option<Vec<usize>>,
    f32_data: Option<Vec<f32>>,
    f64_data: Option<Vec<f64>>,
    binary_data: Option<Vec<u8>>,
}

impl TensorAccumulator {
    fn has_any(&self) -> bool {
        self.dtype.is_some()
            || self.shape.is_some()
            || self.f32_data.is_some()
            || self.f64_data.is_some()
            || self.binary_data.is_some()
    }

    fn into_tensor_data(self) -> Result<Option<TensorData>, CsvDataError> {
        if !self.has_any() {
            return Ok(None);
        }

        let dtype = self.dtype.ok_or_else(|| {
            CsvDataError::TrajectoryBuildFailure("Missing tensor dtype while parsing CSV record".to_string())
        })?;
        let shape = self.shape.ok_or_else(|| {
            CsvDataError::TrajectoryBuildFailure("Missing tensor shape while parsing CSV record".to_string())
        })?;

        let tensor = build_tensor_data(dtype, shape, self.f32_data, self.f64_data, self.binary_data)?;
        Ok(Some(tensor))
    }
}

pub struct CsvTrajectory {
    pub trajectory: Option<RelayRLTrajectory>,
    writer_cache: Option<WriterCache>,
    reader_cache: Option<ReaderCache>,
}

impl CsvTrajectory {
    pub fn new(trajectory: Option<RelayRLTrajectory>) -> Self {
        Self {
            trajectory,
            writer_cache: None,
            reader_cache: None,
        }
    }

    pub fn get_records(&self) -> Result<&[StringRecord], CsvDataError> {
        self.reader_cache
            .as_ref()
            .map(|cache| cache.records.as_slice())
            .ok_or_else(|| {
                CsvDataError::ReaderCacheNotInitialized("reader cache is not initialized".to_string())
            })
    }

    pub fn to_csv<P: AsRef<Path>>(mut self, path: P, byte_capacity: usize) -> Result<Self, CsvDataError> {
        let trajectory = self.trajectory.clone().ok_or_else(|| {
            CsvDataError::TrajectoryNotInitialized("No trajectory initialized to write to CSV".to_string())
        })?;

        let target_path = path.as_ref().to_path_buf();
        let reuse_cache = matches!(
            &self.writer_cache,
            Some(cache) if cache.path == target_path && cache.byte_capacity == byte_capacity
        );

        let mut write_header = false;
        if !reuse_cache {
            let had_existing_data = std::fs::metadata(&target_path)
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false);

            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&target_path)
                .map_err(|error| {
                    CsvDataError::TrajectoryBuildFailure(format!(
                        "Failed to open CSV file for writing '{}': {error}",
                        target_path.display()
                    ))
                })?;

            let writer = WriterBuilder::new()
                .has_headers(false)
                .flexible(true)
                .buffer_capacity(byte_capacity)
                .from_writer(file);

            self.writer_cache = Some(WriterCache {
                writer,
                byte_capacity,
                path: target_path,
                rows_written: 0,
            });

            write_header = !had_existing_data;
        }

        let start_row = if reuse_cache {
            let cache = self.writer_cache.as_ref().ok_or_else(|| {
                CsvDataError::WriterCacheNotInitialized("writer cache is not initialized".to_string())
            })?;
            if trajectory.len() >= cache.rows_written {
                cache.rows_written
            } else {
                0
            }
        } else {
            0
        };

        {
            let cache = self
                .writer_cache
                .as_mut()
                .ok_or_else(|| {
                    CsvDataError::WriterCacheNotInitialized("writer cache is not initialized".to_string())
                })?;
            Self::write_trajectory(
                &mut cache.writer,
                &trajectory,
                start_row,
                trajectory.len(),
                write_header,
            )?;
            cache.rows_written = trajectory.len();
        }

        Ok(self)
    }

    pub fn from_csv<P: AsRef<Path>>(
        mut self,
        path: P,
        byte_capacity: usize,
        episode: Option<u64>,
        training_step: Option<u64>,
    ) -> Result<Self, CsvDataError> {
        let target_path = path.as_ref().to_path_buf();
        let reload_from_file = match &self.reader_cache {
            Some(cache) => cache.path != target_path || cache.byte_capacity != byte_capacity,
            None => true,
        };

        if reload_from_file {
            let reader = ReaderBuilder::new()
                .has_headers(true)
                .flexible(true)
                .trim(Trim::All)
                .buffer_capacity(byte_capacity)
                .from_path(&target_path)
                .map_err(CsvDataError::CsvFailure)?;

            self.reader_cache = Some(ReaderCache {
                reader,
                byte_capacity,
                path: target_path,
                records: Vec::new(),
            });
        }

        {
            let cache = self.reader_cache.as_mut().ok_or_else(|| {
                CsvDataError::ReaderCacheNotInitialized("reader cache is not initialized".to_string())
            })?;

            if reload_from_file || cache.records.is_empty() {
                let mut records = Vec::new();
                for row in cache.reader.records() {
                    records.push(row.map_err(CsvDataError::CsvFailure)?);
                }
                cache.records = records;
            }
        }

        let records = self
            .reader_cache
            .as_ref()
            .ok_or_else(|| {
                CsvDataError::ReaderCacheNotInitialized("reader cache is not initialized".to_string())
            })?
            .records
            .clone();

        self.build_trajectory(&records, episode, training_step)?;
        Ok(self)
    }

    fn build_trajectory(
        &mut self,
        records: &[StringRecord],
        episode: Option<u64>,
        training_step: Option<u64>,
    ) -> Result<(), CsvDataError> {
        let max_length = records.len();
        let mut trajectory = RelayRLTrajectory {
            actions: Vec::with_capacity(max_length),
            max_length,
            agent_id: None,
            timestamp: 0,
            episode,
            training_step,
        };

        let mut validation_cache = TrajectoryValidationCache {
            backend: None,
            actor_id: None,
            timestamp: None,
            episode,
            training_step,
        };

        for row in records {
            let action = RelayRLAction::from_csv_data(row.clone(), &mut validation_cache)?;
            trajectory.add_action(action);
        }

        if let Some(agent_id) = validation_cache.actor_id {
            trajectory.agent_id = Some(agent_id);
        }

        if let Some(timestamp) = validation_cache.timestamp {
            trajectory.timestamp = timestamp;
        }

        self.trajectory = Some(trajectory);
        Ok(())
    }

    fn write_trajectory(
        writer: &mut Writer<File>,
        trajectory: &RelayRLTrajectory,
        start_row: usize,
        end_row: usize,
        write_header: bool,
    ) -> Result<(), CsvDataError> {
        if write_header {
            writer
                .write_record(HEADER_NAMES)
                .map_err(CsvDataError::CsvFailure)?;
        }

        for action in trajectory
            .actions
            .iter()
            .skip(start_row)
            .take(end_row.saturating_sub(start_row))
        {
            let row = RelayRLAction::to_csv_data(action)?;
            writer.write_record(&row).map_err(CsvDataError::CsvFailure)?;
        }

        writer.flush().map_err(|error| {
            CsvDataError::TrajectoryBuildFailure(format!("Failed to flush CSV writer: {error}"))
        })?;
        Ok(())
    }
}

trait CsvAction {
    type Action;

    fn to_csv_data(action: &RelayRLAction) -> Result<StringRecord, CsvDataError>;
    fn from_csv_data(
        row: StringRecord,
        validation_cache: &mut TrajectoryValidationCache,
    ) -> Result<Self::Action, CsvDataError>;
}

impl CsvAction for RelayRLAction {
    type Action = RelayRLAction;

    fn to_csv_data(action: &RelayRLAction) -> Result<StringRecord, CsvDataError> {
        let mut fields = vec![String::new(); HEADER_FIELD_COUNT];

        if let Some(obs) = &action.obs {
            let (dtype, shape, f32_data, f64_data, binary_data) = tensor_to_csv_fields(obs)?;
            fields[CsvHeaderFields::ObsDType as usize] = dtype;
            fields[CsvHeaderFields::ObsShape as usize] = shape;
            fields[CsvHeaderFields::ObsF32 as usize] = f32_data;
            fields[CsvHeaderFields::ObsF64 as usize] = f64_data;
            fields[CsvHeaderFields::ObsBinary as usize] = binary_data;
            fields[CsvHeaderFields::Backend as usize] = format!("{:?}", obs.supported_backend);
        }

        if let Some(act) = &action.act {
            let (dtype, shape, f32_data, f64_data, binary_data) = tensor_to_csv_fields(act)?;
            fields[CsvHeaderFields::ActDType as usize] = dtype;
            fields[CsvHeaderFields::ActShape as usize] = shape;
            fields[CsvHeaderFields::ActF32 as usize] = f32_data;
            fields[CsvHeaderFields::ActF64 as usize] = f64_data;
            fields[CsvHeaderFields::ActBinary as usize] = binary_data;
            if fields[CsvHeaderFields::Backend as usize].is_empty() {
                fields[CsvHeaderFields::Backend as usize] = format!("{:?}", act.supported_backend);
            }
        }

        if let Some(mask) = &action.mask {
            let (dtype, shape, f32_data, f64_data, binary_data) = tensor_to_csv_fields(mask)?;
            fields[CsvHeaderFields::MaskDType as usize] = dtype;
            fields[CsvHeaderFields::MaskShape as usize] = shape;
            fields[CsvHeaderFields::MaskF32 as usize] = f32_data;
            fields[CsvHeaderFields::MaskF64 as usize] = f64_data;
            fields[CsvHeaderFields::MaskBinary as usize] = binary_data;
            if fields[CsvHeaderFields::Backend as usize].is_empty() {
                fields[CsvHeaderFields::Backend as usize] = format!("{:?}", mask.supported_backend);
            }
        }

        fields[CsvHeaderFields::ActorID as usize] =
            action.agent_id.map(|id| id.to_string()).unwrap_or_default();
        fields[CsvHeaderFields::Timestamp as usize] = action.timestamp.to_string();
        fields[CsvHeaderFields::Reward as usize] = action.rew.to_string();
        fields[CsvHeaderFields::Done as usize] = action.done.to_string();
        fields[CsvHeaderFields::AuxData as usize] = match &action.data {
            Some(data) => serde_json::to_string(data).map_err(|error| {
                CsvDataError::TrajectoryBuildFailure(format!(
                    "Failed to serialize auxiliary action data as JSON: {error}"
                ))
            })?,
            None => String::new(),
        };

        Ok(StringRecord::from(fields))
    }

    fn from_csv_data(
        row: StringRecord,
        validation_cache: &mut TrajectoryValidationCache,
    ) -> Result<Self::Action, CsvDataError> {
        if row.len() != HEADER_FIELD_COUNT {
            return Err(CsvDataError::TrajectoryBuildFailure(format!(
                "Invalid CSV record: expected {HEADER_FIELD_COUNT} fields, got {}",
                row.len()
            )));
        }

        let mut action = RelayRLAction {
            obs: None,
            act: None,
            mask: None,
            rew: 0.0,
            done: false,
            data: None,
            agent_id: None,
            timestamp: 0,
        };

        let mut reward_seen = false;
        let mut obs_acc = TensorAccumulator::default();
        let mut act_acc = TensorAccumulator::default();
        let mut mask_acc = TensorAccumulator::default();

        for (index, value) in row.iter().enumerate() {
            let field = CSV_FIELDS[index];
            let value = value.trim();

            match field {
                CsvHeaderFields::Backend => {
                    if value.is_empty() {
                        continue;
                    }
                    match &validation_cache.backend {
                        Some(cached) if cached != value => {
                            return Err(CsvDataError::TrajectoryBuildFailure(format!(
                                "Backend mismatch in CSV: expected '{cached}', got '{value}'"
                            )));
                        }
                        Some(_) => {}
                        None => validation_cache.backend = Some(value.to_string()),
                    }
                }
                CsvHeaderFields::ActorID => {
                    if value.is_empty() {
                        continue;
                    }
                    let parsed_actor_id = Uuid::parse_str(value).map_err(|error| {
                        CsvDataError::TrajectoryBuildFailure(format!(
                            "Invalid actor_id UUID '{value}': {error}"
                        ))
                    })?;

                    match validation_cache.actor_id {
                        Some(cached) if cached != parsed_actor_id => {
                            return Err(CsvDataError::TrajectoryBuildFailure(format!(
                                "Actor ID mismatch in CSV: expected '{cached}', got '{parsed_actor_id}'"
                            )));
                        }
                        Some(_) => {}
                        None => validation_cache.actor_id = Some(parsed_actor_id),
                    }
                    action.agent_id = Some(parsed_actor_id);
                }
                CsvHeaderFields::Timestamp => {
                    if value.is_empty() {
                        continue;
                    }
                    let parsed_timestamp: u64 = value.parse().map_err(|error| {
                        CsvDataError::TrajectoryBuildFailure(format!(
                            "Invalid timestamp '{value}': {error}"
                        ))
                    })?;

                    if let Some(cached_timestamp) = validation_cache.timestamp
                        && parsed_timestamp < cached_timestamp
                    {
                        return Err(CsvDataError::TrajectoryBuildFailure(format!(
                            "Timestamp must be monotonic; got {parsed_timestamp} after {cached_timestamp}"
                        )));
                    }
                    validation_cache.timestamp = Some(parsed_timestamp);
                    action.timestamp = parsed_timestamp;
                }
                CsvHeaderFields::ObsDType => {
                    if !value.is_empty() {
                        obs_acc.dtype = Some(parse_dtype(value)?);
                    }
                }
                CsvHeaderFields::ObsShape => {
                    if !value.is_empty() {
                        obs_acc.shape = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::ObsF32 => {
                    if !value.is_empty() {
                        obs_acc.f32_data = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::ObsF64 => {
                    if !value.is_empty() {
                        obs_acc.f64_data = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::ObsBinary => {
                    if !value.is_empty() {
                        obs_acc.binary_data = Some(parse_json_array(value)?);
                    }
                    action.obs = obs_acc.into_tensor_data()?;
                    obs_acc = TensorAccumulator::default();
                }
                CsvHeaderFields::ActDType => {
                    if !value.is_empty() {
                        act_acc.dtype = Some(parse_dtype(value)?);
                    }
                }
                CsvHeaderFields::ActShape => {
                    if !value.is_empty() {
                        act_acc.shape = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::ActF32 => {
                    if !value.is_empty() {
                        act_acc.f32_data = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::ActF64 => {
                    if !value.is_empty() {
                        act_acc.f64_data = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::ActBinary => {
                    if !value.is_empty() {
                        act_acc.binary_data = Some(parse_json_array(value)?);
                    }
                    action.act = act_acc.into_tensor_data()?;
                    act_acc = TensorAccumulator::default();
                }
                CsvHeaderFields::MaskDType => {
                    if !value.is_empty() {
                        mask_acc.dtype = Some(parse_dtype(value)?);
                    }
                }
                CsvHeaderFields::MaskShape => {
                    if !value.is_empty() {
                        mask_acc.shape = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::MaskF32 => {
                    if !value.is_empty() {
                        mask_acc.f32_data = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::MaskF64 => {
                    if !value.is_empty() {
                        mask_acc.f64_data = Some(parse_json_array(value)?);
                    }
                }
                CsvHeaderFields::MaskBinary => {
                    if !value.is_empty() {
                        mask_acc.binary_data = Some(parse_json_array(value)?);
                    }
                    action.mask = mask_acc.into_tensor_data()?;
                    mask_acc = TensorAccumulator::default();
                }
                CsvHeaderFields::Reward => {
                    if value.is_empty() {
                        return Err(CsvDataError::TrajectoryBuildFailure(
                            "Reward field cannot be empty".to_string(),
                        ));
                    }
                    action.rew = value.parse::<f32>().map_err(|error| {
                        CsvDataError::TrajectoryBuildFailure(format!(
                            "Invalid reward value '{value}': {error}"
                        ))
                    })?;
                    reward_seen = true;
                }
                CsvHeaderFields::Done => {
                    action.done = match value {
                        "true" => true,
                        "false" | "" => false,
                        _ => {
                            return Err(CsvDataError::TrajectoryBuildFailure(format!(
                                "Invalid done value '{value}', expected 'true' or 'false'"
                            )));
                        }
                    };
                }
                CsvHeaderFields::AuxData => {
                    if value.is_empty() {
                        action.data = None;
                    } else {
                        let data: HashMap<String, RelayRLData> =
                            serde_json::from_str(value).map_err(|error| {
                                CsvDataError::TrajectoryBuildFailure(format!(
                                    "Failed to deserialize aux_data JSON: {error}"
                                ))
                            })?;
                        action.data = Some(data);
                    }
                }
            }
        }

        if !reward_seen {
            return Err(CsvDataError::TrajectoryBuildFailure(
                "Reward field is required in CSV row".to_string(),
            ));
        }

        Ok(action)
    }
}

fn parse_json_array<T: DeserializeOwned>(value: &str) -> Result<Vec<T>, CsvDataError> {
    serde_json::from_str::<Vec<T>>(value).map_err(|error| {
        CsvDataError::TrajectoryBuildFailure(format!("Failed to parse JSON array '{value}': {error}"))
    })
}

fn vec_to_json_string<T: Serialize>(value: &[T]) -> Result<String, CsvDataError> {
    serde_json::to_string(value).map_err(|error| {
        CsvDataError::TrajectoryBuildFailure(format!("Failed to serialize JSON array: {error}"))
    })
}

fn parse_dtype(value: &str) -> Result<DType, CsvDataError> {
    let trimmed = value.trim();

    #[cfg(feature = "ndarray-backend")]
    if trimmed.starts_with("NdArray(") && trimmed.ends_with(')') {
        let inner = &trimmed["NdArray(".len()..trimmed.len() - 1];
        let dtype = match inner {
            "F16" => NdArrayDType::F16,
            "F32" => NdArrayDType::F32,
            "F64" => NdArrayDType::F64,
            "I8" => NdArrayDType::I8,
            "I16" => NdArrayDType::I16,
            "I32" => NdArrayDType::I32,
            "I64" => NdArrayDType::I64,
            "Bool" => NdArrayDType::Bool,
            _ => {
                return Err(CsvDataError::TrajectoryBuildFailure(format!(
                    "Unsupported NdArray dtype '{inner}'"
                )))
            }
        };
        return Ok(DType::NdArray(dtype));
    }

    #[cfg(feature = "tch-backend")]
    if trimmed.starts_with("Tch(") && trimmed.ends_with(')') {
        let inner = &trimmed["Tch(".len()..trimmed.len() - 1];
        let dtype = match inner {
            "F16" => TchDType::F16,
            "Bf16" => TchDType::Bf16,
            "F32" => TchDType::F32,
            "F64" => TchDType::F64,
            "I8" => TchDType::I8,
            "I16" => TchDType::I16,
            "I32" => TchDType::I32,
            "I64" => TchDType::I64,
            "U8" => TchDType::U8,
            "Bool" => TchDType::Bool,
            _ => {
                return Err(CsvDataError::TrajectoryBuildFailure(format!(
                    "Unsupported Tch dtype '{inner}'"
                )))
            }
        };
        return Ok(DType::Tch(dtype));
    }

    Err(CsvDataError::TrajectoryBuildFailure(format!(
        "Unsupported dtype string '{trimmed}'"
    )))
}

fn build_tensor_data(
    dtype: DType,
    shape: Vec<usize>,
    f32_data: Option<Vec<f32>>,
    f64_data: Option<Vec<f64>>,
    binary_data: Option<Vec<u8>>,
) -> Result<TensorData, CsvDataError> {
    let data = if let Some(values) = f32_data {
        values
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else if let Some(values) = f64_data {
        values
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else if let Some(values) = binary_data {
        values
    } else {
        return Err(CsvDataError::TrajectoryBuildFailure(
            "Tensor dtype/shape were present but tensor payload is missing".to_string(),
        ));
    };

    let supported_backend = TensorData::get_backend_from_dtype(&dtype);
    Ok(TensorData::new(shape, dtype, data, supported_backend))
}

fn tensor_to_csv_fields(
    tensor: &TensorData,
) -> Result<(String, String, String, String, String), CsvDataError> {
    let dtype = tensor.dtype.to_string();
    let shape = vec_to_json_string(&tensor.shape)?;

    if is_f32_dtype(&tensor.dtype) {
        let f32_values = bytes_to_f32_vec(&tensor.data)?;
        return Ok((dtype, shape, vec_to_json_string(&f32_values)?, String::new(), String::new()));
    }

    if is_f64_dtype(&tensor.dtype) {
        let f64_values = bytes_to_f64_vec(&tensor.data)?;
        return Ok((dtype, shape, String::new(), vec_to_json_string(&f64_values)?, String::new()));
    }

    Ok((
        dtype,
        shape,
        String::new(),
        String::new(),
        vec_to_json_string(&tensor.data)?,
    ))
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, CsvDataError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(CsvDataError::TrajectoryBuildFailure(format!(
            "Invalid f32 tensor byte length {}, expected a multiple of 4",
            bytes.len()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn bytes_to_f64_vec(bytes: &[u8]) -> Result<Vec<f64>, CsvDataError> {
    if !bytes.len().is_multiple_of(8) {
        return Err(CsvDataError::TrajectoryBuildFailure(format!(
            "Invalid f64 tensor byte length {}, expected a multiple of 8",
            bytes.len()
        )));
    }

    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn is_f32_dtype(dtype: &DType) -> bool {
    match dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(NdArrayDType::F32) => true,
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F32) => true,
        _ => false,
    }
}

fn is_f64_dtype(dtype: &DType) -> bool {
    match dtype {
        #[cfg(feature = "ndarray-backend")]
        DType::NdArray(NdArrayDType::F64) => true,
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F64) => true,
        _ => false,
    }
}
