use super::{get_backend_str, tensor_to_data_frame};
use crate::data::action::RelayRLAction;
use crate::data::tensor::{DType, NdArrayDType, TensorData};
#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
use crate::data::trajectory::RelayRLTrajectory;
use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanArray, Float32Array, Float32Builder, Float64Array,
    Float64Builder, ListArray, ListBuilder, StringArray, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArrowDataError {
    #[error("Failed to build Arrow schema: {0}")]
    SchemaBuildFailure(String),
    #[error("Failed to build Arrow record batch: {0}")]
    RecordBatchBuildFailure(String),
    #[error("Failed to write Arrow record batch: {0}")]
    RecordBatchWriteFailure(String),
    #[error("Failed to read Arrow file: {0}")]
    ArrowReadFailure(String),
    #[error("Trajectory not initialized: {0}")]
    TrajectoryNotInitialized(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type ArrowTrajectoryError = ArrowDataError;

pub struct ArrowTrajectory {
    pub trajectory: Option<RelayRLTrajectory>,
}

impl ArrowTrajectory {
    pub fn new(trajectory: Option<RelayRLTrajectory>) -> Self {
        Self { trajectory }
    }

    pub fn to_arrow<P: AsRef<Path>>(self, path: P) -> Result<Self, ArrowDataError> {
        let trajectory = self.trajectory.as_ref().ok_or_else(|| {
            ArrowDataError::TrajectoryNotInitialized("No trajectory initialized to write".to_string())
        })?;
        let file = File::create(path.as_ref())?;
        self.write_trajectory(file, trajectory)?;
        Ok(self)
    }

    pub fn from_arrow<P: AsRef<Path>>(
        mut self,
        path: P,
        episode: Option<u64>,
        training_step: Option<u64>,
    ) -> Result<Self, ArrowDataError> {
        let file = File::open(path.as_ref())?;
        let reader = FileReader::try_new(file, None)
            .map_err(|error| ArrowDataError::ArrowReadFailure(error.to_string()))?;

        for maybe_batch in reader {
            let batch =
                maybe_batch.map_err(|error| ArrowDataError::ArrowReadFailure(error.to_string()))?;
            self.build_trajectory(batch, episode, training_step)?;
        }

        Ok(self)
    }

    fn build_trajectory(
        &mut self,
        batch: RecordBatch,
        episode: Option<u64>,
        training_step: Option<u64>,
    ) -> Result<(), ArrowDataError> {
        let rewards = get_required_column::<Float32Array>(&batch, "reward")?;
        let dones = get_required_column::<BooleanArray>(&batch, "done")?;
        let timestamps = get_required_column::<UInt64Array>(&batch, "timestamp")?;
        let agent_ids = get_required_column::<StringArray>(&batch, "agent_id")?;

        let obs_dtype = get_required_column::<StringArray>(&batch, "obs_dtype")?;
        let obs_shape = get_required_column::<ListArray>(&batch, "obs_shape")?;
        let obs_f32 = get_required_column::<ListArray>(&batch, "obs_f32")?;
        let obs_f64 = get_required_column::<ListArray>(&batch, "obs_f64")?;
        let obs_binary = get_required_column::<BinaryArray>(&batch, "obs_binary")?;

        let act_dtype = get_required_column::<StringArray>(&batch, "act_dtype")?;
        let act_shape = get_required_column::<ListArray>(&batch, "act_shape")?;
        let act_f32 = get_required_column::<ListArray>(&batch, "act_f32")?;
        let act_f64 = get_required_column::<ListArray>(&batch, "act_f64")?;
        let act_binary = get_required_column::<BinaryArray>(&batch, "act_binary")?;

        let mask_dtype = get_required_column::<StringArray>(&batch, "mask_dtype")?;
        let mask_shape = get_required_column::<ListArray>(&batch, "mask_shape")?;
        let mask_f32 = get_required_column::<ListArray>(&batch, "mask_f32")?;
        let mask_f64 = get_required_column::<ListArray>(&batch, "mask_f64")?;
        let mask_binary = get_required_column::<BinaryArray>(&batch, "mask_binary")?;

        let mut trajectory = self.trajectory.take().unwrap_or_else(|| RelayRLTrajectory {
            actions: Vec::with_capacity(batch.num_rows()),
            max_length: batch.num_rows(),
            agent_id: None,
            timestamp: 0,
            episode,
            training_step,
        });

        for index in 0..batch.num_rows() {
            let obs = reconstruct_tensor(obs_dtype, obs_shape, obs_f32, obs_f64, obs_binary, index)?;
            let act = reconstruct_tensor(act_dtype, act_shape, act_f32, act_f64, act_binary, index)?;
            let mask =
                reconstruct_tensor(mask_dtype, mask_shape, mask_f32, mask_f64, mask_binary, index)?;

            let agent_id = if agent_ids.is_null(index) {
                None
            } else {
                let value = agent_ids.value(index);
                Some(Uuid::parse_str(value).map_err(|error| {
                    ArrowDataError::RecordBatchBuildFailure(format!(
                        "Invalid agent_id UUID '{value}': {error}"
                    ))
                })?)
            };

            let action = RelayRLAction {
                obs,
                act,
                mask,
                rew: rewards.value(index),
                done: dones.value(index),
                data: None,
                agent_id,
                timestamp: timestamps.value(index),
            };
            trajectory.add_action(action);
        }

        if trajectory.timestamp == 0 && !trajectory.actions.is_empty() {
            trajectory.timestamp = trajectory.actions[0].timestamp;
        }
        if trajectory.agent_id.is_none() {
            trajectory.agent_id = trajectory.actions.iter().find_map(|action| action.agent_id);
        }

        self.trajectory = Some(trajectory);
        Ok(())
    }

    fn write_trajectory(&self, file: File, trajectory: &RelayRLTrajectory) -> Result<(), ArrowDataError> {
        let num_actions = trajectory.actions.len();
        let schema = create_arrow_schema();

        if num_actions == 0 {
            let mut writer = FileWriter::try_new(file, &schema)
                .map_err(|error| ArrowDataError::RecordBatchWriteFailure(error.to_string()))?;
            writer
                .finish()
                .map_err(|error| ArrowDataError::RecordBatchWriteFailure(error.to_string()))?;
            return Ok(());
        }

        let backend_str = get_backend_str(trajectory);
        let mut backends: Vec<String> = Vec::with_capacity(num_actions);
        let mut rewards: Vec<f32> = Vec::with_capacity(num_actions);
        let mut dones: Vec<bool> = Vec::with_capacity(num_actions);
        let mut timestamps: Vec<u64> = Vec::with_capacity(num_actions);
        let mut agent_ids: Vec<Option<String>> = Vec::with_capacity(num_actions);

        let mut obs_dtypes: Vec<Option<String>> = Vec::with_capacity(num_actions);
        let mut obs_shapes: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_actions);
        let mut obs_f32: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_actions);
        let mut obs_f64: Vec<Option<Vec<f64>>> = Vec::with_capacity(num_actions);
        let mut obs_binary: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_actions);

        let mut act_dtypes: Vec<Option<String>> = Vec::with_capacity(num_actions);
        let mut act_shapes: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_actions);
        let mut act_f32: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_actions);
        let mut act_f64: Vec<Option<Vec<f64>>> = Vec::with_capacity(num_actions);
        let mut act_binary: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_actions);

        let mut mask_dtypes: Vec<Option<String>> = Vec::with_capacity(num_actions);
        let mut mask_shapes: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_actions);
        let mut mask_f32: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_actions);
        let mut mask_f64: Vec<Option<Vec<f64>>> = Vec::with_capacity(num_actions);
        let mut mask_binary: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_actions);

        for action in trajectory.actions.iter() {
            backends.push(backend_str.clone());
            rewards.push(action.get_rew());
            dones.push(action.get_done());
            timestamps.push(action.get_timestamp());
            agent_ids.push(action.get_agent_id().map(|id| id.to_string()));

            if let Some(obs) = action.get_obs() {
                let frame = tensor_to_data_frame(obs);
                obs_dtypes.push(Some(frame.dtype_str));
                obs_shapes.push(Some(frame.shape));
                obs_f32.push(frame.f32_data);
                obs_f64.push(frame.f64_data);
                obs_binary.push(frame.binary_data);
            } else {
                obs_dtypes.push(None);
                obs_shapes.push(None);
                obs_f32.push(None);
                obs_f64.push(None);
                obs_binary.push(None);
            }

            if let Some(act) = action.get_act() {
                let frame = tensor_to_data_frame(act);
                act_dtypes.push(Some(frame.dtype_str));
                act_shapes.push(Some(frame.shape));
                act_f32.push(frame.f32_data);
                act_f64.push(frame.f64_data);
                act_binary.push(frame.binary_data);
            } else {
                act_dtypes.push(None);
                act_shapes.push(None);
                act_f32.push(None);
                act_f64.push(None);
                act_binary.push(None);
            }

            if let Some(mask) = action.get_mask() {
                let frame = tensor_to_data_frame(mask);
                mask_dtypes.push(Some(frame.dtype_str));
                mask_shapes.push(Some(frame.shape));
                mask_f32.push(frame.f32_data);
                mask_f64.push(frame.f64_data);
                mask_binary.push(frame.binary_data);
            } else {
                mask_dtypes.push(None);
                mask_shapes.push(None);
                mask_f32.push(None);
                mask_f64.push(None);
                mask_binary.push(None);
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(backends)) as ArrayRef,
                Arc::new(Float32Array::from(rewards)) as ArrayRef,
                Arc::new(BooleanArray::from(dones)) as ArrayRef,
                Arc::new(UInt64Array::from(timestamps)) as ArrayRef,
                Arc::new(StringArray::from(agent_ids)) as ArrayRef,
                Arc::new(StringArray::from(obs_dtypes)) as ArrayRef,
                build_shape_list_array(obs_shapes),
                build_f32_list_array(obs_f32),
                build_f64_list_array(obs_f64),
                build_binary_array(obs_binary),
                Arc::new(StringArray::from(act_dtypes)) as ArrayRef,
                build_shape_list_array(act_shapes),
                build_f32_list_array(act_f32),
                build_f64_list_array(act_f64),
                build_binary_array(act_binary),
                Arc::new(StringArray::from(mask_dtypes)) as ArrayRef,
                build_shape_list_array(mask_shapes),
                build_f32_list_array(mask_f32),
                build_f64_list_array(mask_f64),
                build_binary_array(mask_binary),
            ],
        )
        .map_err(|error| ArrowDataError::RecordBatchBuildFailure(error.to_string()))?;

        let mut writer = FileWriter::try_new(file, &schema)
            .map_err(|error| ArrowDataError::RecordBatchWriteFailure(error.to_string()))?;
        writer
            .write(&batch)
            .map_err(|error| ArrowDataError::RecordBatchWriteFailure(error.to_string()))?;
        writer
            .finish()
            .map_err(|error| ArrowDataError::RecordBatchWriteFailure(error.to_string()))?;

        Ok(())
    }
}

fn create_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("backend", DataType::Utf8, false),
        Field::new("reward", DataType::Float32, false),
        Field::new("done", DataType::Boolean, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("agent_id", DataType::Utf8, true),
        Field::new("obs_dtype", DataType::Utf8, true),
        Field::new(
            "obs_shape",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            true,
        ),
        Field::new(
            "obs_f32",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ),
        Field::new(
            "obs_f64",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
        Field::new("obs_binary", DataType::Binary, true),
        Field::new("act_dtype", DataType::Utf8, true),
        Field::new(
            "act_shape",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            true,
        ),
        Field::new(
            "act_f32",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ),
        Field::new(
            "act_f64",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
        Field::new("act_binary", DataType::Binary, true),
        Field::new("mask_dtype", DataType::Utf8, true),
        Field::new(
            "mask_shape",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            true,
        ),
        Field::new(
            "mask_f32",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ),
        Field::new(
            "mask_f64",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
        Field::new("mask_binary", DataType::Binary, true),
    ]))
}

fn build_f32_list_array(data: Vec<Option<Vec<f32>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for item in data {
        match item {
            Some(values) => {
                for value in values {
                    builder.values().append_value(value);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_f64_list_array(data: Vec<Option<Vec<f64>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(Float64Builder::new());
    for item in data {
        match item {
            Some(values) => {
                for value in values {
                    builder.values().append_value(value);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_shape_list_array(data: Vec<Option<Vec<u64>>>) -> ArrayRef {
    let mut builder = ListBuilder::new(UInt64Builder::new());
    for item in data {
        match item {
            Some(values) => {
                for value in values {
                    builder.values().append_value(value);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

fn build_binary_array(data: Vec<Option<Vec<u8>>>) -> ArrayRef {
    let mut builder = BinaryBuilder::new();
    for item in data {
        match item {
            Some(bytes) => builder.append_value(bytes),
            None => builder.append_null(),
        }
    }
    Arc::new(builder.finish())
}

fn get_required_column<'a, T: 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a T, ArrowDataError> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<T>())
        .ok_or_else(|| {
            ArrowDataError::RecordBatchBuildFailure(format!(
                "Missing or invalid column '{name}' in Arrow record batch"
            ))
        })
}

fn parse_dtype(dtype: &str) -> Result<DType, ArrowDataError> {
    let value = dtype.trim();

    #[cfg(feature = "ndarray-backend")]
    if value.starts_with("NdArray(") && value.ends_with(')') {
        let inner = &value["NdArray(".len()..value.len() - 1];
        let parsed = match inner {
            "F16" => NdArrayDType::F16,
            "F32" => NdArrayDType::F32,
            "F64" => NdArrayDType::F64,
            "I8" => NdArrayDType::I8,
            "I16" => NdArrayDType::I16,
            "I32" => NdArrayDType::I32,
            "I64" => NdArrayDType::I64,
            "Bool" => NdArrayDType::Bool,
            _ => {
                return Err(ArrowDataError::RecordBatchBuildFailure(format!(
                    "Unsupported NdArray dtype '{inner}'"
                )))
            }
        };
        return Ok(DType::NdArray(parsed));
    }

    #[cfg(feature = "tch-backend")]
    if value.starts_with("Tch(") && value.ends_with(')') {
        let inner = &value["Tch(".len()..value.len() - 1];
        let parsed = match inner {
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
                return Err(ArrowDataError::RecordBatchBuildFailure(format!(
                    "Unsupported Tch dtype '{inner}'"
                )))
            }
        };
        return Ok(DType::Tch(parsed));
    }

    Err(ArrowDataError::RecordBatchBuildFailure(format!(
        "Unsupported dtype string '{value}'"
    )))
}

fn extract_u64_list(list: &ListArray, index: usize) -> Result<Option<Vec<u64>>, ArrowDataError> {
    if list.is_null(index) {
        return Ok(None);
    }
    let values = list.value(index);
    let array = values.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
        ArrowDataError::RecordBatchBuildFailure("Expected UInt64 list values".to_string())
    })?;
    Ok(Some((0..array.len()).map(|i| array.value(i)).collect()))
}

fn extract_f32_list(list: &ListArray, index: usize) -> Result<Option<Vec<f32>>, ArrowDataError> {
    if list.is_null(index) {
        return Ok(None);
    }
    let values = list.value(index);
    let array = values.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
        ArrowDataError::RecordBatchBuildFailure("Expected Float32 list values".to_string())
    })?;
    Ok(Some((0..array.len()).map(|i| array.value(i)).collect()))
}

fn extract_f64_list(list: &ListArray, index: usize) -> Result<Option<Vec<f64>>, ArrowDataError> {
    if list.is_null(index) {
        return Ok(None);
    }
    let values = list.value(index);
    let array = values.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        ArrowDataError::RecordBatchBuildFailure("Expected Float64 list values".to_string())
    })?;
    Ok(Some((0..array.len()).map(|i| array.value(i)).collect()))
}

fn reconstruct_tensor(
    dtype_array: &StringArray,
    shape_array: &ListArray,
    f32_array: &ListArray,
    f64_array: &ListArray,
    binary_array: &BinaryArray,
    index: usize,
) -> Result<Option<TensorData>, ArrowDataError> {
    if dtype_array.is_null(index) {
        return Ok(None);
    }

    let dtype = parse_dtype(dtype_array.value(index))?;
    let shape_u64 = extract_u64_list(shape_array, index)?.ok_or_else(|| {
        ArrowDataError::RecordBatchBuildFailure("Missing shape for non-null tensor".to_string())
    })?;
    let shape = shape_u64
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                ArrowDataError::RecordBatchBuildFailure(format!(
                    "Tensor shape value '{value}' does not fit in usize"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let f32_values = extract_f32_list(f32_array, index)?;
    let f64_values = extract_f64_list(f64_array, index)?;
    let binary_values = if binary_array.is_null(index) {
        None
    } else {
        Some(binary_array.value(index).to_vec())
    };

    let data = if let Some(values) = f32_values {
        values
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else if let Some(values) = f64_values {
        values
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else if let Some(values) = binary_values {
        values
    } else {
        return Err(ArrowDataError::RecordBatchBuildFailure(
            "Tensor is missing data payload".to_string(),
        ));
    };

    Ok(Some(TensorData::new(
        shape,
        dtype.clone(),
        data,
        TensorData::get_backend_from_dtype(&dtype),
    )))
}