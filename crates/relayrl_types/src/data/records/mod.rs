pub mod csv;
pub mod arrow;

use crate::data::action::RelayRLAction;
use crate::data::tensor::{DType, NdArrayDType, TensorData};
#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
use crate::data::trajectory::RelayRLTrajectory;

pub(super) struct TensorDataFrame {
    dtype_str: String,
    shape: Vec<u64>,
    f32_data: Option<Vec<f32>>,
    f64_data: Option<Vec<f64>>,
    binary_data: Option<Vec<u8>>,
}

pub(super) fn tensor_to_data_frame(tensor: &TensorData) -> TensorDataFrame {
    let dtype_str = tensor.dtype.to_string();
    let shape: Vec<u64> = tensor.shape.iter().map(|&s| s as u64).collect();

    match &tensor.dtype {
        DType::NdArray(NdArrayDType::F32) => {
            let floats: Vec<f32> = tensor.data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
            TensorDataFrame {
                dtype_str,
                shape,
                f32_data: Some(floats),
                f64_data: None,
                binary_data: None,
            }
        }
        DType::NdArray(NdArrayDType::F64) => {
            let floats: Vec<f64> = tensor.data.chunks_exact(8).map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
            TensorDataFrame {
                dtype_str,
                shape,
                f32_data: None,
                f64_data: Some(floats),
                binary_data: None,
            }
        }
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F32) => {
            let floats: Vec<f32> = tensor.data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
            TensorDataFrame {
                dtype_str,
                shape,
                f32_data: Some(floats),
                f64_data: None,
                binary_data: None,
            }
        }
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F64) => {
            let floats: Vec<f64> = tensor.data.chunks_exact(8).map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
            TensorDataFrame {
                dtype_str,
                shape,
                f32_data: None,
                f64_data: Some(floats),
                binary_data: None,
            }
        }
        _ => TensorDataFrame {
            dtype_str,
            shape,
            f32_data: None,
            f64_data: None,
            binary_data: Some(tensor.data.clone()),
        },
    }
}

pub(super) fn get_backend_str(trajectory: &RelayRLTrajectory) -> String {
    trajectory.actions.iter().find_map(|a: &RelayRLAction| {
        a.get_obs().map(|t| format!("{:?}", t.supported_backend))
    })
        .unwrap_or_else(|| "None".to_string())
}