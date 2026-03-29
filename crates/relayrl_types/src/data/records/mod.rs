pub mod csv;
pub mod arrow;

use crate::data::action::RelayRLAction;
use crate::data::tensor::{DType, TensorData};
#[cfg(feature = "ndarray-backend")]
use crate::data::tensor::NdArrayDType;
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
        #[cfg(feature = "ndarray-backend")]
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
        #[cfg(feature = "ndarray-backend")]
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

#[cfg(all(test, feature = "ndarray-backend"))]
mod unit_tests {
    use super::*;

    fn f32_tensor(values: &[f32]) -> TensorData {
        TensorData::new(
            vec![values.len()],
            DType::NdArray(NdArrayDType::F32),
            values.iter().flat_map(|value| value.to_le_bytes()).collect(),
            crate::data::tensor::SupportedTensorBackend::NdArray,
        )
    }

    fn f64_tensor(values: &[f64]) -> TensorData {
        TensorData::new(
            vec![values.len()],
            DType::NdArray(NdArrayDType::F64),
            values.iter().flat_map(|value| value.to_le_bytes()).collect(),
            crate::data::tensor::SupportedTensorBackend::NdArray,
        )
    }

    fn i32_tensor(values: &[i32]) -> TensorData {
        TensorData::new(
            vec![values.len()],
            DType::NdArray(NdArrayDType::I32),
            values.iter().flat_map(|value| value.to_le_bytes()).collect(),
            crate::data::tensor::SupportedTensorBackend::NdArray,
        )
    }

    #[test]
    fn tensor_to_data_frame_extracts_f32_values() {
        let frame = tensor_to_data_frame(&f32_tensor(&[1.0, 2.0]));

        assert_eq!(frame.dtype_str, "NdArray(F32)");
        assert_eq!(frame.shape, vec![2]);
        assert_eq!(frame.f32_data, Some(vec![1.0, 2.0]));
        assert!(frame.f64_data.is_none());
        assert!(frame.binary_data.is_none());
    }

    #[test]
    fn tensor_to_data_frame_extracts_f64_values() {
        let frame = tensor_to_data_frame(&f64_tensor(&[1.5, 2.5]));

        assert_eq!(frame.dtype_str, "NdArray(F64)");
        assert_eq!(frame.f64_data, Some(vec![1.5, 2.5]));
        assert!(frame.f32_data.is_none());
    }

    #[test]
    fn tensor_to_data_frame_falls_back_to_binary_for_non_float_types() {
        let tensor = i32_tensor(&[1, -2, 3]);
        let frame = tensor_to_data_frame(&tensor);

        assert_eq!(frame.dtype_str, "NdArray(I32)");
        assert!(frame.f32_data.is_none());
        assert!(frame.f64_data.is_none());
        assert_eq!(frame.binary_data, Some(tensor.data));
    }

    #[test]
    fn get_backend_str_uses_the_first_observation_tensor() {
        let action = RelayRLAction::new(Some(f32_tensor(&[1.0])), None, None, 0.0, false, None, None);
        let trajectory = RelayRLTrajectory {
            actions: vec![action],
            max_length: 1,
            agent_id: None,
            timestamp: 0,
            episode: None,
            training_step: None,
        };

        assert_eq!(get_backend_str(&trajectory), "NdArray");
    }

    #[test]
    fn get_backend_str_returns_none_when_observations_are_absent() {
        let action = RelayRLAction::minimal(0.0, false);
        let trajectory = RelayRLTrajectory {
            actions: vec![action],
            max_length: 1,
            agent_id: None,
            timestamp: 0,
            episode: None,
            training_step: None,
        };

        assert_eq!(get_backend_str(&trajectory), "None");
    }
}