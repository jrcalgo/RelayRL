use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use super::types::ArchLayer;

/// Serializes flat MLP layer specs into a runnable `ModelModule` (ONNX, or LibTorch under `tch-model`).
pub fn acquire_model_module<B: Backend + BackendMatcher<Backend = B>>(
    model_name: &str,
    layer_specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)>,
    input_dtype: relayrl_types::data::tensor::DType,
    output_dtype: relayrl_types::data::tensor::DType,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    device: Option<relayrl_types::data::tensor::DeviceType>,
) -> Option<relayrl_types::model::ModelModule<B>> {
    use relayrl_types::data::tensor::SupportedTensorBackend;
    use relayrl_types::model::{ModelFileType, ModelMetadata, ModelModule};

    if layer_specs.is_empty() {
        return None;
    }

    match B::get_supported_backend() {
        SupportedTensorBackend::NdArray => {
            use crate::algorithms::onnx_builder::build_onnx_mlp_bytes;

            let onnx_bytes = build_onnx_mlp_bytes(&layer_specs);
            if onnx_bytes.is_empty() {
                return None;
            }

            let model_file = format!("{}.onnx", model_name);

            let metadata = ModelMetadata {
                model_file,
                model_type: ModelFileType::Onnx,
                input_dtype,
                output_dtype,
                input_shape,
                output_shape,
                default_device: device,
            };

            ModelModule::from_onnx_bytes(onnx_bytes, metadata).ok()
        }
        #[cfg(all(feature = "tch-backend", feature = "tch-model"))]
        SupportedTensorBackend::Tch => {
            use crate::algorithms::torch_builder::build_pt_mlp_temp;

            let (pt_bytes, _temp_path) = build_pt_mlp_temp(&layer_specs).ok()?;
            if pt_bytes.is_empty() {
                return None;
            }

            let model_file = format!("{}.pt", model_name);

            let metadata = ModelMetadata {
                model_file,
                model_type: ModelFileType::Pt,
                input_dtype,
                output_dtype,
                input_shape,
                output_shape,
                default_device: device,
            };

            ModelModule::from_pt_bytes(pt_bytes, metadata).ok()
        }
        _ => None,
    }
}

/// Build a `ModelModule` from an architecture-aware layer sequence.
///
/// Used for `ConvNetPolicy` and any other network that cannot be described as
/// flat dense `LayerSpecs`.  Dispatches to the correct backend builder:
/// - `NdArray` → ONNX via `build_onnx_conv_bytes`
/// - `Tch`     → TorchScript `.pt` via `build_pt_conv_temp`
pub fn acquire_conv_model_module<B: Backend + BackendMatcher<Backend = B>>(
    model_name: &str,
    arch: Vec<ArchLayer>,
    input_dtype: relayrl_types::data::tensor::DType,
    output_dtype: relayrl_types::data::tensor::DType,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    device: Option<relayrl_types::data::tensor::DeviceType>,
) -> Option<relayrl_types::model::ModelModule<B>> {
    use relayrl_types::data::tensor::SupportedTensorBackend;
    use relayrl_types::model::{ModelFileType, ModelMetadata, ModelModule};

    if arch.is_empty() {
        return None;
    }

    let obs_dim = input_shape.get(1).copied().unwrap_or(0);
    let act_dim = output_shape.get(1).copied().unwrap_or(0);

    match B::get_supported_backend() {
        SupportedTensorBackend::NdArray => {
            use crate::algorithms::onnx_builder::build_onnx_conv_bytes;

            let onnx_bytes = build_onnx_conv_bytes(&arch, obs_dim, act_dim);
            if onnx_bytes.is_empty() {
                return None;
            }

            let metadata = ModelMetadata {
                model_file: format!("{}.onnx", model_name),
                model_type: ModelFileType::Onnx,
                input_dtype,
                output_dtype,
                input_shape,
                output_shape,
                default_device: device,
            };
            ModelModule::from_onnx_bytes(onnx_bytes, metadata).ok()
        }
        #[cfg(all(feature = "tch-backend", feature = "tch-model"))]
        SupportedTensorBackend::Tch => {
            use crate::algorithms::torch_builder::build_pt_conv_temp;

            let (pt_bytes, _temp_path) = build_pt_conv_temp(&arch, obs_dim).ok()?;
            if pt_bytes.is_empty() {
                return None;
            }

            let metadata = ModelMetadata {
                model_file: format!("{}.pt", model_name),
                model_type: ModelFileType::Pt,
                input_dtype,
                output_dtype,
                input_shape,
                output_shape,
                default_device: device,
                obs_preprocess: relayrl_types::model::obs_preprocess::ObservationPreprocessMetadata::caller_normalized(preprocess_width),
            };
            ModelModule::from_pt_bytes(pt_bytes, metadata).ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod preprocess_export_tests {
    use super::*;
    use relayrl_types::data::tensor::{
        DType, DeviceType, NdArrayDType, SupportedTensorBackend, TensorData,
    };
    use relayrl_types::model::obs_preprocess::{
        ObservationNormalizerState, ObservationPreprocessMetadata, ObservationPreprocessMode,
        fold_observation_norm_into_first_layer,
    };

    fn sample_state() -> ObservationNormalizerState {
        let mut state = ObservationNormalizerState::identity(2);
        state.update_row(&[1.0, 2.0]);
        state.update_row(&[3.0, 5.0]);
        state.bind_policy_version(7);
        state
    }

    fn sample_specs() -> Vec<(usize, usize, Vec<f32>, Vec<f32>)> {
        vec![
            (
                2,
                3,
                vec![0.2, -0.1, 0.5, 0.3, 0.0, -0.4],
                vec![0.1, -0.2, 0.3],
            ),
            (3, 1, vec![1.0, -1.0, 0.5], vec![0.25]),
        ]
    }

    fn f32_tensor(values: &[f32], dtype: DType, backend: SupportedTensorBackend) -> TensorData {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        TensorData::new(vec![1, values.len()], dtype, bytes, backend)
    }

    fn f32_output(data: &TensorData) -> Vec<f32> {
        data.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (actual, expected) in left.iter().zip(right) {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }

    fn parity<B>(dtype: DType, backend: SupportedTensorBackend, device: DeviceType, label: &str)
    where
        B: burn_tensor::backend::Backend
            + relayrl_types::prelude::tensor::relayrl::BackendMatcher<Backend = B>,
    {
        let state = sample_state();
        let specs = sample_specs();
        let unfolded = acquire_model_module::<B>(
            "ppo_pi",
            specs.clone(),
            dtype.clone(),
            dtype.clone(),
            vec![1, 2],
            vec![1, 1],
            Some(device.clone()),
        )
        .expect("unfolded module");
        assert_eq!(
            unfolded.metadata.obs_preprocess.mode,
            ObservationPreprocessMode::CallerNormalized
        );
        assert_eq!(unfolded.metadata.obs_preprocess.state.schema_version, 1);
        assert_eq!(unfolded.metadata.obs_preprocess.state.count, 0);
        let mut folded_specs = specs.clone();
        fold_observation_norm_into_first_layer(&mut folded_specs[0], &state).unwrap();
        let folded = acquire_model_module_with_preprocess::<B>(
            "ppo_pi",
            folded_specs,
            dtype.clone(),
            dtype.clone(),
            vec![1, 2],
            vec![1, 1],
            Some(device.clone()),
            ObservationPreprocessMetadata::folded(state.clone(), 7),
        )
        .expect("folded module");
        let external = acquire_model_module_with_preprocess::<B>(
            "ppo_pi",
            specs,
            dtype.clone(),
            dtype.clone(),
            vec![1, 2],
            vec![1, 1],
            Some(device),
            ObservationPreprocessMetadata::external_frozen(state.clone(), 7),
        )
        .expect("external module");
        assert_eq!(
            external.metadata.obs_preprocess.mode,
            ObservationPreprocessMode::ExternalFrozen
        );

        let raw = [2.5f32, -1.0];
        let normalized = state.normalize_row(&[f64::from(raw[0]), f64::from(raw[1])]);
        let normalized = [normalized[0] as f32, normalized[1] as f32];
        let expected = f32_output(
            &unfolded
                .flat_batch_inference(f32_tensor(&normalized, dtype.clone(), backend.clone()))
                .unwrap(),
        );
        assert_close(
            &f32_output(
                &folded
                    .flat_batch_inference(f32_tensor(&raw, dtype.clone(), backend.clone()))
                    .unwrap(),
            ),
            &expected,
        );
        assert_close(
            &f32_output(
                &external
                    .flat_batch_inference(f32_tensor(&raw, dtype.clone(), backend.clone()))
                    .unwrap(),
            ),
            &expected,
        );
        assert_eq!(
            folded.metadata.obs_preprocess.mode,
            ObservationPreprocessMode::FoldedIntoGraph
        );

        let dir =
            std::env::temp_dir().join(format!("relayrl-obs-norm-{}-{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        folded.save(&dir).unwrap();
        let loaded = relayrl_types::model::ModelModule::<B>::load_from_path(&dir).unwrap();
        assert_eq!(loaded.metadata.obs_preprocess.state.policy_version, 7);
        assert_close(
            &f32_output(
                &loaded
                    .flat_batch_inference(f32_tensor(&raw, dtype, backend))
                    .unwrap(),
            ),
            &expected,
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn onnx_folded_and_external_match_caller_normalized_input() {
        parity::<burn_ndarray::NdArray>(
            DType::NdArray(NdArrayDType::F32),
            SupportedTensorBackend::NdArray,
            DeviceType::Cpu,
            "onnx",
        );
    }

    #[cfg(all(feature = "tch-backend", feature = "tch-model"))]
    #[test]
    fn torchscript_folded_and_external_match_caller_normalized_input() {
        use relayrl_types::data::tensor::TchDType;
        parity::<burn_tch::LibTorch>(
            DType::Tch(TchDType::F32),
            SupportedTensorBackend::Tch,
            DeviceType::Cpu,
            "torchscript",
        );
    }
}
