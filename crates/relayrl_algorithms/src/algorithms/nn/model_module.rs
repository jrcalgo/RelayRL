use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use super::types::ArchLayer;

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
            };
            ModelModule::from_pt_bytes(pt_bytes, metadata).ok()
        }
        _ => None,
    }
}
