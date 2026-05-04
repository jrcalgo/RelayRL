/// LibTorch (TorchScript) model builder for fully-connected MLPs.
///
/// This module creates TorchScript models from layer specifications that match
/// the format used by `WeightProvider::get_pi_layer_specs`. It serves as the
/// LibTorch counterpart to `onnx_builder.rs`.
///
/// # Layout
///
/// The layer specifications follow Burn's `Linear` convention: weights are stored
/// in row-major `[in_features, out_features]` order. This builder constructs a
/// sequential model with Linear layers and ReLU activations between layers (but
/// not after the final layer).
///
/// # Temporary File Approach
///
/// Since `tch::CModule::load` only supports loading from filesystem paths (no
/// in-memory buffer API like ONNX Runtime's `commit_from_memory`), this builder:
/// 1. Constructs the model in memory using `tch::nn`
/// 2. Saves it to a temporary file via `CModule::save`
/// 3. Reads the file bytes back for storage
/// 4. Returns both the bytes and the temp path (caller must load from path)
use std::io::Read;
use std::path::PathBuf;

/// Build a TorchScript MLP model from layer specifications.
///
/// `layer_specs`: `(in_dim, out_dim, flat_weights, flat_biases)` per layer, ordered
/// input→output. ReLU is inserted between every consecutive layer pair; the last
/// layer has no activation.
///
/// Returns `(bytes, temp_path)`:
/// - `bytes`: The serialized TorchScript model for storage/transmission
/// - `temp_path`: Path to the temporary file (needed for `CModule::load`)
///
/// The caller is responsible for loading the model via `CModule::load(&temp_path)`
/// and managing the temporary file lifecycle.
#[cfg(feature = "tch-model")]
pub fn build_pt_mlp_temp(
    layer_specs: &[(usize, usize, Vec<f32>, Vec<f32>)],
) -> Result<(Vec<u8>, PathBuf), String> {
    use tch::nn::Module;
    use tch::{Device, Kind, Tensor, nn};

    if layer_specs.is_empty() {
        return Err("Empty layer specs".to_string());
    }

    // Create a variable store for building the model
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();

    // Build sequential layers
    let mut seq = nn::seq();

    for (idx, (layer_in, layer_out, weights, biases)) in layer_specs.iter().enumerate() {
        // Create a linear layer
        let layer_path = root.sub(&format!("layer_{}", idx));
        let mut linear_config = nn::LinearConfig::default();
        linear_config.bias = true;

        let mut linear = nn::linear(
            &layer_path,
            *layer_in as i64,
            *layer_out as i64,
            linear_config,
        );

        // Load the weights and biases from layer_specs into the linear layer
        // Weights are [in_features, out_features] in Burn format
        // tch expects [out_features, in_features] for nn::Linear weight parameter
        let weight_tensor = Tensor::from_slice(weights)
            .reshape([*layer_in as i64, *layer_out as i64])
            .transpose(0, 1); // Transpose to [out, in] for PyTorch convention

        let bias_tensor = Tensor::from_slice(biases);

        // Copy the tensors into the linear layer's parameters
        tch::no_grad(|| {
            linear.ws.copy_(&weight_tensor);
            if let Some(ref mut bs) = linear.bs {
                bs.copy_(&bias_tensor);
            }
        });

        // Add linear layer to sequence
        seq = seq.add(linear);

        // Add ReLU between layers (but not after the last layer)
        if idx < layer_specs.len() - 1 {
            seq = seq.add_fn(|x| x.relu());
        }
    }

    // Create a temporary file for saving the model
    let temp_file = tempfile::Builder::new()
        .prefix("relayrl_pt_model_")
        .suffix(".pt")
        .tempfile()
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    let temp_path = temp_file.path().to_path_buf();

    // Trace the model to create a TorchScript module
    // We need an example input to trace - use the first layer's input dimension
    let in_dim = layer_specs[0].0 as i64;
    let example_input = Tensor::zeros([1, in_dim], (Kind::Float, Device::Cpu));

    // Create a traced module using create_by_tracing
    // The closure must return Vec<Tensor> and be passed as &mut
    let mut trace_closure = |inputs: &[Tensor]| -> Vec<Tensor> { vec![seq.forward(&inputs[0])] };

    let module =
        tch::CModule::create_by_tracing("mlp", "forward", &[example_input], &mut trace_closure)
            .map_err(|e| format!("Failed to create traced module: {}", e))?;

    // Save the traced module
    module
        .save(&temp_path)
        .map_err(|e| format!("Failed to save model: {}", e))?;

    // Read the bytes back from the file
    let mut file = std::fs::File::open(&temp_path)
        .map_err(|e| format!("Failed to open saved model: {}", e))?;

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read model bytes: {}", e))?;

    // Keep the temp file alive by forgetting it - caller manages cleanup
    std::mem::forget(temp_file);

    Ok((bytes, temp_path))
}

#[cfg(not(feature = "tch-model"))]
pub fn build_pt_mlp_temp(
    _layer_specs: &[(usize, usize, Vec<f32>, Vec<f32>)],
) -> Result<(Vec<u8>, PathBuf), String> {
    Err("tch-model feature not enabled".to_string())
}

#[cfg(all(test, feature = "tch-model"))]
mod tests {
    use super::*;

    #[test]
    fn test_build_pt_mlp_single_layer() {
        let weights = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity-ish
        let biases = vec![0.0f32, 0.0];
        let specs = vec![(2usize, 2usize, weights, biases)];

        let result = build_pt_mlp_temp(&specs);
        assert!(result.is_ok(), "Should successfully build PT model");

        let (bytes, path) = result.unwrap();
        assert!(!bytes.is_empty(), "PT bytes should not be empty");
        assert!(path.exists(), "Temp file should exist");

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_build_pt_mlp_empty_layers() {
        let result = build_pt_mlp_temp(&[]);
        assert!(result.is_err(), "Should fail on empty layer specs");
    }

    #[test]
    fn test_build_pt_mlp_two_layers() {
        let w1 = vec![0.1f32; 4 * 8]; // 4→8
        let b1 = vec![0.0f32; 8];
        let w2 = vec![0.2f32; 8 * 2]; // 8→2
        let b2 = vec![0.0f32; 2];
        let specs = vec![(4, 8, w1, b1), (8, 2, w2, b2)];

        let result = build_pt_mlp_temp(&specs);
        assert!(result.is_ok(), "Should successfully build 2-layer PT model");

        let (bytes, path) = result.unwrap();
        assert!(bytes.len() > 100, "Expected non-trivial PT model bytes");

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
