//! LibTorch (TorchScript) model builders for fully-connected MLPs and ConvNets.
//!
//! Two public entry points:
//! - `build_pt_mlp_temp`  — dense MLP from flat `LayerSpecs`.
//! - `build_pt_conv_temp` — architecture-aware conv+dense network from `ArchLayer` slice.
//!
//! Both follow the same temporary-file approach:
//! 1. Construct the model in memory using `tch::nn`.
//! 2. Freeze parameters and trace via `CModule::create_by_tracing`.
//! 3. Save to a temp file with `CModule::save`.
//! 4. Read the bytes back and return `(bytes, temp_path)`.
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
    let mut vs = nn::VarStore::new(Device::Cpu);
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

    // Freeze all VarStore parameters before tracing: the JIT tracer bakes captured
    // tensors as constants, and PyTorch refuses tensors with requires_grad=true in
    // that role. no_grad() only disables gradient *computation*; freeze() clears the
    // requires_grad flag on every parameter so the tracer accepts them.
    vs.freeze();

    // Trace the model to create a TorchScript module
    // We need an example input to trace - use the first layer's input dimension
    let in_dim = layer_specs[0].0 as i64;
    let example_input = Tensor::zeros([1, in_dim], (Kind::Float, Device::Cpu));

    // Create a traced module using create_by_tracing
    // The closure must return Vec<Tensor> and be passed as &mut
    let mut trace_closure = |inputs: &[Tensor]| -> Vec<Tensor> { vec![seq.forward(&inputs[0])] };

    let module = tch::no_grad(|| {
        tch::CModule::create_by_tracing("mlp", "forward", &[example_input], &mut trace_closure)
    })
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

/// Stub that errors because TorchScript export requires the `tch-model` feature.
#[cfg(not(feature = "tch-model"))]
pub fn build_pt_mlp_temp(
    _layer_specs: &[(usize, usize, Vec<f32>, Vec<f32>)],
) -> Result<(Vec<u8>, PathBuf), String> {
    Err("tch-model feature not enabled".to_string())
}

// ── Conv model builder ────────────────────────────────────────────────────────

/// Build a TorchScript `.pt` model from an `ArchLayer` sequence (conv + dense).
///
/// `arch` — the layer sequence returned by `ConvNetPolicy::get_arch_spec()`.
/// `obs_dim` — flat input size (e.g. 27 648 for VizDoom); used for the example
///             input during JIT tracing.
///
/// Returns `(bytes, temp_path)`.  The caller is responsible for keeping the
/// temp file alive if they need to reload via `CModule::load(&temp_path)`.
#[cfg(feature = "tch-model")]
pub fn build_pt_conv_temp(
    arch: &[crate::algorithms::ArchLayer],
    obs_dim: usize,
) -> Result<(Vec<u8>, PathBuf), String> {
    use crate::algorithms::ArchLayer;
    use tch::nn::Module;
    use tch::{Device, Kind, Tensor, nn};

    if arch.is_empty() {
        return Err("Empty arch spec".to_string());
    }

    let mut vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let mut seq = nn::seq();

    let mut conv_idx = 0usize;
    let mut fc_idx = 0usize;

    for layer in arch {
        match layer {
            ArchLayer::Reshape { shape } => {
                let shape: Vec<i64> = shape.clone();
                seq = seq.add_fn(move |x| x.reshape(shape.as_slice()));
            }
            ArchLayer::Conv2d {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                weights,
                biases,
            } => {
                let in_ch = *in_channels as i64;
                let out_ch = *out_channels as i64;
                let k = *kernel_size as i64;
                let s = *stride as i64;
                let path = root.sub(&format!("conv_{conv_idx}"));
                let mut conv = nn::conv2d(
                    &path,
                    in_ch,
                    out_ch,
                    k,
                    nn::ConvConfig {
                        stride: s,
                        padding: 0,
                        ..Default::default()
                    },
                );
                // Load weights (OIHW: same layout as Burn and ONNX, no transpose needed).
                let w_t = Tensor::from_slice(weights).reshape([out_ch, in_ch, k, k]);
                let b_t = Tensor::from_slice(biases.as_slice());
                tch::no_grad(|| {
                    conv.ws.copy_(&w_t);
                    if let Some(ref mut bs) = conv.bs {
                        bs.copy_(&b_t);
                    }
                });
                seq = seq.add(conv);
                conv_idx += 1;
            }
            ArchLayer::Elu => {
                seq = seq.add_fn(|x| x.elu());
            }
            ArchLayer::Flatten => {
                seq = seq.add_fn(|x| x.flatten(1, -1));
            }
            ArchLayer::Linear {
                in_dim,
                out_dim,
                weights,
                biases,
            } => {
                let in_d = *in_dim as i64;
                let out_d = *out_dim as i64;
                let path = root.sub(&format!("fc_{fc_idx}"));
                let mut linear = nn::linear(
                    &path,
                    in_d,
                    out_d,
                    nn::LinearConfig {
                        bias: true,
                        ..Default::default()
                    },
                );
                // Burn stores [in, out]; PyTorch/tch expects [out, in] for nn.Linear.
                let w_t = Tensor::from_slice(weights.as_slice())
                    .reshape([in_d, out_d])
                    .transpose(0, 1);
                let b_t = Tensor::from_slice(biases.as_slice());
                tch::no_grad(|| {
                    linear.ws.copy_(&w_t);
                    if let Some(ref mut bs) = linear.bs {
                        bs.copy_(&b_t);
                    }
                });
                seq = seq.add(linear);
                fc_idx += 1;
            }
        }
    }

    let temp_file = tempfile::Builder::new()
        .prefix("relayrl_pt_conv_")
        .suffix(".pt")
        .tempfile()
        .map_err(|e| format!("Failed to create temp file: {e}"))?;
    let temp_path = temp_file.path().to_path_buf();

    vs.freeze();
    let example_input = Tensor::zeros([1, obs_dim as i64], (Kind::Float, Device::Cpu));
    let mut trace_fn = |inputs: &[Tensor]| -> Vec<Tensor> { vec![seq.forward(&inputs[0])] };
    let module = tch::no_grad(|| {
        tch::CModule::create_by_tracing("convnet", "forward", &[example_input], &mut trace_fn)
    })
    .map_err(|e| format!("Failed to create traced module: {e}"))?;

    module
        .save(&temp_path)
        .map_err(|e| format!("Failed to save model: {e}"))?;

    let mut bytes = Vec::new();
    std::fs::File::open(&temp_path)
        .map_err(|e| format!("Failed to open saved model: {e}"))?
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read model bytes: {e}"))?;

    std::mem::forget(temp_file);
    Ok((bytes, temp_path))
}

/// Stub that errors because convolutional TorchScript export requires the `tch-model` feature.
#[cfg(not(feature = "tch-model"))]
pub fn build_pt_conv_temp(
    _arch: &[crate::algorithms::ArchLayer],
    _obs_dim: usize,
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
