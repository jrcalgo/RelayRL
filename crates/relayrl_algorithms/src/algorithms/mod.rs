#[allow(non_snake_case)]
pub mod PPO;

pub mod nn;
pub mod onnx_builder;
#[cfg(feature = "tch-model")]
pub mod torch_builder;

pub use nn::{
    acquire_conv_model_module, acquire_model_module, convert_byte_dtype_to_f32,
    convert_byte_dtype_to_i64, dtype_to_byte_count, ActivationKind, ArchLayer, GenericMlp,
    LayerSpecs, NeuralNetwork, NeuralNetworkError, NeuralNetworkForward, NeuralNetworkSpec,
    ValueFunction, WeightProvider,
};
pub use nn::conv_policy;

#[inline(always)]
pub(crate) fn discounted_cumsum(x: &[f32], discount: f32) -> Vec<f32> {
    let n = x.len();
    let mut result = vec![0.0f32; n];
    let mut running = 0.0f32;
    for i in (0..n).rev() {
        running = x[i] + discount * running;
        result[i] = running;
    }
    result
}

#[inline(always)]
pub(crate) fn scalar_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, variance.sqrt())
}

#[inline(always)]
pub(crate) fn compute_normed_advantages(advantages: &[f32], mean: f32, std: f32) -> Vec<f32> {
    advantages.iter().map(|a| (a - mean) / std).collect()
}
