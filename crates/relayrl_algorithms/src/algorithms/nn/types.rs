use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

#[derive(Clone, Debug)]
pub enum ActivationKind<B: Backend + BackendMatcher<Backend = B>> {
    ReLU(burn_nn::activation::Relu),
    LeakyReLU(burn_nn::activation::LeakyRelu),
    Tanh(burn_nn::activation::Tanh),
    Sigmoid(burn_nn::activation::Sigmoid),
    HardSigmoid(burn_nn::activation::HardSigmoid),
    HardSwish(burn_nn::activation::HardSwish),
    PReLU(burn_nn::activation::PRelu<B>),
    Gelu(burn_nn::activation::Gelu),
    SoftPlus(burn_nn::activation::Softplus),
    None,
}

pub type Dim0 = usize;
pub type Dim1 = usize;
pub type Weights = Vec<f32>;
pub type Biases = Vec<f32>;
pub type LayerSpecs = Vec<(Dim0, Dim1, Weights, Biases)>;

/// Describes a single operation in an architecture-aware forward pass.
///
/// Weights follow the same conventions as the underlying burn layers:
/// - `Conv2d`: OIHW flat `[out_ch, in_ch, kH, kW]`
/// - `Linear`: row-major `[in, out]` (Burn convention, no transposition needed
///   for the ONNX Gemm builder with `transB=0`)
#[derive(Clone)]
pub enum ArchLayer {
    /// 2-D convolution with square kernel and equal H/W stride (no padding).
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        /// Square kernel side length.
        kernel_size: usize,
        /// Equal H/W stride.
        stride: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },
    /// Fully-connected layer.
    Linear {
        in_dim: usize,
        out_dim: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },
    /// ELU activation (alpha=1.0), matching Sample Factory's default nonlinearity.
    Elu,
    /// Flatten all dimensions from axis 1 onward.
    Flatten,
    /// Reshape the tensor; `-1` in `shape` is the dynamic batch dimension.
    Reshape { shape: Vec<i64> },
}
