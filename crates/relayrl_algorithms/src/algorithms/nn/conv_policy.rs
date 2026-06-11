//! Sample Factory's `convnet_simple` ConvNet policy for ViZDoom.
//!
//! Architecture (matches SF `doom_benchmark` `encoder_conv_architecture="convnet_simple"`):
//! ```text
//! Input: flat [N, 27648]  (OBS_DIM = 3×72×128)
//! Reshape →  [N, 3, 72, 128]  (NCHW)
//! Conv2d(3→32,  k=8, stride=4, no pad) + ELU  → [N, 32, 17, 31]
//! Conv2d(32→64, k=4, stride=2, no pad) + ELU  → [N, 64,  7, 14]
//! Conv2d(64→128,k=3, stride=2, no pad) + ELU  → [N,128,  3,  6]
//! Flatten                                      → [N, 2304]
//! Linear(2304→512) + ELU
//! Linear(512→act_dim)                          → [N, act_dim]
//! ```
//!
//! `WeightProvider::get_layer_specs` returns an empty `LayerSpecs` (this policy
//! is exported via `get_arch_spec` → `acquire_conv_model_module`).  The MLP
//! trainer in `PPOActorCriticTrainer` derives hidden sizes from `get_layer_specs`
//! and therefore creates a single-layer linear model — acceptable for a
//! throughput benchmark where the Conv Pi is used for inference.

use std::marker::PhantomData;

use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{Linear, LinearConfig};
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Tensor, TensorKind};
use relayrl_types::data::tensor::DType;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use super::traits::{NeuralNetwork, NeuralNetworkForward, NeuralNetworkSpec, WeightProvider};
use super::types::{ArchLayer, LayerSpecs};

// ── Architecture constants (matching SF convnet_simple for 72×128 input) ────

/// Input channels (RGB).
pub const CONV_CHANNELS: usize = 3;
/// Input height after SF resize.
pub const CONV_H: usize = 72;
/// Input width after SF resize.
pub const CONV_W: usize = 128;
/// Flattened conv-head output: 128 × 3 × 6.
pub const CONV_HEAD_OUT: usize = 2304;
/// Dense hidden size between conv head and policy head.
pub const CONV_FC_HIDDEN: usize = 512;

// ── ELU helper (alpha = 1.0) ─────────────────────────────────────────────────

/// Element-wise ELU with alpha=1.0: `x if x≥0 else exp(x)−1`.
///
/// Implemented as `relu(x) + (exp(x − relu(x)) − 1)` to avoid
/// mask/select ops that may not be available on all backends:
/// - `x − relu(x) = min(0, x)` for any x
/// - For x≥0: relu(x)=x, exp(0)−1=0 → x
/// - For x<0: relu(x)=0, exp(x)−1 → exp(x)−1
fn elu<B: Backend, const D: usize>(x: Tensor<B, D, Float>) -> Tensor<B, D, Float> {
    let pos = burn_nn::activation::Relu::new().forward(x.clone()); // max(0, x)
    let neg_exp = (x - pos.clone()).exp(); // exp(min(0, x))
    pos + neg_exp - 1.0_f32
}

// ── ConvNetPolicy ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ConvNetPolicy<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> {
    input_dim: usize,
    input_dtype: DType,
    output_dim: usize,
    output_dtype: DType,
    /// Conv2d(3→32, k=8, stride=4)
    conv1: Conv2d<B>,
    /// Conv2d(32→64, k=4, stride=2)
    conv2: Conv2d<B>,
    /// Conv2d(64→128, k=3, stride=2)
    conv3: Conv2d<B>,
    /// Linear(CONV_HEAD_OUT → CONV_FC_HIDDEN)
    fc1: Linear<B>,
    /// Linear(CONV_FC_HIDDEN → output_dim)
    head: Linear<B>,
    _in_k: PhantomData<KindIn>,
    _out_k: PhantomData<KindOut>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> ConvNetPolicy<B, KindIn, KindOut>
{
    pub fn new(
        input_dim: usize,
        input_dtype: DType,
        output_dim: usize,
        output_dtype: DType,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv2dConfig::new([CONV_CHANNELS, 32], [8, 8])
            .with_stride([4, 4])
            .init(device);
        let conv2 = Conv2dConfig::new([32, 64], [4, 4])
            .with_stride([2, 2])
            .init(device);
        let conv3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2])
            .init(device);
        let fc1 = LinearConfig::new(CONV_HEAD_OUT, CONV_FC_HIDDEN).init(device);
        let head = LinearConfig::new(CONV_FC_HIDDEN, output_dim).init(device);
        Self {
            input_dim,
            input_dtype,
            output_dim,
            output_dtype,
            conv1,
            conv2,
            conv3,
            fc1,
            head,
            _in_k: PhantomData,
            _out_k: PhantomData,
        }
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> NeuralNetwork<B, KindIn, KindOut> for ConvNetPolicy<B, KindIn, KindOut>
{
    fn default(
        input_dim: usize,
        input_dtype: DType,
        output_dim: usize,
        output_dtype: DType,
        device: &B::Device,
    ) -> Self {
        Self::new(input_dim, input_dtype, output_dim, output_dtype, device)
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> NeuralNetworkSpec<B, KindIn, KindOut> for ConvNetPolicy<B, KindIn, KindOut>
{
    fn input_dim(&self) -> &usize {
        &self.input_dim
    }
    fn input_dtype(&self) -> &DType {
        &self.input_dtype
    }
    fn output_dim(&self) -> &usize {
        &self.output_dim
    }
    fn output_dtype(&self) -> &DType {
        &self.output_dtype
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> NeuralNetworkForward<B, KindIn, KindOut> for ConvNetPolicy<B, KindIn, KindOut>
{
    fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        input: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut> {
        let device = input.device();
        let n = input.shape().dims[0];

        // Cast to f32, reshape flat [N, OBS_DIM] → [N, C, H, W].
        let x: Tensor<B, 2, Float> = Tensor::from_data(
            input.into_data().convert::<f32>(),
            &device,
        );
        let x = x.reshape([n, CONV_CHANNELS, CONV_H, CONV_W]);

        // Conv stack + ELU.
        let x = elu(self.conv1.forward(x));
        let x = elu(self.conv2.forward(x));
        let x = elu(self.conv3.forward(x));

        // Flatten [N, 128, 3, 6] → [N, 2304].
        let x = x.flatten(1, 3);

        // Dense head.
        let x = elu(self.fc1.forward(x));
        let x: Tensor<B, 2, Float> = self.head.forward(x);

        Tensor::<B, OUT_D, KindOut>::from_data(
            x.into_data().convert::<KindOut::Elem>(),
            &device,
        )
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> WeightProvider for ConvNetPolicy<B, KindIn, KindOut>
{
    /// Returns empty `LayerSpecs`; the full architecture is exposed via
    /// `get_arch_spec`.  The PPO trainer derives hidden sizes from this and
    /// therefore builds a simple linear training surrogate — acceptable for a
    /// throughput benchmark.
    fn get_layer_specs(&self) -> LayerSpecs {
        Vec::new()
    }

    fn get_arch_spec(&self) -> Option<Vec<ArchLayer>> {
        let n_ch = CONV_CHANNELS as i64;
        let h = CONV_H as i64;
        let w = CONV_W as i64;

        let conv_w = |layer: &Conv2d<B>| -> Vec<f32> {
            layer.weight.val().into_data().to_vec::<f32>().unwrap_or_default()
        };
        let conv_b = |layer: &Conv2d<B>| -> Vec<f32> {
            layer
                .bias
                .as_ref()
                .map(|b| b.val().into_data().to_vec::<f32>().unwrap_or_default())
                .unwrap_or_default()
        };
        let lin_w = |layer: &Linear<B>| -> Vec<f32> {
            layer.weight.val().into_data().to_vec::<f32>().unwrap_or_default()
        };
        let lin_b = |layer: &Linear<B>, out: usize| -> Vec<f32> {
            layer
                .bias
                .as_ref()
                .map(|b| b.val().into_data().to_vec::<f32>().unwrap_or_default())
                .unwrap_or_else(|| vec![0.0; out])
        };

        Some(vec![
            ArchLayer::Reshape { shape: vec![-1, n_ch, h, w] },
            ArchLayer::Conv2d {
                in_channels: CONV_CHANNELS,
                out_channels: 32,
                kernel_size: 8,
                stride: 4,
                weights: conv_w(&self.conv1),
                biases: conv_b(&self.conv1),
            },
            ArchLayer::Elu,
            ArchLayer::Conv2d {
                in_channels: 32,
                out_channels: 64,
                kernel_size: 4,
                stride: 2,
                weights: conv_w(&self.conv2),
                biases: conv_b(&self.conv2),
            },
            ArchLayer::Elu,
            ArchLayer::Conv2d {
                in_channels: 64,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                weights: conv_w(&self.conv3),
                biases: conv_b(&self.conv3),
            },
            ArchLayer::Elu,
            ArchLayer::Flatten,
            ArchLayer::Linear {
                in_dim: CONV_HEAD_OUT,
                out_dim: CONV_FC_HIDDEN,
                weights: lin_w(&self.fc1),
                biases: lin_b(&self.fc1, CONV_FC_HIDDEN),
            },
            ArchLayer::Elu,
            ArchLayer::Linear {
                in_dim: CONV_FC_HIDDEN,
                out_dim: self.output_dim,
                weights: lin_w(&self.head),
                biases: lin_b(&self.head, self.output_dim),
            },
        ])
    }
}
