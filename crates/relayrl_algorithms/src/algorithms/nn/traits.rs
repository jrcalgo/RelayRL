use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Tensor, TensorKind};
use relayrl_types::data::tensor::DType;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use super::types::{ArchLayer, LayerSpecs};

pub trait NeuralNetwork<B, KindIn, KindOut>:
    NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut> + WeightProvider
where
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
{
    fn default(
        input_dim: usize,
        input_dtype: DType,
        output_dim: usize,
        output_dtype: DType,
        device: &B::Device,
    ) -> Self;
}

pub trait NeuralNetworkSpec<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
>
{
    fn input_dim(&self) -> &usize;
    fn input_dtype(&self) -> &DType;
    fn output_dim(&self) -> &usize;
    fn output_dtype(&self) -> &DType;
}

pub trait NeuralNetworkForward<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
>
{
    fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        input: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut>;
}

/// Trait for extracting per-layer weight specs from a network.
pub trait WeightProvider {
    fn get_layer_specs(&self) -> LayerSpecs;

    /// Return an architecture-aware layer sequence for conv / mixed networks.
    ///
    /// The default implementation returns `None`, preserving backward
    /// compatibility for all MLP implementations.  Override this for networks
    /// that cannot be represented as flat `LayerSpecs` (e.g. `ConvNetPolicy`).
    fn get_arch_spec(&self) -> Option<Vec<ArchLayer>> {
        None
    }
}
