use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Tensor, TensorKind};
use relayrl_types::data::tensor::{DType, NdArrayDType};
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, SupportedTensorBackend};

use super::error::NeuralNetworkError;
use super::generic_mlp::GenericMlp;
use super::traits::{NeuralNetwork, NeuralNetworkForward, NeuralNetworkSpec, WeightProvider};
use super::types::{ActivationKind, LayerSpecs};

/// A critic network wrapping a `GenericMlp` and enforcing a single f32 output — the value estimate.
#[derive(Clone, Debug)]
pub struct ValueFunction<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
>(GenericMlp<B, KindIn, Float>);

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B> + BasicOps<B>>
    ValueFunction<B, KindIn>
{
    /// Wraps a `GenericMlp` as a value function, erroring unless its output is a single f32.
    pub fn new(vf_mlp: GenericMlp<B, KindIn, Float>) -> Result<Self, NeuralNetworkError> {
        match (vf_mlp.output_dtype(), vf_mlp.output_dim()) {
            (DType::NdArray(NdArrayDType::F32), 1) => Ok(Self(vf_mlp)),
            #[cfg(feature = "tch-backend")]
            (DType::Tch(TchDType::F32), 1) => Ok(Self(vf_mlp)),
            _ => Err(NeuralNetworkError::UnsupportedOutputParams(
                vf_mlp.output_dtype().to_string(),
                vf_mlp.output_dim().to_string(),
            )),
        }
    }

    /// Builds a value function from a fresh MLP with the given hidden sizes and activation.
    pub fn new_generic_mlp(
        input_dim: usize,
        input_dtype: DType,
        hidden_sizes: &[usize],
        activation: ActivationKind<B>,
        device: &B::Device,
    ) -> Result<Self, NeuralNetworkError> {
        let output_dype: DType = match B::get_supported_backend() {
            SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
            #[cfg(feature = "tch-backend")]
            SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
            _ => {
                return Err(NeuralNetworkError::BackendUnavailable(
                    match B::get_supported_backend() {
                        SupportedTensorBackend::NdArray => "NdArray",
                        #[cfg(feature = "tch-backend")]
                        SupportedTensorBackend::Tch => "Tch",
                        _ => "None",
                    }
                    .to_string(),
                ));
            }
        };
        Self::new(GenericMlp::new(
            input_dim,
            input_dtype,
            hidden_sizes,
            1,
            output_dype,
            activation,
            device,
        ))
    }

    /// Builds a value function from a default-sized MLP for the given input.
    pub fn new_default_mlp(
        input_dim: usize,
        input_dtype: DType,
        device: &B::Device,
    ) -> Result<Self, NeuralNetworkError> {
        Ok(Self(GenericMlp::default(
            input_dim,
            input_dtype,
            1,
            DType::NdArray(NdArrayDType::F32),
            device,
        )))
    }

    /// Returns the value network's per-layer weight specs for model export.
    pub fn get_vf_layer_specs(&self) -> LayerSpecs {
        self.0.get_layer_specs()
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> NeuralNetworkSpec<B, KindIn, KindOut> for ValueFunction<B, KindIn>
{
    fn input_dim(&self) -> &usize {
        self.0.input_dim()
    }

    fn input_dtype(&self) -> &DType {
        self.0.input_dtype()
    }

    fn output_dim(&self) -> &usize {
        self.0.output_dim()
    }

    fn output_dtype(&self) -> &DType {
        self.0.output_dtype()
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B> + BasicOps<B>>
    NeuralNetworkForward<B, KindIn, Float> for ValueFunction<B, KindIn>
where
    KindIn: BasicOps<B>,
{
    fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        input: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, Float>
    where
        KindIn: BasicOps<B>,
    {
        self.0.forward(input)
    }
}
