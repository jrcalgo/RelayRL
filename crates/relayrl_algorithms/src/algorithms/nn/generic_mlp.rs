use burn_nn::{Linear, LinearConfig};
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Tensor, TensorKind};
use relayrl_types::data::tensor::DType;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use super::traits::{NeuralNetwork, NeuralNetworkForward, NeuralNetworkSpec, WeightProvider};
use super::types::{ActivationKind, LayerSpecs};

#[derive(Clone, Debug)]
pub struct GenericMlp<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> {
    input_dim: usize,
    input_dtype: DType,
    output_dim: usize,
    output_dtype: DType,
    layers: Vec<Linear<B>>,
    activation: ActivationKind<B>,
    _in_k: std::marker::PhantomData<KindIn>,
    _out_k: std::marker::PhantomData<KindOut>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> GenericMlp<B, KindIn, KindOut>
{
    pub fn new(
        input_dim: usize,
        input_dtype: DType,
        hidden_sizes: &[usize],
        output_dim: usize,
        output_dtype: DType,
        activation: ActivationKind<B>,
        device: &B::Device,
    ) -> Self {
        let mut dims = Vec::with_capacity(hidden_sizes.len() + 2);
        dims.push(input_dim);
        dims.extend_from_slice(hidden_sizes);
        dims.push(output_dim);

        let layers = dims
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();

        Self {
            input_dim,
            input_dtype,
            output_dim,
            output_dtype,
            layers,
            activation,
            _in_k: std::marker::PhantomData,
            _out_k: std::marker::PhantomData,
        }
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> NeuralNetwork<B, KindIn, KindOut> for GenericMlp<B, KindIn, KindOut>
{
    fn default(
        input_dim: usize,
        input_dtype: DType,
        output_dim: usize,
        output_dtype: DType,
        device: &B::Device,
    ) -> Self {
        Self::new(
            input_dim,
            input_dtype,
            &[512, 512],
            output_dim,
            output_dtype,
            ActivationKind::ReLU(burn_nn::activation::Relu::new()),
            device,
        )
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
    KindOut: TensorKind<B> + BasicOps<B>,
> NeuralNetworkSpec<B, KindIn, KindOut> for GenericMlp<B, KindIn, KindOut>
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

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    NeuralNetworkForward<B, KindIn, KindOut> for GenericMlp<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    fn forward<const IN_D: usize, const OUT_D: usize>(
        &self,
        input: Tensor<B, IN_D, KindIn>,
    ) -> Tensor<B, OUT_D, KindOut>
    where
        KindIn: BasicOps<B>,
        KindOut: BasicOps<B>,
    {
        let device = input.device();
        let mut x_float: Tensor<B, IN_D, Float> =
            Tensor::from_data(input.into_data().convert::<f32>(), &device);
        for (i, layer) in self.layers.iter().enumerate() {
            x_float = layer.forward(x_float);
            if i < self.layers.len() - 1 {
                x_float = match &self.activation {
                    ActivationKind::ReLU(relu) => relu.forward(x_float),
                    ActivationKind::LeakyReLU(leaky_relu) => leaky_relu.forward(x_float),
                    ActivationKind::Tanh(tanh) => tanh.forward(x_float),
                    ActivationKind::Sigmoid(sigmoid) => sigmoid.forward(x_float),
                    ActivationKind::HardSigmoid(hard_sigmoid) => hard_sigmoid.forward(x_float),
                    ActivationKind::HardSwish(hard_swish) => hard_swish.forward(x_float),
                    ActivationKind::PReLU(prelu) => prelu.forward(x_float),
                    ActivationKind::Gelu(gelu) => gelu.forward(x_float),
                    ActivationKind::SoftPlus(softplus) => softplus.forward(x_float),
                    ActivationKind::None => x_float,
                }
            }
        }

        Tensor::<B, OUT_D, KindOut>::from_data(
            x_float.into_data().convert::<KindOut::Elem>(),
            &device,
        )
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B>, KindOut: TensorKind<B>>
    WeightProvider for GenericMlp<B, KindIn, KindOut>
where
    KindIn: BasicOps<B>,
    KindOut: BasicOps<B>,
{
    fn get_layer_specs(&self) -> LayerSpecs {
        self.layers
            .iter()
            .map(|layer| -> (usize, usize, Vec<f32>, Vec<f32>) {
                let w = layer.weight.val();
                let dims = w.dims();
                let weights: Vec<f32> = w.into_data().to_vec::<f32>().unwrap_or_default();
                let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                    bias_param
                        .val()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                } else {
                    vec![0.0; dims[1]]
                };
                (dims[0], dims[1], weights, biases)
            })
            .collect()
    }
}
