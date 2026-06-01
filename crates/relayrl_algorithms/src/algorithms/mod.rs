use burn_nn::{Linear, LinearConfig};
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Tensor, TensorKind};
use relayrl_types::data::tensor::DType;
use relayrl_types::data::tensor::NdArrayDType;
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, SupportedTensorBackend};

use half;

#[allow(non_snake_case)]
pub mod PPO;

pub mod onnx_builder;
#[cfg(feature = "tch-model")]
pub mod torch_builder;

#[derive(thiserror::Error, Debug, Clone)]
pub enum NeuralNetworkError {
    #[error("Unsupported device: {0}")]
    UnsupportedDevice(String),
    #[error("Unsupported DType: {0}")]
    UnsupportedDType(String),
    #[error("Unsupported output params: {0}")]
    UnsupportedOutputParams(String, String),
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("Input dimension mismatch: {0} != {1}")]
    InputDimMismatch(usize, usize),
    #[error("Invalid distribution")]
    InvalidDistribution,
}

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

pub type Dim0 = usize;
pub type Dim1 = usize;
pub type Weights = Vec<f32>;
pub type Biases = Vec<f32>;
pub type LayerSpecs = Vec<(Dim0, Dim1, Weights, Biases)>;

/// Trait for extracting per-layer weight specs from a network.
pub trait WeightProvider {
    fn get_layer_specs(&self) -> LayerSpecs;
}

// ---- generic MLP for easy usage ----
// implements NeuralNetworkSpec and NeuralNetworkForward, is compatible with all algorithms

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

// ---- value function ----
// wraps GenericMlp and ensures output is Float
// implements NeuralNetworkSpec and NeuralNetworkForward, is compatible with all algorithms

#[derive(Clone, Debug)]
pub struct ValueFunction<
    B: Backend + BackendMatcher<Backend = B>,
    KindIn: TensorKind<B> + BasicOps<B>,
>(GenericMlp<B, KindIn, Float>);

impl<B: Backend + BackendMatcher<Backend = B>, KindIn: TensorKind<B> + BasicOps<B>>
    ValueFunction<B, KindIn>
{
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

#[inline(always)]
pub fn dtype_to_byte_count(dtype: DType) -> usize {
    match dtype {
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => 2usize,
            NdArrayDType::F32 => 4usize,
            NdArrayDType::F64 => 8usize,
            NdArrayDType::I8 => 1usize,
            NdArrayDType::I16 => 2usize,
            NdArrayDType::I32 => 4usize,
            NdArrayDType::I64 => 8usize,
            NdArrayDType::Bool => 1usize,
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => 2usize,
            TchDType::Bf16 => 2usize,
            TchDType::F32 => 4usize,
            TchDType::F64 => 8usize,
            TchDType::I8 => 1usize,
            TchDType::I16 => 2usize,
            TchDType::I32 => 4usize,
            TchDType::I64 => 8usize,
            TchDType::U8 => 1usize,
            TchDType::Bool => 1usize,
        },
    }
}

#[inline(always)]
pub fn convert_byte_dtype_to_f32(
    bytes: Vec<u8>,
    byte_dtype: DType,
) -> Result<Vec<f32>, NeuralNetworkError> {
    Ok(match byte_dtype {
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => bytemuck::cast_slice::<u8, half::f16>(&bytes)
                .iter()
                .map(|&x| f32::from(x))
                .collect::<Vec<f32>>(),
            NdArrayDType::F32 => bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
            NdArrayDType::F64 => bytemuck::cast_slice::<u8, f64>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            NdArrayDType::I8 => bytemuck::cast_slice::<u8, i8>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            NdArrayDType::I16 => bytemuck::cast_slice::<u8, i16>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            NdArrayDType::I32 => bytemuck::cast_slice::<u8, i32>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            NdArrayDType::I64 => bytemuck::cast_slice::<u8, i64>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            NdArrayDType::Bool => bytes
                .iter()
                .map(|&x| if x != 0 { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<f32>>(),
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => bytemuck::cast_slice::<u8, half::f16>(&bytes)
                .iter()
                .map(|&x| f32::from(x))
                .collect::<Vec<f32>>(),
            TchDType::Bf16 => bytemuck::cast_slice::<u8, half::bf16>(&bytes)
                .iter()
                .map(|&x| f32::from(x))
                .collect::<Vec<f32>>(),
            TchDType::F32 => bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
            TchDType::F64 => bytemuck::cast_slice::<u8, f64>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            TchDType::I8 => bytemuck::cast_slice::<u8, i8>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            TchDType::I16 => bytemuck::cast_slice::<u8, i16>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            TchDType::I32 => bytemuck::cast_slice::<u8, i32>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            TchDType::I64 => bytemuck::cast_slice::<u8, i64>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            TchDType::U8 => bytemuck::cast_slice::<u8, u8>(&bytes)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>(),
            TchDType::Bool => bytes
                .iter()
                .map(|&x| if x != 0 { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<f32>>(),
        },
    })
}

#[inline(always)]
pub fn convert_byte_dtype_to_i64(
    bytes: &[u8],
    byte_dtype: &DType,
) -> Result<Vec<i64>, NeuralNetworkError> {
    Ok(match byte_dtype {
        DType::NdArray(nd) => match nd {
            NdArrayDType::F16 => bytemuck::cast_slice::<u8, half::f16>(bytes)
                .iter()
                .map(|&x| f32::from(x) as i64)
                .collect::<Vec<i64>>(),
            NdArrayDType::F32 => bytemuck::cast_slice::<u8, f32>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            NdArrayDType::F64 => bytemuck::cast_slice::<u8, f64>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            NdArrayDType::I8 => bytemuck::cast_slice::<u8, i8>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            NdArrayDType::I16 => bytemuck::cast_slice::<u8, i16>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            NdArrayDType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            NdArrayDType::I64 => bytemuck::cast_slice::<u8, i64>(bytes).to_vec(),
            NdArrayDType::Bool => bytes
                .iter()
                .map(|&x| if x != 0 { 1i64 } else { 0i64 })
                .collect::<Vec<i64>>(),
        },
        #[cfg(feature = "tch-backend")]
        DType::Tch(tch) => match tch {
            TchDType::F16 => bytemuck::cast_slice::<u8, half::f16>(bytes)
                .iter()
                .map(|&x| f32::from(x) as i64)
                .collect::<Vec<i64>>(),
            TchDType::Bf16 => bytemuck::cast_slice::<u8, half::bf16>(bytes)
                .iter()
                .map(|&x| f32::from(x) as i64)
                .collect::<Vec<i64>>(),
            TchDType::F32 => bytemuck::cast_slice::<u8, f32>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            TchDType::F64 => bytemuck::cast_slice::<u8, f64>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            TchDType::I8 => bytemuck::cast_slice::<u8, i8>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            TchDType::I16 => bytemuck::cast_slice::<u8, i16>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            TchDType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            TchDType::I64 => bytemuck::cast_slice::<u8, i64>(bytes).to_vec(),
            TchDType::U8 => bytemuck::cast_slice::<u8, u8>(bytes)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<i64>>(),
            TchDType::Bool => bytes
                .iter()
                .map(|&x| if x != 0 { 1i64 } else { 0i64 })
                .collect::<Vec<i64>>(),
        },
    })
}
