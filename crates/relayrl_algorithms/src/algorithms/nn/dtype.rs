use relayrl_types::data::tensor::{DType, NdArrayDType};
#[cfg(feature = "tch-backend")]
use relayrl_types::data::tensor::TchDType;

use half;

use super::error::NeuralNetworkError;

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
