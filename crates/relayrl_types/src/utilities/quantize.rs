//! Quantization utilities for reducing tensor size with minimal accuracy loss

use serde::{Deserialize, Serialize};

#[cfg(feature = "quantization")]
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationScheme {
    None,
    Float16,
    BFloat16,
    Int8Symmetric,
    Int8Asymmetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedData {
    pub data: Vec<u8>,
    pub scheme: QuantizationScheme,
    pub scale: f32,
    pub zero_point: i32,
    pub shape: Vec<usize>,
}

impl QuantizedData {
    pub fn quantize_int8_symmetric(data: &[f32]) -> Self {
        let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = max_abs / 127.0;
        let quantized: Vec<i8> = data
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Self {
            data: bytemuck::cast_slice(&quantized).to_vec(),
            scheme: QuantizationScheme::Int8Symmetric,
            scale,
            zero_point: 0,
            shape: vec![data.len()],
        }
    }

    pub fn quantize_int8_asymmetric(data: &[f32]) -> Self {
        let min_val = data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i32;
        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| ((x / scale).round() as i32 + zero_point).clamp(0, 255) as u8)
            .collect();
        Self {
            data: quantized,
            scheme: QuantizationScheme::Int8Asymmetric,
            scale,
            zero_point,
            shape: vec![data.len()],
        }
    }

    #[cfg(feature = "quantization")]
    pub fn quantize_f16(data: &[f32]) -> Self {
        let quantized: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        Self {
            data: bytemuck::cast_slice(&quantized).to_vec(),
            scheme: QuantizationScheme::Float16,
            scale: 1.0,
            zero_point: 0,
            shape: vec![data.len()],
        }
    }

    #[cfg(feature = "quantization")]
    pub fn quantize_bf16(data: &[f32]) -> Self {
        let quantized: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
        Self {
            data: bytemuck::cast_slice(&quantized).to_vec(),
            scheme: QuantizationScheme::BFloat16,
            scale: 1.0,
            zero_point: 0,
            shape: vec![data.len()],
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        match self.scheme {
            QuantizationScheme::None => {
                bytemuck::cast_slice(&self.data).to_vec()
            }
            QuantizationScheme::Int8Symmetric => {
                let quantized: &[i8] = bytemuck::cast_slice(&self.data);
                quantized.iter().map(|&x| x as f32 * self.scale).collect()
            }
            QuantizationScheme::Int8Asymmetric => {
                self.data
                    .iter()
                    .map(|&x| (x as i32 - self.zero_point) as f32 * self.scale)
                    .collect()
            }
            #[cfg(feature = "quantization")]
            QuantizationScheme::Float16 => {
                let quantized: &[f16] = bytemuck::cast_slice(&self.data);
                quantized.iter().map(|&x| x.to_f32()).collect()
            }
            #[cfg(feature = "quantization")]
            QuantizationScheme::BFloat16 => {
                let quantized: &[bf16] = bytemuck::cast_slice(&self.data);
                quantized.iter().map(|&x| x.to_f32()).collect()
            }
            #[cfg(not(feature = "quantization"))]
            _ => Vec::new(),
        }
    }

    pub fn size_reduction_ratio(&self) -> f32 {
        let original_size = self.shape.iter().product::<usize>() * 4;
        original_size as f32 / self.data.len() as f32
    }
}
