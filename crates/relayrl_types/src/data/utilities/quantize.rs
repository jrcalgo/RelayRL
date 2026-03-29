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
            QuantizationScheme::None => bytemuck::cast_slice(&self.data).to_vec(),
            QuantizationScheme::Int8Symmetric => {
                let quantized: &[i8] = bytemuck::cast_slice(&self.data);
                quantized.iter().map(|&x| x as f32 * self.scale).collect()
            }
            QuantizationScheme::Int8Asymmetric => self
                .data
                .iter()
                .map(|&x| (x as i32 - self.zero_point) as f32 * self.scale)
                .collect(),
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

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!((lhs - rhs).abs() <= tolerance, "{lhs} != {rhs}");
        }
    }

    #[test]
    fn none_scheme_dequantizes_raw_f32_bytes() {
        let values = [1.0f32, -2.0, 3.5];
        let data = QuantizedData {
            data: values
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            scheme: QuantizationScheme::None,
            scale: 1.0,
            zero_point: 0,
            shape: vec![3],
        };

        assert_eq!(data.dequantize(), values.to_vec());
    }

    #[test]
    fn int8_symmetric_quantization_round_trips_approximately() {
        let values = [-3.5f32, -1.0, 0.5, 2.25];
        let quantized = QuantizedData::quantize_int8_symmetric(&values);

        assert_eq!(quantized.scheme, QuantizationScheme::Int8Symmetric);
        assert_eq!(quantized.shape, vec![4]);
        assert_eq!(quantized.size_reduction_ratio(), 4.0);
        assert_close(&quantized.dequantize(), &values, 0.05);
    }

    #[test]
    fn int8_asymmetric_quantization_round_trips_approximately() {
        let values = [0.25f32, 1.5, 2.75, 4.0];
        let quantized = QuantizedData::quantize_int8_asymmetric(&values);

        assert_eq!(quantized.scheme, QuantizationScheme::Int8Asymmetric);
        assert_eq!(quantized.data.len(), values.len());
        assert_close(&quantized.dequantize(), &values, 0.1);
    }

    #[test]
    #[cfg(feature = "quantization")]
    fn float16_quantization_round_trips_approximately() {
        let values = [1.25f32, -2.5, 4.75];
        let quantized = QuantizedData::quantize_f16(&values);

        assert_eq!(quantized.scheme, QuantizationScheme::Float16);
        assert_close(&quantized.dequantize(), &values, 0.01);
    }

    #[test]
    #[cfg(feature = "quantization")]
    fn bfloat16_quantization_round_trips_approximately() {
        let values = [1.25f32, -2.5, 4.75];
        let quantized = QuantizedData::quantize_bf16(&values);

        assert_eq!(quantized.scheme, QuantizationScheme::BFloat16);
        assert_close(&quantized.dequantize(), &values, 0.05);
    }
}
