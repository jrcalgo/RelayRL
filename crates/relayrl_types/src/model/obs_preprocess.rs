//! Frozen or mutable Welford observation normalization shared by training and export.
//!
//! The accumulator matches the PPO collector: `m2` starts at one, `count` counts observation
//! rows, and the standard deviation is floored at `std_floor` (1e-4).

use serde::{Deserialize, Serialize};

use super::ModelError;
#[cfg(feature = "tch-backend")]
use crate::data::tensor::TchDType;
use crate::data::tensor::{DType, NdArrayDType, TensorData};

/// Standard-deviation floor used by the live PPO observation normalizer.
pub const PPO_OBS_STD_FLOOR: f64 = 1e-4;

/// How a saved model expects its observation input to be prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPreprocessMode {
    /// The caller supplies observations in the distribution expected by the graph.
    ///
    /// Generic and unnormalized models use this mode with an identity state.
    CallerNormalized,
    /// The graph is unfolded; inference applies the frozen state once.
    ExternalFrozen,
    /// The frozen affine is already inside the first linear layer.
    FoldedIntoGraph,
}

/// Exact Welford state used while collecting and the snapshot sealed with a policy version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationNormalizerState {
    pub schema_version: u32,
    pub mean: Vec<f64>,
    pub m2: Vec<f64>,
    pub count: u64,
    pub std_floor: f64,
    pub policy_version: i64,
}

/// Metadata describing whether and how observation normalization belongs to a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationPreprocessMetadata {
    pub state: ObservationNormalizerState,
    pub mode: ObservationPreprocessMode,
}

impl ObservationNormalizerState {
    /// Starts at mean 0, M2 1, count 0, matching the collector's initial accumulator.
    pub fn identity(width: usize) -> Self {
        Self {
            schema_version: 1,
            mean: vec![0.0; width],
            m2: vec![1.0; width],
            count: 0,
            std_floor: PPO_OBS_STD_FLOOR,
            policy_version: -1,
        }
    }

    /// Validates dimensions and numeric values against the model input width.
    pub fn validate(&self, feature_width: usize) -> Result<(), ModelError> {
        if self.schema_version != 1 {
            return Err(ModelError::InvalidMetadata(format!(
                "unsupported observation normalizer schema {}",
                self.schema_version
            )));
        }
        if self.mean.len() != feature_width || self.m2.len() != feature_width {
            return Err(ModelError::InvalidMetadata(format!(
                "observation normalizer width {} does not match input width {feature_width}",
                self.mean.len()
            )));
        }
        if !self.std_floor.is_finite() || self.std_floor <= 0.0 {
            return Err(ModelError::InvalidMetadata(
                "observation normalizer std_floor must be finite and positive".to_string(),
            ));
        }
        if self
            .m2
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ModelError::InvalidMetadata(
                "observation normalizer M2 must be finite and nonnegative".to_string(),
            ));
        }
        if self.mean.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidMetadata(
                "observation normalizer mean must be finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Standard deviation for each feature, floored exactly like the collector.
    pub fn stds(&self) -> Vec<f64> {
        self.m2
            .iter()
            .map(|m2| (m2 / self.count.max(1) as f64).sqrt().max(self.std_floor))
            .collect()
    }

    /// Immutable snapshot. Later updates to this copy do not affect the source.
    pub fn seal(&self) -> Self {
        self.clone()
    }

    /// Records the policy generation published with this snapshot.
    pub fn bind_policy_version(&mut self, version: i64) {
        self.policy_version = version;
    }

    /// Continues the collector update for one observation row.
    pub fn update_row(&mut self, row: &[f64]) {
        self.count = self.count.saturating_add(1);
        let count = self.count as f64;
        for (index, value) in row.iter().copied().enumerate() {
            let delta = value - self.mean[index];
            self.mean[index] += delta / count;
            let delta2 = value - self.mean[index];
            self.m2[index] += delta * delta2;
        }
    }

    /// Normalizes one row without changing the accumulator.
    pub fn normalize_row(&self, row: &[f64]) -> Vec<f64> {
        let stds = self.stds();
        row.iter()
            .zip(self.mean.iter())
            .zip(stds.iter())
            .map(|((value, mean), std)| (value - mean) / std)
            .collect()
    }

    /// Normalizes a row-major f64 batch in place without changing the accumulator.
    pub fn normalize_flat(&self, values: &mut [f64], width: usize) {
        if width == 0 || values.len() % width != 0 {
            return;
        }
        let stds = self.stds();
        for row in values.chunks_mut(width) {
            for (value, (mean, std)) in row.iter_mut().zip(self.mean.iter().zip(stds.iter())) {
                *value = (*value - mean) / std;
            }
        }
    }
}

impl ObservationPreprocessMetadata {
    /// Explicit no-transform contract for generic, unnormalized, and caller-prepared models.
    pub fn caller_normalized(width: usize) -> Self {
        Self {
            state: ObservationNormalizerState::identity(width),
            mode: ObservationPreprocessMode::CallerNormalized,
        }
    }

    pub fn external_frozen(mut state: ObservationNormalizerState, policy_version: i64) -> Self {
        state.bind_policy_version(policy_version);
        Self {
            state,
            mode: ObservationPreprocessMode::ExternalFrozen,
        }
    }

    pub fn folded(mut state: ObservationNormalizerState, policy_version: i64) -> Self {
        state.bind_policy_version(policy_version);
        Self {
            state,
            mode: ObservationPreprocessMode::FoldedIntoGraph,
        }
    }

    pub fn validate(&self, feature_width: usize) -> Result<(), ModelError> {
        self.state.validate(feature_width)
    }
}

/// Folds `(x - mean) / std` into a fresh first linear layer stored `[input, output]`.
pub fn fold_observation_norm_into_first_layer(
    first: &mut (usize, usize, Vec<f32>, Vec<f32>),
    state: &ObservationNormalizerState,
) -> Result<(), ModelError> {
    let (in_dim, out_dim, weights, biases) = first;
    state.validate(*in_dim)?;
    if weights.len() != in_dim.saturating_mul(*out_dim) || biases.len() != *out_dim {
        return Err(ModelError::InvalidMetadata(
            "first-layer weight or bias length does not match its dimensions".to_string(),
        ));
    }
    let stds = state.stds();
    let mut folded_weights = Vec::with_capacity(weights.len());
    let mut folded_biases = vec![0.0f64; *out_dim];
    for (input, (mean, std)) in state.mean.iter().zip(stds.iter()).enumerate() {
        for output in 0..*out_dim {
            let original = weights[input * *out_dim + output] as f64;
            let folded = original / std;
            if !folded.is_finite() {
                return Err(ModelError::InvalidMetadata(
                    "folded observation weight is non-finite".to_string(),
                ));
            }
            folded_weights.push(folded as f32);
            folded_biases[output] -= mean * folded;
        }
    }
    for (output, bias) in biases.iter().enumerate() {
        let folded = *bias as f64 + folded_biases[output];
        if !folded.is_finite() {
            return Err(ModelError::InvalidMetadata(
                "folded observation bias is non-finite".to_string(),
            ));
        }
        folded_biases[output] = folded;
    }
    *weights = folded_weights;
    *biases = folded_biases
        .into_iter()
        .map(|value| value as f32)
        .collect();
    Ok(())
}

/// Inverts [`fold_observation_norm_into_first_layer`] for a `[input, output]` first layer.
///
/// Training applies `(x - mean) / std` before the network, so a folded raw-input graph has to be
/// unfolded before those weights can initialize a `normalize_obs` policy.
pub fn unfold_observation_norm_from_first_layer(
    first: &mut (usize, usize, Vec<f32>, Vec<f32>),
    state: &ObservationNormalizerState,
) -> Result<(), ModelError> {
    let (in_dim, out_dim, weights, biases) = first;
    state.validate(*in_dim)?;
    if weights.len() != in_dim.saturating_mul(*out_dim) || biases.len() != *out_dim {
        return Err(ModelError::InvalidMetadata(
            "first-layer weight or bias length does not match its dimensions".to_string(),
        ));
    }
    let stds = state.stds();
    let mut unfolded_weights = Vec::with_capacity(weights.len());
    let mut bias_adjust = vec![0.0f64; *out_dim];
    for (input, (mean, std)) in state.mean.iter().zip(stds.iter()).enumerate() {
        for output in 0..*out_dim {
            let folded = weights[input * *out_dim + output] as f64;
            let original = folded * std;
            if !original.is_finite() {
                return Err(ModelError::InvalidMetadata(
                    "unfolded observation weight is non-finite".to_string(),
                ));
            }
            unfolded_weights.push(original as f32);
            bias_adjust[output] += mean * folded;
        }
    }
    let mut unfolded_biases = Vec::with_capacity(*out_dim);
    for (output, bias) in biases.iter().enumerate() {
        let original = *bias as f64 + bias_adjust[output];
        if !original.is_finite() {
            return Err(ModelError::InvalidMetadata(
                "unfolded observation bias is non-finite".to_string(),
            ));
        }
        unfolded_biases.push(original as f32);
    }
    *weights = unfolded_weights;
    *biases = unfolded_biases;
    Ok(())
}

/// Applies frozen normalization once to a row-major floating-point tensor.
pub fn apply_frozen_normalizer_to_tensor_data(
    data: &mut TensorData,
    state: &ObservationNormalizerState,
) -> Result<(), ModelError> {
    let width = *data.shape.last().ok_or_else(|| {
        ModelError::InvalidInputDimension(
            "observation tensor is missing a feature dimension".to_string(),
        )
    })?;
    state.validate(width)?;
    if width == 0 || data.num_el() % width != 0 {
        return Err(ModelError::InvalidInputDimension(
            "observation tensor length is not a multiple of the normalizer width".to_string(),
        ));
    }
    match &data.dtype {
        DType::NdArray(NdArrayDType::F32) => normalize_f32_bytes(&mut data.data, state, width),
        DType::NdArray(NdArrayDType::F64) => normalize_f64_bytes(&mut data.data, state, width),
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F32) => normalize_f32_bytes(&mut data.data, state, width),
        #[cfg(feature = "tch-backend")]
        DType::Tch(TchDType::F64) => normalize_f64_bytes(&mut data.data, state, width),
        _ => Err(ModelError::DTypeError(
            "frozen observation normalization supports f32 and f64 inputs".to_string(),
        )),
    }
}

fn normalize_f32_bytes(
    bytes: &mut [u8],
    state: &ObservationNormalizerState,
    width: usize,
) -> Result<(), ModelError> {
    if bytes.len() % 4 != 0 {
        return Err(ModelError::DTypeError(
            "f32 observation buffer is not 4-byte aligned".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
    }
    state.normalize_flat(&mut values, width);
    for (chunk, value) in bytes.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&(value as f32).to_le_bytes());
    }
    Ok(())
}

fn normalize_f64_bytes(
    bytes: &mut [u8],
    state: &ObservationNormalizerState,
    width: usize,
) -> Result<(), ModelError> {
    if bytes.len() % 8 != 0 {
        return Err(ModelError::DTypeError(
            "f64 observation buffer is not 8-byte aligned".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    state.normalize_flat(&mut values, width);
    for (chunk, value) in bytes.chunks_exact_mut(8).zip(values) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_then_one_update_matches_the_collector_formula() {
        let mut state = ObservationNormalizerState::identity(2);
        assert_eq!(state.normalize_row(&[3.0, -4.0]), vec![3.0, -4.0]);
        state.update_row(&[3.0, -4.0]);
        assert_eq!(state.count, 1);
        assert_eq!(state.mean, vec![3.0, -4.0]);
        assert_eq!(state.m2, vec![1.0, 1.0]);
        let normalized = state.normalize_row(&[3.0, -4.0]);
        assert!(normalized.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn continued_updates_match_one_uninterrupted_accumulator() {
        let rows = [[1.0, 2.0], [3.0, 5.0], [8.0, -1.0]];
        let mut whole = ObservationNormalizerState::identity(2);
        for row in rows {
            whole.update_row(&row);
        }
        let mut restored = ObservationNormalizerState::identity(2);
        restored.update_row(&rows[0]);
        let saved = restored.seal();
        restored.update_row(&rows[1]);
        restored.update_row(&rows[2]);
        saved.validate(2).unwrap();
        assert_eq!(restored.mean, whole.mean);
        assert_eq!(restored.m2, whole.m2);
        assert_eq!(restored.count, whole.count);
        assert_eq!(saved.count, 1);
    }

    #[test]
    fn frozen_normalization_does_not_change_the_accumulator() {
        let mut state = ObservationNormalizerState::identity(1);
        state.update_row(&[2.0]);
        state.update_row(&[4.0]);
        let before = state.clone();
        let _ = state.normalize_row(&[4.0]);
        assert_eq!(state, before);
    }

    #[test]
    fn validation_rejects_invalid_schema_width_and_numeric_state() {
        let mut state = ObservationNormalizerState::identity(2);
        state.schema_version = 0;
        assert!(state.validate(2).is_err());

        state = ObservationNormalizerState::identity(2);
        assert!(state.validate(3).is_err());

        state = ObservationNormalizerState::identity(2);
        state.mean[0] = f64::NAN;
        assert!(state.validate(2).is_err());

        state = ObservationNormalizerState::identity(2);
        state.m2[0] = -1.0;
        assert!(state.validate(2).is_err());

        state = ObservationNormalizerState::identity(2);
        state.std_floor = 0.0;
        assert!(state.validate(2).is_err());
    }

    #[test]
    fn fold_matches_normalize_then_linear() {
        let mut state = ObservationNormalizerState::identity(2);
        state.update_row(&[1.0, 3.0]);
        state.update_row(&[5.0, 7.0]);
        let weights = vec![2.0f32, -1.0, 4.0, 0.5];
        let biases = vec![0.25f32, -0.5];
        let mut folded = (2usize, 2usize, weights.clone(), biases.clone());
        fold_observation_norm_into_first_layer(&mut folded, &state).unwrap();
        let raw = [9.0f64, -2.0];
        let normalized = state.normalize_row(&raw);
        for output in 0..2 {
            let mut expected = biases[output] as f64;
            for input in 0..2 {
                expected += normalized[input] * weights[input * 2 + output] as f64;
            }
            let mut actual = folded.3[output] as f64;
            for input in 0..2 {
                actual += raw[input] * folded.2[input * 2 + output] as f64;
            }
            assert!((expected - actual).abs() < 1e-5, "{expected} vs {actual}");
        }
    }

    #[test]
    fn unfold_inverts_fold_and_matches_normalized_linear() {
        let mut state = ObservationNormalizerState::identity(2);
        state.update_row(&[1.0, 3.0]);
        state.update_row(&[5.0, 7.0]);
        let weights = vec![2.0f32, -1.0, 4.0, 0.5];
        let biases = vec![0.25f32, -0.5];
        let mut folded = (2usize, 2usize, weights.clone(), biases.clone());
        fold_observation_norm_into_first_layer(&mut folded, &state).unwrap();
        let mut unfolded = folded.clone();
        unfold_observation_norm_from_first_layer(&mut unfolded, &state).unwrap();
        for (actual, expected) in unfolded.2.iter().zip(&weights) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
        for (actual, expected) in unfolded.3.iter().zip(&biases) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }

        let raw = [9.0f64, -2.0];
        let normalized = state.normalize_row(&raw);
        for output in 0..2 {
            let mut from_folded = folded.3[output] as f64;
            let mut from_unfolded = unfolded.3[output] as f64;
            for input in 0..2 {
                from_folded += raw[input] * folded.2[input * 2 + output] as f64;
                from_unfolded += normalized[input] * unfolded.2[input * 2 + output] as f64;
            }
            assert!(
                (from_folded - from_unfolded).abs() < 1e-5,
                "{from_folded} vs {from_unfolded}"
            );
        }
    }
}
