//! This module provides utilities for converting between internal RelayRL action and tensor types
//! and their serialized representations using safetensors. It also defines error types and
//! conversion functions to support these operations.

use crate::proto::{Action as GrpcRelayRLAction, Trajectory};
use crate::types::NetworkParticipant;
use crate::types::action::{RelayRLAction, SafeTensorError, TensorData};
use crate::types::trajectory::{RelayRLTrajectory, RelayRLTrajectoryTrait};

use tch::{CModule, Device, TchError};
use tempfile::NamedTempFile;

use crate::types::action::RelayRLData;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

/// Serializes an internal [`RelayRLAction`] into its gRPC representation.
///
/// For each tensor field (observation, action, and mask), if present, the function extracts its
/// underlying serialized byte vector; otherwise, it produces an empty vector. Additionally, if
/// auxiliary data is present, each value is serialized to JSON bytes.
///
/// # Arguments
///
/// * `action` - A reference to the [`RelayRLAction`] instance to be serialized.
///
/// # Returns
///
/// A [`GrpcRelayRLAction`] struct containing the serialized observation, action, mask, reward,
/// auxiliary data, and flags.
pub(crate) fn serialize_action(action: &RelayRLAction) -> GrpcRelayRLAction {
    // Retrieve serialized bytes for each tensor field; use an empty vector if absent.
    let obs_bytes = action
        .obs
        .as_ref()
        .map_or_else(Vec::new, |td| td.data.clone());
    let act_bytes = action
        .act
        .as_ref()
        .map_or_else(Vec::new, |td| td.data.clone());
    let mask_bytes = action
        .mask
        .as_ref()
        .map_or_else(Vec::new, |td| td.data.clone());

    // Serialize auxiliary data (if any) into JSON bytes.
    let data: HashMap<String, Vec<u8>> = action.data.as_ref().map_or_else(HashMap::new, |map| {
        map.iter()
            .map(|(k, v)| {
                let serialized = serde_json::to_vec(v).expect("Serialization of RelayRLData failed");
                (k.clone(), serialized)
            })
            .collect()
    });

    GrpcRelayRLAction {
        obs: obs_bytes,
        action: act_bytes,
        mask: mask_bytes,
        reward: action.rew,
        data,
        done: action.done,
    }
}

/// Deserializes a gRPC action message into an internal [`RelayRLAction`].
///
/// For each tensor field, if the provided byte vector is nonempty, it attempts to convert the bytes
/// back into a tensor representation via [`RelayRLAction::from_bytes`]. It also deserializes any
/// auxiliary data from JSON bytes.
///
/// # Arguments
///
/// * `grpc_action` - A [`GrpcRelayRLAction`] containing the serialized action data.
///
/// # Returns
///
/// A [`Result`] which is:
/// - `Ok(RelayRLAction)` if deserialization succeeds, or
/// - `Err(SafeTensorError)` if any conversion fails.
pub(crate) fn deserialize_action(
    grpc_action: GrpcRelayRLAction,
) -> Result<RelayRLAction, SafeTensorError> {
    // Convert observation bytes to tensor if available.
    let obs: Option<TensorData> = if grpc_action.obs.is_empty() {
        None
    } else {
        Some(RelayRLAction::from_bytes(grpc_action.obs)?)
    };

    // Convert action bytes to tensor if available.
    let act: Option<TensorData> = if grpc_action.action.is_empty() {
        None
    } else {
        Some(RelayRLAction::from_bytes(grpc_action.action)?)
    };

    // Convert mask bytes to tensor if available.
    let mask: Option<TensorData> = if grpc_action.mask.is_empty() {
        None
    } else {
        Some(RelayRLAction::from_bytes(grpc_action.mask)?)
    };

    // Deserialize auxiliary data from JSON bytes if available.
    let data: Option<HashMap<String, RelayRLData>> = if grpc_action.data.is_empty() {
        None
    } else {
        let mut map = HashMap::new();
        for (k, v) in grpc_action.data.into_iter() {
            let deserialized: RelayRLData = serde_json::from_slice(&v)
                .map_err(|e| SafeTensorError::SerializationError(e.to_string()))?;
            map.insert(k, deserialized);
        }
        Some(map)
    };

    Ok(RelayRLAction {
        obs,
        act,
        mask,
        rew: grpc_action.reward,
        data,
        done: grpc_action.done,
    })
}

/// Converts a gRPC [`Trajectory`] into an internal [`RelayRLTrajectory`].
///
/// This function deserializes each action in the provided gRPC trajectory and adds it to a new
/// [`RelayRLTrajectory`] with the specified maximum trajectory length.
///
/// # Arguments
///
/// * `trajectory` - The gRPC [`Trajectory`] to be converted.
/// * `max_traj_length` - The maximum allowed length for the trajectory.
///
/// # Returns
///
/// An [`RelayRLTrajectory`] constructed from the deserialized actions.
pub(crate) fn grpc_trajectory_to_relayrl_trajectory(
    trajectory: Trajectory,
    max_traj_length: u128,
    config_path: &PathBuf,
) -> RelayRLTrajectory {
    let mut relayrl_trajectory: RelayRLTrajectory = RelayRLTrajectory::new(
        Some(max_traj_length),
        NetworkParticipant::RelayRLTrainingServer,
        config_path,
    );

    for action in trajectory.actions {
        let action: RelayRLAction =
            deserialize_action(action).expect("failed to deserialize action");
        relayrl_trajectory.add_action(&action);
    }

    relayrl_trajectory
}

/// Serializes a TorchScript model (`CModule`) into a vector of bytes.
///
/// The model is saved to a temporary file and then read back into a byte vector.
///
/// # Arguments
///
/// * `model` - A reference to the [`CModule`] (TorchScript model) to be serialized.
///
/// # Returns
///
/// A vector of bytes representing the serialized model.
pub(crate) fn serialize_model(model: &CModule, dir: PathBuf) -> Vec<u8> {
    let temp_file = tempfile::Builder::new()
        .prefix("_model")
        .suffix(".pt")
        .tempfile_in(dir)
        .expect("Failed to create temp file");
    let temp_path = temp_file.path();

    model.save(temp_path).expect("Failed to save model");

    std::fs::read(temp_path).expect("Failed to read model bytes")
}

/// Deserializes a vector of bytes into a TorchScript model (`CModule`).
///
/// The function writes the provided bytes to a temporary file, flushes it, and then loads
/// the model from that file.
///
/// # Arguments
///
/// * `model_bytes` - A vector of bytes containing the serialized model.
///
/// # Returns
///
/// A [`CModule`] representing the deserialized TorchScript model.
pub(crate) fn deserialize_model(model_bytes: Vec<u8>) -> Result<CModule, TchError> {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(&model_bytes)
        .expect("Failed to write model bytes");
    temp_file.flush().expect("Failed to flush temp file");

    Ok(CModule::load_on_device(temp_file.path(), Device::Cpu)
        .expect("Failed to load model from bytes"))
}
