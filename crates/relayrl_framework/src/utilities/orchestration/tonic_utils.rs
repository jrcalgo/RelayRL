//! This module provides utilities for converting between internal RelayRL action and tensor types
//! and their serialized representations using safetensors. It also defines error types and
//! conversion functions to support these operations.

use crate::proto::{Action as GrpcRelayRLAction, Trajectory};

use relayrl_types::prelude::{RelayRLAction, RelayRLData, TensorData};
use relayrl_types::types::trajectory::{RelayRLTrajectory, RelayRLTrajectoryTrait};

use tempfile::NamedTempFile;

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

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
    max_traj_length: usize,
    config_path: &PathBuf,
) -> RelayRLTrajectory {
    let mut relayrl_trajectory: RelayRLTrajectory = RelayRLTrajectory::new(max_traj_length);

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
