//! This module provides utilities for converting between internal RelayRL action and tensor types
//! and their serialized representations using safetensors. It also defines error types and
//! conversion functions to support these operations.

use crate::model::{ModelError, ModelModule};
use crate::proto::{
    EncodedAction as GrpcEncodedAction, EncodedTrajectory as GrpcEncodedTrajectory,
};

use relayrl_types::prelude::{RelayRLAction, RelayRLData, TensorData};
use relayrl_types::types::trajectory::{EncodedTrajectory, EncodedTrajectoryTrait};

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
pub(crate) fn grpc_encoded_trajectory_to_relayrl_encoded_trajectory(
    encoded_trajectory: GrpcEncodedTrajectory,
    max_traj_length: usize,
    config_path: &PathBuf,
) -> EncodedTrajectory {
    let data = encoded_trajectory.data;

    EncodedTrajectory {
        data: data,
        metadata: encoded_trajectory.metadata,
        compressed: encoded_trajectory.compressed,
        encrypted: encoded_trajectory.encrypted,
        checksum: encoded_trajectory.checksum,
        num_actions: encoded_trajectory.num_actions,
        original_size: encoded_trajectory.original_size,
    }
}
