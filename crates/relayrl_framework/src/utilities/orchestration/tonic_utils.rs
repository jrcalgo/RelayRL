//! This module provides utilities for converting between internal RelayRL action and tensor types
//! and their serialized representations using safetensors. It also defines error types and
//! conversion functions to support these operations.

use bincode::config;
use relayrl_types::prelude::{RelayRLAction, RelayRLData, TensorData};
use relayrl_types::types::data::trajectory::EncodedTrajectory;

#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::tonic::rl_service::EncodedTrajectory as GrpcEncodedTrajectory;

use tempfile::NamedTempFile;

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

pub(crate) fn grpc_encoded_trajectory_to_relayrl_encoded_trajectory(
    encoded_trajectory: GrpcEncodedTrajectory,
) -> EncodedTrajectory {
    let data = encoded_trajectory.data.clone();

    // Deserialize metadata from bytes if present
    let metadata = if let Some(metadata_bytes) = encoded_trajectory.metadata {
        match relayrl_types::types::data::utilities::metadata::TensorMetadata::from_binary(
            &metadata_bytes,
        ) {
            Ok((meta, _)) => Some(meta),
            Err(_) => None, // If deserialization fails, set to None
        }
    } else {
        None
    };

    // Convert optional bools to bools with defaults
    let compressed = encoded_trajectory.compressed.unwrap_or(false);
    let encrypted = encoded_trajectory.encrypted.unwrap_or(false);
    // Deserialize checksum from bytes if present
    let checksum = if let Some(checksum_bytes) = encoded_trajectory.checksum {
        if checksum_bytes.len() == 32 {
            let mut checksum_array = [0u8; 32];
            checksum_array.copy_from_slice(&checksum_bytes);
            Some(checksum_array)
        } else {
            None
        }
    } else {
        None
    };

    // Deserialize num_actions from bytes
    let num_actions = bincode::serde::decode_from_slice::<usize, _>(
        &encoded_trajectory.num_actions,
        config::standard(),
    )
    .map(|(val, _)| val)
    .unwrap_or(0); // Default to 0 if deserialization fails

    // Deserialize original_size from bytes
    let original_size = bincode::serde::decode_from_slice::<usize, _>(
        &encoded_trajectory.original_size,
        config::standard(),
    )
    .map(|(val, _)| val)
    .unwrap_or(0); // Default to 0 if deserialization fails

    EncodedTrajectory {
        data,
        metadata,
        compressed,
        encrypted,
        checksum,
        num_actions,
        original_size,
    }
}

pub(crate) fn relayrl_encoded_trajectory_to_grpc_encoded_trajectory(
    encoded_trajectory: EncodedTrajectory,
) -> GrpcEncodedTrajectory {
    let metadata = if let Some(meta) = encoded_trajectory.metadata {
        meta.to_binary().ok()
    } else {
        None
    };

    let compressed = Some(encoded_trajectory.compressed);
    let encrypted = Some(encoded_trajectory.encrypted);
    let checksum = encoded_trajectory.checksum.map(|c| c.to_vec());

    // Serialize num_actions to bytes
    let num_actions =
        bincode::serde::encode_to_vec(&encoded_trajectory.num_actions, config::standard())
            .unwrap_or_default();

    // Serialize original_size to bytes
    let original_size =
        bincode::serde::encode_to_vec(&encoded_trajectory.original_size, config::standard())
            .unwrap_or_default();

    GrpcEncodedTrajectory {
        data: encoded_trajectory.data.clone(),
        metadata,
        compressed,
        encrypted,
        checksum,
        num_actions,
        original_size,
    }
}
