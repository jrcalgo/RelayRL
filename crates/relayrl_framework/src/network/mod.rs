use relayrl_types::Hyperparams;
use relayrl_types::types::data::action::{RelayRLAction, RelayRLData};
use relayrl_types::types::data::tensor::{AnyBurnTensor, TensorData};

use rand::Rng;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

/// **Client Modules**: Handles client-side runtime coordination and actor management.
///
/// The client module provides a comprehensive runtime system for managing RL agents:
/// - `agent`: Public interface for agent implementations and wrappers
/// - `runtime`: Internal runtime system including:
///   - `actor`: Individual agent actor implementations
///   - `coordination`: Manages lifecycle, scaling, metrics, and state across actors
///   - `router`: Message routing between actors and transport layers
///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
pub mod client {
    pub mod agent;
    pub(crate) mod runtime {
        pub(crate) mod actor;
        pub(crate) mod coordination {
            pub(crate) mod coordinator;
            pub(crate) mod lifecycle_manager;
            pub(crate) mod scale_manager;
            pub(crate) mod state_manager;
        }
        pub(crate) mod router;
        pub(crate) mod router_dispatcher;
        #[cfg(feature = "networks")]
        pub(crate) mod transport;
    }
}

/// **Server Modules**: Implements RelayRL training servers and communication channels.
///
/// The server module provides a comprehensive runtime system for managing RL training:
/// - `training_server`: Public interface for training server implementations
/// - `runtime`: Internal runtime system including:
///   - `coordination`: Manages lifecycle, scaling, metrics, and state for training
///   - `python_subprocesses`: Python subprocess management for server interactions
///     - `python_algorithm_request`: Manages Python-command-based algorithm interactions
///     - `python_training_tensorboard`: Manages TensorBoard integration for training visualization
///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
///   - `router`: Message routing between workers and transport layers
///   - `worker`: Individual training worker implementations
pub mod server {
    pub mod inference_server;
    pub mod training_server;
    pub(crate) mod runtime {
        pub(crate) mod coordination {
            pub(crate) mod coordinator;
            pub(crate) mod lifecycle_manager;
            pub(crate) mod scale_manager;
            pub(crate) mod state_manager;
        }
        pub(crate) mod router;
        #[cfg(feature = "networks")]
        pub(crate) mod transport;
        pub(crate) mod worker;
    }
}

/// UUID generation code with thread-safe pool management.
///
/// This module provides functions for generating unique UUIDs and tracking them in a thread-safe pool.

#[derive(Debug, thiserror::Error)]
pub enum UuidPoolError {
    #[error("Failed to generate unique UUID: {0}")]
    FailedToGenerateUniqueUuidError(String),
    #[error("Failed to find UUID in pool: {0}")]
    FailedToFindUuidInPoolError(String),
    #[error("Failed to set UUID in pool: {0}")]
    FailedToSetUuidInPoolError(String),
}

// Thread-safe UUID pool using Mutex
pub(crate) static GLOBAL_UUID_POOL: Mutex<Vec<(Uuid, String)>> = Mutex::new(Vec::new());

pub(crate) fn random_uuid(
    context: &str,
    base: u32,
    max_retries: usize,
    retry_count: usize,
) -> Result<Uuid, UuidPoolError> {
    if retry_count >= max_retries {
        return Err(UuidPoolError::FailedToGenerateUniqueUuidError(format!(
            "Failed to generate unique UUID after {} attempts",
            max_retries
        )));
    }

    let mut rng: rand::prelude::ThreadRng = rand::rng();
    let mut uuid_bytes: [u8; 16] = [0u8; 16];

    uuid_bytes[0..4].copy_from_slice(&base.to_be_bytes());
    for i in 4..16 {
        uuid_bytes[i] = rng.random_range(0..=255);
    }

    let uuid: Uuid = Uuid::new_v8(uuid_bytes);

    let mut pool = GLOBAL_UUID_POOL.lock().map_err(|e| {
        UuidPoolError::FailedToGenerateUniqueUuidError(format!("Failed to lock UUID pool: {}", e))
    })?;

    if pool.contains(&(uuid, context.to_string())) {
        drop(pool); // Release lock before recursion
        return random_uuid(context, base, max_retries, retry_count + 1);
    }

    pool.push((uuid, context.to_string()));
    drop(pool);

    Ok(uuid)
}

pub(crate) fn add_uuid_to_pool(context: &str, uuid: &Uuid) -> Result<(), UuidPoolError> {
    let mut pool = GLOBAL_UUID_POOL.lock().map_err(|e| {
        UuidPoolError::FailedToGenerateUniqueUuidError(format!("Failed to lock UUID pool: {}", e))
    })?;

    if pool.contains(&(uuid.clone(), context.to_string())) {
        drop(pool);
        return Err(UuidPoolError::FailedToGenerateUniqueUuidError(format!(
            "UUID already exists in pool: {}",
            uuid.to_string()
        )));
    }

    pool.push((uuid.clone(), context.to_string()));
    drop(pool);
    Ok(())
}

pub(crate) fn remove_uuid_from_pool(context: &str, uuid: &Uuid) -> Result<(), UuidPoolError> {
    let mut pool = GLOBAL_UUID_POOL.lock().map_err(|e| {
        UuidPoolError::FailedToGenerateUniqueUuidError(format!("Failed to lock UUID pool: {}", e))
    })?;

    if let Some(pos) = pool
        .iter()
        .position(|x| x.0 == *uuid && x.1 == context.to_string())
    {
        pool.remove(pos);
    }

    drop(pool);
    Ok(())
}

pub(crate) fn set_uuid_in_pool(
    context: &str,
    old_uuid: &Uuid,
    new_uuid: &Uuid,
) -> Result<(), UuidPoolError> {
    let mut pool = GLOBAL_UUID_POOL.lock().map_err(|e| {
        UuidPoolError::FailedToGenerateUniqueUuidError(format!("Failed to lock UUID pool: {}", e))
    })?;

    if pool.contains(&(old_uuid.clone(), context.to_string())) {
        let confirmed_pos = pool
            .iter()
            .position(|x| x.0 == *old_uuid && x.1 == context.to_string());
        match confirmed_pos {
            Some(pos) => {
                pool.remove(pos);
                pool.push((new_uuid.clone(), context.to_string()));
                return Ok(());
            }
            None => {
                drop(pool);
                return Err(UuidPoolError::FailedToFindUuidInPoolError(format!(
                    "Failed to find UUID in pool: {}",
                    old_uuid.to_string()
                )));
            }
        }
    }
    drop(pool);
    Ok(())
}

/// Extend for future utility with other transport protocols (extend transport.rs accordingly)
#[derive(Clone, Copy, Debug)]
pub enum TransportType {
    GRPC,
    ZMQ,
}

/// Parses hyperparameter arguments into a HashMap.
///
/// The function accepts an optional `Hyperparams` enum value, which may be provided as either
/// a map or a vector of argument strings. It returns a HashMap mapping hyperparameter keys to
/// their corresponding string values.
///
/// # Arguments
///
/// * `hyperparams` - An optional [Hyperparams] enum that contains either a map or vector of strings.
///
/// # Returns
///
/// A [HashMap] where the keys and values are both strings.
pub fn parse_args(hyperparams: &Option<Hyperparams>) -> HashMap<String, String> {
    let mut hyperparams_map: HashMap<String, String> = HashMap::new();

    match hyperparams {
        Some(Hyperparams::Map(map)) => {
            for entry in map.iter() {
                hyperparams_map.insert(entry.key().to_string(), entry.value().to_string());
            }
        }
        Some(Hyperparams::Args(args)) => {
            for arg in args {
                // Split the argument string on '=' or ' ' if possible.
                let split: Vec<&str> = if arg.contains("=") {
                    arg.split('=').collect()
                } else if arg.contains(' ') {
                    arg.split(' ').collect()
                } else {
                    panic!(
                        "[TrainingServer - new] Invalid hyperparameter argument: {}",
                        arg
                    );
                };
                // Ensure exactly two parts are obtained: key and value.
                if split.len() != 2 {
                    panic!(
                        "[TrainingServer - new] Invalid hyperparameter argument: {}",
                        arg
                    );
                }
                hyperparams_map.insert(split[0].to_string(), split[1].to_string());
            }
        }
        None => {}
    }

    hyperparams_map
}
