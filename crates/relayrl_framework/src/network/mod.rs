use relayrl_types::Hyperparams;
use relayrl_types::types::action::{RelayRLAction, RelayRLData};
use relayrl_types::types::tensor::TensorData;

use rand::Rng;
use std::collections::HashMap;
use std::path::Path;
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

pub fn random_uuid(base: u32) -> Uuid {
    let random_num = base * rand::thread_rng().gen_range(11..100)
        + base
        + 1 * rand::thread_rng().gen_range(11..100)
        - rand::thread_rng().gen_range(1..10);
    Uuid::new_v8([random_num as u8; 16])
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
            for (key, value) in map {
                hyperparams_map.insert(key.to_string(), value.to_string());
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
