use std::collections::HashMap;

/// **Client Modules**: Handles client-side runtime coordination and actor management.
///
/// The client module provides a comprehensive runtime system for managing RL agents:
/// - `agent`: Public interface for agent implementations and wrappers
/// - `runtime`: Internal runtime system including:
///   - `actor`: Individual agent actor implementations
///   - `coordination`: Manages lifecycle, scaling, metrics, and state across actors
///   - `router`: Message routing between actors and transport layers
///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
pub mod client;

/// **Server Modules**: Implements RelayRL training servers and communication channels.
///
/// The server module provides a comprehensive runtime system for managing RL training:
/// - `training_server`: Public interface for training server implementation
/// - `inference_server`: Public interface for inference server implementation
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
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        pub(crate) mod transport;
        pub(crate) mod worker;
    }
}

/// Extend for future utility with other transport protocols (extend transport.rs accordingly)
#[derive(Clone, Copy, Debug)]
pub enum TransportType {
    #[cfg(feature = "zmq_transport")]
    ZMQ,
}

impl Default for TransportType {
    fn default() -> Self {
        TransportType::ZMQ
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HyperparameterArgs {
    Map(HashMap<String, String>),
    List(Vec<String>),
}

/// Parses hyperparameter arguments into a HashMap.
///
/// The function accepts an optional `HyperparameterArgs` enum value, which may be provided as either
/// a map or a vector of argument strings. It returns a HashMap mapping hyperparameter keys to
/// their corresponding string values.
///
/// # Arguments
///
/// * `hyperparameter_args` - An optional [HyperparameterArgs] enum that contains either a map or vector of strings.
///
/// # Returns
///
/// A [DashMap] where the keys and values are both strings.
pub fn parse_args(hyperparameter_args: &Option<HyperparameterArgs>) -> HashMap<String, String> {
    let mut hyperparams_map: HashMap<String, String> = HashMap::new();

    match hyperparameter_args {
        Some(HyperparameterArgs::Map(map)) => {
            for entry in map.iter() {
                hyperparams_map.insert(entry.0.to_string(), entry.1.to_string());
            }
        }
        Some(HyperparameterArgs::List(args)) => {
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
