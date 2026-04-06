use relayrl_types::HyperparameterArgs;
use std::collections::HashMap;

/// **Client Constants**: Constants for client-side runtime coordination and actor management.
#[cfg(feature = "client")]
pub(super) const CLIENT_NAMESPACE_PREFIX: &str = "client";
#[cfg(feature = "client")]
pub(super) const ACTOR_CONTEXT: &str = "actor";
#[cfg(feature = "client")]
pub(super) const SCALE_MANAGER_CONTEXT: &str = "scaler";
#[cfg(all(feature = "client", feature = "zmq-transport"))]
pub(super) const ZMQ_CLIENT_CONTEXT: &str = "zmq-client";
#[cfg(all(feature = "client", feature = "nats-transport"))]
pub(super) const NATS_CLIENT_CONTEXT: &str = "nats-client";

#[cfg(feature = "client")]
pub(super) const ROUTER_NAMESPACE_PREFIX: &str = "router";
#[cfg(all(feature = "client", feature = "nats-transport"))]
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(super) const RECEIVER_CONTEXT: &str = "receiver";
#[cfg(feature = "client")]
pub(super) const BUFFER_CONTEXT: &str = "buffer";

/// **Server Constants**: Constants for server-side runtime coordination and actor management.
#[cfg(all(
    feature = "training-server",
    any(feature = "nats-transport", feature = "zmq-transport")
))]
pub(super) const TRAINING_SERVER_NAMESPACE_PREFIX: &str = "training-server";
#[cfg(all(
    feature = "inference-server",
    any(feature = "nats-transport", feature = "zmq-transport")
))]
pub(super) const INFERENCE_SERVER_NAMESPACE_PREFIX: &str = "inference-server";
#[cfg(all(
    any(feature = "training-server", feature = "inference-server"),
    any(feature = "nats-transport", feature = "zmq-transport")
))]
pub(super) const WORKER_CONTEXT: &str = "worker";
#[cfg(all(
    any(feature = "training-server", feature = "inference-server"),
    feature = "zmq-transport"
))]
pub(super) const ZMQ_SERVER_CONTEXT: &str = "zmq-server";
#[cfg(all(
    any(feature = "training-server", feature = "inference-server"),
    feature = "nats-transport"
))]
pub(super) const NATS_SERVER_CONTEXT: &str = "nats-server";

/// **Client Modules**: Handles client-side runtime coordination and actor management.
///
/// The client module provides a comprehensive runtime system for managing RL agents:
/// - `agent`: Public interface for agent implementations and wrappers
/// - `runtime`: Internal runtime system including:
///   - `actor`: Individual agent actor implementations
///   - `coordination`: Manages lifecycle, scaling, metrics, and state across actors
///   - `router`: Message routing between actors and transport layers
///   - `transport`: Network transport implementations (ZeroMQ)
///   - `database`: Database implementations (PostgreSQL, SQLite)
#[cfg(feature = "client")]
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
#[cfg(any(feature = "inference-server", feature = "training-server"))]
pub mod server;

/// Extend for future utility with other transport protocols (extend transport.rs accordingly)
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Clone, Copy, Debug)]
pub enum TransportType {
    #[cfg(feature = "nats-transport")]
    NATS,
    #[cfg(feature = "zmq-transport")]
    ZMQ,
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
impl Default for TransportType {
    fn default() -> Self {
        #[cfg(all(feature = "zmq-transport", not(feature = "nats-transport")))]
        return TransportType::ZMQ;
        #[cfg(all(not(feature = "zmq-transport"), feature = "nats-transport"))]
        return TransportType::NATS;
        #[cfg(all(feature = "zmq-transport", feature = "nats-transport"))]
        return TransportType::NATS;
    }
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
