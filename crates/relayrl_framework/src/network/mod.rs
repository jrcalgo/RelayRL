use relayrl_types::HyperparameterArgs;
use std::collections::HashMap;

/// **Client Constants**: Constants for client-side runtime coordination and actor management.
#[cfg(feature = "client")]
pub(super) const CLIENT_NAMESPACE_PREFIX: &str = "client";
#[cfg(feature = "client")]
pub(super) const ACTOR_CONTEXT: &str = "actor";
#[cfg(feature = "client")]
pub(super) const ENVIRONMENT_CONTEXT_PREFIX: &str = "env";
#[cfg(feature = "client")]
pub(super) const SCALE_MANAGER_CONTEXT: &str = "scaler";
#[cfg(all(feature = "client", feature = "zmq-transport"))]
pub(super) const ZMQ_CLIENT_CONTEXT: &str = "zmq-client";
#[cfg(all(feature = "client", feature = "nats-transport"))]
pub(super) const NATS_CLIENT_CONTEXT: &str = "nats-client";

#[cfg(feature = "client")]
pub(super) const ROUTER_NAMESPACE_PREFIX: &str = "router";
#[cfg(all(
    feature = "client",
    any(feature = "nats-transport", feature = "zmq-transport")
))]
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
/// The client module provides the multi-actor RL runtime:
/// - `agent` / `builder`: public construction and control APIs
/// - `runtime`: internal runtime system including:
///   - `actor`: per-actor inference and trajectory building
///   - `control`: coordinator, lifecycle, scaling, and state management
///   - `data::router`: message routing between actors and data sinks
///   - `data::sinks`: Arrow/CSV file sinks, in-memory trajectory cache, and
///     experimental ZMQ/NATS transport sinks
#[cfg(feature = "client")]
pub mod client;

/// Server runtime support is reserved for a future implementation.
///
/// The `training-server` and `inference-server` feature flags currently do not
/// compile a `network::server` module in this branch.

/// Transport mode used by client runtimes when a transport feature is enabled.
/// Extend for future utility with other transport protocols.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Clone, Copy, Debug)]
pub enum TransportMode {
    #[cfg(feature = "nats-transport")]
    NATS,
    #[cfg(feature = "zmq-transport")]
    ZMQ,
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
impl Default for TransportMode {
    fn default() -> Self {
        #[cfg(all(feature = "zmq-transport", not(feature = "nats-transport")))]
        return TransportMode::ZMQ;
        #[cfg(all(not(feature = "zmq-transport"), feature = "nats-transport"))]
        return TransportMode::NATS;
        #[cfg(all(feature = "zmq-transport", feature = "nats-transport"))]
        return TransportMode::NATS;
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
/// A [`HashMap`] where the keys and values are both strings.
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
