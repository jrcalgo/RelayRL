//! RelayRL client runtime.
//!
//! This module is split into a small public API surface and a larger internal runtime:
//! - [`agent`](crate::agent::client::agent): public construction and control APIs for client applications
//! - `runtime::control`: coordinator, lifecycle, scaling, and state management
//! - `runtime::data::router`: message routing between actors and data sinks
//! - `runtime::data`: local file sinks plus experimental transport-backed sinks
//!
//! In `0.5.0`, the supported path is the local/default runtime exposed through
//! [`agent`](crate::agent::client::agent). Transport-backed flows behind `zmq-transport` and
//! `nats-transport` remain experimental.
//!
//! The local/default runtime follows this flow:
//! `AgentBuilder` -> `RelayRLAgent` -> coordinator -> router/actors -> local file sink.
pub mod process;
mod builder;
pub(crate) mod runtime {
    pub(crate) mod actor;
    pub(crate) mod control {
        pub(crate) mod coordinator;
        pub(crate) mod lifecycle_manager;
        pub(crate) mod scale_manager;
        pub(crate) mod state_manager;
    }

    pub(crate) mod data {
        pub(crate) mod environments;
        pub(crate) mod router;
        pub(crate) mod sinks {
            pub(crate) mod file_sink;
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            pub(crate) mod transport_sink;
        }
        pub(crate) mod training;
    }
}

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
