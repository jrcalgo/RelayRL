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
/// A [`HashMap`] where the keys and values are both strings, or a
/// [`HyperparameterParseError`] when a list entry is not exactly one key/value pair.
pub fn parse_args(
    hyperparameter_args: &Option<HyperparameterArgs>,
) -> Result<HashMap<String, String>, HyperparameterParseError> {
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
                    return Err(HyperparameterParseError {
                        argument: arg.clone(),
                    });
                };
                // Ensure exactly two parts are obtained: key and value.
                if split.len() != 2 {
                    return Err(HyperparameterParseError {
                        argument: arg.clone(),
                    });
                }
                hyperparams_map.insert(split[0].to_string(), split[1].to_string());
            }
        }
        None => {}
    }

    Ok(hyperparams_map)
}

pub(crate) async fn drive_on_runtime_thread<Build, Fut, Output>(
    build: Build,
) -> Result<Output, String>
where
    Build: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Output> + 'static,
    Output: Send + 'static,
{
    let runtime = tokio::runtime::Handle::try_current()
        .map_err(|error| format!("Tokio runtime unavailable: {error}"))?;
    let (reply_to, reply_from_thread) = tokio::sync::oneshot::channel();

    std::thread::Builder::new()
        .name("relayrl-env-rollout".to_string())
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| runtime.block_on(build()))).map_err(
                |panic_payload| {
                    if let Some(message) = panic_payload.downcast_ref::<&str>() {
                        (*message).to_string()
                    } else if let Some(message) = panic_payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        "rollout thread panicked with a non-string payload".to_string()
                    }
                },
            );
            let _ = reply_to.send(result);
        })
        .map_err(|error| format!("Failed to spawn rollout thread: {error}"))?;

    reply_from_thread
        .await
        .map_err(|error| format!("Rollout thread ended without returning a result: {error}"))?
}

#[cfg(test)]
mod unit_tests {
    use super::{drive_on_runtime_thread, parse_args};
    use relayrl_types::HyperparameterArgs;
    use std::collections::HashMap;
    use std::rc::Rc;

    #[test]
    fn parse_args_accepts_none_map_and_empty_list() {
        assert_eq!(parse_args(&None).unwrap(), HashMap::new());
        assert_eq!(
            parse_args(&Some(HyperparameterArgs::List(Vec::new()))).unwrap(),
            HashMap::new()
        );

        let expected = HashMap::from([("gamma".to_string(), "0.99".to_string())]);
        assert_eq!(
            parse_args(&Some(HyperparameterArgs::Map(expected.clone()))).unwrap(),
            expected
        );
    }

    #[test]
    fn parse_args_preserves_valid_list_semantics() {
        let parsed = parse_args(&Some(HyperparameterArgs::List(vec![
            "gamma=0.99".to_string(),
            "lam 0.95".to_string(),
            "key=".to_string(),
            "mixed=value with spaces".to_string(),
            "gamma=0.5".to_string(),
        ])))
        .unwrap();

        assert_eq!(parsed.get("gamma").map(String::as_str), Some("0.5"));
        assert_eq!(parsed.get("lam").map(String::as_str), Some("0.95"));
        assert_eq!(parsed.get("key").map(String::as_str), Some(""));
        assert_eq!(
            parsed.get("mixed").map(String::as_str),
            Some("value with spaces")
        );
    }

    #[test]
    fn parse_args_reports_the_malformed_entry() {
        for invalid in ["gamma", "a=b=c", "a b c"] {
            let error =
                parse_args(&Some(HyperparameterArgs::List(vec![invalid.to_string()]))).unwrap_err();
            assert_eq!(error.argument, invalid);
        }
    }

    #[tokio::test]
    async fn runtime_thread_returns_output() {
        let output = drive_on_runtime_thread(|| async { 42 }).await.unwrap();
        assert_eq!(output, 42);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn runtime_thread_accepts_non_send_future() {
        let output = drive_on_runtime_thread(|| async {
            let value = Rc::new(42);
            tokio::task::yield_now().await;
            *value
        })
        .await
        .unwrap();
        assert_eq!(output, 42);
    }

    #[tokio::test]
    async fn runtime_thread_reports_panics() {
        let error = drive_on_runtime_thread(|| async {
            panic!("rollout panic");
        })
        .await
        .unwrap_err();
        assert!(error.contains("rollout panic"));
    }
}
