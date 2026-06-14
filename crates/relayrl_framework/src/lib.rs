#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, deny(rustdoc::broken_intra_doc_links))]

//! # RelayRL Framework
//!
//! **Version:** 0.5.0 &middot; **Status:** stable
//!
//! RelayRL is a high-performance, multi-actor reinforcement learning framework built for
//! concurrent actor execution and efficient trajectory collection. This crate is the
//! top-level runtime: it composes the data model from [`relayrl_types`] and the learning
//! logic from [`relayrl_algorithms`] into a controllable, scalable client that runs many
//! actors, performs local inference, and streams trajectories to data sinks.
//!
//! ## Architecture overview
//!
//! The client runtime is layered, with a small public API over an internal,
//! concurrency-oriented runtime:
//!
//! ```text
//! Public API ......... RelayRLAgent + AgentBuilder
//!        |
//! Coordination ....... ClientCoordinator (orchestrator)
//!        |             ScaleManager (router scaling)
//!        |             StateManager (actor state)
//!        |             LifecycleManager (config, shutdown)
//!        |
//! Routing ............ RouterDispatcher + scalable Router workers
//!        |
//! Actors ............. concurrent actors, local model inference, trajectory building
//!        |
//! Data sinks ......... file sink (Arrow/CSV), transport sink (ZMQ/NATS, experimental)
//! ```
//!
//! The local/default control flow is:
//! `AgentBuilder` -> `RelayRLAgent` -> `ClientCoordinator` -> routers/actors -> data sinks.
//!
//! ## Module structure
//!
//! - [`network`]: the runtime.
//!   - [`network::client`]: the multi-actor client runtime (rewritten in v0.5.0).
//!     - [`agent`](network::client::agent): the public [`RelayRLAgent`](network::client::agent)
//!       facade and [`AgentBuilder`](network::client::agent) construction API.
//!     - The internal `runtime` holds `coordination` (coordinator, lifecycle, scaling, state),
//!       `router` (message routing), and `data` (file sinks plus experimental transport sinks).
//!   - `network::server`: optional, experimental training/inference servers behind feature
//!     flags.
//! - [`utilities`]: JSON configuration loading/builders, logging (log4rs), metrics
//!   (Prometheus/OpenTelemetry), and Tokio helpers.
//! - [`prelude`]: grouped re-exports spanning this crate plus [`relayrl_types`],
//!   [`relayrl_algorithms`], and `relayrl_env_trait` (see the [`prelude`] docs for the
//!   available paths).
//!
//! ## Design notes
//!
//! - **Generic over a Burn backend.** [`RelayRLAgent`](network::client::agent) and
//!   [`AgentBuilder`](network::client::agent) are parameterized by a single backend `B`
//!   (e.g. `burn_ndarray::NdArray`); observation/action tensor ranks and kinds are supplied
//!   per call to `request_action`, not on the builder.
//! - **Builder produces a `(RelayRLAgent, AgentStartParameters)` pair.** The agent handle and
//!   its startup parameters are separated so the runtime can be started, restarted, and
//!   shut down without rebuilding the agent.
//! - **Async, concurrent runtime.** The runtime is Tokio-based; routers can be scaled live
//!   via `scale_throughput`, and actors run concurrently with interior-mutable shared state.
//! - **What lives elsewhere.** Algorithms are in [`relayrl_algorithms`]; data types, tensors,
//!   and codecs are in [`relayrl_types`]; the environment contract is in `relayrl_env_trait`.
//!
//! ## Quick start
//!
//! Build the agent, start the runtime, request actions, and shut down. The example is
//! `no_run` because it expects a model directory and config on disk:
//!
//! ```rust,no_run
//! use relayrl_framework::prelude::network::*;
//! use relayrl_framework::prelude::types::model::ModelModule;
//! use burn_ndarray::NdArray;
//! use burn_tensor::{Tensor, Float};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Construct the agent and its startup parameters (single backend type parameter).
//! let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
//! let (mut agent, params) = AgentBuilder::<NdArray>::builder()
//!     .router_scale(2)
//!     .default_model(default_model)
//!     .config_path(PathBuf::from("client_config.json"))
//!     .build()
//!     .await?;
//!
//! // Start the coordinator, routers, and actors.
//! agent.start(params).await?;
//!
//! // Request actions: const generics are the observation/action tensor ranks.
//! let ids = agent.get_actor_ids()?;
//! let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//! let _actions = agent
//!     .request_action::<2, 2, Float, Float>(ids, observation, None, 0.0)
//!     .await?;
//!
//! // Tear everything down gracefully.
//! agent.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature flags
//!
//! - `client` (default): core client runtime.
//! - `logging` (default): log4rs logging.
//! - `tch-backend`: LibTorch (`tch`) backend support via [`relayrl_types`].
//! - `metrics`: Prometheus/OpenTelemetry metrics.
//! - `profile`: flamegraph and tokio-console profiling.
//! - `zmq-transport` / `nats-transport`: experimental network transports.
//! - `inference-server` / `training-server`: experimental server integrations.

/// Core networking functionality for RelayRL.
///
/// This module provides the multi-actor client runtime and optional server implementations.
///
/// ## Client Runtime
///
/// The [`client`](network::client) module contains the complete rewrite (v0.5.0) of the
/// multi-actor client runtime.
///
/// In `0.5.0`, the supported path is the local/default client runtime, including:
/// - Public [`agent`](network::client::agent) API for agent construction and control
/// - Internal runtime coordination (scaling, lifecycle, state management)
/// - Router-based message dispatching
/// - Actor execution with local inference
/// - Vectorized environment execution per actor
/// - Data collection via Arrow/CSV file sinks
///
/// Transport-backed workflows remain experimental even when the corresponding feature flags are
/// enabled.
///
pub mod network;

/// Configuration, logging, metrics, and system utilities.
///
/// This module contains:
/// - `configuration`: JSON-based configuration loading and builders
/// - `observability`: Logging (log4rs) and metrics (Prometheus/OpenTelemetry) systems
/// - `tokio`: Tokio runtime utilities
pub mod utilities {
    pub(crate) mod config_json;
    pub mod configuration;
    pub(crate) mod observability;
}

/// Prelude module for convenient imports.
///
/// This module re-exports commonly used types and traits for easier access:
///
/// ```rust
/// use relayrl_framework::prelude::network::*;  // Agent API
/// use relayrl_framework::prelude::config::*;  // Configuration
/// use relayrl_framework::prelude::config::network_codec::*;  // Codec types
/// use relayrl_framework::prelude::types::tensor::burn::*;  // Burn tensor types
/// use relayrl_framework::prelude::types::tensor::relayrl::*;  // RelayRL tensor types
/// use relayrl_framework::prelude::types::action::*;  // Action types
/// use relayrl_framework::prelude::types::trajectory::*;  // Trajectory types
/// use relayrl_framework::prelude::types::model::*;  // Model types
/// use relayrl_framework::prelude::templates::environment::*;  // Environment types
/// use relayrl_framework::prelude::templates::algorithms::*;  // Algorithm types
/// ```
pub mod prelude {
    pub mod algorithms {
        pub use relayrl_algorithms::algorithms::*;
    }

    pub mod network {
        pub use crate::network::client::agent::*;
        // pub use crate::network::server::inference_server::*;
        // pub use crate::network::server::training_server::*;
    }

    pub mod templates {
        pub mod algorithms {
            pub use relayrl_algorithms::templates::base_algorithm::*;
            pub use relayrl_algorithms::templates::base_replay_buffer::*;
        }

        pub mod environment {
            pub use relayrl_env_trait::*;
        }
    }

    pub mod types {
        pub mod action {
            pub use relayrl_types::prelude::action::*;
        }

        pub mod tensor {
            pub mod burn {
                pub use relayrl_types::prelude::tensor::burn::*;
            }
            pub mod relayrl {
                pub use relayrl_types::prelude::tensor::relayrl::*;
            }
        }

        pub mod trajectory {
            pub use relayrl_types::prelude::trajectory::*;
        }

        pub mod model {
            pub use relayrl_types::prelude::model::*;
        }

        pub mod records {
            pub use relayrl_types::prelude::records::*;
        }
    }

    pub mod utilities {
        pub mod config {
            pub use crate::utilities::configuration::{
                ClientConfigBuilder, ClientConfigLoader, ClientConfigParams,
                TrainingServerConfigBuilder, TrainingServerConfigLoader,
                TrainingServerConfigParams, TransportConfigBuilder, TransportConfigParams,
            };
            pub use relayrl_types::HyperparameterArgs;
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            pub mod network_codec {
                pub use relayrl_types::data::utilities::chunking::*;
                pub use relayrl_types::data::utilities::compress::*;
                pub use relayrl_types::data::utilities::encrypt::*;
                pub use relayrl_types::data::utilities::integrity::*;
                pub use relayrl_types::data::utilities::metadata::*;
                pub use relayrl_types::data::utilities::quantize::*;
            }
        }

        pub mod uuid {
            pub use active_uuid_registry::registry_uuid::*;
        }
    }
}
