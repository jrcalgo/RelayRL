#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, deny(rustdoc::broken_intra_doc_links))]

//! # RelayRL Framework
//!
//! **Version:** 0.5.0-beta
//! **Status:** Under active development, expect breaking changes
//! **Beta Scope:** The supported beta path is the local/default client runtime. Transport-backed
//! and server-backed workflows are still experimental.
//!
//! RelayRL is a high-performance, multi-actor native reinforcement learning framework designed for
//! concurrent actor execution and efficient trajectory collection. This crate currently provides the core
//! client runtime infrastructure for distributed RL experiments.
//!
//! ## Architecture Overview
//!
//! The framework follows a layered architecture optimized for concurrent multi-actor execution:
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Public API (RelayRLAgent, AgentBuilder)        │
//! └─────────────────────────────────────────────────┘
//!                         │
//! ┌─────────────────────────────────────────────────┐
//! │  Runtime Coordination Layer                     │
//! │  - ClientCoordinator (orchestrator)                            │
//! │  - ScaleManager (router scaling)                │
//! │  - StateManager (actor state)                   │
//! │  - LifecycleManager (config, shutdown)          │
//! └─────────────────────────────────────────────────┘
//!                         │
//! ┌─────────────────────────────────────────────────┐
//! │  Message Routing Layer                          │
//! │  - RouterDispatcher                             │
//! │  - Router instances (scalable workers)          │
//! └─────────────────────────────────────────────────┘
//!                         │
//! ┌─────────────────────────────────────────────────┐
//! │  Actor Execution Layer                          │
//! │  - Concurrent Actor instances                   │
//! │  - Local model inference                        │
//! │  - Trajectory building                          │
//! └─────────────────────────────────────────────────┘
//!                         │
//! ┌─────────────────────────────────────────────────┐
//! │  Data Collection Layer                          │
//! │  - TrajectoryBuffer (priority scheduling)       │
//! │  - File Sink (Arrow/CSV)                        │
//! │  - Transport Sink (ZMQ/NATS, experimental)      │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Module Structure
//!
//! - **[`network::client`]**: Multi-actor client runtime (complete rewrite in v0.5.0)
//!   - [`agent`](network::client::agent): Public API for agent construction and interaction
//!   - `runtime`: Internal runtime components
//!     - `actor`: Individual actor implementations with local inference
//!     - `coordination`: Lifecycle, scaling, metrics, and state management
//!     - `router`: Message routing between actors and data sinks
//!     - `data`: Transport layers (ZMQ/NATS) and file sinks (Arrow/CSV)
//!
//! - **[`utilities`]**: Configuration loading, logging, metrics, and system utilities
//!
//! ## Current Status
//!
//! ### Available
//! - Local/default multi-actor client runtime with concurrent execution
//! - Local Arrow/CSV file sink for trajectory data
//! - Builder pattern API for ergonomic agent construction
//! - Router-based message dispatching with scaling support
//! - Actor lifecycle management (create, remove, scale)
//!
//! ### Under Development
//! - Transport-backed client workflows (`zmq-transport`, `nats-transport`)
//! - Inference server integration
//! - Training server integration
//!
//! ### Not In This Crate
//! - **Algorithms**: See `relayrl_algorithms` crate
//! - **Type Definitions**: See `relayrl_types` crate
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use relayrl_framework::prelude::network::*;
//! use relayrl_framework::prelude::types::model::ModelModule;
//! use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;
//! use burn_ndarray::NdArray;
//! use burn_tensor::{Tensor, Float};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Build agent with 4 concurrent actors
//! let default_model = ModelModule::<NdArray>::load_from_path("model_dir")?;
//! let (mut agent, params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
//!     .actor_count(4)
//!     .router_scale(2)
//!     .default_device(DeviceType::Cpu)
//!     .default_model(default_model)
//!     .config_path(PathBuf::from("client_config.json"))
//!     .build()
//!     .await?;
//!
//! // Start runtime
//! agent.start(params).await?;
//!
//! // Request actions from actors
//! let ids = agent.get_actor_ids()?;
//! let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//! let _actions = agent.request_action(
//!     ids,
//!     observation,
//!     None,
//!     0.0
//! ).await?;
//!
//! // Shutdown gracefully
//! agent.shutdown().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! - `client` (default): Core client runtime
//! - `tch-backend`: Tch backend support
//! - `inference-server`: Inference server support (under development)
//! - `training-server`: Training server support (under development)
//! - `zmq-transport`: Experimental network transport (ZMQ)
//! - `nats-transport`: Experimental network transport (NATS)
//! - `logging`: Log4rs logging
//! - `metrics`: Prometheus/OpenTelemetry metrics
//! - `profile`: Flamegraph and tokio-console profiling

/// Core networking functionality for RelayRL.
///
/// This module provides the multi-actor client runtime and optional server implementations.
///
/// ## Client Runtime
///
/// The [`client`](network::client) module contains the complete rewrite (v0.5.0) of the
/// multi-actor client runtime.
///
/// In `0.5.0-beta`, the supported path is the local/default client runtime, including:
/// - Public [`agent`](network::client::agent) API for agent construction and control
/// - Internal runtime coordination (scaling, lifecycle, state management)
/// - Router-based message dispatching
/// - Actor execution with local inference
/// - Data collection via Arrow/CSV file sinks
///
/// Transport-backed workflows remain experimental even when the corresponding feature flags are
/// enabled.
///
/// ## Server Components (Optional)
///
/// The [`server`](network::server) module provides training and inference server implementations,
/// available via feature flags (`training_server`, `inference_server`). These are still
/// experimental and not part of the `0.5.0-beta` support promise.
pub mod network;

/// Configuration, logging, metrics, and system utilities.
///
/// This module contains:
/// - `configuration`: JSON-based configuration loading and builders
/// - `observability`: Logging (log4rs) and metrics (Prometheus/OpenTelemetry) systems
/// - `tokio`: Tokio runtime utilities
pub mod utilities {
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

    pub mod config {
        pub use crate::utilities::configuration::{
            ClientConfigBuilder, ClientConfigLoader, ClientConfigParams,
            TrainingServerConfigBuilder, TrainingServerConfigLoader, TrainingServerConfigParams,
            TransportConfigBuilder, TransportConfigParams,
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
        pub mod tensor {
            pub mod burn {
                pub use relayrl_types::prelude::tensor::burn::*;
            }
            pub mod relayrl {
                pub use relayrl_types::prelude::tensor::relayrl::*;
            }
        }

        pub mod action {
            pub use relayrl_types::prelude::action::*;
        }

        pub mod trajectory {
            pub use relayrl_types::prelude::trajectory::*;
        }

        pub mod model {
            pub use relayrl_types::prelude::model::*;
        }
    }
}
