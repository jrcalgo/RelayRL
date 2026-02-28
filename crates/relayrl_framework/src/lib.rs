#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, deny(rustdoc::broken_intra_doc_links))]

//! # RelayRL Framework
//!
//! **Version:** 0.5.0-alpha  
//! **Status:** Under active development, expect breaking changes
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
//! │  - ClientCoordinator                            │
//! │  - ScaleManager (router scaling)                │
//! │  - StateManager (actor state)                   │
//! │  - LifecycleManager (shutdown coordination)     │
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
//! │  - Arrow File Sink (available)                  │
//! │  - Transport Sink (under development)           │
//! │  - Database Sink (under development)            │
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
//!     - `data`: Transport and database layers (under development)
//!
//! - **[`network::server`]**: Training and inference server implementations (optional features)
//!
//! - **[`templates`]**: Environment trait definitions for training and testing
//!
//! - **[`utilities`]**: Configuration loading, logging, metrics, and system utilities
//!
//! ## Current Status
//!
//! ### Available
//! - Multi-actor client runtime with concurrent execution
//! - Local Arrow file sink for trajectory data
//! - Builder pattern API for ergonomic agent construction
//! - Router-based message dispatching with scaling support
//! - Actor lifecycle management (create, remove, scale)
//!
//! ### Under Development
//! - Network transport layer (ZMQ)
//! - Database trajectory sinks (PostgreSQL/SQLite)
//! - Server-side inference mode
//! - Training server integration
//!
//! ### Not In This Crate
//! - **Python Bindings**: See `relayrl_python` crate
//! - **Algorithms**: See `relayrl_algorithms` crate
//! - **Type Definitions**: See `relayrl_types` crate
//!
//! ## Quick Example
//!
//! ```rust,no_run
//! use relayrl_framework::prelude::network::*;
//! use relayrl_types::data::tensor::DeviceType;
//! use burn_ndarray::NdArray;
//! use burn_tensor::{Tensor, Float};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Build agent with 4 concurrent actors
//! let (agent, params) = AgentBuilder::<NdArray, 4, 2, Float, Float>::builder()
//!     .actor_count(4)
//!     .router_scale(2)
//!     .default_device(DeviceType::Cpu)
//!     .config_path(PathBuf::from("client_config.json"))
//!     .build()
//!     .await?;
//!
//! // Start runtime
//! agent.start(
//!     params.actor_count,
//!     params.router_scale,
//!     params.default_device,
//!     params.default_model,
//!     params.config_path,
//! ).await?;
//!
//! // Request actions from actors
//! let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &Default::default());
//! let actions = agent.request_action(
//!     vec![/* actor IDs */],
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
//! - `network`: Full network stack (client + servers)
//! - `transport_layer`: Network transport (ZMQ)
//! - `database_layer`: Database support (PostgreSQL, SQLite)
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
/// multi-actor client runtime, including:
/// - Public [`agent`](network::client::agent) API for agent construction and control
/// - Internal runtime coordination (scaling, lifecycle, state management)
/// - Router-based message dispatching
/// - Actor execution with local inference
/// - Data collection via Arrow file sink (transport/database under development)
///
/// ## Server Components (Optional)
///
/// The [`server`](network::server) module provides training and inference server implementations,
/// available via feature flags (`training_server`, `inference_server`).
pub mod network;

/// Environment trait definitions for RL training and testing.
///
/// This module provides:
/// - [`EnvironmentTrainingTrait`](templates::environment_traits::EnvironmentTrainingTrait):
///   Interface for training environments with performance metrics
/// - [`EnvironmentTestingTrait`](templates::environment_traits::EnvironmentTestingTrait):
///   Interface for inference/testing environments
pub mod templates;

/// Configuration, logging, metrics, and system utilities.
///
/// This module contains:
/// - `configuration`: JSON-based configuration loading and builders
/// - `observability`: Logging (log4rs) and metrics (Prometheus/OpenTelemetry) systems
/// - `tokio`: Tokio runtime utilities
pub mod utilities {
    pub mod configuration;
    pub(crate) mod observability;
    pub(crate) mod tokio;
}

/// Prelude module for convenient imports.
///
/// This module re-exports commonly used types and traits for easier access:
///
/// ```rust
/// use relayrl_framework::prelude::network::*;  // Agent API
/// use relayrl_framework::prelude::config::*;  // Configuration
/// use relayrl_framework::prelude::config::network_codec::*;  // Codec types
/// use relayrl_framework::prelude::tensor::burn::*;  // Burn tensor types
/// use relayrl_framework::prelude::tensor::relayrl::*;  // RelayRL tensor types
/// use relayrl_framework::prelude::action::*;  // Action types
/// use relayrl_framework::prelude::trajectory::*;  // Trajectory types
/// use relayrl_framework::prelude::model::*;  // Model types
/// use relayrl_framework::prelude::templates::*;  // Environment types
/// ```
pub mod prelude {
    pub mod network {
        pub use crate::network::client::agent::*;
        // pub use crate::network::server::inference_server::*;
        // pub use crate::network::server::training_server::*;
    }

    pub mod config {
        pub use crate::utilities::configuration::{
            ClientConfigBuilder, ClientConfigLoader, ClientConfigParams, TrainingServerConfigBuilder,
            TrainingServerConfigLoader, TrainingServerConfigParams, TransportConfigBuilder, TransportConfigParams,
        };
        pub use relayrl_types::HyperparameterArgs;
        pub mod network_codec {
            pub use relayrl_types::data::utilities::chunking::*;
            pub use relayrl_types::data::utilities::compress::*;
            pub use relayrl_types::data::utilities::encrypt::*;
            pub use relayrl_types::data::utilities::integrity::*;
            pub use relayrl_types::data::utilities::metadata::*;
            pub use relayrl_types::data::utilities::quantize::*;
        }
    }

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

    pub mod templates {
        pub use crate::templates::environment_traits::{
            EnvironmentError, EnvironmentTestingTrait, EnvironmentTrainingTrait,
        };
    }
}
