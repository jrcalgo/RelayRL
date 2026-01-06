//! # RelayRL Framework Structure
//! RelayRL is a high-performance reinforcement learning framework designed for distributed
//! and asynchronous RL training, particularly in high-performance computing (HPC) environments.
//!
//! RelayRL follows a modular architecture with clearly defined roles for agents, training servers,
//! configuration management, and inter-process communication. These modules are structured into
//! the following submodules:
//!
//! - **Client Modules** (`client::*`): Define agent implementations and wrappers for different communication
//!   methods, such as gRPC and ZMQ.
//! - **Server Modules** (`server::*`): Contain implementations for the RelayRL training server, including
//!   gRPC and ZMQ-based communication layers.
//! - **Core Modules** (`action`, `config_loader`, `trajectory`): Define fundamental RelayRL components,
//!   including action handling, configuration parsing, and trajectory management.
//! - **Python Bindings** (`bindings::*`): Expose the Rust implementation to Python via PyO3, enabling
//!   Python scripts to interact with RelayRL seamlessly.
//!
//! ## Rust-to-Python Bindings
//!
//! RelayRL provides a primary entry point for RelayRL Python bindings using PyO3,
//! allowing seamless integration of RelayRL functionality into Python environments.
//!
//! Agents, training servers, configuration loaders, actions, and trajectories are exposed as
//! Python-accessible classes within the `relayrl_framework` module. This enables Python users to
//! interact with RelayRL's core functionality without directly handling the Rust backend.
//!
//! The exposed Python module includes the following key classes:
//!
//! - **`ConfigLoader`**: Manages configuration settings for RelayRL components, including model paths
//!   and training parameters.
//! - **`TrainingServer`**: Represents the RelayRL training server, which is responsible for processing
//!   and optimizing trajectories sent by agents.
//! - **`RelayRLAgent`**: A Python wrapper for the RelayRL agent, allowing interaction with the reinforcement
//!   learning model and execution of actions.
//! - **`RelayRLTrajectory`**: Handles the storage and management of action sequences (trajectories).
//! - **`RelayRLAction`**: Represents individual actions taken within the RL environment, including
//!   observation, action, reward, and auxiliary data.
//!
//! ## Using RelayRL
//!

/// **Network Modules**: Provides the core networking functionality for RelayRL, including gRPC and ZMQ communication layers.
pub mod network;

/// **Development Templates**: Provides base algorithm and application templates for RelayRL.
/// These templates serve as foundations for extending or customizing RL environments, models,
/// or training strategies.
pub mod templates;

/// **System Utilities**: Provides helper functions for gRPC communication, model serialization,
/// and configuration resolution. These utilities support seamless inter-module communication.
pub mod utilities {
    pub mod configuration;
    pub(crate) mod observability;
    pub(crate) mod tokio;
}

pub mod prelude {
    pub mod config {
        pub use crate::utilities::configuration::{
            ClientConfigBuilder, ClientConfigLoader, ClientConfigParams, ServerConfigBuilder,
            ServerConfigLoader, ServerConfigParams, TransportConfigBuilder, TransportConfigParams,
        };
    }
    pub mod network {
        pub use crate::network::client::agent::*;
        // pub use crate::network::server::inference_server::*;
        // pub use crate::network::server::training_server::*;
    }
    pub mod templates {
        pub use crate::templates::environment_traits::{
            EnvironmentTestingTrait, EnvironmentTrainingTrait,
        };
    }
}
