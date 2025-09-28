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

pub mod network;

#[cfg(feature = "python_bindings")]
use crate::bindings::python::network::client::o3_agent::PyRelayRLAgent;
#[cfg(feature = "python_bindings")]
use crate::bindings::python::network::server::o3_training_server::PyTrainingServer;
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_action::PyRelayRLAction;
#[cfg(feature = "python_bindings")]
use crate::bindings::python::o3_config_loader::PyConfigLoader;
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_trajectory::PyRelayRLTrajectory;
#[cfg(feature = "python_bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python_bindings")]
use pyo3::{Bound, PyResult, pymodule};


/// **Protocol Buffers (Protobuf) for gRPC Communication**
///
/// This module contains Rust code generated from `.proto` files using `tonic::include_proto!`,
/// enabling structured message exchange between RelayRL components.
#[cfg(feature = "grpc_network")]
mod proto {
    tonic::include_proto!("relayrl_grpc");
}

/// **System Utilities**: Provides helper functions for gRPC communication, model serialization,
/// and configuration resolution. These utilities support seamless inter-module communication.
pub(crate) mod sys_utils {
    pub mod config_loader;
    pub(crate) mod misc_utils;
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network"
    ))]
    pub(crate) mod resolve_server_config;
    pub(crate) mod tokio_utils;
    #[cfg(feature = "grpc_network")]
    pub(crate) mod grpc_utils;
}

/// **Core RelayRL Data Types**: Define the primary data structures used
/// throughout the framework. These include:
/// - `trajectory`: Defines trajectory management and serialization logic.
/// - `action`: Handles action structures and data.
#[cfg(feature = "data_types")]
pub mod types {
    pub mod action;
    pub mod trajectory;
}

/// **Python Bindings for RelayRL**: This module contains the Rust-to-Python bindings,
/// exposing RelayRL components as Python classes. The `o3_*` modules implement PyO3-compatible
/// wrappers for core structures, enabling smooth Python interaction.
pub mod bindings {
    pub mod python {
        #[cfg(any(
            feature = "networks",
            feature = "grpc_network",
            feature = "zmq_network",
            feature = "python_bindings"
        ))]
        #[cfg_attr(bench, visibility = "pub")]
        pub mod o3_action;
        #[cfg(feature = "python_bindings")]
        #[cfg_attr(bench, visibility = "pub")]
        pub mod o3_config_loader;
        #[cfg(any(
            feature = "networks",
            feature = "grpc_network",
            feature = "zmq_network",
            feature = "python_bindings"
        ))]
        #[cfg_attr(bench, visibility = "pub")]
        pub mod o3_trajectory;

        /// **Network Python Wrappers**: Exposes the RelayRL network components to Python.
        #[cfg(feature = "python_bindings")]
        pub mod network {
            /// **Client Python Wrappers**: Wraps RelayRL agents for Python integration.
            pub mod client {
                pub mod o3_agent;
            }

            /// **Server Python Wrappers**: Exposes the RelayRL training server to Python.
            pub mod server {
                pub mod o3_training_server;
            }
        }
    }
}

/// ### RelayRL Python Module Definition
///
/// This function defines `relayrl_framework`, the Python module for RelayRL bindings.
///
/// It registers the following Python classes:
/// - `ConfigLoader`
/// - `TrainingServer`
/// - `RelayRLAgent`
/// - `RelayRLTrajectory`
/// - `RelayRLAction`
///
/// This allows Python users to easily import and use RelayRL functionalities via:
///
/// ```python
/// from relayrl_framework import RelayRLAgent, RelayRLTrajectory, RelayRLAction
/// ```
///
#[cfg(feature = "python_bindings")]
#[pymodule(name = "relayrl_framework")]
fn relayrl_framework(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Python-accessible classes from the Rust implementation.
    m.add_class::<PyConfigLoader>()?;
    m.add_class::<PyTrainingServer>()?;
    m.add_class::<PyRelayRLAgent>()?;
    m.add_class::<PyRelayRLTrajectory>()?;
    m.add_class::<PyRelayRLAction>()?;

    // Define Python `__all__` to indicate available imports.
    m.add(
        "__all__",
        vec![
            "ConfigLoader",
            "TrainingServer",
            "RelayRLAgent",
            "RelayRLTrajectory",
            "RelayRLAction",
        ],
    )?;

    Ok(())
}
