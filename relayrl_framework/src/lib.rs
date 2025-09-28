//! # RL4Sys Framework Structure
//! RL4Sys is a high-performance reinforcement learning framework designed for distributed
//! and asynchronous RL training, particularly in high-performance computing (HPC) environments.
//!
//! RL4Sys follows a modular architecture with clearly defined roles for agents, training servers,
//! configuration management, and inter-process communication. These modules are structured into
//! the following submodules:
//!
//! - **Client Modules** (`client::*`): Define agent implementations and wrappers for different communication
//!   methods, such as gRPC and ZMQ.
//! - **Server Modules** (`server::*`): Contain implementations for the RL4Sys training server, including
//!   gRPC and ZMQ-based communication layers.
//! - **Core Modules** (`action`, `config_loader`, `trajectory`): Define fundamental RL4Sys components,
//!   including action handling, configuration parsing, and trajectory management.
//! - **Python Bindings** (`bindings::*`): Expose the Rust implementation to Python via PyO3, enabling
//!   Python scripts to interact with RL4Sys seamlessly.
//!
//! ## Rust-to-Python Bindings
//!
//! RL4Sys provides a primary entry point for RL4Sys Python bindings using PyO3,
//! allowing seamless integration of RL4Sys functionality into Python environments.
//!
//! Agents, training servers, configuration loaders, actions, and trajectories are exposed as
//! Python-accessible classes within the `rl4sys_framework` module. This enables Python users to
//! interact with RL4Sys’s core functionality without directly handling the Rust backend.
//!
//! The exposed Python module includes the following key classes:
//!
//! - **`ConfigLoader`**: Manages configuration settings for RL4Sys components, including model paths
//!   and training parameters.
//! - **`TrainingServer`**: Represents the RL4Sys training server, which is responsible for processing
//!   and optimizing trajectories sent by agents.
//! - **`RL4SysAgent`**: A Python wrapper for the RL4Sys agent, allowing interaction with the reinforcement
//!   learning model and execution of actions.
//! - **`RL4SysTrajectory`**: Handles the storage and management of action sequences (trajectories).
//! - **`RL4SysAction`**: Represents individual actions taken within the RL environment, including
//!   observation, action, reward, and auxiliary data.
//!
//! ## Using RL4Sys
//!

mod network;
mod orchestration;
mod types;

/// **Development Templates**: Provides base algorithm and application templates for RL4Sys.
/// These templates serve as foundations for extending or customizing RL environments, models,
/// or training strategies.
mod dev_templates;

#[cfg(feature = "python_bindings")]
use crate::bindings::python::network::client::o3_agent::PyRL4SysAgent;
#[cfg(feature = "python_bindings")]
use crate::bindings::python::network::server::o3_training_server::PyTrainingServer;
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_action::PyRL4SysAction;
#[cfg(feature = "python_bindings")]
use crate::bindings::python::o3_config_loader::PyConfigLoader;
#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_trajectory::PyRL4SysTrajectory;
#[cfg(feature = "python_bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python_bindings")]
use pyo3::{Bound, PyResult, pymodule};

/// **Protocol Buffers (Protobuf) for gRPC Communication**
///
/// This module contains Rust code generated from `.proto` files using `tonic::include_proto!`,
/// enabling structured message exchange between RL4Sys components.
#[cfg(feature = "grpc_network")]
mod proto {
    tonic::include_proto!("rl4sys");
}

/// **System Utilities**: Provides helper functions for gRPC communication, model serialization,
/// and configuration resolution. These utilities support seamless inter-module communication.
pub(crate) mod utilities {
    pub mod configuration;
    pub(crate) mod misc_utils;
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network"
    ))]
    pub(crate) mod resolve_server_config;

    pub(crate) mod observability {
        pub(crate) mod logging {
            mod sinks {
                pub(crate) mod console;
                pub(crate) mod file;
            }
            pub(crate) mod builder;
            pub(crate) mod filters;
        }
        pub(crate) mod metrics {
            mod export {
                mod open_telemetry;
                mod prometheus;
            }
            pub(crate) mod definitions;
            pub(crate) mod registry;
        }
    }
}

/// **Python Bindings for RL4Sys**: This module contains the Rust-to-Python bindings,
/// exposing RL4Sys components as Python classes. The `o3_*` modules implement PyO3-compatible
/// wrappers for core structures, enabling smooth Python interaction.
pub(crate) mod bindings {
    pub(crate) mod python {
        #[cfg(any(
            feature = "networks",
            feature = "grpc_network",
            feature = "zmq_network",
            feature = "python_bindings"
        ))]
        #[cfg_attr(bench, visibility = "pub")]
        pub(crate) mod o3_action;
        #[cfg(feature = "python_bindings")]
        #[cfg_attr(bench, visibility = "pub")]
        pub(crate) mod o3_config_loader;
        #[cfg(any(
            feature = "networks",
            feature = "grpc_network",
            feature = "zmq_network",
            feature = "python_bindings"
        ))]
        #[cfg_attr(bench, visibility = "pub")]
        pub(crate) mod o3_trajectory;

        /// **Network Python Wrappers**: Exposes the RL4Sys network components to Python.
        #[cfg(feature = "python_bindings")]
        pub(crate) mod network {
            /// **Client Python Wrappers**: Wraps RL4Sys agents for Python integration.
            pub(crate) mod client {
                pub(crate) mod o3_agent;
            }

            /// **Server Python Wrappers**: Exposes the RL4Sys training server to Python.
            pub(crate) mod server {
                pub(crate) mod o3_training_server;
            }
        }
    }

    #[cfg(feature = "wasm_bindings")]
    pub(crate) mod wasm {
        pub(crate) mod wasm_action;
        pub(crate) mod wasm_config_loader;
        pub(crate) mod wasm_trajectory;
        pub(crate) mod client {
            pub(crate) mod wasm_agent;
        }
    }
}

/// ### RL4Sys Python Module Definition
///
/// This function defines `rl4sys_framework`, the Python module for RL4Sys bindings.
///
/// It registers the following Python classes:
/// - `ConfigLoader`
/// - `TrainingServer`
/// - `RL4SysAgent`
/// - `RL4SysTrajectory`
/// - `RL4SysAction`
///
/// This allows Python users to easily import and use RL4Sys functionalities via:
///
/// ```python
/// from rl4sys_framework import RL4SysAgent, RL4SysTrajectory, RL4SysAction
/// ```
///
#[cfg(feature = "python_bindings")]
#[pymodule(name = "rl4sys_framework")]
fn rl4sys_framework(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Python-accessible classes from the Rust implementation.
    m.add_class::<PyConfigLoader>()?;
    m.add_class::<PyTrainingServer>()?;
    m.add_class::<PyRL4SysAgent>()?;
    m.add_class::<PyRL4SysTrajectory>()?;
    m.add_class::<PyRL4SysAction>()?;

    // Define Python `__all__` to indicate available imports.
    m.add(
        "__all__",
        vec![
            "ConfigLoader",
            "TrainingServer",
            "RL4SysAgent",
            "RL4SysTrajectory",
            "RL4SysAction",
        ],
    )?;

    Ok(())
}