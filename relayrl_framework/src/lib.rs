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
pub mod templates;

/// **System Utilities**: Provides helper functions for gRPC communication, model serialization,
/// and configuration resolution. These utilities support seamless inter-module communication.
pub mod utilities {
    pub mod configuration;
    pub(crate) mod misc_utils;
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

/// **Protocol Buffers (Protobuf) for gRPC Communication**
///
/// This module contains Rust code generated from `.proto` files using `tonic::include_proto!`,
/// enabling structured message exchange between RL4Sys components.
#[cfg(feature = "grpc_network")]
mod proto {
    tonic::include_proto!("rl4sys");
}
