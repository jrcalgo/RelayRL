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
}