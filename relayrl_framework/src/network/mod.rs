/// **Client Modules**: Handles client-side runtime coordination and actor management.
///
/// The client module provides a comprehensive runtime system for managing RL agents:
/// - `agent`: Public interface for agent implementations and wrappers
/// - `runtime`: Internal runtime system including:
///   - `actor`: Individual agent actor implementations
///   - `coordination`: Manages lifecycle, scaling, metrics, and state across actors
///   - `router`: Message routing between actors and transport layers
///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
pub mod client {
    pub mod agent;
    pub(crate) mod runtime {
        pub(crate) mod actor;
        pub(crate) mod coordination {
            pub(crate) mod coordinator;
            pub(crate) mod lifecycle_manager;
            pub(crate) mod metrics_manager;
            pub(crate) mod scale_manager;
            pub(crate) mod state_manager;
        }
        pub(crate) mod router;
        #[cfg(feature = "networks")]
        pub(crate) mod transport {
            #[cfg(feature = "grpc_network")]
            pub(crate) mod tonic;
            #[cfg(feature = "zmq_network")]
            pub(crate) mod zmq;
        }
    }
}

    /// **Server Modules**: Implements RelayRL training servers and communication channels.
    ///
    /// The server module provides a comprehensive runtime system for managing RL training:
    /// - `training_server`: Public interface for training server implementations
    /// - `runtime`: Internal runtime system including:
    ///   - `coordination`: Manages lifecycle, scaling, metrics, and state for training
    ///   - `python_subprocesses`: Python subprocess management for server interactions
    ///     - `python_algorithm_request`: Manages Python-command-based algorithm interactions
    ///     - `python_training_tensorboard`: Manages TensorBoard integration for training visualization
    ///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
    ///   - `router`: Message routing between workers and transport layers
    ///   - `worker`: Individual training worker implementations
pub mod server {
    pub mod training_server;
    pub(crate) mod runtime {
        pub(crate) mod coordination {
            pub(crate) mod coordinator;
            pub(crate) mod lifecycle_manager;
            pub(crate) mod metrics_manager;
            pub(crate) mod scale_manager;
            pub(crate) mod state_manager;
        }
        pub(crate) mod python_subprocesses {
            pub(crate) mod python_algorithm_request;
            pub(crate) mod python_training_tensorboard;
        }
        pub(crate) mod transport {
            #[cfg(feature = "grpc_network")]
            pub(crate) mod tonic;
            #[cfg(feature = "zmq_network")]
            pub(crate) mod zmq;
        }
        pub(crate) mod router;
        pub(crate) mod worker;
    }
}