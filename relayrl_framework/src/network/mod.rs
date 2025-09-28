/// **Client Modules**: Handles agent implementations and inter-agent communication.
///
/// These modules include different agent communication strategies, such as:
/// - `agent_wrapper`: Provides Python-accessible agent wrappers for easy integration.
/// - `agent_grpc`: Uses gRPC for agent-server interaction.
/// - `agent_zmq`: Uses ZeroMQ for lightweight, high-speed messaging.
pub mod client {
    #[cfg(feature = "grpc_network")]
    pub mod agent_grpc;
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network"
    ))]
    pub mod agent_wrapper;
    #[cfg(feature = "zmq_network")]
    pub mod agent_zmq;
}

/// **Server Modules**: Implements RelayRL training servers and communication channels.
///
/// The training server is responsible for managing reinforcement learning updates,
/// handling incoming trajectories, and updating models. This module includes:
/// - `training_server_wrapper`: Provides utility functions for handling training requests.
/// - `training_grpc`: Implements gRPC-based training server communication.
/// - `training_zmq`: Implements ZeroMQ-based training server communication.
/// - `mod python_subprocesses`: Contains Python subprocess management for server interactions.
///     - `python_training_tensorboard`: Manages TensorBoard integration for training visualization.
///     - `python_channel_request`: Manages Python-command-based server interactions.
pub mod server {
    #[cfg(feature = "grpc_network")]
    pub mod training_grpc;
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network"
    ))]
    pub mod training_server_wrapper;
    #[cfg(feature = "zmq_network")]
    pub mod training_zmq;
    pub(crate) mod python_subprocesses {
        #[cfg(any(
            feature = "networks",
            feature = "grpc_network",
            feature = "zmq_network"
        ))]
        pub(crate) mod python_algorithm_request;
        #[cfg(any(
            feature = "networks",
            feature = "grpc_network",
            feature = "zmq_network"
        ))]
        pub(crate) mod python_training_tensorboard;
    }
}