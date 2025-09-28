//! # RelayRLAgent Abstraction
//!
//! This module defines the `RelayRLAgent` abstraction, which serves as a wrapper around different
//! communication-based RL agents in the RelayRL framework. The `RelayRLAgent` is responsible for handling
//! interactions between reinforcement learning models and external systems via **gRPC-based** or **ZMQ-based**
//! communication protocols.
//!
//! ## Overview
//!
//! The RelayRL framework supports two primary communication methods for RL agents:
//!
//! - **gRPC-based agent (`RelayRLAgentGrpc`)**: Uses gRPC for structured communication with the training server,
//!   ideal for distributed reinforcement learning deployments in cloud and HPC environments.
//! - **ZeroMQ-based agent (`RelayRLAgentZmq`)**: Uses ZeroMQ, a high-performance messaging library, optimized for
//!   decentralized, high-speed messaging in local or distributed RL setups.
//!
//! The `RelayRLAgent` abstraction enables seamless switching between these implementations based on user preferences
//! and system configurations.
//!
//! ## Key Responsibilities
//!
//! - **Agent Initialization**: Dynamically constructs either a gRPC-based or ZMQ-based RL agent based on
//!   configuration settings.
//! - **Model Validation**: Ensures that the loaded RL model is correctly formatted by performing a dummy inference
//!   check before deployment.
//! - **Generic Dictionary Conversion**: Converts raw data structures (such as generic dictionaries) into structured
//!   `HashMap<String, RelayRLData>` representations.
//!
//! ## How It Works
//!
//! 1. **Agent Selection**: During initialization, the module determines whether to use a gRPC or ZMQ agent, based on
//!    provided configuration parameters.
//! 2. **Model Validation**: If a model is supplied, it is validated to ensure compatibility with the RelayRL execution pipeline.
//! 3. **Data Conversion Utilities**: Provides functions to transform unstructured dictionaries into structured RelayRLData objects.
//!
//! ## Example Usage
//! ```rust
//! let agent = RelayRLAgent::new(Some(model), Some("config_path".to_string()), Some("grpc".to_string()), None, None, None).await;
//! ```
//! In this example, an RelayRL agent is created using the **gRPC-based** communication protocol.

#[cfg(feature = "grpc_network")]
use crate::network::client::agent_grpc::RelayRLAgentGrpc;
#[cfg(feature = "zmq_network")]
use crate::network::client::agent_zmq::RelayRLAgentZmq;
use crate::sys_utils::config_loader::{ConfigLoader, DEFAULT_CONFIG_CONTENT, DEFAULT_CONFIG_PATH};
use crate::types::action::{RelayRLData, TensorData};
use crate::{get_or_create_config_json_path, resolve_config_json_path};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tch::{CModule, Device, IValue, Kind, Tensor, no_grad};

fn agent_type_to_string(agent: Arc<RelayRLAgent>) -> Option<&'static str> {
    if agent.agent_zmq.is_some() {
        Some("zmq")
    } else if agent.agent_grpc.is_some() {
        Some("grpc")
    } else {
        None
    }
}

/// The RelayRLAgent struct serves as a unified interface that wraps either a gRPC or a ZMQ agent.
///
/// Depending on the configuration, one of the agent implementations is instantiated. The other remains None.
pub struct RelayRLAgent {
    /// An optional ZMQ-based RelayRL agent.
    #[cfg(any(feature = "networks", feature = "zmq_network"))]
    pub agent_zmq: Option<RelayRLAgentZmq>,
    /// An optional gRPC-based RelayRL agent.
    #[cfg(any(feature = "networks", feature = "grpc_network"))]
    pub agent_grpc: Option<RelayRLAgentGrpc>,
}

/// Validates a TorchScript model (CModule) by running a forward pass with a dummy tensor.
///
/// This function checks that:
/// 1. The model's forward pass returns a tuple (IValue::Tuple) with exactly two elements.
/// 2. The first element is a Tensor.
/// 3. The second element is a dictionary (IValue::GenericDict) that could be empty.
///
/// # Arguments
///
/// * `model` - A reference to the TorchScript model (CModule) to be validated.
/// * `input_dim` - The dimensionality of the input vector.
pub fn validate_model(model: &CModule) {
    // check if input_dim is a model attribute.
    let input_dim: IValue = model
        .method_is::<IValue>("get_input_dim", &[])
        .expect("Failed to get input dimension");
    let input_dim_usize: usize = if let IValue::Int(dim) = input_dim {
        if dim < 0 {
            panic!("Input dimension must be non-negative");
        }
        usize::try_from(dim).expect("Input dimension too large")
    } else {
        panic!("Input dimension must be an integer");
    };

    // check if output_dim is a model attribute.
    let output_dim: IValue = model
        .method_is::<IValue>("get_output_dim", &[])
        .expect("Failed to get output dimension");
    let output_dim_usize: usize = if let IValue::Int(dim) = output_dim {
        if dim < 0 {
            panic!("Output dimension must be non-negative");
        }
        usize::try_from(dim).expect("Output dimension too large")
    } else {
        panic!("Output dimension must be an integer");
    };

    // Create a dummy input vector filled with zeros.
    let input_test_vec: Vec<f64> = vec![0.0; input_dim_usize];
    // Convert the vector into a tensor and reshape it to have a batch dimension.
    let input_test_tensor: Tensor = Tensor::f_from_slice(&input_test_vec)
        .expect("Failed to convert slice to tensor")
        .reshape([1, input_dim_usize as i64]);
    // Create a dummy output vector filled with zeros.
    let output_test_vec: Vec<f64> = vec![0.0; output_dim_usize];
    // Convert the vector into a tensor and reshape it to have a batch dimension.
    let output_test_tensor: Tensor = Tensor::f_from_slice(&output_test_vec)
        .expect("Failed to convert slice to tensor")
        .reshape([1, output_dim_usize as i64]);
    // Convert the tensor to a device tensor.
    let obs_tensor: Tensor = input_test_tensor.to_device(Device::Cpu).contiguous();
    let mask_tensor: Tensor = output_test_tensor.to_device(Device::Cpu).contiguous();
    // Construct IValue tensor input vec
    let obs_ivalue = IValue::Tensor(obs_tensor.to_kind(Kind::Float));
    let mask_ivalue = IValue::Tensor(mask_tensor.to_kind(Kind::Float));
    let test_input: Vec<IValue> = vec![obs_ivalue, mask_ivalue];

    // Run the forward pass (step).
    let output: IValue = no_grad(|| model.method_is::<IValue>("step", &test_input))
        .expect("Failed to run forward 'step' pass");

    // Validate that the output is a tuple.
    match output {
        IValue::Tuple(ref values) => {
            // Assert that the tuple has exactly two elements.
            assert_eq!(
                values.len(),
                2,
                "Model forward must return a tuple of length 2"
            );

            // Check that the first element is a Tensor.
            if let IValue::Tensor(ref _tensor) = values[0] {
                // Optionally: Add additional checks for tensor shape or content here.
            } else {
                panic!("First element of tuple must be a Tensor");
            }

            // Check that the second element is a dictionary (GenericDict).
            if let IValue::GenericDict(ref dict) = values[1] {
                assert!(
                    !dict.is_empty(),
                    "Second element of tuple must be a non-empty dictionary"
                );
            } else {
                panic!("Second element of tuple must be a Dictionary");
            }
        }
        _ => panic!("Model forward must return a tuple"),
    }
}

/// Converts a generic dictionary (represented as a Vec of (IValue, IValue)) into a HashMap
/// with String keys and RelayRLData values.
///
/// The function iterates over each key-value pair in the generic dictionary. If the key is a
/// string and the value is one of the supported types (Tensor, Int, Double), it converts the value
/// into the corresponding RelayRLData variant. For tensors, the value is first converted to a Float
/// tensor before being transformed into TensorData.
///
/// # Arguments
///
/// * `dict` - A reference to a vector of (IValue, IValue) tuples representing a generic dictionary.
///
/// # Returns
///
/// An Option containing a HashMap with String keys and RelayRLData values if conversion is successful;
/// otherwise, None.
pub fn convert_generic_dict(dict: &Vec<(IValue, IValue)>) -> Option<HashMap<String, RelayRLData>> {
    let mut map: HashMap<String, RelayRLData> = HashMap::new();

    for (k, v) in dict {
        if let IValue::String(s) = k {
            if let IValue::Tensor(tensor) = v {
                map.insert(
                    s.clone(),
                    RelayRLData::Tensor(
                        TensorData::try_from(&tensor.to_kind(Kind::Float))
                            .expect("Failed to convert tensor to TensorData"),
                    ),
                );
            } else if let IValue::Int(i) = v {
                map.insert(
                    s.clone(),
                    RelayRLData::Int((*i).try_into().expect("Failed to convert int to i32")),
                );
            } else if let IValue::Double(f) = v {
                map.insert(s.clone(), RelayRLData::Double(*f));
            }
        }
    }

    Some(map)
}

impl RelayRLAgent {
    /// Constructs a new RelayRLAgent instance.
    ///
    /// Based on the provided `server_type` parameter, this function instantiates either a gRPC-based agent
    /// or a ZMQ-based agent. If no server type is specified, it defaults to using the ZMQ-based agent.
    ///
    /// # Arguments
    ///
    /// * `model` - An optional TorchScript model (CModule) that the agent will use for inference.
    /// * `config_path` - An optional path to the configuration file.
    /// * `server_type` - An optional string specifying the type of server to use ("grpc" or "zmq").
    /// * `training_port` - An optional string specifying the training port.
    /// * `training_prefix` - An optional string for the training server prefix.
    /// * `training_host` - An optional string for the training server host.
    ///
    /// # Returns
    ///
    /// An Arc-wrapped RelayRLAgent instance.
    pub async fn new(
        model: Option<CModule>,
        config_path: Option<PathBuf>,
        server_type: Option<String>,
        training_prefix: Option<String>,
        training_port: Option<String>,
        training_host: Option<String>,
    ) -> RelayRLAgent {
        let config_path: Option<PathBuf> = resolve_config_json_path!(config_path);

        let training_server: String;
        let config_path_clone: Option<PathBuf> = config_path.clone();
        {
            let config: ConfigLoader = ConfigLoader::new(None, config_path_clone);

            // Construct the training server address.
            let prefix: String = training_prefix.unwrap_or(config.train_server.prefix);
            let host: String = training_host.unwrap_or(config.train_server.host);
            let port: String = training_port.unwrap_or(config.train_server.port);
            training_server = format!("{}{}:{}", prefix, host, port);
        }

        // Construct the agent based on the specified server type.
        // If no server type is provided, default to using the ZMQ-based agent.
        // Return the constructed agent.
        match server_type {
            Some(_) => {
                let server_type_str: String =
                    server_type.expect("Server type is None").to_lowercase();
                if server_type_str == "grpc" {
                    new_grpc_agent(model, config_path, Some(training_server)).await
                } else if server_type_str == "zmq" {
                    new_zmq_agent(model, config_path, Some(training_server)).await
                } else {
                    panic!("[RelayRLAgent - new] Server type unavailable: Input 'zmq' or 'grpc'")
                }
            }
            None => new_zmq_agent(model, config_path, Some(training_server)).await,
        }
    }

    pub async fn restart_agent(
        self,
        training_server_address: Option<String>,
    ) -> Option<Vec<Result<(), Box<dyn std::error::Error>>>> {
        match (self.agent_zmq, self.agent_grpc) {
            (Some(mut zmq_agent), _) => {
                Some(zmq_agent.restart_agent(training_server_address).await)
            }
            (_, Some(mut grpc_agent)) => {
                Some(grpc_agent.restart_agent(training_server_address).await)
            }
            _ => {
                eprintln!("Agent instance not available");
                None
            }
        }
    }

    pub async fn enable_agent(
        self,
        training_server_address: Option<String>,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        match (self.agent_zmq, self.agent_grpc) {
            (Some(mut zmq_agent), _) => Some(zmq_agent.enable_agent(training_server_address).await),
            (_, Some(mut grpc_agent)) => {
                Some(grpc_agent.enable_agent(training_server_address).await)
            }
            _ => {
                eprintln!("Agent instance not available");
                None
            }
        }
    }

    pub async fn disable_agent(self) -> Option<Result<(), Box<dyn std::error::Error>>> {
        match (self.agent_zmq, self.agent_grpc) {
            (Some(mut zmq_agent), _) => Some(zmq_agent.disable_agent().await),
            (_, Some(mut grpc_agent)) => Some(grpc_agent.disable_agent().await),
            _ => {
                eprintln!("Agent instance not available");
                None
            }
        }
    }
}

#[cfg(feature = "grpc_network")]
async fn new_grpc_agent(
    model: Option<CModule>,
    config_path: Option<PathBuf>,
    training_server: Option<String>,
) -> RelayRLAgent {
    RelayRLAgent {
        #[cfg(feature = "zmq_network")]
        agent_zmq: None,
        agent_grpc: Some(RelayRLAgentGrpc::init_agent(model, config_path, training_server).await),
    }
}

#[cfg(not(feature = "grpc_network"))]
async fn new_grpc_agent(
    _model: Option<CModule>,
    _config_path: Option<PathBuf>,
    _training_server: Option<String>,
) -> RelayRLAgent {
    panic!("[RelayRLAgent - new] gRPC feature not enabled")
}

#[cfg(feature = "zmq_network")]
async fn new_zmq_agent(
    model: Option<CModule>,
    config_path: Option<PathBuf>,
    training_server: Option<String>,
) -> RelayRLAgent {
    RelayRLAgent {
        agent_zmq: Some(RelayRLAgentZmq::init_agent(
            model,
            config_path,
            training_server,
        )),
        #[cfg(feature = "grpc_network")]
        agent_grpc: None,
    }
}

#[cfg(not(feature = "zmq_network"))]
async fn new_zmq_agent(
    _model: Option<CModule>,
    _config_path: Option<PathBuf>,
    _training_server: Option<String>,
) -> RelayRLAgent {
    panic!("[RelayRLAgent - new] ZMQ feature not enabled")
}
