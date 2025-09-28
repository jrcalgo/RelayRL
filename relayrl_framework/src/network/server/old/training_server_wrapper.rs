//! # Training Server Abstraction for RelayRL
//!
//! This module defines the `TrainingServer` abstraction for the RelayRL framework, which serves as the
//! backbone for reinforcement learning training and model management. The training server is responsible
//! for receiving agent trajectories, managing model updates, handling hyperparameters, and coordinating
//! multi-actor training settings in distributed environments.
//!
//! ## Overview
//!
//! RelayRL supports both **gRPC-based** and **ZeroMQ (ZMQ)-based** communication protocols, allowing users
//! to select the most suitable inter-process communication method for their specific use case. This module
//! dynamically configures and instantiates the appropriate server type based on user input.
//!
//! The key responsibilities of the `TrainingServer` module include:
//!
//! - **Hyperparameter Parsing**: Extracts and processes hyperparameters from user input, supporting both
//!   map-based and argument-string-based formats.
//! - **Multi-Actor Configuration**: Determines whether multi-actor training is enabled and manages actor states.
//! - **Server Type Selection**: Instantiates either a **gRPC-based** or **ZMQ-based** training server, depending
//!   on user preferences or configuration settings.
//! - **Training Server Initialization**: Loads configuration files, sets up communication endpoints, and launches
//!   the training process.
//!
//! ## Supported Server Types
//!
//! - **gRPC Server (`TrainingServerGrpc`)**: Enables robust, structured communication between agents and the server
//!   using gRPC, supporting large-scale deployments.
//! - **ZMQ Server (`TrainingServerZmq`)**: Provides lightweight and high-speed messaging using ZeroMQ, particularly
//!   useful for low-latency, decentralized communication in high-performance computing (HPC) environments.
//!
//! ## Configuration and Initialization Flow
//!
//! 1. **Hyperparameter Parsing**: Extracts hyperparameters from user input and converts them into a HashMap format.
//! 2. **Server Address Resolution**: Determines the training server's host and port, using either user-provided values
//!    or defaults from the configuration file.
//! 3. **Algorithm and Environment Setup**: Resolves paths for the reinforcement learning algorithm and environment files.
//! 4. **Server Selection & Instantiation**:
//!     - If `server_type = "grpc"`, initializes `TrainingServerGrpc`.
//!     - If `server_type = "zmq"`, initializes `TrainingServerZmq`.
//!     - If no server type is specified, defaults to **ZMQ**.
//! 5. **Multi-Actor Support**: Configures multi-actor training if enabled, setting up actor identifiers and managing
//!    concurrency constraints.

#[cfg(any(feature = "networks", feature = "grpc_network"))]
use crate::network::server::training_grpc::TrainingServerGrpc;
#[cfg(any(feature = "networks", feature = "zmq_network"))]
use crate::network::server::training_zmq::TrainingServerZmq;
use crate::sys_utils::config_loader::{
    ConfigLoader, DEFAULT_CONFIG_CONTENT, DEFAULT_CONFIG_PATH, ServerParams,
};
use crate::{get_or_create_config_json_path, resolve_config_json_path};

use crate::network::server::python_subprocesses::python_algorithm_request::{
    PythonAlgorithmCommand, PythonAlgorithmRequest,
};
use crate::network::server::python_subprocesses::python_training_tensorboard::PythonTrainingTensorboard;
use std::collections::HashMap;
use std::fs;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc::Sender as TokioSender;
use tokio::sync::{RwLock as TokioRwLock, RwLockReadGuard, RwLockWriteGuard};

const ALGORITHMS_PATH: &str = "src/native/python/algorithms";

pub(crate) async fn resolve_new_training_server_address(
    old_training_server_address: &str,
    new_training_server_address: Option<String>,
) -> String {
    match new_training_server_address {
        Some(new_address) => {
            if new_address.eq(&old_training_server_address) {
                old_training_server_address
                    .parse()
                    .expect("Failed to parse old training server address")
            } else {
                new_address
            }
        }
        None => {
            eprintln!("No new training server address provided, using original address.");
            old_training_server_address
                .parse()
                .expect("Failed to parse old training server address")
        }
    }
}

pub(crate) fn server_type_to_string(
    server: &RwLockReadGuard<TrainingServer>,
) -> Option<&'static str> {
    if server.ts_zmq.is_some() {
        Some("zmq")
    } else if server.ts_grpc.is_some() {
        Some("grpc")
    } else {
        eprintln!("Training server instance not active.");
        None
    }
}

/// Parses hyperparameter arguments into a HashMap.
///
/// The function accepts an optional `Hyperparams` enum value, which may be provided as either
/// a map or a vector of argument strings. It returns a HashMap mapping hyperparameter keys to
/// their corresponding string values.
///
/// # Arguments
///
/// * `hyperparams` - An optional [Hyperparams] enum that contains either a map or vector of strings.
///
/// # Returns
///
/// A [HashMap] where the keys and values are both strings.
pub fn parse_args(hyperparams: &Option<Hyperparams>) -> HashMap<String, String> {
    let mut hyperparams_map: HashMap<String, String> = HashMap::new();

    match hyperparams {
        Some(Hyperparams::Map(map)) => {
            for (key, value) in map {
                hyperparams_map.insert(key.to_string(), value.to_string());
            }
        }
        Some(Hyperparams::Args(args)) => {
            for arg in args {
                // Split the argument string on '=' or ' ' if possible.
                let split: Vec<&str> = if arg.contains("=") {
                    arg.split('=').collect()
                } else if arg.contains(' ') {
                    arg.split(' ').collect()
                } else {
                    panic!(
                        "[TrainingServer - new] Invalid hyperparameter argument: {}",
                        arg
                    );
                };
                // Ensure exactly two parts are obtained: key and value.
                if split.len() != 2 {
                    panic!(
                        "[TrainingServer - new] Invalid hyperparameter argument: {}",
                        arg
                    );
                }
                hyperparams_map.insert(split[0].to_string(), split[1].to_string());
            }
        }
        None => {}
    }

    hyperparams_map
}

/// MultiactorParams struct is used to store information about multiple actors
/// including whether multiactor training is enabled, the current count of actors,
/// and their respective identifiers.
pub struct MultiactorParams {
    pub(crate) multiactor: bool,
    pub(crate) current_actor_count: u32,
    pub(crate) agent_ids: Vec<String>,
}

/// Struct wrapping essential values and instances of Python interface instances.
///
/// Extend this with additional python subprocesses variables if necessary.
pub struct PythonSubprocesses {
    /// Arguments passed to `training_tensorboard.py` script for initialization.
    pub ptt_args: Vec<String>,
    /// Reference to the `PythonTrainingTensorboard` interface used to communicate with tensorboard script.
    pub ptt_obj: Option<Arc<PythonTrainingTensorboard>>,
    /// Arguments passed to `python_algorithm_reply.py` script for initialization.
    pub par_args: Vec<String>,
    /// Reference to the `PythonAlgorithmRequest` interface used to communication with pytorch scripts.
    pub par_obj: Option<Arc<PythonAlgorithmRequest>>,
    /// Channel sender used to dispatch commands to the PythonAlgorithmRequest interface.
    pub command_sender: Option<TokioSender<PythonAlgorithmCommand>>,
    /// Shared atomic flag indicating whether the python algorithm script is active.
    pub algorithm_pyscript_status: Arc<AtomicBool>,
}

/// Hyperparams enum represents hyperparameter inputs which can be provided either as a map
/// or as a list of argument strings.
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(HashMap<String, String>),
    Args(Vec<String>),
}

/// The TrainingServer struct is the main abstraction that encapsulates the training server
/// functionality for RelayRL. It wraps either a ZMQ-based training server or a gRPC-based training
/// server (or both) and provides a unified interface.
///
/// # Fields
///
/// * `ts_zmq` - An optional Arc-wrapped TokioRwLock for the ZMQ-based training server.
/// * `ts_grpc` - An optional Arc-wrapped TokioRwLock for the gRPC-based training server.
pub struct TrainingServer {
    /// An optional ZMQ-based training server.
    #[cfg(any(feature = "networks", feature = "zmq_network"))]
    pub ts_zmq: Option<Arc<TokioRwLock<TrainingServerZmq>>>,
    /// An optional gRPC-based training server.
    #[cfg(any(feature = "networks", feature = "grpc_network"))]
    pub ts_grpc: Option<Arc<TokioRwLock<TrainingServerGrpc>>>,
}

impl TrainingServer {
    /// Creates a new TrainingServer instance.
    ///
    /// This asynchronous function configures the training server based on provided parameters,
    /// parses hyperparameters, resolves configuration settings, and instantiates either a ZMQ-based
    /// or gRPC-based training server. The function prints various configuration details for debugging.
    ///
    /// # Arguments
    ///
    /// * `algorithm_name` - The name of the algorithm (e.g., "PPO", "DQN") in uppercase.
    /// * `obs_dim` - The observation dimension hyperparameter.
    /// * `act_dim` - The action dimension hyperparameter.
    /// * `buf_size` - The buffer size hyperparameter.
    /// * `tensorboard` - A flag indicating whether Tensorboard integration is enabled.
    /// * `multiactor` - A flag indicating whether multiactor training is enabled.
    /// * `env_dir` - An optional directory for the environment.
    /// * `algorithm_dir` - The directory where algorithm code is stored.
    /// * `config_path` - An optional path to the configuration file.
    /// * `hyperparams` - Optional hyperparameters provided as a [Hyperparams] enum.
    /// * `server_type` - Optional server type string ("grpc" or "zmq"); if not provided, defaults to ZMQ.
    /// * `training_prefix` - Optional prefix for the ZMQ training server address.
    /// * `training_host` - Optional host for the ZMQ training server.
    /// * `training_port` - Optional port for the ZMQ training server.
    ///
    /// # Returns
    ///
    /// An Arc-wrapped TokioRwLock containing the new TrainingServer instance.
    pub async fn new(
        algorithm_name: String,
        obs_dim: i32,
        act_dim: i32,
        buf_size: i32,
        tensorboard: bool,
        multiactor: bool,
        env_dir: Option<String>,
        algorithm_dir: Option<String>,
        config_path: Option<PathBuf>,
        hyperparams: Option<Hyperparams>,
        server_type: Option<String>,
        training_prefix: Option<String>,
        training_host: Option<String>,
        training_port: Option<String>,
    ) -> Arc<TokioRwLock<TrainingServer>> {
        // Resolve config path
        let config_path: Option<PathBuf> = resolve_config_json_path!(config_path);

        // Load configuration using ConfigLoader.
        let config: Arc<ConfigLoader> = Arc::new(ConfigLoader::new(
            Some(algorithm_name.to_uppercase()),
            config_path.clone(),
        ));
        println!(
            "[TrainingServer - new] Resolved configuration path: {:?}",
            config_path.clone()
        );

        // Parse hyperparameters into a HashMap.
        let mut hyperparams_map: HashMap<String, String> = parse_args(&hyperparams);
        hyperparams_map.insert(
            "env_dir".to_string(),
            env_dir
                .clone()
                .unwrap_or_else(|| "default_env_dir".to_string()),
        );
        hyperparams_map.insert("obs_dim".to_string(), obs_dim.to_string());
        hyperparams_map.insert("act_dim".to_string(), act_dim.to_string());
        hyperparams_map.insert("buf_size".to_string(), buf_size.to_string());

        // Resolve the algorithm directory; default to ALGORITHMS_PATH if empty.
        let resolved_algorithm_dir: String = match algorithm_dir {
            Some(dir) => {
                if dir.is_empty() {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join(ALGORITHMS_PATH)
                        .to_str()
                        .expect("Failed to convert algorithm path to &str")
                        .to_string()
                } else {
                    dir
                }
            }
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(ALGORITHMS_PATH)
                .to_str()
                .expect("Failed to convert algorithm path to &str")
                .to_string(),
        };
        println!(
            "[TrainingServer - new] Resolved algorithm directory: {}",
            resolved_algorithm_dir
        );

        let server_type_str: String = server_type
            .clone()
            .expect("server_type is None")
            .to_lowercase();

        // Determine the training server address either from the provided parameters or configuration.
        let resolved_training_server_address: String =
            if let (Some(prefix), Some(host), Some(port)) =
                (training_prefix, training_host, training_port)
            {
                if server_type_str == "grpc" {
                    format!("{}:{}", host, port)
                } else {
                    // ZMQ
                    format!("{}{}:{}", prefix, host, port)
                }
            } else {
                let train_server: &ServerParams = config.get_train_server();
                if server_type_str == "grpc" {
                    format!("{}:{}", train_server.host, train_server.port)
                } else {
                    // ZMQ
                    format!(
                        "{}{}:{}",
                        train_server.prefix, train_server.host, train_server.port
                    )
                }
            };

        // Instantiate the appropriate TrainingServer based on the server_type argument.
        {
            match server_type {
                Some(_) => {
                    let server_type_str: String =
                        server_type.expect("server_type is None").to_lowercase();
                    if server_type_str == "grpc" {
                        new_grpc_training_server(
                            algorithm_name,
                            resolved_algorithm_dir,
                            tensorboard,
                            multiactor,
                            env_dir,
                            config_path,
                            Some(hyperparams_map),
                            Some(resolved_training_server_address),
                        )
                        .await
                    } else if server_type_str == "zmq" {
                        new_zmq_training_server(
                            algorithm_name,
                            resolved_algorithm_dir,
                            tensorboard,
                            multiactor,
                            env_dir,
                            config_path,
                            Some(hyperparams_map),
                            Some(resolved_training_server_address),
                        )
                        .await
                    } else {
                        panic!(
                            "[TrainingServer - new] Server type unavailable: Input 'zmq' or 'grpc'"
                        )
                    }
                }
                None => {
                    new_zmq_training_server(
                        algorithm_name,
                        resolved_algorithm_dir,
                        tensorboard,
                        multiactor,
                        env_dir,
                        config_path,
                        Some(hyperparams_map),
                        Some(resolved_training_server_address),
                    )
                    .await
                }
            }
        }
    }

    pub async fn restart_server(
        self,
        training_server_address: Option<String>,
    ) -> Option<Vec<Result<(), Box<dyn std::error::Error>>>> {
        match (self.ts_zmq, self.ts_grpc) {
            (Some(ts_zmq), _) => {
                let mut zmq_server = ts_zmq.write().await;
                Some(zmq_server.restart_server(training_server_address).await)
            }
            (_, Some(ts_grpc)) => {
                let mut grpc_server = ts_grpc.write().await;
                Some(grpc_server.restart_server(training_server_address).await)
            }
            _ => {
                eprintln!("Training server instance not available.");
                None
            }
        }
    }

    pub async fn enable_server(
        self,
        training_server_address: Option<String>,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        match (self.ts_zmq, self.ts_grpc) {
            (Some(ts_zmq), _) => {
                let mut zmq_server: RwLockWriteGuard<TrainingServerZmq> = ts_zmq.write().await;
                let enable_result = zmq_server.enable_server(training_server_address).await;
                Some(enable_result)
            }
            (_, Some(ts_grpc)) => {
                let mut grpc_server: RwLockWriteGuard<TrainingServerGrpc> = ts_grpc.write().await;
                let enable_result = grpc_server.enable_server(training_server_address).await;
                Some(enable_result)
            }
            _ => {
                eprintln!("Training server instance not available.");
                None
            }
        }
    }

    /// Disable the gRPC/ZMQ server.
    pub async fn disable_server(self) -> Option<Result<(), Box<dyn std::error::Error>>> {
        match (self.ts_zmq, self.ts_grpc) {
            (Some(ts_zmq), _) => {
                let mut zmq_server: RwLockWriteGuard<TrainingServerZmq> = ts_zmq.write().await;
                let disable_result = zmq_server.disable_server().await;
                Some(disable_result)
            }
            (_, Some(ts_grpc)) => {
                let mut grpc_server: RwLockWriteGuard<TrainingServerGrpc> = ts_grpc.write().await;
                let disable_result = grpc_server.disable_server().await;
                Some(disable_result)
            }
            _ => {
                eprintln!("Training server instance not available.");
                None
            }
        }
    }
}

#[cfg(feature = "grpc_network")]
async fn new_grpc_training_server(
    algorithm_name: String,
    algorithm_dir: String,
    tensorboard: bool,
    multiactor: bool,
    env_dir: Option<String>,
    config_path: Option<PathBuf>,
    hyperparams: Option<HashMap<String, String>>,
    training_server_address: Option<String>,
) -> Arc<TokioRwLock<TrainingServer>> {
    Arc::new(TokioRwLock::new(TrainingServer {
        #[cfg(feature = "zmq_network")]
        ts_zmq: None,
        ts_grpc: Some(
            TrainingServerGrpc::init_server(
                training_server_address.expect("training server address is None"),
                algorithm_name,
                algorithm_dir,
                tensorboard,
                multiactor,
                hyperparams,
                env_dir,
                config_path,
            )
            .await,
        ),
    }))
}

#[cfg(not(feature = "grpc_network"))]
async fn new_grpc_training_server(
    _algorithm_name: String,
    _algorithm_dir: String,
    _tensorboard: bool,
    _multiactor: bool,
    _env_dir: Option<String>,
    _config_path: Option<&str>,
    _hyperparams: Option<HashMap<String, String>>,
    _training_server_address: Option<String>,
) -> Arc<TokioRwLock<TrainingServer>> {
    panic!("[TrainingServer - new] gRPC feature not enabled.");
}

#[cfg(feature = "zmq_network")]
async fn new_zmq_training_server(
    algorithm_name: String,
    algorithm_dir: String,
    tensorboard: bool,
    multiactor: bool,
    env_dir: Option<String>,
    config_path: Option<PathBuf>,
    hyperparams: Option<HashMap<String, String>>,
    training_server_address: Option<String>,
) -> Arc<TokioRwLock<TrainingServer>> {
    Arc::new(TokioRwLock::new(TrainingServer {
        ts_zmq: Some(
            TrainingServerZmq::init_server(
                algorithm_name,
                algorithm_dir,
                tensorboard,
                multiactor,
                env_dir,
                config_path,
                hyperparams,
                training_server_address,
            )
            .await,
        ),
        #[cfg(feature = "grpc_network")]
        ts_grpc: None,
    }))
}

#[cfg(not(feature = "zmq_network"))]
async fn new_zmq_training_server(
    _algorithm_name: String,
    _algorithm_dir: String,
    _tensorboard: bool,
    _multiactor: bool,
    _env_dir: Option<String>,
    _config_path: Option<&str>,
    _hyperparams: Option<HashMap<String, String>>,
    _training_server_address: Option<String>,
) -> Arc<TokioRwLock<TrainingServer>> {
    panic!("[TrainingServer - new] ZMQ feature not enabled.");
}
