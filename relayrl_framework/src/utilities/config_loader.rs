//! This module provides configuration loading and parsing for the RL4Sys framework.
//! It reads a JSON configuration file, deserializes it into Rust structs, and provides helper
//! functions to retrieve various configuration parameters such as algorithm settings, server
//! addresses, tensorboard parameters, and model paths.

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::{fs, fs::File, io::Read, path::PathBuf};

use crate::get_or_create_config_json_path;

#[macro_use]
pub mod config_macros {
    /// Resolves config json file between argument and default value.
    #[macro_export]
    macro_rules! resolve_config_json_path {
        ($path: expr) => {
            match $path {
                Some(p) => get_or_create_config_json_path!(p.clone()),
                None => DEFAULT_CONFIG_PATH.clone(),
            }
        };
        ($path: literal) => {
            get_or_create_config_json_path!(std::path::PathBuf::from($path))
        };
    }

    /// Will write config file if not found in provided path.
    /// Reads file if found, writes new file if not
    #[macro_export]
    macro_rules! get_or_create_config_json_path {
        ($path: expr) => {
            if $path.exists() {
                println!(
                    "[ConfigLoader - load_config] Found config.json in current directory: {:?}",
                    $path
                );
                Some($path)
            } else {
                match fs::write($path, DEFAULT_CONFIG_CONTENT).await {
                    Ok(_) => {
                        println!(
                            "[ConfigLoader - load_config] Created new config at: {:?}",
                            $path
                        );
                        Some($path)
                    }
                    Err(e) => {
                        eprintln!(
                            "[ConfigLoader - load_config] Failed to create config file: {}",
                            e
                        );
                        None
                    }
                }
            }
        };
    }
}

/// The default configuration file path, loaded lazily at runtime.
/// If not overridden, the configuration will be retrieved or created in the cwd.
pub static DEFAULT_CONFIG_PATH: Lazy<Option<PathBuf>> =
    Lazy::new(|| get_or_create_config_json_path!(PathBuf::from("rl4sys_config.json")));

pub const DEFAULT_CONFIG_CONTENT: &str = r#"{
    "algorithms": {
        "C51": {
            "batch_size": 64,
            "act_dim": 4,
            "seed": 0,
            "traj_per_epoch": 5,
            "n_atoms": 51,
            "v_min": -500,
            "v_max": 1000,
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.999,
            "train_update_freq": 4,
            "target_update_freq": 50,
            "q_lr": 1e-5,
            "train_q_iters": 50
        },
        "DDPG": {
            "seed": 1,
            "gamma": 0.99,
            "tau": 1e-2,
            "learning_rate": 3e-3,
            "batch_size": 128,
            "buffer_size": 50000,
            "learning_starts": 128,
            "policy_frequency": 1,
            "noise_scale": 0.1,
            "train_iters": 50
        },
        "DQN": {
            "batch_size": 32,
            "seed": 0,
            "traj_per_epoch": 3,
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.001,
            "train_update_freq": 8,
            "q_lr": 5e-4,
            "train_q_iters": 80
        },
        "PPO": {
            "seed": 0,
            "traj_per_epoch": 1,
            "clip_ratio": 0.1,
            "gamma": 0.99,
            "lam": 0.97,
            "pi_lr": 3e-4,
            "vf_lr": 3e-4,
            "train_pi_iters": 40,
            "train_v_iters": 40,
            "target_kl": 0.01
        },
        "REINFORCE": {
            "discrete": true,
            "with_vf_baseline": false,
            "seed": 1,
            "traj_per_epoch": 8,
            "gamma": 0.98,
            "lam": 0.97,
            "pi_lr": 3e-4,
            "vf_lr": 1e-3,
            "train_vf_iters": 80
        },
        "RPO": {
            "seed": 0,
            "learning_rate": 3e-4,
            "num_steps": 2048,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "rpo_alpha": 0.5,
            "update_epochs": 10,
            "num_minibatches": 32,
            "anneal_lr": true,
            "clip_vloss": true,
            "target_kl": null,
            "total_timesteps_anneal_lr": 1000000
        },
        "SAC": {
            "discrete": true,
            "adaptive_alpha": false,
            "act_dim": 4,
            "batch_size": 128,
            "seed": 0,
            "traj_per_epoch": 10,
            "log_std_min": -20,
            "log_std_max": 2,
            "gamma": 0.99,
            "polyak": 1e-2,
            "alpha": 0.1,
            "lr": 5e-4,
            "clip_grad_norm": 1,
            "train_update_freq": 8,
            "train_iters": 50
        },
        "TD3": {
            "seed": 1,
            "gamma": 0.99,
            "tau": 0.005,
            "learning_rate": 3e-4,
            "batch_size": 128,
            "buffer_size": 50000,
            "exploration_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "learning_starts": 25000,
            "policy_frequency": 2
        }
    },
    "grpc_idle_timeout": 30,
    "max_traj_length": 1000,
    "model_paths": {
        "client_model": "client_model.pt",
        "server_model": "server_model.pt"
    },
    "server": {
        "_comment": "gRPC uses only this address (prefix is unused).",
        "training_server": {
            "prefix": "tcp://",
            "host": "127.0.0.1",
            "port": "50051"
        },
        "trajectory_server": {
            "prefix": "tcp://",
            "host": "127.0.0.1",
            "port": "7776"
        },
        "agent_listener": {
            "prefix": "tcp://",
            "host": "127.0.0.1",
            "port": "7777"
        }
    },
    "tensorboard": {
        "training_tensorboard": {
            "_comment1": "Runs `tensorboard --logdir /logs` in cwd on start up of server.",
            "launch_tb_on_startup": true,
            "_comment2": "scalar tags can be any column header from `progress.txt` files.",
            "scalar_tags": "AverageEpRet;LossQ",
            "global_step_tag": "Epoch"
        }
    }
}"#;

/// The root configuration structure for RL4Sys.
///
/// This struct contains optional configuration sections for algorithms,
/// server settings, tensorboard configuration, model paths, maximum trajectory length,
/// and gRPC idle timeout.
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "algorithms")]
    pub algorithms: Option<AlgorithmConfig>,
    pub server: Option<ServerConfig>,
    pub tensorboard: Option<TensorboardConfig>,
    pub model_paths: Option<ModelPaths>,
    pub max_traj_length: Option<u32>,
    pub grpc_idle_timeout: Option<u32>,
}

/// Configuration parameters for various algorithms.
///
/// Each field is optional and holds algorithm-specific parameters.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AlgorithmConfig {
    #[serde(rename = "C51")]
    pub c51: Option<C51Params>,
    #[serde(rename = "DDPG")]
    pub ddpg: Option<DDPGParams>,
    #[serde(rename = "DQN")]
    pub dqn: Option<DQNParams>,
    #[serde(rename = "PPO")]
    pub ppo: Option<PPOParams>,
    #[serde(rename = "REINFORCE")]
    pub reinforce: Option<REINFORCEParams>,
    #[serde(rename = "RPO")]
    pub rpo: Option<RPOParams>,
    #[serde(rename = "SAC")]
    pub sac: Option<SACParams>,
    #[serde(rename = "TD3")]
    pub td3: Option<TD3Params>,
    // Add other fields depending on the algorithm
}

/// An enum representing loaded algorithm parameters.
/// Each variant corresponds to one algorithm's parameter struct.
#[derive(Debug, Clone)]
pub enum LoadedAlgorithmParams {
    C51(C51Params),
    DDPG(DDPGParams),
    DQN(DQNParams),
    PPO(PPOParams),
    REINFORCE(REINFORCEParams),
    SAC(SACParams),
    TD3(TD3Params),
}

/// Parameters for the C51 algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct C51Params {
    pub batch_size: u32,
    pub act_dim: u32,
    pub seed: u32,
    pub traj_per_epoch: u32,
    pub n_atoms: u32,
    pub v_min: f32,
    pub v_max: f32,
    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
    pub train_update_freq: u32,
    pub target_update_freq: u32,
    pub q_lr: f32,
    pub train_q_iters: u32,
}

/// Parameters for the DDPG algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DDPGParams {
    pub seed: u32,
    pub gamma: f32,
    pub tau: f32,
    pub learning_rate: f32,
    pub batch_size: u32,
    pub buffer_size: u32,
    pub learning_starts: u32,
    pub policy_frequency: u32,
    pub noise_scale: f32,
    pub train_iters: u32,
}

/// Parameters for the DQN algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DQNParams {
    pub batch_size: u32,
    pub seed: u32,
    pub traj_per_epoch: u32,
    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
    pub train_update_freq: u32,
    pub q_lr: f32,
    pub train_q_iters: u32,
}

/// Parameters for the PPO algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PPOParams {
    pub seed: u32,
    pub traj_per_epoch: u32,
    pub clip_ratio: f32,
    pub gamma: f32,
    pub lam: f32,
    pub pi_lr: f32,
    pub vf_lr: f32,
    pub train_pi_iters: u32,
    pub train_v_iters: u32,
    pub target_kl: f32,
}

/// Parameters for the REINFORCE algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct REINFORCEParams {
    pub discrete: bool,
    pub with_vf_baseline: bool,
    pub seed: u32,
    pub traj_per_epoch: u32,
    pub gamma: f32,
    pub lam: f32,
    pub pi_lr: f32,
    pub vf_lr: f32,
    pub train_vf_iters: u32,
}

/// Parameters for the RPO algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RPOParams {
    pub seed: u32,
    pub learning_rate: f32,
    pub num_steps: u32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_coef: f32,
    pub ent_coef: f32,
    pub vf_coef: f32,
    pub max_grad_norm: f32,
    pub rpo_alpha: f32,
    pub update_epochs: u32,
    pub num_minibatches: u32,
    pub anneal_lr: bool,
    pub clip_vloss: bool,
    pub target_kl: Option<f32>,
    pub total_timesteps_anneal_lr: u32,
}

/// Parameters for the SAC algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SACParams {
    pub discrete: bool,
    pub adaptive_alpha: bool,
    pub act_dim: u32,
    pub batch_size: u32,
    pub seed: u32,
    pub traj_per_epoch: u32,
    pub log_std_min: f32,
    pub log_std_max: f32,
    pub gamma: f32,
    pub polyak: f32,
    pub alpha: f32,
    pub lr: f32,
    pub clip_grad_norm: u32,
    pub train_update_freq: u32,
    pub train_iters: u32,
}

/// Parameters for the TD3 algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TD3Params {
    pub seed: u32,
    pub gamma: f32,
    pub tau: f32,
    pub learning_rate: f32,
    pub batch_size: u32,
    pub buffer_size: u32,
    pub exploration_noise: f32,
    pub policy_noise: f32,
    pub noise_clip: f32,
    pub learning_starts: u32,
    pub policy_frequency: u32,
}

/// Configuration parameters for servers.
///
/// This struct holds optional server parameters for training, trajectory, and agent listener.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub training_server: Option<ServerParams>,
    pub trajectory_server: Option<ServerParams>,
    pub agent_listener: Option<ServerParams>,
}

/// Server address parameters.
///
/// Each server parameter includes a prefix, host, and port.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServerParams {
    pub prefix: String,
    pub host: String,
    pub port: String,
}

/// Tensorboard configuration structure.
///
/// Contains optional tensorboard writer parameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorboardConfig {
    pub training_tensorboard: Option<TensorboardParams>,
}

/// Parameters for Training Tensorboard Writer, used for real-time plotting.
///
/// The scalar_tags field is deserialized from a semicolon-separated string.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorboardParams {
    pub launch_tb_on_startup: bool,
    #[serde(deserialize_with = "vec_scalar_tags")]
    pub scalar_tags: Vec<String>,
    pub global_step_tag: String,
}

/// Helper function to deserialize a semicolon-separated string into a vector of strings.
///
/// # Arguments
///
/// * `deserializer` - A serde deserializer.
///
/// # Returns
///
/// A [Result] containing a vector of strings on success.
fn vec_scalar_tags<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(s.split(';').map(|s| s.to_string()).collect())
}

/// Paths for loading (client operation) and saving (server operation) models.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelPaths {
    pub client_model: Option<String>,
    pub server_model: Option<String>,
}

/// The main configuration loader for RL4Sys.
///
/// This struct holds the parsed configuration, including algorithm parameters, server settings,
/// tensorboard parameters, model paths, maximum trajectory length, and gRPC idle timeout.
#[derive(Clone)]
pub struct ConfigLoader {
    pub algorithm_params: Option<LoadedAlgorithmParams>,
    pub train_server: ServerParams,
    pub traj_server: ServerParams,
    pub agent_listener: ServerParams,
    pub tb_params: TensorboardParams,
    pub client_model_path: PathBuf,
    pub server_model_path: PathBuf,
    pub max_traj_length: u32,
    pub grpc_idle_timeout: u32,
}

impl ConfigLoader {
    /// Constructs a new [ConfigLoader] instance.
    ///
    /// This function loads the configuration from the specified path (or the default path if none is provided),
    /// deserializes it into a [Config] struct, and extracts individual configuration parameters.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - An optional algorithm name to select specific algorithm parameters.
    /// * `config_path` - An optional path to the configuration file.
    ///
    /// # Returns
    ///
    /// A new [ConfigLoader] populated with configuration parameters.
    pub fn new(algorithm: Option<String>, config_path: Option<PathBuf>) -> Self {
        // Determine the configuration file path.
        let config: PathBuf = if config_path.is_none() {
            DEFAULT_CONFIG_PATH
                .clone()
                .expect("[ConfigLoader - new] Invalid config path")
        } else {
            config_path.expect("[ConfigLoader - new] Invalid config path")
        };

        // Load the configuration file into a Config struct.
        let config: Config = Self::load_config(&config);

        // Set algorithm-specific parameters if an algorithm name is provided.
        let algorithm_params: Option<LoadedAlgorithmParams> =
            algorithm.and_then(|algo| Self::set_algorithm_params(&config, &algo));

        // Retrieve server parameters from the configuration.
        let train_server: ServerParams = Self::set_train_server(&config);
        let traj_server: ServerParams = Self::set_traj_server(&config);
        let agent_listener: ServerParams = Self::set_agent_listener(&config);

        // Retrieve gRPC idle timeout and tensorboard parameters.
        let grpc_idle_timeout: u32 = Self::set_grpc_idle_timeout(&config);
        let tb_params: TensorboardParams = Self::set_tensorboard_params(&config);

        // Retrieve model paths and maximum trajectory length.
        let client_model_path: PathBuf = Self::set_client_model_path(&config);
        let server_model_path: PathBuf = Self::set_server_model_path(&config);
        let max_traj_length: u32 = Self::set_max_traj_length(&config);

        Self {
            algorithm_params,
            train_server,
            traj_server,
            agent_listener,
            tb_params,
            client_model_path,
            server_model_path,
            max_traj_length,
            grpc_idle_timeout,
        }
    }

    /// Loads and deserializes the configuration file at the given path.
    ///
    /// # Arguments
    ///
    /// * `config_path` - A reference to the path of the configuration file.
    ///
    /// # Returns
    ///
    /// A [Config] instance populated with configuration data.
    pub fn load_config(config_path: &PathBuf) -> Config {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents)
                    .expect("[ConfigLoader - load_config] Failed to read configuration file");
                serde_json::from_str(&contents).unwrap_or_else(|_| {
                    eprintln!("[ConfigLoader - load_config] Failed to parse configuration, loading empty defaults...");
                    Config {
                        algorithms: None,
                        server: None,
                        tensorboard: None,
                        model_paths: None,
                        max_traj_length: None,
                        grpc_idle_timeout: None,
                    }
                })
            }
            Err(e) => {
                eprintln!(
                    "[ConfigLoader - load_config] Failed to load configuration from {:?}, loading defaults. Error: {:?}",
                    config_path, e
                );
                Config {
                    algorithms: None,
                    server: None,
                    tensorboard: None,
                    model_paths: None,
                    max_traj_length: None,
                    grpc_idle_timeout: None,
                }
            }
        }
    }

    /// Returns a reference to the loaded algorithm parameters.
    pub fn get_algorithm_params(&self) -> &Option<LoadedAlgorithmParams> {
        &self.algorithm_params
    }

    /// Returns a reference to the training server parameters.
    pub fn get_train_server(&self) -> &ServerParams {
        &self.train_server
    }

    /// Returns a reference to the trajectory server parameters.
    pub fn get_traj_server(&self) -> &ServerParams {
        &self.traj_server
    }

    /// Returns a reference to the agent listener parameters.
    pub fn get_agent_listener(&self) -> &ServerParams {
        &self.agent_listener
    }

    /// Returns a reference to the tensorboard parameters.
    pub fn get_tb_params(&self) -> &TensorboardParams {
        &self.tb_params
    }

    /// Returns a reference to the client model path.
    pub fn get_client_model_path(&self) -> &PathBuf {
        &self.client_model_path
    }

    /// Returns a reference to the server model path.
    pub fn get_server_model_path(&self) -> &PathBuf {
        &self.server_model_path
    }

    /// Returns a reference to the maximum trajectory length.
    pub fn get_max_traj_length(&self) -> &u32 {
        &self.max_traj_length
    }

    /// Sets the algorithm parameters for the specified algorithm.
    ///
    /// This function checks if the given algorithm name is among the available algorithms.
    /// If so, it extracts the corresponding parameters from the configuration; otherwise,
    /// it logs an error and returns None.
    ///
    /// # Arguments
    ///
    /// * `config` - A reference to the loaded [Config] object.
    /// * `algo` - The algorithm name.
    ///
    /// # Returns
    ///
    /// An [Option] containing [LoadedAlgorithmParams] if found, or None otherwise.
    fn set_algorithm_params(config: &Config, algo: &str) -> Option<LoadedAlgorithmParams> {
        let available_algorithms: [&str; 7] =
            ["C51", "DDPG", "DQN", "PPO", "REINFORCE", "SAC", "TD3"];
        if !available_algorithms.contains(&algo) {
            eprintln!(
                "[ConfigLoader - set_algorithm_params] Failed to load algorithm hyperparameters, loading defaults..."
            );
            return None;
        }
        match algo {
            "C51" => {
                let params = config
                    .algorithms
                    .as_ref()
                    .and_then(|alg| alg.c51.clone())
                    .unwrap_or_else(|| C51Params {
                        batch_size: 32,
                        act_dim: 4,
                        seed: 0,
                        traj_per_epoch: 3,
                        n_atoms: 51,
                        v_min: -10.0,
                        v_max: 10.0,
                        gamma: 0.95,
                        epsilon: 1.0,
                        epsilon_min: 0.01,
                        epsilon_decay: 5e-4,
                        train_update_freq: 8,
                        target_update_freq: 20,
                        q_lr: 1e-3,
                        train_q_iters: 80,
                    });
                Some(LoadedAlgorithmParams::C51(params))
            }
            "DQN" => {
                let params = config
                    .algorithms
                    .as_ref()
                    .and_then(|alg| alg.dqn.clone())
                    .unwrap_or(DQNParams {
                        batch_size: 32,
                        seed: 0,
                        traj_per_epoch: 3,
                        gamma: 0.95,
                        epsilon: 1.0,
                        epsilon_min: 0.01,
                        epsilon_decay: 5e-4,
                        train_update_freq: 4,
                        q_lr: 1e-3,
                        train_q_iters: 80,
                    });
                Some(LoadedAlgorithmParams::DQN(params))
            }
            "PPO" => {
                let params = config
                    .algorithms
                    .as_ref()
                    .and_then(|alg| alg.ppo.clone())
                    .unwrap_or(PPOParams {
                        seed: 0,
                        traj_per_epoch: 3,
                        clip_ratio: 0.2,
                        gamma: 0.99,
                        lam: 0.97,
                        pi_lr: 3e-4,
                        vf_lr: 1e-3,
                        train_pi_iters: 80,
                        train_v_iters: 80,
                        target_kl: 0.01,
                    });
                Some(LoadedAlgorithmParams::PPO(params))
            }
            "REINFORCE" => {
                let params = config
                    .algorithms
                    .as_ref()
                    .and_then(|alg| alg.reinforce.clone())
                    .unwrap_or(REINFORCEParams {
                        discrete: true,
                        with_vf_baseline: true,
                        seed: 0,
                        traj_per_epoch: 12,
                        gamma: 0.99,
                        lam: 0.97,
                        pi_lr: 3e-4,
                        vf_lr: 1e-3,
                        train_vf_iters: 80,
                    });
                Some(LoadedAlgorithmParams::REINFORCE(params))
            }
            "SAC" => {
                let params = config
                    .algorithms
                    .as_ref()
                    .and_then(|alg| alg.sac.clone())
                    .unwrap_or_else(|| SACParams {
                        discrete: true,
                        adaptive_alpha: false,
                        act_dim: 1,
                        batch_size: 32,
                        seed: 0,
                        traj_per_epoch: 3,
                        log_std_min: -20.0,
                        log_std_max: 2.0,
                        gamma: 0.99,
                        polyak: 0.995,
                        alpha: 0.2,
                        lr: 3e-4,
                        clip_grad_norm: 1,
                        train_update_freq: 1,
                        train_iters: 80,
                    });
                Some(LoadedAlgorithmParams::SAC(params))
            }
            _ => {
                eprintln!(
                    "[ConfigLoader - set_algorithm_params] Algorithm {} is not implemented, loading defaults...",
                    algo
                );
                None
            }
        }
    }

    /// Retrieves the training server parameters from the configuration.
    ///
    /// If the training server configuration is missing, it logs an error and returns default parameters.
    fn set_train_server(config: &Config) -> ServerParams {
        config.server.as_ref().and_then(|s| s.training_server.clone()).unwrap_or_else(|| {
            eprintln!("[ConfigLoader - set_train_server] Failed to load training server configuration, loading defaults...");
            ServerParams {
                prefix: "tcp://".to_string(),
                host: "*".to_string(),
                port: "7776".to_string(),
            }
        })
    }

    /// Retrieves the trajectory server parameters from the configuration.
    ///
    /// If the trajectory server configuration is missing, it logs an error and returns default parameters.
    fn set_traj_server(config: &Config) -> ServerParams {
        config.server.as_ref().and_then(|s| s.trajectory_server.clone()).unwrap_or_else(|| {
            eprintln!("[ConfigLoader - set_traj_server] Failed to load trajectory server configuration, loading defaults...");
            ServerParams {
                prefix: "tcp://".to_string(),
                host: "*".to_string(),
                port: "7777".to_string(),
            }
        })
    }

    /// Retrieves the agent listener parameters from the configuration.
    ///
    /// If the agent listener configuration is missing, it logs an error and returns default parameters.
    fn set_agent_listener(config: &Config) -> ServerParams {
        config.server.as_ref().and_then(|s| s.agent_listener.clone()).unwrap_or_else(|| {
            eprintln!("[ConfigLoader - set_agent_listener] Failed to load agent listener configuration, loading defaults...");
            ServerParams {
                prefix: "tcp://".to_string(),
                host: "*".to_string(),
                port: "7778".to_string(),
            }
        })
    }

    /// Retrieves the tensorboard writer parameters from the configuration.
    ///
    /// If the tensorboard parameters are missing, it logs an error and returns default tensorboard parameters.
    fn set_tensorboard_params(config: &Config) -> TensorboardParams {
        config
            .tensorboard
            .as_ref()
            .and_then(|tb| tb.training_tensorboard.clone())
            .unwrap_or_else(|| {
                eprintln!(
                    "[ConfigLoader - set_tensorboard_params] Failed to load tensorboard parameters, loading defaults..."
                );
                TensorboardParams {
                    launch_tb_on_startup: false,
                    scalar_tags: "AverageEpRet;StdEpRet"
                        .split(';')
                        .map(|s| s.to_string())
                        .collect(),
                    global_step_tag: "Epoch".to_string(),
                }
            })
    }

    /// Determines the path where the model should be loaded from. Used by training server after PCR
    ///     command execution.
    ///
    /// If the configuration does not specify a client model path, it logs an error and returns a default path.
    fn set_client_model_path(config: &Config) -> PathBuf {
        let current_dir: PathBuf =
            std::env::current_dir().expect("failed to load current directory");
        config
            .model_paths
            .as_ref()
            .and_then(|mp| mp.client_model.clone())
            .map(|path| current_dir.join(path))
            .unwrap_or_else(|| {
                eprintln!("[ConfigLoader - set_client_model_path] Failed to client model path, loading defaults...");
                current_dir.join("server_model.pt")
            })
    }

    /// Determines the path where the model should be saved. Used by agent to save the model it receives
    ///     from the training server.
    ///
    /// If the configuration does not specify a server model path, it logs an error and returns a default path.
    fn set_server_model_path(config: &Config) -> PathBuf {
        let current_dir: PathBuf =
            std::env::current_dir().expect("failed to load current directory");
        config
            .model_paths
            .as_ref()
            .and_then(|mp| mp.server_model.clone())
            .map(|path| current_dir.join(path))
            .unwrap_or_else(|| {
                eprintln!("[ConfigLoader - set_server_model_path] Failed to load server model path, loading defaults...");
                current_dir.join("client_model.pt")
            })
    }

    /// Retrieves the maximum trajectory length from the configuration.
    ///
    /// If the maximum trajectory length is missing, it logs an error and returns a default value of 1000.
    fn set_max_traj_length(config: &Config) -> u32 {
        config.max_traj_length.unwrap_or_else(|| {
            eprintln!("[ConfigLoader - set_max_traj_length] Failed to load max trajectory length, loading defaults...");
            1000
        })
    }

    /// Retrieves the gRPC idle timeout value from the configuration.
    ///
    /// If the idle timeout is missing, it logs an error and returns a default value of 30 seconds.
    fn set_grpc_idle_timeout(config: &Config) -> u32 {
        config.grpc_idle_timeout.unwrap_or_else(|| {
            eprintln!("[ConfigLoader - set_grpc_idle_timeout] Failed to load idle timeout, loading defaults...");
            30
        })
    }
}
