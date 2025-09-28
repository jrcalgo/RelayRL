

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::{fs, fs::File, io::Read, path::PathBuf};

use crate::get_or_create_client_config_json_path;

#[macro_use]
pub mod client_config_macros {
    /// Resolves config json file between argument and default value.
    #[macro_export]
    macro_rules! resolve_config_json_path {
        ($path: expr) => {
            match $path {
                Some(p) => get_or_create_client_config_json_path!(p.clone()),
                None => DEFAULT_CLIENT_CONFIG_PATH.clone(),
            }
        };
        ($path: literal) => {
            get_or_create_config_json_path!(std::path::PathBuf::from($path))
        };
    }

    /// Will write config file if not found in provided path.
    /// Reads file if found, writes new file if not
    #[macro_export]
    macro_rules! get_or_create_client_config_json_path {
        ($path: expr) => {
            if $path.exists() {
                println!(
                    "[ConfigLoader - load_config] Found config.json in current directory: {:?}",
                    $path
                );
                Some($path)
            } else {
                match fs::write($path, DEFAULT_CLIENT_CONFIG_CONTENT).await {
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
pub static DEFAULT_CLIENT_CONFIG_PATH: Lazy<Option<PathBuf>> =
    Lazy::new(|| get_or_create_client_config_json_path!(PathBuf::from("client_config.json")));

#[macro_use]
pub mod server_config_macros {
    /// Resolves config json file between argument and default value.
    #[macro_export]
    macro_rules! resolve_config_json_path {
        ($path: expr) => {
            match $path {
                Some(p) => get_or_create_server_config_json_path!(p.clone()),
                None => DEFAULT_SERVER_CONFIG_PATH.clone(),
            }
        };
        ($path: literal) => {
            get_or_create_server_config_json_path!(std::path::PathBuf::from($path))
        };
    }

    /// Will write config file if not found in provided path.
    /// Reads file if found, writes new file if not
    #[macro_export]
    macro_rules! get_or_create_server_config_json_path {
        ($path: expr) => {
            if $path.exists() {
                println!(
                    "[ConfigLoader - load_config] Found config.json in current directory: {:?}",
                    $path
                );
                Some($path)
            } else {
                match fs::write($path, DEFAULT_SERVER_CONFIG_CONTENT).await {
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

pub static DEFAULT_SERVER_CONFIG_PATH: Lazy<Option<PathBuf>> = 
    Lazy::new(|| get_or_create_server_config_json_path!(PathBuf::from("server_config.json")));

pub(crate) const DEFAULT_CLIENT_CONFIG_CONTENT: &str = r#"{
    "client_config": {
        "actor_count": 2,
        "_comment1": "For >= 2 distinct actors, algorithm_name value format as PPO;REINFORCE;...",
        "_comment2": "For multiagent algorithms with single inference (1 actor), use single algorithm name.",
        "_comment3": "If > 1 actors and single algorithm name, default to algorithm name for all actors.",
        "algorithm_name": "PPO;REINFORCE",
        "default_device": "cuda",
        "default_model": "",
        "client_model": "client_model.pt",
        "config_path": "client_config.json"
    },
    "transport_config": {
        "addresses": {
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
        "grpc_idle_timeout": 30,
        "max_traj_length": 1000
    }
}"#;

pub(crate) const DEFAULT_SERVER_CONFIG_CONTENT: &str = r#"{
    "server_config": {
        "config_path": "server_config.json",
        "default_hyperparameters": {
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
        "server_model": "server_model.pt",
        "training_tensorboard": {
            "_comment1": "Runs `tensorboard --logdir /logs` in cwd on start up of server.",
            "launch_tb_on_startup": true,
            "_comment2": "scalar tags can be any column header from `progress.txt` files.",
            "_comment3": "For more than one tag, separate by semi-colon (;)",
            "scalar_tags": "AverageEpRet;LossQ",
            "global_step_tag": "Epoch"
        },
        "transport_config": {
            "addresses": {
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
            "grpc_idle_timeout": 30,
            "max_traj_length": 1000
        }
    }
}"#;

// ============================================================================
// Algorithm Parameter Structs (ported from config_loader.rs)
// ============================================================================

/// An enum representing loaded algorithm parameters.
/// Each variant corresponds to one algorithm's parameter struct.
#[derive(Debug, Clone)]
pub enum LoadedAlgorithmParams {
    C51(C51Params),
    DDPG(DDPGParams),
    DQN(DQNParams),
    PPO(PPOParams),
    REINFORCE(REINFORCEParams),
    RPO(RPOParams),
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
}

// ============================================================================
// Server Configuration Structs
// ============================================================================

/// Server address parameters.
///
/// Each server parameter includes a prefix, host, and port.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServerParams {
    pub prefix: String,
    pub host: String,
    pub port: String,
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

// ============================================================================
// Builder Pattern Implementation
// ============================================================================

pub trait ClientConfigBuilder {
    fn actor_count(&mut self, actor_count: i64) -> &mut Self;
    fn algorithm_name(&mut self, algorithm_name: &str) -> &mut Self;
    fn default_device(&mut self, default_device: &str) -> &mut Self;
    fn default_model(&mut self, default_model: &str) -> &mut Self;
    fn client_model(&mut self, client_model: &str) -> &mut Self;
    fn config_path(&mut self, config_path: &str) -> &mut Self;
    fn build_transport_config(&mut self) -> TransportConfigBuilder;
    fn build(&self) -> ClientConfig;
    fn build_default() -> ClientConfig;
}

pub trait ServerConfigBuilder {
    fn config_path(&mut self, config_path: &str) -> &mut Self;
    fn default_hyperparameters(&mut self, default_hyperparameters: AlgorithmConfig) -> &mut Self;
    fn server_model(&mut self, server_model: &str) -> &mut Self;
    fn training_tensorboard(&mut self, training_tensorboard: TensorboardParams) -> &mut Self;
    fn build_transport_config(&mut self) -> TransportConfigBuilder;
    fn build(&self) -> ServerConfig;
    fn build_default() -> ServerConfig;
}

pub trait TransportConfigBuilder {
    fn training_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn trajectory_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn agent_listener_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn grpc_idle_timeout(&mut self, timeout: u64) -> &mut Self;
    fn max_traj_length(&mut self, max_traj_length: u64) -> &mut Self;
    fn model_paths(&mut self, client_model: &str, server_model: &str) -> &mut Self;
    fn build(&self) -> TransportConfig;
    fn build_default() -> TransportConfig;
}

// ============================================================================
// Configuration Structs
// ============================================================================

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub actor_count: i64,
    pub algorithm_name: String,
    pub default_device: String,
    pub default_model: String,
    pub client_model: String,
    pub config_path: String,
    pub transport_config: TransportConfig,
}

impl ClientConfig {
    pub fn new() -> ClientConfigBuilder {
        ClientConfigBuilder::new()
    }

    pub fn load_config(algorithm_name: Option<String>, config_path: Option<PathBuf>) -> Self {
        let config: PathBuf = if config_path.is_none() {
            DEFAULT_CLIENT_CONFIG_PATH
                .clone()
                .expect("[ConfigLoader - new] Invalid config path")
        } else {
            config_path.expect("[ConfigLoader - new] Invalid config path")
        };
        
        Self::_load_config_file(&config)
    }

    fn _load_config_file(config_path: &PathBuf) -> Self {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents)
                    .expect("[ConfigLoader - load_config] Failed to read configuration file");
                serde_json::from_str(&contents).unwrap_or_else(|_| {
                    eprintln!("[ConfigLoader - load_config] Failed to parse configuration, loading empty defaults...");
                    Self::build_default()
                })
            }
            Err(e) => {
                eprintln!(
                    "[ConfigLoader - load_config] Failed to load configuration from {:?}, loading defaults. Error: {:?}",
                    config_path, e
                );
                Self::build_default()
            }
        }
    }

    pub fn get_algorithm_params(&self) -> Option<LoadedAlgorithmParams> {
        // Parse algorithm names (support multiple algorithms separated by semicolon)
        let algorithm_names: Vec<&str> = self.algorithm_name.split(';').collect();
        
        // For now, return the first algorithm's parameters
        // In the future, this could be extended to handle multiple algorithms
        if let Some(first_algo) = algorithm_names.first() {
            Self::set_algorithm_params(first_algo)
        } else {
            None
        }
    }

    fn set_algorithm_params(algo: &str) -> Option<LoadedAlgorithmParams> {
        let available_algorithms: [&str; 8] =
            ["C51", "DDPG", "DQN", "PPO", "REINFORCE", "RPO", "SAC", "TD3"];
        
        if !available_algorithms.contains(&algo) {
            eprintln!(
                "[ConfigLoader - set_algorithm_params] Algorithm {} not found, loading defaults...",
                algo
            );
            return None;
        }

        match algo {
            "C51" => Some(LoadedAlgorithmParams::C51(C51Params {
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
            })),
            "DQN" => Some(LoadedAlgorithmParams::DQN(DQNParams {
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
            })),
            "PPO" => Some(LoadedAlgorithmParams::PPO(PPOParams {
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
            })),
            "REINFORCE" => Some(LoadedAlgorithmParams::REINFORCE(REINFORCEParams {
                discrete: true,
                with_vf_baseline: true,
                seed: 0,
                traj_per_epoch: 12,
                gamma: 0.99,
                lam: 0.97,
                pi_lr: 3e-4,
                vf_lr: 1e-3,
                train_vf_iters: 80,
            })),
            "SAC" => Some(LoadedAlgorithmParams::SAC(SACParams {
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
            })),
            _ => {
                eprintln!(
                    "[ConfigLoader - set_algorithm_params] Algorithm {} is not implemented, loading defaults...",
                    algo
                );
                None
            }
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self::build_default()
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub config_path: String,
    pub default_hyperparameters: AlgorithmConfig,
    pub server_model: String,
    pub training_tensorboard: TensorboardParams,
    pub transport_config: TransportConfig,
}

impl ServerConfig {
    pub fn new() -> ServerConfigBuilder {
        ServerConfigBuilder::new()
    }

    pub fn load_config(config_path: Option<PathBuf>) -> Self {
        let config: PathBuf = if config_path.is_none() {
            DEFAULT_SERVER_CONFIG_PATH
                .clone()
                .expect("[ConfigLoader - new] Invalid config path")
        } else {
            config_path.expect("[ConfigLoader - new] Invalid config path")
        };
        
        Self::_load_config_file(&config)
    }

    fn _load_config_file(config_path: &PathBuf) -> Self {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents)
                    .expect("[ConfigLoader - load_config] Failed to read configuration file");
                serde_json::from_str(&contents).unwrap_or_else(|_| {
                    eprintln!("[ConfigLoader - load_config] Failed to parse configuration, loading empty defaults...");
                    Self::build_default()
                })
            }
            Err(e) => {
                eprintln!(
                    "[ConfigLoader - load_config] Failed to load configuration from {:?}, loading defaults. Error: {:?}",
                    config_path, e
                );
                Self::build_default()
            }
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::build_default()
    }
}

#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub training_server_address: ServerParams,
    pub trajectory_server_address: ServerParams,
    pub agent_listener_address: ServerParams,
    pub grpc_idle_timeout: u64,
    pub max_traj_length: u64,
    pub model_paths: ModelPaths,
}

impl TransportConfig {
    pub fn new() -> TransportConfigBuilder {
        TransportConfigBuilder::new()
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self::build_default()
    }
}

// ============================================================================
// Builder Implementations
// ============================================================================

pub struct ClientConfigBuilder {
    actor_count: Option<i64>,
    algorithm_name: Option<String>,
    default_device: Option<String>,
    default_model: Option<String>,
    client_model: Option<String>,
    config_path: Option<String>,
    transport_config: Option<TransportConfig>,
}

impl ClientConfigBuilder {
    pub fn new() -> Self {
        Self {
            actor_count: None,
            algorithm_name: None,
            default_device: None,
            default_model: None,
            client_model: None,
            config_path: None,
            transport_config: None,
        }
    }
}

impl ClientConfigBuilder for ClientConfigBuilder {
    fn actor_count(&mut self, actor_count: i64) -> &mut Self {
        self.actor_count = Some(actor_count);
        self
    }

    fn algorithm_name(&mut self, algorithm_name: &str) -> &mut Self {
        self.algorithm_name = Some(algorithm_name.to_string());
        self
    }

    fn default_device(&mut self, default_device: &str) -> &mut Self {
        self.default_device = Some(default_device.to_string());
        self
    }

    fn default_model(&mut self, default_model: &str) -> &mut Self {
        self.default_model = Some(default_model.to_string());
        self
    }

    fn client_model(&mut self, client_model: &str) -> &mut Self {
        self.client_model = Some(client_model.to_string());
        self
    }

    fn config_path(&mut self, config_path: &str) -> &mut Self {
        self.config_path = Some(config_path.to_string());
        self
    }

    fn build_transport_config(&mut self) -> TransportConfigBuilder {
        TransportConfigBuilder::new()
    }

    fn build(&self) -> ClientConfig {
        ClientConfig {
            actor_count: self.actor_count.unwrap_or(1),
            algorithm_name: self.algorithm_name.clone().unwrap_or_else(|| "PPO".to_string()),
            default_device: self.default_device.clone().unwrap_or_else(|| "cuda".to_string()),
            default_model: self.default_model.clone().unwrap_or_else(|| "".to_string()),
            client_model: self.client_model.clone().unwrap_or_else(|| "client_model.pt".to_string()),
            config_path: self.config_path.clone().unwrap_or_else(|| "client_config.json".to_string()),
            transport_config: self.transport_config.clone().unwrap_or_else(|| TransportConfig::build_default()),
        }
    }

    fn build_default() -> ClientConfig {
        ClientConfig {
            actor_count: 1,
            algorithm_name: "PPO".to_string(),
            default_device: "cuda".to_string(),
            default_model: "".to_string(),
            client_model: "client_model.pt".to_string(),
            config_path: "client_config.json".to_string(),
            transport_config: TransportConfig::build_default(),
        }
    }
}

pub struct ServerConfigBuilder {
    config_path: Option<String>,
    default_hyperparameters: Option<AlgorithmConfig>,
    server_model: Option<String>,
    training_tensorboard: Option<TensorboardParams>,
    transport_config: Option<TransportConfig>,
}

impl ServerConfigBuilder {
    pub fn new() -> Self {
        Self {
            config_path: None,
            default_hyperparameters: None,
            server_model: None,
            training_tensorboard: None,
            transport_config: None,
        }
    }
}

impl ServerConfigBuilder for ServerConfigBuilder {
    fn config_path(&mut self, config_path: &str) -> &mut Self {
        self.config_path = Some(config_path.to_string());
        self
    }

    fn default_hyperparameters(&mut self, default_hyperparameters: AlgorithmConfig) -> &mut Self {
        self.default_hyperparameters = Some(default_hyperparameters);
        self
    }

    fn server_model(&mut self, server_model: &str) -> &mut Self {
        self.server_model = Some(server_model.to_string());
        self
    }

    fn training_tensorboard(&mut self, training_tensorboard: TensorboardParams) -> &mut Self {
        self.training_tensorboard = Some(training_tensorboard);
        self
    }

    fn build_transport_config(&mut self) -> TransportConfigBuilder {
        TransportConfigBuilder::new()
    }

    fn build(&self) -> ServerConfig {
        ServerConfig {
            config_path: self.config_path.clone().unwrap_or_else(|| "server_config.json".to_string()),
            default_hyperparameters: self.default_hyperparameters.clone().unwrap_or_default(),
            server_model: self.server_model.clone().unwrap_or_else(|| "server_model.pt".to_string()),
            training_tensorboard: self.training_tensorboard.clone().unwrap_or_else(|| TensorboardParams {
                launch_tb_on_startup: false,
                scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                global_step_tag: "Epoch".to_string(),
            }),
            transport_config: self.transport_config.clone().unwrap_or_else(|| TransportConfig::build_default()),
        }
    }

    fn build_default() -> ServerConfig {
        ServerConfig {
            config_path: "server_config.json".to_string(),
            default_hyperparameters: AlgorithmConfig {
                c51: None,
                ddpg: None,
                dqn: None,
                ppo: None,
                reinforce: None,
                rpo: None,
                sac: None,
                td3: None,
            },
            server_model: "server_model.pt".to_string(),
            training_tensorboard: TensorboardParams {
                launch_tb_on_startup: false,
                scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                global_step_tag: "Epoch".to_string(),
            },
            transport_config: TransportConfig::build_default(),
        }
    }
}

pub struct TransportConfigBuilder {
    training_server_address: Option<ServerParams>,
    trajectory_server_address: Option<ServerParams>,
    agent_listener_address: Option<ServerParams>,
    grpc_idle_timeout: Option<u64>,
    max_traj_length: Option<u64>,
    model_paths: Option<ModelPaths>,
}

impl TransportConfigBuilder {
    pub fn new() -> Self {
        Self {
            training_server_address: None,
            trajectory_server_address: None,
            agent_listener_address: None,
            grpc_idle_timeout: None,
            max_traj_length: None,
            model_paths: None,
        }
    }
}

impl TransportConfigBuilder for TransportConfigBuilder {
    fn training_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.training_server_address = Some(ServerParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn trajectory_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.trajectory_server_address = Some(ServerParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn agent_listener_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.agent_listener_address = Some(ServerParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn grpc_idle_timeout(&mut self, timeout: u64) -> &mut Self {
        self.grpc_idle_timeout = Some(timeout);
        self
    }

    fn max_traj_length(&mut self, max_traj_length: u64) -> &mut Self {
        self.max_traj_length = Some(max_traj_length);
        self
    }

    fn model_paths(&mut self, client_model: &str, server_model: &str) -> &mut Self {
        self.model_paths = Some(ModelPaths {
            client_model: Some(client_model.to_string()),
            server_model: Some(server_model.to_string()),
        });
        self
    }

    fn build(&self) -> TransportConfig {
        TransportConfig {
            training_server_address: self.training_server_address.clone().unwrap_or_else(|| ServerParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "50051".to_string(),
            }),
            trajectory_server_address: self.trajectory_server_address.clone().unwrap_or_else(|| ServerParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7776".to_string(),
            }),
            agent_listener_address: self.agent_listener_address.clone().unwrap_or_else(|| ServerParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7777".to_string(),
            }),
            grpc_idle_timeout: self.grpc_idle_timeout.unwrap_or(30),
            max_traj_length: self.max_traj_length.unwrap_or(1000),
            model_paths: self.model_paths.clone().unwrap_or_else(|| ModelPaths {
                client_model: Some("client_model.pt".to_string()),
                server_model: Some("server_model.pt".to_string()),
            }),
        }
    }

    fn build_default() -> TransportConfig {
        TransportConfig {
            training_server_address: ServerParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "50051".to_string(),
            },
            trajectory_server_address: ServerParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7776".to_string(),
            },
            agent_listener_address: ServerParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7777".to_string(),
            },
            grpc_idle_timeout: 30,
            max_traj_length: 1000,
            model_paths: ModelPaths {
                client_model: Some("client_model.pt".to_string()),
                server_model: Some("server_model.pt".to_string()),
            },
        }
    }
}
