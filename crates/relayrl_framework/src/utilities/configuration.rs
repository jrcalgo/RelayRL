use relayrl_types::Hyperparams;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{fs, fs::File, io::Read, path::PathBuf};

use crate::get_or_create_client_config_json_path;

#[macro_use]
pub mod client_config_macros {
    /// Resolves config json file between argument and default value.
    #[macro_export]
    macro_rules! resolve_client_config_json_path {
        ($path: expr) => {
            match $path {
                Some(p) => get_or_create_client_config_json_path!(p.clone()),
                None => DEFAULT_CLIENT_CONFIG_PATH.clone(),
            }
        };
        ($path: literal) => {
            get_or_create_client_config_json_path!(std::path::PathBuf::from($path))
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
                match fs::write($path, DEFAULT_CLIENT_CONFIG_CONTENT) {
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
    macro_rules! resolve_server_config_json_path {
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
                match fs::write($path, DEFAULT_SERVER_CONFIG_CONTENT) {
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
        "config_path": "client_config.json",
        "default_device": "cuda",
        "default_model_path": ""
    },
    "transport_config": {
        "addresses": {
            "_comment": "gRPC uses only this address (prefix is unused).",
            "model_server": {
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
            },
            "_comment2": "Only used when client-side inference is disabled or when client-side inference is used as fallback.",
            "inference_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7778"
            }
        },
        "config_update_polling": 10,
        "grpc_idle_timeout": 30,
        "local_model_module": {
            "directory_name": "model_module",
            "model_name": "client_model"
        },
        "max_traj_length": 1000
    }
}"#;

pub(crate) const DEFAULT_SERVER_CONFIG_CONTENT: &str = r#"{
    "server_config": {
        "config_path": "server_config.json",
        "default_hyperparameters": {
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
            "PPO": {
                "discrete": true,
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
                "with_vf_baseline": true,
                "seed": 1,
                "traj_per_epoch": 8,
                "gamma": 0.98,
                "lam": 0.97,
                "pi_lr": 3e-4,
                "vf_lr": 1e-3,
                "train_vf_iters": 80
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
        "training_tensorboard": {
            "_comment1": "Runs `tensorboard --logdir /logs` in cwd on start up of server.",
            "launch_tb_on_startup": true,
            "_comment2": "scalar tags can be any column header from `progress.txt` files.",
            "_comment3": "For more than one tag, separate by semi-colon (;)",
            "scalar_tags": "AverageEpRet;LossQ",
            "global_step_tag": "Epoch"
        }
    },
    "transport_config": {
        "addresses": {
            "_comment1": "gRPC uses only this address (prefix is unused).",
            "model_server": {
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
            },
            "_comment2": "Only available when client-side inference is disabled or when client-side inference is used as fallback.",
            "inference_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7778"
            }
        },
        "config_update_polling": 10,
        "grpc_idle_timeout": 30,
        "local_model_module": {
            "directory_name": "model_module",
            "model_name": "server_model"
        },
        "max_traj_length": 1000
    }
}"#;

/// Configuration parameters for various algorithms.
///
/// Each field is optional and holds algorithm-specific parameters.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AlgorithmConfig {
    #[serde(rename = "DDPG")]
    pub ddpg: Option<DDPGParams>,
    #[serde(rename = "PPO")]
    pub ppo: Option<PPOParams>,
    #[serde(rename = "REINFORCE")]
    pub reinforce: Option<REINFORCEParams>,
    #[serde(rename = "TD3")]
    pub td3: Option<TD3Params>,
    // Add other fields depending on the algorithm
    #[serde(rename = "custom")]
    pub custom: Option<CustomAlgorithmParams>,
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

impl Default for DDPGParams {
    fn default() -> Self {
        Self {
            seed: 1,
            gamma: 0.99,
            tau: 1e-2,
            learning_rate: 3e-3,
            batch_size: 128,
            buffer_size: 50000,
            learning_starts: 128,
            policy_frequency: 1,
            noise_scale: 0.1,
            train_iters: 50,
        }
    }
}

/// Parameters for the PPO algorithm.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PPOParams {
    pub discrete: bool,
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

impl Default for PPOParams {
    fn default() -> Self {
        Self {
            discrete: true,
            seed: 0,
            traj_per_epoch: 1,
            clip_ratio: 0.1,
            gamma: 0.99,
            lam: 0.97,
            pi_lr: 3e-4,
            vf_lr: 3e-4,
            train_pi_iters: 40,
            train_v_iters: 40,
            target_kl: 0.01,
        }
    }
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

impl Default for REINFORCEParams {
    fn default() -> Self {
        Self {
            discrete: true,
            with_vf_baseline: false,
            seed: 1,
            traj_per_epoch: 8,
            gamma: 0.98,
            lam: 0.97,
            pi_lr: 3e-4,
            vf_lr: 1e-3,
            train_vf_iters: 80,
        }
    }
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

impl Default for TD3Params {
    fn default() -> Self {
        Self {
            seed: 1,
            gamma: 0.99,
            tau: 0.005,
            learning_rate: 3e-4,
            batch_size: 128,
            buffer_size: 50000,
            exploration_noise: 0.1,
            policy_noise: 0.2,
            noise_clip: 0.5,
            learning_starts: 25000,
            policy_frequency: 2,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CustomAlgorithmParams {
    pub algorithm_name: String,
    pub hyperparams: HashMap<String, String>,
}

impl Default for CustomAlgorithmParams {
    fn default() -> Self {
        Self {
            algorithm_name: "".to_string(),
            hyperparams: HashMap::new(),
        }
    }
}

/// Server address parameters.
///
/// Each server parameter includes a prefix, host, and port.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkParams {
    pub prefix: String,
    pub host: String,
    pub port: String,
}

/// Configuration parameters for servers.
///
/// This struct holds optional server parameters for training, trajectory, and agent listener.
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkConfigData {
    pub training_server: Option<NetworkParams>,
    pub trajectory_server: Option<NetworkParams>,
    pub agent_listener: Option<NetworkParams>,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientConfigParams {
    pub actor_count: u64,
    pub algorithm_name: String,
    pub config_path: PathBuf,
    pub default_device: String,
    pub default_model_path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientConfigLoader {
    pub client_config: ClientConfigParams,
    pub transport_config: TransportConfigParams,
}

impl ClientConfigLoader {
    pub fn new_config(algorithm_name: Option<String>, config_path: Option<PathBuf>) -> Self {
        let _config_path: PathBuf = if config_path.is_none() {
            DEFAULT_CLIENT_CONFIG_PATH
                .clone()
                .expect("[ClientConfigParams - new] Invalid config path")
        } else {
            config_path.expect("[ClientConfigParams - new] Invalid config path")
        };

        let config: ClientConfigLoader = Self::load_config(&_config_path);

        let client_config: ClientConfigParams = config.client_config;
        let transport_config: TransportConfigParams = config.transport_config;

        let _algorithm_name: String = match algorithm_name {
            Some(algorithm_name) => algorithm_name,
            None => client_config.algorithm_name,
        };

        Self {
            client_config: ClientConfigParams {
                actor_count: client_config.actor_count,
                algorithm_name: _algorithm_name,
                config_path: _config_path,
                default_device: client_config.default_device,
                default_model_path: client_config.default_model_path,
            },
            transport_config,
        }
    }

    pub fn load_config(config_path: &PathBuf) -> Self {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents)
                    .expect("[ClientConfigParams - load_config] Failed to read configuration file");
                serde_json::from_str(&contents).unwrap_or_else(|_| {
                    eprintln!("[ClientConfigParams - load_config] Failed to parse configuration, loading empty defaults...");
                    ClientConfigLoader {
                        client_config: ClientConfigParams {
                            actor_count: 1,
                            algorithm_name: "".to_string(),
                            config_path: PathBuf::from("client_config.json"),
                            default_device: "cpu".to_string(),
                            default_model_path: PathBuf::from(""),
                        },
                        transport_config: TransportConfigBuilder::build_default()
                    }
                })
            }
            Err(e) => {
                panic!(
                    "[ClientConfigParams - load_config] Failed to open configuration file: {}",
                    e
                );
            }
        }
    }

    pub fn get_actor_count(&self) -> &u64 {
        &self.client_config.actor_count
    }

    pub fn get_algorithm_name(&self) -> &str {
        &self.client_config.algorithm_name
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.client_config.config_path
    }

    pub fn get_default_device(&self) -> &str {
        &self.client_config.default_device
    }

    pub fn get_default_model_path(&self) -> &PathBuf {
        &self.client_config.default_model_path
    }

    pub fn get_transport_config(&self) -> &TransportConfigParams {
        &self.transport_config
    }
}

pub trait ClientConfigBuildParams {
    fn set_actor_count(&mut self, actor_count: u64) -> &mut Self;
    fn set_algorithm_name(&mut self, algorithm_name: &str) -> &mut Self;
    fn set_config_path(&mut self, config_path: &str) -> &mut Self;
    fn set_default_device(&mut self, default_device: &str) -> &mut Self;
    fn set_default_model_path(&mut self, initial_model: &str) -> &mut Self;
    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self;
    fn build(&self) -> ClientConfigLoader;
    fn build_default() -> ClientConfigLoader;
}

pub struct ClientConfigBuilder {
    actor_count: Option<u64>,
    algorithm_name: Option<String>,
    config_path: Option<PathBuf>,
    default_device: Option<String>,
    default_model_path: Option<PathBuf>,
    transport_config: Option<TransportConfigParams>,
}

impl ClientConfigBuildParams for ClientConfigBuilder {
    fn set_actor_count(&mut self, actor_count: u64) -> &mut Self {
        self.actor_count = Some(actor_count);
        self
    }

    fn set_algorithm_name(&mut self, algorithm_name: &str) -> &mut Self {
        self.algorithm_name = Some(algorithm_name.to_string());
        self
    }

    fn set_config_path(&mut self, config_path: &str) -> &mut Self {
        self.config_path = Some(PathBuf::from(config_path));
        self
    }

    fn set_default_device(&mut self, default_device: &str) -> &mut Self {
        self.default_device = Some(default_device.to_string());
        self
    }

    fn set_default_model_path(&mut self, initial_model: &str) -> &mut Self {
        self.default_model_path = Some(PathBuf::from(initial_model));
        self
    }

    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self {
        self.transport_config = Some(transport_config);
        self
    }

    fn build(&self) -> ClientConfigLoader {
        let client_config: ClientConfigParams = ClientConfigParams {
            actor_count: self.actor_count.unwrap_or(1),
            algorithm_name: self
                .algorithm_name
                .clone()
                .unwrap_or_else(|| "".to_string()),
            config_path: self
                .config_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("client_config.json")),
            default_device: self
                .default_device
                .clone()
                .unwrap_or_else(|| "cpu".to_string()),
            default_model_path: self
                .default_model_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("")),
        };

        let transport_config: TransportConfigParams = match &self.transport_config {
            Some(transport_config) => TransportConfigParams {
                agent_listener_address: transport_config.agent_listener_address.clone(),
                model_server_address: transport_config.model_server_address.clone(),
                trajectory_server_address: transport_config.trajectory_server_address.clone(),
                grpc_idle_timeout: transport_config.grpc_idle_timeout,
                max_traj_length: transport_config.max_traj_length,
                local_model_path: transport_config.local_model_path.clone(),
                config_update_polling: transport_config.config_update_polling,
            },
            None => TransportConfigBuilder::build_default(),
        };

        ClientConfigLoader {
            client_config,
            transport_config,
        }
    }

    fn build_default() -> ClientConfigLoader {
        ClientConfigLoader {
            client_config: ClientConfigParams {
                actor_count: 1,
                algorithm_name: "".to_string(),
                config_path: PathBuf::from("client_config.json"),
                default_device: "cpu".to_string(),
                default_model_path: PathBuf::from(""),
            },
            transport_config: TransportConfigBuilder::build_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfigParams {
    pub config_path: PathBuf,
    pub default_hyperparameters: Option<AlgorithmConfig>,
    pub training_tensorboard: TensorboardParams,
    pub default_model_path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfigLoader {
    pub server_config: ServerConfigParams,
    pub transport_config: TransportConfigParams,
}

impl ServerConfigLoader {
    pub fn new_config(config_path: Option<PathBuf>) -> Self {
        let _config_path: PathBuf = if config_path.is_none() {
            DEFAULT_SERVER_CONFIG_PATH
                .clone()
                .expect("[ServerConfigParams - new] Invalid config path")
        } else {
            config_path.expect("[ServerConfigParams - new] Invalid config path")
        };

        let config: ServerConfigLoader = Self::load_config(&_config_path);

        let server_config: ServerConfigParams = config.server_config;
        let transport_config: TransportConfigParams = config.transport_config;

        Self {
            server_config,
            transport_config,
        }
    }

    pub fn load_config(config_path: &PathBuf) -> Self {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents)
                    .expect("[ServerConfigParams - load_config] Failed to read configuration file");
                serde_json::from_str(&contents).unwrap_or_else(|_| {
                    eprintln!("[ServerConfigParams - load_config] Failed to parse configuration, loading empty defaults...");
                    ServerConfigLoader {
                        server_config: ServerConfigParams {
                            config_path: PathBuf::from("server_config.json"),
                            default_hyperparameters: None,
                            training_tensorboard: TensorboardParams {
                                launch_tb_on_startup: false,
                                scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                                global_step_tag: "Epoch".to_string(),
                            },
                            default_model_path: PathBuf::from(""),
                        },
                        transport_config: TransportConfigBuilder::build_default(),
                    }
                })
            }
            Err(e) => {
                panic!(
                    "[ServerConfigParams - load_config] Failed to open configuration file: {}",
                    e
                );
            }
        }
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.server_config.config_path
    }

    pub fn get_hyperparameters(&self) -> &Option<AlgorithmConfig> {
        &self.server_config.default_hyperparameters
    }

    pub fn get_training_tensorboard(&self) -> &TensorboardParams {
        &self.server_config.training_tensorboard
    }

    pub fn get_default_model_path(&self) -> &PathBuf {
        &self.server_config.default_model_path
    }

    pub fn get_transport_config(&self) -> &TransportConfigParams {
        &self.transport_config
    }
}

pub trait ServerConfigBuildParams {
    fn set_config_path(&mut self, config_path: &str) -> &mut Self;
    fn set_hyperparameters(
        &mut self,
        algorithm_name: &str,
        hyperparameters: Hyperparams,
    ) -> &mut Self;
    fn set_training_tensorboard_params(
        &mut self,
        launch_tb_on_startup: bool,
        scalar_tags: &str,
        global_step_tag: &str,
    ) -> &mut Self;
    fn set_default_model_path(&mut self, initial_model: &str) -> &mut Self;
    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self;
    fn build(&self) -> ServerConfigLoader;
    fn build_default() -> ServerConfigLoader;
}

pub struct ServerConfigBuilder {
    config_path: Option<PathBuf>,
    default_hyperparameters: Option<AlgorithmConfig>,
    training_tensorboard: Option<TensorboardParams>,
    default_model_path: Option<PathBuf>,
    transport_config: Option<TransportConfigParams>,
}

impl ServerConfigBuildParams for ServerConfigBuilder {
    fn set_config_path(&mut self, config_path: &str) -> &mut Self {
        self.config_path = Some(PathBuf::from(config_path));
        self
    }

    fn set_hyperparameters(
        &mut self,
        algorithm_name: &str,
        hyperparameters: Hyperparams,
    ) -> &mut Self {
        let hp_map: HashMap<String, String> =
            crate::network::parse_args(&Some(hyperparameters.clone()));

        // Start from defaults for all supported algorithms.
        let mut all_cfg = AlgorithmConfig {
            ddpg: Some(DDPGParams::default()),
            ppo: Some(PPOParams::default()),
            reinforce: Some(REINFORCEParams::default()),
            td3: Some(TD3Params::default()),
            custom: None,
        };

        let algo_upper = algorithm_name.to_uppercase();

        match algo_upper.as_str() {
            "DDPG" => {
                if let Some(params) = &mut all_cfg.ddpg {
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u32>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_rate")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.learning_rate = v;
                    }
                    if let Some(v) = hp_map.get("batch_size").and_then(|s| s.parse::<u32>().ok()) {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_starts")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.learning_starts = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("noise_scale")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.noise_scale = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_iters = v;
                    }
                }
            }
            "PPO" => {
                if let Some(params) = &mut all_cfg.ppo {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u32>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map.get("clip_ratio").and_then(|s| s.parse::<f32>().ok()) {
                        params.clip_ratio = v;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lam = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_pi_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_pi_iters = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_v_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_v_iters = v;
                    }
                    if let Some(v) = hp_map.get("target_kl").and_then(|s| s.parse::<f32>().ok()) {
                        params.target_kl = v;
                    }
                }
            }
            "REINFORCE" => {
                if let Some(params) = &mut all_cfg.reinforce {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("with_vf_baseline") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.with_vf_baseline = vv;
                    }
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u32>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lam = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_vf_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_vf_iters = v;
                    }
                }
            }
            "TD3" => {
                if let Some(params) = &mut all_cfg.td3 {
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u32>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_rate")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.learning_rate = v;
                    }
                    if let Some(v) = hp_map.get("batch_size").and_then(|s| s.parse::<u32>().ok()) {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("exploration_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.exploration_noise = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.policy_noise = v;
                    }
                    if let Some(v) = hp_map.get("noise_clip").and_then(|s| s.parse::<f32>().ok()) {
                        params.noise_clip = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_starts")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.learning_starts = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.policy_frequency = v;
                    }
                }
            }
            custom_algorithm => {
                let mut custom = all_cfg.custom.take().unwrap_or_default();
                custom.algorithm_name = custom_algorithm.to_string();
                custom.hyperparams = hp_map.clone();
                all_cfg.custom = Some(custom);
            }
        }

        self.default_hyperparameters = Some(all_cfg);
        self
    }

    fn set_training_tensorboard_params(
        &mut self,
        launch_tb_on_startup: bool,
        scalar_tags: &str,
        global_step_tag: &str,
    ) -> &mut Self {
        self.training_tensorboard = Some(TensorboardParams {
            launch_tb_on_startup,
            scalar_tags: scalar_tags.split(';').map(|s| s.to_string()).collect(),
            global_step_tag: global_step_tag.to_string(),
        });
        self
    }

    fn set_default_model_path(&mut self, initial_model: &str) -> &mut Self {
        self.default_model_path = Some(PathBuf::from(initial_model));
        self
    }

    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self {
        self.transport_config = Some(transport_config);
        self
    }

    fn build(&self) -> ServerConfigLoader {
        let server_config: ServerConfigParams = ServerConfigParams {
            config_path: self
                .config_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("server_config.json")),
            default_hyperparameters: self.default_hyperparameters.clone(),
            training_tensorboard: self.training_tensorboard.clone().unwrap_or_else(|| {
                TensorboardParams {
                    launch_tb_on_startup: false,
                    scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                    global_step_tag: "Epoch".to_string(),
                }
            }),
            default_model_path: self
                .default_model_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("")),
        };

        let transport_config: TransportConfigParams = match &self.transport_config {
            Some(transport_config) => TransportConfigParams {
                agent_listener_address: transport_config.agent_listener_address.clone(),
                model_server_address: transport_config.model_server_address.clone(),
                trajectory_server_address: transport_config.trajectory_server_address.clone(),
                grpc_idle_timeout: transport_config.grpc_idle_timeout,
                max_traj_length: transport_config.max_traj_length,
                local_model_path: transport_config.local_model_path.clone(),
                config_update_polling: transport_config.config_update_polling,
            },
            None => TransportConfigBuilder::build_default(),
        };

        ServerConfigLoader {
            server_config,
            transport_config,
        }
    }

    fn build_default() -> ServerConfigLoader {
        ServerConfigLoader {
            server_config: ServerConfigParams {
                config_path: PathBuf::from("server_config.json"),
                default_hyperparameters: None,
                training_tensorboard: TensorboardParams {
                    launch_tb_on_startup: false,
                    scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                    global_step_tag: "Epoch".to_string(),
                },
                default_model_path: PathBuf::from(""),
            },
            transport_config: TransportConfigBuilder::build_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransportConfigParams {
    pub agent_listener_address: NetworkParams,
    pub model_server_address: NetworkParams,
    pub trajectory_server_address: NetworkParams,
    pub grpc_idle_timeout: u32,
    pub max_traj_length: u128,
    pub local_model_path: PathBuf,
    pub config_update_polling: u32,
}

impl TransportConfigParams {
    pub fn get_agent_listener_address(&self) -> &NetworkParams {
        &self.agent_listener_address
    }

    pub fn get_model_server_address(&self) -> &NetworkParams {
        &self.model_server_address
    }

    pub fn get_trajectory_server_address(&self) -> &NetworkParams {
        &self.trajectory_server_address
    }
}

pub trait TransportConfigBuildParams {
    fn set_agent_listener_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_model_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_trajectory_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_grpc_idle_timeout(&mut self, grpc_idle_timeout: u32) -> &mut Self;
    fn set_max_traj_length(&mut self, max_traj_length: u128) -> &mut Self;
    fn set_config_update_polling(&mut self, config_update_polling: u32) -> &mut Self;
    fn set_local_model_path(&mut self, model_path: PathBuf) -> &mut Self;
    fn build(&self) -> TransportConfigParams;
    fn build_default() -> TransportConfigParams;
}

pub struct TransportConfigBuilder {
    agent_listener_address: Option<NetworkParams>,
    model_server_address: Option<NetworkParams>,
    trajectory_server_address: Option<NetworkParams>,
    grpc_idle_timeout: Option<u32>,
    max_traj_length: Option<u128>,
    config_update_polling: Option<u32>,
    local_model_path: Option<PathBuf>,
}

impl TransportConfigBuildParams for TransportConfigBuilder {
    fn set_agent_listener_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.agent_listener_address = Some(NetworkParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_model_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.model_server_address = Some(NetworkParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_trajectory_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.trajectory_server_address = Some(NetworkParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_grpc_idle_timeout(&mut self, grpc_idle_timeout: u32) -> &mut Self {
        self.grpc_idle_timeout = Some(grpc_idle_timeout);
        self
    }

    fn set_max_traj_length(&mut self, max_traj_length: u128) -> &mut Self {
        self.max_traj_length = Some(max_traj_length);
        self
    }

    fn set_config_update_polling(&mut self, config_update_polling: u32) -> &mut Self {
        self.config_update_polling = Some(config_update_polling);
        self
    }

    fn set_local_model_path(&mut self, model_path: PathBuf) -> &mut Self {
        self.local_model_path = Some(model_path);
        self
    }

    fn build(&self) -> TransportConfigParams {
        let agent_listener_address: NetworkParams = match &self.agent_listener_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7778".to_string(),
            },
        };

        let model_server_address: NetworkParams = match &self.model_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "50051".to_string(),
            },
        };

        let trajectory_server_address: NetworkParams = match &self.trajectory_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7776".to_string(),
            },
        };

        let grpc_idle_timeout: u32 = match &self.grpc_idle_timeout {
            Some(timeout) => *timeout,
            None => 30,
        };

        let max_traj_length: u128 = match &self.max_traj_length {
            Some(length) => *length,
            None => 1000,
        };

        let config_update_polling: u32 = match &self.config_update_polling {
            Some(polling) => *polling,
            None => 10,
        };

        let local_model_path: PathBuf = match &self.local_model_path {
            Some(path) => path.to_path_buf(),
            None => PathBuf::from(format!("{}_model.pt", std::process::id().to_string())),
        };

        TransportConfigParams {
            agent_listener_address,
            model_server_address,
            trajectory_server_address,
            grpc_idle_timeout,
            max_traj_length,
            local_model_path,
            config_update_polling,
        }
    }

    fn build_default() -> TransportConfigParams {
        let random_model_path_id: String = std::process::id().to_string();
        TransportConfigParams {
            agent_listener_address: NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7778".to_string(),
            },
            model_server_address: NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "50051".to_string(),
            },
            trajectory_server_address: NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7776".to_string(),
            },
            config_update_polling: 10,
            grpc_idle_timeout: 30,
            max_traj_length: 1000,
            local_model_path: PathBuf::from(format!("{}_model.pt", random_model_path_id)),
        }
    }
}
