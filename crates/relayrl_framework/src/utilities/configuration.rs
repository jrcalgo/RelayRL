use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{fs, fs::File, io::Read, path::PathBuf};

use dashmap::DashMap;

use crate::get_or_create_client_config_json_path;
use crate::network::HyperparameterArgs;

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
        "algorithm_name": "REINFORCE",
        "config_update_polling_seconds": 10.0,
        "init_hyperparameters": {
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
            },
            "_comment": "Add custom algorithm hyperparams here",
            "_comment2": "Make sure to add the algorithm name to the algorithm_name field",
            "_comment3": "These key-values will be sent to the server for initialization"
        },
        "trajectory_file_output": {
            "enabled": true,
            "encode": true,
            "output": {
                "directory": "data",
                "file_name": "action_data",
                "_comment": "csv, txt, json are supported.",
                "format": "json"
            }
        }
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
            "scaling_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7778"
            },
            "_comment2": "Only used when client-side inference is disabled or when client-side inference is used as fallback.",
            "inference_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7779"
            }
        },
        "local_model_module": {
            "directory": "model_module",
            "model_name": "client_model",
            "format": "pt"
        },
        "max_traj_length": 1000
    }
}"#;

pub(crate) const DEFAULT_SERVER_CONFIG_CONTENT: &str = r#"{
    "server_config": {
        "config_update_polling_seconds": 10.0,
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
            "scaling_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7778"
            },
            "_comment2": "Only available when client-side inference is disabled or when client-side inference is used as fallback.",
            "inference_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7779"
            }
        },
        "local_model_module": {
            "directory_name": "model_module",
            "model_name": "server_model"
        },
        "max_traj_length": 1000
    }
}"#;

/// TODO: Implement infernece server configuration file and builder components.
pub(crate) const DEFAULT_INFERENCE_SERVER_CONFIG_CONTENT: &str = r#"{
    "server_config": {
        "
    },
    "transport_config": {
        "addresses": {
            "inference_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7779"
            }
        },
        "config_update_polling": 10,
        "grpc_idle_timeout": 30,
        "local_model_module": {
            "directory_name": "model_module",
            "model_name": "inference_model.pt"
        },
        "max_traj_length": 1000
    },
}"#;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub enum Algorithm {
    DDPG,
    PPO,
    REINFORCE,
    TD3,
    CUSTOM(String),
    ConfigInit,
}

impl Algorithm {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "DDPG" => Some(Algorithm::DDPG),
            "PPO" => Some(Algorithm::PPO),
            "REINFORCE" => Some(Algorithm::REINFORCE),
            "TD3" => Some(Algorithm::TD3),
            "CONFIG_INIT" => Some(Algorithm::ConfigInit),
            _ => Some(Algorithm::CUSTOM(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Algorithm::DDPG => "DDPG",
            Algorithm::PPO => "PPO",
            Algorithm::REINFORCE => "REINFORCE",
            Algorithm::TD3 => "TD3",
            Algorithm::CUSTOM(custom) => custom,
            Algorithm::ConfigInit => "CONFIG_INIT",
        }
    }
}

/// Configuration parameters for various algorithms.
///
/// Each field is optional and holds algorithm-specific parameters.
///
/// In a future edition, this struct will be useful when multiple algorithm init is supported.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HyperparameterConfig {
    #[serde(rename = "DDPG")]
    pub ddpg: Option<DDPGParams>,
    #[serde(rename = "PPO")]
    pub ppo: Option<PPOParams>,
    #[serde(rename = "REINFORCE")]
    pub reinforce: Option<REINFORCEParams>,
    #[serde(rename = "TD3")]
    pub td3: Option<TD3Params>,
    // Add other fields later for in-house algorithms
    #[serde(rename = "custom")]
    pub custom: Option<CustomAlgorithmParams>,
}

impl HyperparameterConfig {
    /// Converts the hyperparameter config to a map of algorithm names to hyperparameter arguments.
    ///
    /// If an a specific algorithm is provided, only the hyperparameters for that algorithm are returned.
    ///
    /// Otherwise, all hyperparameters for all loaded algorithms are returned.
    pub fn to_args(&self, algorithm: Option<&Algorithm>) -> HashMap<Algorithm, HyperparameterArgs> {
        match algorithm {
            Some(algo) => match algo {
                Algorithm::DDPG => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(ddpg) = &self.ddpg {
                        let mut map = HashMap::new();
                        map.insert("seed".to_string(), ddpg.seed.to_string());
                        map.insert("gamma".to_string(), ddpg.gamma.to_string());
                        map.insert("tau".to_string(), ddpg.tau.to_string());
                        map.insert("learning_rate".to_string(), ddpg.learning_rate.to_string());
                        map.insert("batch_size".to_string(), ddpg.batch_size.to_string());
                        map.insert("buffer_size".to_string(), ddpg.buffer_size.to_string());
                        map.insert(
                            "learning_starts".to_string(),
                            ddpg.learning_starts.to_string(),
                        );
                        map.insert(
                            "policy_frequency".to_string(),
                            ddpg.policy_frequency.to_string(),
                        );
                        map.insert("noise_scale".to_string(), ddpg.noise_scale.to_string());
                        map.insert("train_iters".to_string(), ddpg.train_iters.to_string());
                        args.insert(Algorithm::DDPG, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::PPO => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(ppo) = &self.ppo {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), ppo.discrete.to_string());
                        map.insert("seed".to_string(), ppo.seed.to_string());
                        map.insert("traj_per_epoch".to_string(), ppo.traj_per_epoch.to_string());
                        map.insert("clip_ratio".to_string(), ppo.clip_ratio.to_string());
                        map.insert("gamma".to_string(), ppo.gamma.to_string());
                        map.insert("lam".to_string(), ppo.lam.to_string());
                        map.insert("pi_lr".to_string(), ppo.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), ppo.vf_lr.to_string());
                        map.insert("train_pi_iters".to_string(), ppo.train_pi_iters.to_string());
                        map.insert("train_v_iters".to_string(), ppo.train_v_iters.to_string());
                        map.insert("target_kl".to_string(), ppo.target_kl.to_string());
                        args.insert(Algorithm::PPO, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::REINFORCE => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(reinforce) = &self.reinforce {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), reinforce.discrete.to_string());
                        map.insert(
                            "with_vf_baseline".to_string(),
                            reinforce.with_vf_baseline.to_string(),
                        );
                        map.insert("seed".to_string(), reinforce.seed.to_string());
                        map.insert(
                            "traj_per_epoch".to_string(),
                            reinforce.traj_per_epoch.to_string(),
                        );
                        map.insert("gamma".to_string(), reinforce.gamma.to_string());
                        map.insert("lam".to_string(), reinforce.lam.to_string());
                        map.insert("pi_lr".to_string(), reinforce.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), reinforce.vf_lr.to_string());
                        map.insert(
                            "train_vf_iters".to_string(),
                            reinforce.train_vf_iters.to_string(),
                        );
                        args.insert(Algorithm::REINFORCE, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::TD3 => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(td3) = &self.td3 {
                        let mut map = HashMap::new();
                        map.insert("seed".to_string(), td3.seed.to_string());
                        map.insert("gamma".to_string(), td3.gamma.to_string());
                        map.insert("tau".to_string(), td3.tau.to_string());
                        map.insert("learning_rate".to_string(), td3.learning_rate.to_string());
                        map.insert("batch_size".to_string(), td3.batch_size.to_string());
                        map.insert("buffer_size".to_string(), td3.buffer_size.to_string());
                        map.insert(
                            "exploration_noise".to_string(),
                            td3.exploration_noise.to_string(),
                        );
                        map.insert("policy_noise".to_string(), td3.policy_noise.to_string());
                        map.insert("noise_clip".to_string(), td3.noise_clip.to_string());
                        map.insert(
                            "learning_starts".to_string(),
                            td3.learning_starts.to_string(),
                        );
                        map.insert(
                            "policy_frequency".to_string(),
                            td3.policy_frequency.to_string(),
                        );
                        args.insert(Algorithm::TD3, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::CUSTOM(custom_name) => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(custom) = &self.custom {
                        let mut map = HashMap::new();
                        map.insert(
                            "custom_algorithm_name".to_string(),
                            custom.algorithm_name.as_str().to_string(),
                        );
                        for (key, value) in custom.hyperparams.iter() {
                            map.insert(key.to_string(), value.to_string());
                        }
                        args.insert(
                            Algorithm::CUSTOM(custom_name.clone()),
                            HyperparameterArgs::Map(map),
                        );
                    }
                    args
                }
                Algorithm::ConfigInit => {
                    return HashMap::new();
                }
            },
            None => {
                let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                if let Some(ddpg) = &self.ddpg {
                    let mut map = HashMap::new();
                    map.insert("seed".to_string(), ddpg.seed.to_string());
                    map.insert("gamma".to_string(), ddpg.gamma.to_string());
                    map.insert("tau".to_string(), ddpg.tau.to_string());
                    map.insert("learning_rate".to_string(), ddpg.learning_rate.to_string());
                    map.insert("batch_size".to_string(), ddpg.batch_size.to_string());
                    map.insert("buffer_size".to_string(), ddpg.buffer_size.to_string());
                    map.insert(
                        "learning_starts".to_string(),
                        ddpg.learning_starts.to_string(),
                    );
                    map.insert(
                        "policy_frequency".to_string(),
                        ddpg.policy_frequency.to_string(),
                    );
                    map.insert("noise_scale".to_string(), ddpg.noise_scale.to_string());
                    map.insert("train_iters".to_string(), ddpg.train_iters.to_string());
                    args.insert(Algorithm::DDPG, HyperparameterArgs::Map(map));
                }

                if let Some(ppo) = &self.ppo {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), ppo.discrete.to_string());
                    map.insert("seed".to_string(), ppo.seed.to_string());
                    map.insert("traj_per_epoch".to_string(), ppo.traj_per_epoch.to_string());
                    map.insert("clip_ratio".to_string(), ppo.clip_ratio.to_string());
                    map.insert("gamma".to_string(), ppo.gamma.to_string());
                    map.insert("lam".to_string(), ppo.lam.to_string());
                    map.insert("pi_lr".to_string(), ppo.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), ppo.vf_lr.to_string());
                    map.insert("train_pi_iters".to_string(), ppo.train_pi_iters.to_string());
                    map.insert("train_v_iters".to_string(), ppo.train_v_iters.to_string());
                    map.insert("target_kl".to_string(), ppo.target_kl.to_string());
                    args.insert(Algorithm::PPO, HyperparameterArgs::Map(map));
                }

                if let Some(reinforce) = &self.reinforce {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), reinforce.discrete.to_string());
                    map.insert(
                        "with_vf_baseline".to_string(),
                        reinforce.with_vf_baseline.to_string(),
                    );
                    map.insert("seed".to_string(), reinforce.seed.to_string());
                    map.insert(
                        "traj_per_epoch".to_string(),
                        reinforce.traj_per_epoch.to_string(),
                    );
                    map.insert("gamma".to_string(), reinforce.gamma.to_string());
                    map.insert("lam".to_string(), reinforce.lam.to_string());
                    map.insert("pi_lr".to_string(), reinforce.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), reinforce.vf_lr.to_string());
                    map.insert(
                        "train_vf_iters".to_string(),
                        reinforce.train_vf_iters.to_string(),
                    );
                    args.insert(Algorithm::REINFORCE, HyperparameterArgs::Map(map));
                }

                if let Some(td3) = &self.td3 {
                    let mut map = HashMap::new();
                    map.insert("seed".to_string(), td3.seed.to_string());
                    map.insert("gamma".to_string(), td3.gamma.to_string());
                    map.insert("tau".to_string(), td3.tau.to_string());
                    map.insert("learning_rate".to_string(), td3.learning_rate.to_string());
                    map.insert("batch_size".to_string(), td3.batch_size.to_string());
                    map.insert("buffer_size".to_string(), td3.buffer_size.to_string());
                    map.insert(
                        "exploration_noise".to_string(),
                        td3.exploration_noise.to_string(),
                    );
                    map.insert("policy_noise".to_string(), td3.policy_noise.to_string());
                    map.insert("noise_clip".to_string(), td3.noise_clip.to_string());
                    map.insert(
                        "learning_starts".to_string(),
                        td3.learning_starts.to_string(),
                    );
                    map.insert(
                        "policy_frequency".to_string(),
                        td3.policy_frequency.to_string(),
                    );
                    args.insert(Algorithm::TD3, HyperparameterArgs::Map(map));
                }

                if let Some(custom) = &self.custom {
                    let mut map = HashMap::new();
                    map.insert(
                        "custom_algorithm_name".to_string(),
                        custom.algorithm_name.as_str().to_string(),
                    );
                    for (key, value) in custom.hyperparams.iter() {
                        map.insert(key.to_string(), value.to_string());
                    }
                    args.insert(
                        Algorithm::CUSTOM(custom.algorithm_name.as_str().to_string()),
                        HyperparameterArgs::Map(map),
                    );
                }

                args
            }
        }
    }
}

impl Default for HyperparameterConfig {
    fn default() -> Self {
        Self {
            ddpg: None,
            ppo: None,
            reinforce: Some(REINFORCEParams::default()),
            td3: None,
            custom: None,
        }
    }
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
    pub algorithm_name: Algorithm,
    pub hyperparams: HashMap<String, String>,
}

impl Default for CustomAlgorithmParams {
    fn default() -> Self {
        Self {
            algorithm_name: Algorithm::CUSTOM("".to_string()),
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

/// Tensorboard configuration structure.
///
/// Contains optional tensorboard writer parameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorboardConfig {
    pub training_tensorboard: Option<TensorboardParams>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrajectoryFileOutput {
    pub directory: String,
    pub file_name: String,
    pub format: String,
}

impl Default for TrajectoryFileOutput {
    fn default() -> Self {
        Self {
            directory: "data".to_string(),
            file_name: "experiment_data".to_string(),
            format: "json".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrajectoryFileOutputParams {
    pub enabled: bool,
    pub encode: bool,
    pub output: TrajectoryFileOutput,
}

impl Default for TrajectoryFileOutputParams {
    fn default() -> Self {
        Self {
            enabled: false,
            encode: true,
            output: TrajectoryFileOutput::default(),
        }
    }
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
    pub algorithm_name: String,
    pub config_path: PathBuf,
    pub config_update_polling_seconds: f32,
    pub init_hyperparameters: HyperparameterConfig,
    pub trajectory_file_output: TrajectoryFileOutputParams,
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
                algorithm_name: _algorithm_name,
                config_path: _config_path,
                config_update_polling_seconds: client_config.config_update_polling_seconds,
                init_hyperparameters: client_config.init_hyperparameters,
                trajectory_file_output: client_config.trajectory_file_output,
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
                            algorithm_name: "REINFORCE".to_string(),
                            config_path: PathBuf::from("client_config.json"),
                            config_update_polling_seconds: 10.0,
                            init_hyperparameters: HyperparameterConfig::default(),
                            trajectory_file_output: TrajectoryFileOutputParams::default(),
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

    pub fn get_algorithm_name(&self) -> &str {
        &self.client_config.algorithm_name
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.client_config.config_path
    }

    pub fn get_init_hyperparameters(&self) -> &HyperparameterConfig {
        &self.client_config.init_hyperparameters
    }

    pub fn get_init_hyperparameter_args(
        &self,
        algorithm: Option<&Algorithm>,
    ) -> HashMap<Algorithm, HyperparameterArgs> {
        self.client_config.init_hyperparameters.to_args(algorithm)
    }

    pub fn get_trajectory_file_output(&self) -> &TrajectoryFileOutputParams {
        &self.client_config.trajectory_file_output
    }

    pub fn get_transport_config(&self) -> &TransportConfigParams {
        &self.transport_config
    }
}

pub trait ClientConfigBuildParams {
    fn set_algorithm_name(&mut self, algorithm_name: &str) -> &mut Self;
    fn set_config_path(&mut self, config_path: &str) -> &mut Self;
    fn set_init_hyperparameters(&mut self, init_hyperparameters: HyperparameterConfig)
    -> &mut Self;
    fn set_trajectory_file_output(
        &mut self,
        trajectory_file_output: TrajectoryFileOutputParams,
    ) -> &mut Self;
    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self;
    fn build(&self) -> ClientConfigLoader;
    fn build_default() -> ClientConfigLoader;
}

pub struct ClientConfigBuilder {
    algorithm_name: Option<String>,
    config_path: Option<PathBuf>,
    config_update_polling_seconds: Option<f32>,
    init_hyperparameters: Option<HyperparameterConfig>,
    transport_config: Option<TransportConfigParams>,
    trajectory_file_output: Option<TrajectoryFileOutputParams>,
}

impl ClientConfigBuildParams for ClientConfigBuilder {
    fn set_algorithm_name(&mut self, algorithm_name: &str) -> &mut Self {
        self.algorithm_name = Some(algorithm_name.to_string());
        self
    }

    fn set_config_path(&mut self, config_path: &str) -> &mut Self {
        self.config_path = Some(PathBuf::from(config_path));
        self
    }

    fn set_init_hyperparameters(
        &mut self,
        init_hyperparameters: HyperparameterConfig,
    ) -> &mut Self {
        self.init_hyperparameters = Some(init_hyperparameters);
        self
    }

    fn set_trajectory_file_output(
        &mut self,
        trajectory_file_output: TrajectoryFileOutputParams,
    ) -> &mut Self {
        self.trajectory_file_output = Some(trajectory_file_output);
        self
    }

    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self {
        self.transport_config = Some(transport_config);
        self
    }

    fn build(&self) -> ClientConfigLoader {
        let client_config: ClientConfigParams = ClientConfigParams {
            algorithm_name: self
                .algorithm_name
                .clone()
                .unwrap_or_else(|| "REINFORCE".to_string()),
            config_path: self
                .config_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("client_config.json")),
            config_update_polling_seconds: self
                .config_update_polling_seconds
                .clone()
                .unwrap_or_else(|| 10.0),
            init_hyperparameters: self
                .init_hyperparameters
                .clone()
                .unwrap_or_else(|| HyperparameterConfig::default()),
            trajectory_file_output: self
                .trajectory_file_output
                .clone()
                .unwrap_or_else(|| TrajectoryFileOutputParams::default()),
        };

        let transport_config: TransportConfigParams = match &self.transport_config {
            Some(transport_config) => TransportConfigParams {
                inference_server_address: transport_config.inference_server_address.clone(),
                agent_listener_address: transport_config.agent_listener_address.clone(),
                model_server_address: transport_config.model_server_address.clone(),
                trajectory_server_address: transport_config.trajectory_server_address.clone(),
                scaling_server_address: transport_config.scaling_server_address.clone(),
                max_traj_length: transport_config.max_traj_length,
                local_model_module: transport_config.local_model_module.clone(),
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
                algorithm_name: "REINFORCE".to_string(),
                config_path: PathBuf::from("client_config.json"),
                config_update_polling_seconds: 10.0,
                init_hyperparameters: HyperparameterConfig::default(),
                trajectory_file_output: TrajectoryFileOutputParams::default(),
            },
            transport_config: TransportConfigBuilder::build_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfigParams {
    pub config_path: PathBuf,
    pub default_hyperparameters: Option<HyperparameterConfig>,
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

    pub fn get_hyperparameters(&self) -> &Option<HyperparameterConfig> {
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
        hyperparameter_args: HyperparameterArgs,
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
    default_hyperparameters: Option<HyperparameterConfig>,
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
        hyperparameter_args: HyperparameterArgs,
    ) -> &mut Self {
        let hp_map: HashMap<String, String> =
            crate::network::parse_args(&Some(hyperparameter_args.clone()));

        // Start from defaults for all supported algorithms.
        let mut all_cfg = HyperparameterConfig {
            ddpg: Some(DDPGParams::default()),
            ppo: Some(PPOParams::default()),
            reinforce: Some(REINFORCEParams::default()),
            td3: Some(TD3Params::default()),
            custom: None,
        };

        let algo_upper = algorithm_name.to_uppercase();
        let algorithm = Algorithm::from_str(algo_upper.as_str()).unwrap_or(Algorithm::REINFORCE);

        match algorithm {
            Algorithm::DDPG => {
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
            Algorithm::PPO => {
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
            Algorithm::REINFORCE => {
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
            Algorithm::TD3 => {
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
            Algorithm::CUSTOM(custom_algorithm) => {
                let mut custom = all_cfg.custom.take().unwrap_or_default();
                custom.algorithm_name = Algorithm::CUSTOM(custom_algorithm.clone());
                custom.hyperparams = hp_map.clone();
                all_cfg.custom = Some(custom);
            }
            Algorithm::ConfigInit => {
                eprintln!(
                    "[ServerConfigBuilder] ConfigInit is not a valid algorithm for hyperparameters"
                );
                return self;
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
                inference_server_address: transport_config.inference_server_address.clone(),
                agent_listener_address: transport_config.agent_listener_address.clone(),
                model_server_address: transport_config.model_server_address.clone(),
                trajectory_server_address: transport_config.trajectory_server_address.clone(),
                scaling_server_address: transport_config.scaling_server_address.clone(),
                max_traj_length: transport_config.max_traj_length,
                local_model_module: transport_config.local_model_module.clone(),
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
    pub inference_server_address: NetworkParams,
    pub agent_listener_address: NetworkParams,
    pub model_server_address: NetworkParams,
    pub trajectory_server_address: NetworkParams,
    pub scaling_server_address: NetworkParams,
    pub max_traj_length: u128,
    pub local_model_module: LocalModelModuleParams,
}

impl TransportConfigParams {
    pub fn get_inference_server_address(&self) -> &NetworkParams {
        &self.inference_server_address
    }

    pub fn get_agent_listener_address(&self) -> &NetworkParams {
        &self.agent_listener_address
    }

    pub fn get_model_server_address(&self) -> &NetworkParams {
        &self.model_server_address
    }

    pub fn get_trajectory_server_address(&self) -> &NetworkParams {
        &self.trajectory_server_address
    }

    pub fn get_scaling_server_address(&self) -> &NetworkParams {
        &self.scaling_server_address
    }
}

pub trait TransportConfigBuildParams {
    fn set_inference_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_agent_listener_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_model_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_trajectory_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_scaling_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self;
    fn set_max_traj_length(&mut self, max_traj_length: u128) -> &mut Self;
    fn set_local_model_module(&mut self, directory_name: &str, model_name: &str) -> &mut Self;
    fn build(&self) -> TransportConfigParams;
    fn build_default() -> TransportConfigParams;
}

pub struct TransportConfigBuilder {
    inference_server_address: Option<NetworkParams>,
    agent_listener_address: Option<NetworkParams>,
    model_server_address: Option<NetworkParams>,
    trajectory_server_address: Option<NetworkParams>,
    scaling_server_address: Option<NetworkParams>,
    max_traj_length: Option<u128>,
    local_model_module: Option<LocalModelModuleParams>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocalModelModuleParams {
    pub directory: String,
    pub model_name: String,
    pub format: String,
}

impl TransportConfigBuildParams for TransportConfigBuilder {
    fn set_inference_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.inference_server_address = Some(NetworkParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

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

    fn set_scaling_server_address(&mut self, prefix: &str, host: &str, port: &str) -> &mut Self {
        self.scaling_server_address = Some(NetworkParams {
            prefix: prefix.to_string(),
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_max_traj_length(&mut self, max_traj_length: u128) -> &mut Self {
        self.max_traj_length = Some(max_traj_length);
        self
    }

    fn set_local_model_module(&mut self, directory_name: &str, model_name: &str) -> &mut Self {
        self.local_model_module = Some(LocalModelModuleParams {
            directory: directory_name.to_string(),
            format: "pt".to_string(),
            model_name: model_name.to_string(),
        });
        self
    }

    fn build(&self) -> TransportConfigParams {
        let inference_server_address: NetworkParams = match &self.inference_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "50050".to_string(),
            },
        };
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

        let scaling_server_address: NetworkParams = match &self.scaling_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7777".to_string(),
            },
        };

        let max_traj_length: u128 = match &self.max_traj_length {
            Some(length) => *length,
            None => 1000,
        };

        let local_model_module: LocalModelModuleParams = match &self.local_model_module {
            Some(module) => module.clone(),
            None => LocalModelModuleParams {
                directory: "model_module".to_string(),
                format: "pt".to_string(),
                model_name: "model".to_string(),
            },
        };

        TransportConfigParams {
            inference_server_address,
            agent_listener_address,
            model_server_address,
            trajectory_server_address,
            scaling_server_address,
            max_traj_length,
            local_model_module,
        }
    }

    fn build_default() -> TransportConfigParams {
        TransportConfigParams {
            inference_server_address: NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "50050".to_string(),
            },
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
            scaling_server_address: NetworkParams {
                prefix: "tcp://".to_string(),
                host: "127.0.0.1".to_string(),
                port: "7777".to_string(),
            },
            max_traj_length: 1000,
            local_model_module: LocalModelModuleParams {
                directory: "model_module".to_string(),
                format: "pt".to_string(),
                model_name: "model".to_string(),
            },
        }
    }
}
