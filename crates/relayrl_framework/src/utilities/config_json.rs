use once_cell::sync::Lazy;
use std::fs;
use std::path::PathBuf;

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
                log::info!(
                    "[ConfigLoader - load_config] Found config.json in current directory: {:?}",
                    $path
                );
                Some($path)
            } else {
                match fs::write($path, DEFAULT_CLIENT_CONFIG_JSON) {
                    Ok(_) => {
                        log::info!(
                            "[ConfigLoader - load_config] Created new config at: {:?}",
                            $path
                        );
                        Some($path)
                    }
                    Err(e) => {
                        log::error!(
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
    macro_rules! resolve_training_server_config_json_path {
        ($path: expr) => {
            match $path {
                Some(p) => get_or_create_training_server_config_json_path!(p.clone()),
                None => DEFAULT_TRAINING_SERVER_CONFIG_PATH.clone(),
            }
        };
        ($path: literal) => {
            get_or_create_training_server_config_json_path!(std::path::PathBuf::from($path))
        };
    }

    /// Will write config file if not found in provided path.
    /// Reads file if found, writes new file if not
    #[macro_export]
    macro_rules! get_or_create_training_server_config_json_path {
        ($path: expr) => {
            if $path.exists() {
                log::info!(
                    "[ConfigLoader - load_config] Found config.json in current directory: {:?}",
                    $path
                );
                Some($path)
            } else {
                match fs::write($path, DEFAULT_TRAINING_SERVER_CONFIG_JSON) {
                    Ok(_) => {
                        log::info!(
                            "[ConfigLoader - load_config] Created new config at: {:?}",
                            $path
                        );
                        Some($path)
                    }
                    Err(e) => {
                        log::error!(
                            "[ConfigLoader - load_config] Failed to create config file: {}",
                            e
                        );
                        None
                    }
                }
            }
        };
    }

    #[macro_export]
    macro_rules! resolve_inference_server_config_json_path {
        ($path: expr) => {
            match $path {
                Some(p) => get_or_create_inference_server_config_json_path!(p.clone()),
                None => DEFAULT_INFERENCE_SERVER_CONFIG_PATH.clone(),
            }
        };
        ($path: literal) => {
            get_or_create_inference_server_config_json_path!(std::path::PathBuf::from($path))
        };
    }

    #[macro_export]
    macro_rules! get_or_create_inference_server_config_json_path {
        ($path: expr) => {
            if $path.exists() {
                log::info!(
                    "[ConfigLoader - load_config] Found config.json in current directory: {:?}",
                    $path
                );
                Some($path)
            } else {
                match fs::write($path, DEFAULT_INFERENCE_SERVER_CONFIG_JSON) {
                    Ok(_) => {
                        log::info!(
                            "[ConfigLoader - load_config] Created new config at: {:?}",
                            $path
                        );
                        Some($path)
                    }
                    Err(e) => {
                        log::error!(
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

pub static DEFAULT_TRAINING_SERVER_CONFIG_PATH: Lazy<Option<PathBuf>> = Lazy::new(|| {
    get_or_create_training_server_config_json_path!(PathBuf::from("training_server_config.json"))
});

pub static DEFAULT_INFERENCE_SERVER_CONFIG_PATH: Lazy<Option<PathBuf>> = Lazy::new(|| {
    get_or_create_inference_server_config_json_path!(PathBuf::from("inference_server_config.json"))
});

pub(crate) const DEFAULT_CLIENT_CONFIG_JSON: &str = r#"{
    "client_config": {
        "config_update_polling_seconds": 10.0,
        "init_hyperparameters": {
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
            "IPPO": {
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
            "MAPPO": {
                "discrete": true,
                "gamma": 0.99,
                "lam": 0.97,
                "clip_ratio": 0.2,
                "pi_lr": 3e-4,
                "vf_lr": 1e-3,
                "train_pi_iters": 80,
                "train_vf_iters": 80,
                "target_kl": 0.01,
                "traj_per_epoch": 8
            },
            "CUSTOM": {
                "_comment": "Add custom algorithm hyperparams here formatted just like the other algorithms. i.e. \"MAPPO\": {...}",
                "_comment2": "Make sure to add the algorithm name to the algorithm_name field",
                "_comment3": "These key-values will be sent to the server for initialization"
            }
        },
        "router_buffer_size_per_actor": 1000,
        "trajectory_file_output": {
            "directory": "experiment_data",
            "_comment": "use `Csv` or `Arrow`",
            "file_type": "Csv"
        },
        "metrics_meter_name": "relayrl-client",
        "metrics_otlp_endpoint": {
            "prefix": "http://",
            "host": "127.0.0.1",
            "port": "4317"
        }
    },
    "transport_config": {
        "nats_addresses": {
            "inference_server_address": {
                "host": "127.0.0.1",
                "port": "50050"
            },
            "training_server_address": {
                "host": "127.0.0.1",
                "port": "50051"
            }
        },
        "zmq_addresses": {
            "inference_addresses": {
                "inference_server_address": {
                    "host": "127.0.0.1",
                    "port": "7800"
                },
                "inference_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7801"
                }
            },
            "training_addresses": {
                "model_server_address": {
                    "host": "127.0.0.1",
                    "port": "50051"
                },
                "trajectory_server_address": {
                    "host": "127.0.0.1",
                    "port": "7776"
                },
                "agent_listener_address": {
                    "host": "127.0.0.1",
                    "port": "7777"
                },
                "training_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7778"
                }
            }
        },
        "local_model_module": {
            "directory": "model_module",
            "model_name": "client_model",
            "format": "pt"
        }
    }
}"#;

pub(crate) const DEFAULT_TRAINING_SERVER_CONFIG_JSON: &str = r#"{
    "training_server_config": {
        "config_update_polling_seconds": 10.0,
        "default_hyperparameters": {
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
            "IPPO": {
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
            "MAPPO": {
                "discrete": true,
                "seed": 0,
                "traj_per_epoch": 1,
                "gamma": 0.99,
                "lam": 0.97,
                "clip_ratio": 0.2,
                "pi_lr": 3e-4,
                "vf_lr": 1e-3,
                "train_pi_iters": 80,
                "train_vf_iters": 80,
                "target_kl": 0.01,
                "traj_per_epoch": 8
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
        "nats_addresses": {
            "inference_server_address": {
                "host": "127.0.0.1",
                "port": "50050"
            },
            "training_server_address": {
                "host": "127.0.0.1",
                "port": "50051"
            }
        },
        "zmq_addresses": {
            "inference_addresses": {
                "inference_server_address": {
                    "host": "127.0.0.1",
                    "port": "7800"
                },
                "inference_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7801"
                }
            },
            "training_addresses": {
                "model_server_address": {
                    "host": "127.0.0.1",
                    "port": "50051"
                },
                "trajectory_server_address": {
                    "host": "127.0.0.1",
                    "port": "7776"
                },
                "agent_listener_address": {
                    "host": "127.0.0.1",
                    "port": "7777"
                },
                "training_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7778"
                }
            }
        },
        "local_model_module": {
            "directory": "model_module",
            "model_name": "training_server_model",
            "format": "pt"
        }
    }
}"#;

/// TODO: Implement infernece server configuration file and builder components.
pub(crate) const DEFAULT_INFERENCE_SERVER_CONFIG_JSON: &str = r#"{
    "inference_server_config": {
        "config_update_polling_seconds": 10.0,
        "transport_config": {
            "nats_addresses": {
                "inference_server_address": {
                    "host": "127.0.0.1",
                    "port": "50050"
                },
                "training_server_address": {
                    "host": "127.0.0.1",
                    "port": "50051"
                }
            },
            "zmq_addresses": {
                "inference_addresses": {
                    "inference_server_address": {
                        "host": "127.0.0.1",
                        "port": "7800"
                    },
                    "inference_scaling_server_address": {
                        "host": "127.0.0.1",
                        "port": "7801"
                    }
                },
                "training_addresses": {
                    "model_server_address": {
                        "host": "127.0.0.1",
                        "port": "50051"
                    },
                    "trajectory_server_address": {
                        "host": "127.0.0.1",
                        "port": "7776"
                    },
                    "agent_listener_address": {
                        "host": "127.0.0.1",
                        "port": "7777"
                    },
                    "training_scaling_server_address": {
                        "host": "127.0.0.1",
                        "port": "7778"
                    }
                }
            },
            "local_model_module": {
                "directory": "model_module",
                "model_name": "inference_server_model",
                "format": "pt"
            }
        }
    }
}"#;
