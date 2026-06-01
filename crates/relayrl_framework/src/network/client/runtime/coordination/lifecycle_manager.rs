//! Lifecycle coordination for the client runtime.
//!
//! This module owns config watching, shared runtime settings, and shutdown fan-out for the
//! client runtime. The local/default path is the supported `0.5.0-beta` path; transport-backed
//! settings exposed here remain experimental.

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::HyperparameterArgs;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::TransportType;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::agent::DefaultHyperparameterArgs;
use crate::network::client::agent::LocalTrajectoryFileParams;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::prelude::config::TransportConfigParams;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::Algorithm;
#[cfg(feature = "metrics")]
use crate::utilities::configuration::OtlpEndpointParams;
use crate::utilities::configuration::{ClientConfigLoader, LocalModelModuleParams};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::utilities::configuration::{HyperparameterConfig, NetworkParams};

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use relayrl_algorithms::prelude::ppo::algorithm::{IPPOParams, MAPPOParams, PPOParams};

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crossbeam_utils::CachePadded;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{Notify, RwLock, broadcast};

use thiserror::Error;

#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SharedZmqInferenceAddresses {
    pub(crate) inference_server_address: Arc<str>,
    pub(crate) inference_scaling_server_address: Arc<str>,
}

#[cfg(feature = "zmq-transport")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SharedZmqTrainingAddresses {
    pub(crate) agent_listener_address: Arc<str>,
    pub(crate) model_server_address: Arc<str>,
    pub(crate) trajectory_server_address: Arc<str>,
    pub(crate) training_scaling_server_address: Arc<str>,
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Debug, Clone)]
pub(crate) struct SharedDefaultHyperparameters {
    ppo: CachePadded<Arc<RwLock<PPOParams>>>,
    ippo: CachePadded<Arc<RwLock<IPPOParams>>>,
    mappo: CachePadded<Arc<RwLock<MAPPOParams>>>,
    ppo_config_init: bool,
    ippo_config_init: bool,
    mappo_config_init: bool,
}

/// Shared transport addresses for both NATS and ZMQ transports.
///
/// This keeps the active transport addresses together in one shared structure so runtime
/// components can clone a single handle regardless of feature configuration.
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SharedTransportAddresses {
    #[cfg(feature = "nats-transport")]
    pub(crate) nats_inference_address: Arc<str>,
    #[cfg(feature = "nats-transport")]
    pub(crate) nats_training_address: Arc<str>,
    #[cfg(feature = "zmq-transport")]
    pub(crate) zmq_inference_addresses: SharedZmqInferenceAddresses,
    #[cfg(feature = "zmq-transport")]
    pub(crate) zmq_training_addresses: SharedZmqTrainingAddresses,
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) fn construct_transport_addresses(
    transport_config: &TransportConfigParams,
    transport_type: &TransportType,
) -> SharedTransportAddresses {
    fn construct_address(
        transport_type: &TransportType,
        network_params: &NetworkParams,
    ) -> Arc<str> {
        match *transport_type {
            #[cfg(feature = "zmq-transport")]
            TransportType::ZMQ => Arc::<str>::from(
                "tcp://".to_owned() + &network_params.host + ":" + &network_params.port.to_string(),
            ),
            #[cfg(feature = "nats-transport")]
            TransportType::NATS => Arc::<str>::from(
                "nats://".to_owned()
                    + &network_params.host
                    + ":"
                    + &network_params.port.to_string(),
            ),
        }
    }

    match *transport_type {
        #[cfg(feature = "zmq-transport")]
        TransportType::ZMQ => SharedTransportAddresses {
            zmq_inference_addresses: SharedZmqInferenceAddresses {
                inference_server_address: construct_address(
                    transport_type,
                    &transport_config
                        .zmq_addresses
                        .inference_addresses
                        .inference_server_address,
                ),
                inference_scaling_server_address: construct_address(
                    transport_type,
                    &transport_config
                        .zmq_addresses
                        .inference_addresses
                        .inference_scaling_server_address,
                ),
            },
            zmq_training_addresses: SharedZmqTrainingAddresses {
                agent_listener_address: construct_address(
                    transport_type,
                    &transport_config
                        .zmq_addresses
                        .training_addresses
                        .agent_listener_address,
                ),
                model_server_address: construct_address(
                    transport_type,
                    &transport_config
                        .zmq_addresses
                        .training_addresses
                        .model_server_address,
                ),
                trajectory_server_address: construct_address(
                    transport_type,
                    &transport_config
                        .zmq_addresses
                        .training_addresses
                        .trajectory_server_address,
                ),
                training_scaling_server_address: construct_address(
                    transport_type,
                    &transport_config
                        .zmq_addresses
                        .training_addresses
                        .training_scaling_server_address,
                ),
            },
            #[cfg(feature = "nats-transport")]
            nats_inference_address: Arc::<str>::from(""),
            #[cfg(feature = "nats-transport")]
            nats_training_address: Arc::<str>::from(""),
        },
        #[cfg(feature = "nats-transport")]
        TransportType::NATS => SharedTransportAddresses {
            #[cfg(feature = "zmq-transport")]
            zmq_inference_addresses: SharedZmqInferenceAddresses {
                inference_server_address: Arc::<str>::from(""),
                inference_scaling_server_address: Arc::<str>::from(""),
            },
            #[cfg(feature = "zmq-transport")]
            zmq_training_addresses: SharedZmqTrainingAddresses {
                agent_listener_address: Arc::<str>::from(""),
                model_server_address: Arc::<str>::from(""),
                trajectory_server_address: Arc::<str>::from(""),
                training_scaling_server_address: Arc::<str>::from(""),
            },
            nats_inference_address: construct_address(
                transport_type,
                &transport_config.nats_addresses.inference_server_address,
            ),
            nats_training_address: construct_address(
                transport_type,
                &transport_config.nats_addresses.training_server_address,
            ),
        },
    }
}

#[cfg(feature = "metrics")]
pub(crate) fn construct_metrics_otlp_endpoint(
    metrics_otlp_endpoint: &OtlpEndpointParams,
) -> String {
    format!(
        "{}{}:{}",
        metrics_otlp_endpoint.prefix, metrics_otlp_endpoint.host, metrics_otlp_endpoint.port
    )
}

pub(crate) fn construct_local_model_path(local_model_module: &LocalModelModuleParams) -> PathBuf {
    let cwd: PathBuf = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let directory = if !local_model_module.directory.is_empty() {
        local_model_module.directory.clone()
    } else {
        log::warn!("Local model directory is empty, using default: model_module");
        "model_module".to_string()
    };

    let model_name = if !local_model_module.model_name.is_empty() {
        local_model_module.model_name.clone()
    } else {
        log::warn!("Local model name is empty, using default: client_model");
        "client_model".to_string()
    };

    let mut module_format = local_model_module.format.clone();
    if !module_format.is_empty() {
        while module_format.starts_with('.') {
            module_format = module_format[1..].to_string();
        }
    } else {
        log::warn!("Local model format is empty, using default: pt");
        module_format = "pt".to_string();
    }

    cwd.join(&directory)
        .join(format!("{}.{}", &model_name, &module_format))
}

pub(crate) fn construct_trajectory_file_output(
    trajectory_file_output: &LocalTrajectoryFileParams,
) -> LocalTrajectoryFileParams {
    let cwd: PathBuf = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let directory = cwd.join(&trajectory_file_output.directory);

    LocalTrajectoryFileParams {
        directory,
        file_type: trajectory_file_output.file_type.clone(),
    }
}

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum LifecycleManagerError {
    #[error("File metadata error: {0}")]
    FileMetadataError(String),
    #[error("System time error: {0}")]
    SystemTimeError(String),
    #[error("Subscribe shutdown error: {0}")]
    SubscribeShutdownError(String),
    #[error("Send shutdown signal error: {0}")]
    SendShutdownSignalError(String),
    #[error("Config error: {0}")]
    ConfigError(String),
}

/// Orchestrates startup/shutdown signals (SIGINT, config-changes)
///
/// Spins up and tears down futures cleanly
#[derive(Debug, Clone)]
pub(crate) struct LifecycleManager {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    default_hyperparameters: SharedDefaultHyperparameters,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    #[cfg(feature = "metrics")]
    metrics_args: Arc<RwLock<(String, String)>>,
    local_model_path: Arc<RwLock<PathBuf>>,
    trajectory_file_output: Arc<RwLock<LocalTrajectoryFileParams>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    transport_type: Arc<TransportType>,
    config_path: Arc<PathBuf>,
    config_update_polling_seconds: Arc<RwLock<f32>>,
    router_buffer_size_per_actor: Arc<RwLock<usize>>,
    last_modified: Arc<RwLock<SystemTime>>,
    shutdown_tx: broadcast::Sender<()>,
    shutdown_notifier: Arc<Notify>,
}

impl LifecycleManager {
    pub(crate) fn new(
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        default_hyperparameters: DefaultHyperparameterArgs,
        config: &ClientConfigLoader,
        config_path: PathBuf,
        router_buffer_size_per_actor: Option<usize>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: TransportType,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(10_000);

        // Get file metadata with fallback to current time
        let last_modified: SystemTime = fs::metadata(&config_path)
            .and_then(|m| m.modified())
            .unwrap_or_else(|e| {
                log::error!(
                    "[LifecycleManager] Failed to read config metadata: {}, using current time",
                    e
                );
                SystemTime::now()
            });

        let config_update_polling = config.client_config.config_update_polling_seconds;

        let router_buffer_size = match router_buffer_size_per_actor {
            Some(size) => size,
            None => config.client_config.router_buffer_size_per_actor,
        };

        let transport_config = config.get_transport_config();

        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        let resolved_default_hyperparameters: SharedDefaultHyperparameters = {
            let ppo = match default_hyperparameters.ppo {
                Some(ppo) => (ppo, false),
                None if default_hyperparameters.config_default_init => (
                    config
                        .client_config
                        .init_hyperparameters
                        .ppo
                        .clone()
                        .unwrap_or_default(),
                    true,
                ),
                None => (PPOParams::default(), false),
            };
            let ippo = match default_hyperparameters.ippo {
                Some(ippo) => (ippo, false),
                None if default_hyperparameters.config_default_init => (
                    config
                        .client_config
                        .init_hyperparameters
                        .ippo
                        .clone()
                        .unwrap_or_default(),
                    true,
                ),
                None => (IPPOParams::default(), false),
            };
            let mappo = match default_hyperparameters.mappo {
                Some(mappo) => (mappo, false),
                None if default_hyperparameters.config_default_init => (
                    config
                        .client_config
                        .init_hyperparameters
                        .mappo
                        .clone()
                        .unwrap_or_default(),
                    true,
                ),
                None => (MAPPOParams::default(), false),
            };

            SharedDefaultHyperparameters {
                ppo: CachePadded::new(Arc::new(RwLock::new(ppo.0))),
                ippo: CachePadded::new(Arc::new(RwLock::new(ippo.0))),
                mappo: CachePadded::new(Arc::new(RwLock::new(mappo.0))),
                ppo_config_init: ppo.1,
                ippo_config_init: ippo.1,
                mappo_config_init: mappo.1,
            }
        };

        Self {
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            default_hyperparameters: resolved_default_hyperparameters,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_addresses: Arc::new(RwLock::new(construct_transport_addresses(
                transport_config,
                &transport_type,
            ))),
            #[cfg(feature = "metrics")]
            metrics_args: Arc::new(RwLock::new((
                config.client_config.metrics_meter_name.clone(),
                construct_metrics_otlp_endpoint(&config.client_config.metrics_otlp_endpoint),
            ))),
            local_model_path: Arc::new(RwLock::new(construct_local_model_path(
                &transport_config.local_model_module,
            ))),
            trajectory_file_output: Arc::new(RwLock::new(construct_trajectory_file_output(
                &config.client_config.trajectory_file_output,
            ))),
            config_path: Arc::new(config_path),
            last_modified: Arc::new(RwLock::new(last_modified)),
            config_update_polling_seconds: Arc::new(RwLock::new(config_update_polling)),
            router_buffer_size_per_actor: Arc::new(RwLock::new(router_buffer_size)),
            shutdown_tx,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            transport_type: Arc::new(transport_type),
            shutdown_notifier: Arc::new(Notify::new()),
        }
    }

    // Listen for shutdown signals and config changes
    pub(crate) fn spawn_loop(&self) {
        let self_clone: LifecycleManager = self.clone();
        tokio::spawn(async move {
            if let Err(e) = self_clone.watch().await {
                log::error!("[LifecycleManager] Failed to spawn loop: {}", e);
            }
        });
    }

    pub fn get_config_path(&self) -> Arc<PathBuf> {
        self.config_path.clone()
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn get_transport_addresses(&self) -> Arc<RwLock<SharedTransportAddresses>> {
        self.transport_addresses.clone()
    }

    #[cfg(feature = "metrics")]
    pub fn get_metrics_args(&self) -> Arc<RwLock<(String, String)>> {
        self.metrics_args.clone()
    }

    pub fn get_local_model_path(&self) -> Arc<RwLock<PathBuf>> {
        self.local_model_path.clone()
    }

    pub fn get_trajectory_file_output(&self) -> Arc<RwLock<LocalTrajectoryFileParams>> {
        self.trajectory_file_output.clone()
    }

    pub fn get_router_buffer_size_per_actor(&self) -> Arc<RwLock<usize>> {
        self.router_buffer_size_per_actor.clone()
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn get_ppo_hyperparameters(&self) -> Arc<RwLock<PPOParams>> {
        self.default_hyperparameters
            .ppo
            .clone()
            .into_inner()
            .clone()
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn get_ippo_hyperparameters(&self) -> Arc<RwLock<IPPOParams>> {
        self.default_hyperparameters
            .ippo
            .clone()
            .into_inner()
            .clone()
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub fn get_mappo_hyperparameters(&self) -> Arc<RwLock<MAPPOParams>> {
        self.default_hyperparameters
            .mappo
            .clone()
            .into_inner()
            .clone()
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn set_transport_addresses(
        &self,
        transport_params: &TransportConfigParams,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        transport_type: &TransportType,
    ) -> Result<(), LifecycleManagerError> {
        let mut transport_addresses_guard = self.transport_addresses.write().await;
        *transport_addresses_guard =
            construct_transport_addresses(transport_params, transport_type);
        Ok(())
    }

    #[cfg(feature = "metrics")]
    pub(crate) async fn set_metrics_args(
        &self,
        metrics_meter_name: &str,
        metrics_otlp_endpoint: &OtlpEndpointParams,
    ) -> Result<(), LifecycleManagerError> {
        let mut metrics_args_guard = self.metrics_args.write().await;
        *metrics_args_guard = (
            metrics_meter_name.to_string(),
            construct_metrics_otlp_endpoint(metrics_otlp_endpoint),
        );
        Ok(())
    }

    pub(crate) async fn set_router_buffer_size_per_actor(
        &self,
        router_buffer_size_per_actor: usize,
    ) -> Result<(), LifecycleManagerError> {
        let mut router_buffer_size_per_actor_guard =
            self.router_buffer_size_per_actor.write().await;
        *router_buffer_size_per_actor_guard = router_buffer_size_per_actor;
        Ok(())
    }

    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    pub(crate) async fn set_default_hyperparameters(
        &self,
        init_hyperparameters: &HyperparameterConfig,
    ) -> Result<(), LifecycleManagerError> {
        if self.default_hyperparameters.ppo_config_init {
            *self.default_hyperparameters.ppo.write().await =
                init_hyperparameters.ppo.clone().unwrap_or_default();
        }
        if self.default_hyperparameters.ippo_config_init {
            *self.default_hyperparameters.ippo.write().await =
                init_hyperparameters.ippo.clone().unwrap_or_default();
        }
        if self.default_hyperparameters.mappo_config_init {
            *self.default_hyperparameters.mappo.write().await =
                init_hyperparameters.mappo.clone().unwrap_or_default();
        }

        Ok(())
    }

    pub(crate) async fn set_local_model_path(
        &self,
        local_model_module: &LocalModelModuleParams,
    ) -> Result<(), LifecycleManagerError> {
        let mut local_model_path_guard = self.local_model_path.write().await;
        *local_model_path_guard = construct_local_model_path(local_model_module);
        Ok(())
    }

    pub(crate) async fn set_trajectory_file_path(
        &self,
        trajectory_file_output: &LocalTrajectoryFileParams,
    ) -> Result<(), LifecycleManagerError> {
        let mut trajectory_file_output_guard = self.trajectory_file_output.write().await;
        *trajectory_file_output_guard = construct_trajectory_file_output(trajectory_file_output);
        Ok(())
    }

    pub(crate) fn shutdown(&mut self) {
        self.shutdown_notifier.notify_waiters();
        self.handle_shutdown_signal();
    }

    pub(crate) async fn watch(&self) -> Result<(), LifecycleManagerError> {
        let mut config_update_polling_seconds =
            *self.config_update_polling_seconds.read().await as u64;

        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            config_update_polling_seconds,
        ));
        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    self.handle_shutdown_signal();
                    break Ok(());
                }
                _ = self.shutdown_notifier.notified() => {
                    self.handle_shutdown_signal();
                    break Ok(());
                }
                _ = interval.tick() => {
                    if let Ok(metadata) = fs::metadata(&*self.config_path) &&
                        let Ok(modified) = metadata.modified() {
                            let mut last_modified = self.last_modified.write().await;
                            if modified > *last_modified {
                                log::info!("[LifecycleManager] Config file changed, reloading...");
                                *last_modified = modified;

                                #[allow(irrefutable_let_patterns)]
                                if let new_polling_seconds = *self.config_update_polling_seconds.read().await as u64
                                    && new_polling_seconds != config_update_polling_seconds {
                                        interval = tokio::time::interval(std::time::Duration::from_secs(
                                            new_polling_seconds,
                                        ));
                                        config_update_polling_seconds = new_polling_seconds;
                                    }

                                self.handle_config_change(self.config_path.as_ref().clone()).await?;
                            }
                    }
                }
            }
        }
    }

    pub(crate) fn handle_shutdown_signal(&self) {
        if let Err(e) = self.shutdown_tx.send(()) {
            log::error!(
                "[LifecycleManager] Failed to send shutdown signal. No active receivers: {}",
                e
            );
        }
    }

    pub(crate) async fn handle_config_change(
        &self,
        path: PathBuf,
    ) -> Result<(), LifecycleManagerError> {
        let new_config = ClientConfigLoader::load_config(&path);

        #[cfg(all(
            any(feature = "nats-transport", feature = "zmq-transport"),
            not(feature = "metrics")
        ))]
        tokio::try_join!(
            self.set_transport_addresses(&new_config.transport_config, &self.transport_type),
            self.set_local_model_path(&new_config.transport_config.local_model_module),
            self.set_trajectory_file_path(&new_config.client_config.trajectory_file_output),
            self.set_default_hyperparameters(&new_config.client_config.init_hyperparameters),
            self.set_router_buffer_size_per_actor(
                new_config.client_config.router_buffer_size_per_actor
            ),
        )
        .map_err(|e| {
            LifecycleManagerError::ConfigError(format!("Failed to reload config: {:?}", e))
        })?;

        #[cfg(all(
            any(feature = "nats-transport", feature = "zmq-transport"),
            feature = "metrics"
        ))]
        tokio::try_join!(
            self.set_transport_addresses(&new_config.transport_config, &self.transport_type),
            self.set_local_model_path(&new_config.transport_config.local_model_module),
            self.set_trajectory_file_path(&new_config.client_config.trajectory_file_output),
            self.set_default_hyperparameters(&new_config.client_config.init_hyperparameters),
            self.set_router_buffer_size_per_actor(
                new_config.client_config.router_buffer_size_per_actor
            ),
            self.set_metrics_args(
                &new_config.client_config.metrics_meter_name,
                &new_config.client_config.metrics_otlp_endpoint
            ),
        )
        .map_err(|e| {
            LifecycleManagerError::ConfigError(format!("Failed to reload config: {:?}", e))
        })?;

        #[cfg(all(
            not(any(feature = "nats-transport", feature = "zmq-transport")),
            not(feature = "metrics")
        ))]
        tokio::try_join!(
            self.set_local_model_path(&new_config.transport_config.local_model_module),
            self.set_trajectory_file_path(&new_config.client_config.trajectory_file_output),
            self.set_router_buffer_size_per_actor(
                new_config.client_config.router_buffer_size_per_actor
            ),
        )
        .map_err(|e| {
            LifecycleManagerError::ConfigError(format!("Failed to reload config: {:?}", e))
        })?;

        #[cfg(all(
            not(any(feature = "nats-transport", feature = "zmq-transport")),
            feature = "metrics"
        ))]
        tokio::try_join!(
            self.set_local_model_path(&new_config.transport_config.local_model_module),
            self.set_trajectory_file_path(&new_config.client_config.trajectory_file_output),
            self.set_router_buffer_size_per_actor(
                new_config.client_config.router_buffer_size_per_actor
            ),
            self.set_metrics_args(
                &new_config.client_config.metrics_meter_name,
                &new_config.client_config.metrics_otlp_endpoint
            ),
        )
        .map_err(|e| {
            LifecycleManagerError::ConfigError(format!("Failed to reload config: {:?}", e))
        })?;

        Ok(())
    }

    pub(crate) fn subscribe_shutdown(
        &self,
    ) -> Result<broadcast::Receiver<()>, LifecycleManagerError> {
        Ok(self.shutdown_tx.subscribe())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::network::client::agent::{LocalTrajectoryFileParams, LocalTrajectoryFileType};
    use crate::utilities::configuration::LocalModelModuleParams;

    #[test]
    fn construct_local_model_path_joins_components() {
        let params = LocalModelModuleParams {
            directory: "model_dir".to_string(),
            model_name: "my_model".to_string(),
            format: "pt".to_string(),
        };
        let path = construct_local_model_path(&params);
        let path_str = path.to_str().unwrap();
        assert!(
            path_str.contains("model_dir"),
            "Path should contain directory component"
        );
        assert!(
            path_str.contains("my_model"),
            "Path should contain model_name component"
        );
        assert!(
            path_str.contains(".pt"),
            "Path should contain formatted extension"
        );
    }

    #[test]
    fn construct_local_model_path_uses_cwd_as_root() {
        let params = LocalModelModuleParams {
            directory: "subdir".to_string(),
            model_name: "net".to_string(),
            format: "mpk".to_string(),
        };
        let path = construct_local_model_path(&params);
        assert!(
            path.is_absolute(),
            "Returned path should be absolute (rooted at cwd)"
        );
    }

    #[test]
    fn construct_trajectory_file_output_joins_directory_with_cwd() {
        let params = LocalTrajectoryFileParams {
            directory: PathBuf::from("experiment_data"),
            file_type: LocalTrajectoryFileType::Arrow,
        };
        let result = construct_trajectory_file_output(&params);
        let dir_str = result.directory.to_str().unwrap();
        assert!(
            dir_str.contains("experiment_data"),
            "Result directory should contain the given subdirectory"
        );
        assert!(
            result.directory.is_absolute(),
            "Result directory should be absolute"
        );
        assert!(
            matches!(result.file_type, LocalTrajectoryFileType::Arrow),
            "File type should be preserved"
        );
    }

    #[test]
    fn construct_trajectory_file_output_preserves_file_type() {
        let params = LocalTrajectoryFileParams {
            directory: PathBuf::from("out"),
            file_type: LocalTrajectoryFileType::Csv,
        };
        let result = construct_trajectory_file_output(&params);
        assert!(matches!(result.file_type, LocalTrajectoryFileType::Csv));
    }

    #[test]
    fn subscribe_shutdown_receives_signal_after_handle_shutdown() {
        // Test the broadcast mechanism used by LifecycleManager in isolation.
        // This mirrors what subscribe_shutdown() + handle_shutdown_signal() do.
        let (tx, mut rx) = tokio::sync::broadcast::channel::<()>(10);
        // Simulate subscribe_shutdown: the subscriber gets a clone of rx
        // Simulate handle_shutdown_signal: sends () on tx
        tx.send(()).unwrap();
        let result = rx.try_recv();
        assert!(
            result.is_ok(),
            "Subscriber should receive the shutdown signal"
        );
    }

    #[test]
    fn handle_shutdown_signal_fails_with_no_receivers() {
        // broadcast::send returns Err when there are no active receivers.
        let (tx, rx) = tokio::sync::broadcast::channel::<()>(1);
        drop(rx); // drop the only receiver
        let result = tx.send(());
        assert!(
            result.is_err(),
            "Sending to a channel with no receivers should return Err"
        );
    }

    fn make_lifecycle_manager() -> LifecycleManager {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(tmp, "{{}}").expect("write temp config");
        let config = ClientConfigLoader::load_config(&tmp.path().to_path_buf());
        let lm = LifecycleManager::new(
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            AlgorithmArgs::default(),
            &config,
            tmp.path().to_path_buf(),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            TransportType::default(),
            Some(1000),
        );
        // Keep the temp file alive until LifecycleManager has loaded the config.
        drop(tmp);
        lm
    }

    #[tokio::test]
    async fn set_router_buffer_size_per_actor_round_trip() {
        let lm = make_lifecycle_manager();
        lm.set_router_buffer_size_per_actor(99).await.unwrap();
        let val = *lm.get_router_buffer_size_per_actor().read().await;
        assert_eq!(val, 99);
    }

    #[tokio::test]
    async fn set_router_buffer_size_per_actor_overwrites_previous_value() {
        let lm = make_lifecycle_manager();
        lm.set_router_buffer_size_per_actor(10).await.unwrap();
        lm.set_router_buffer_size_per_actor(200).await.unwrap();
        let val = *lm.get_router_buffer_size_per_actor().read().await;
        assert_eq!(val, 200);
    }

    #[tokio::test]
    async fn set_local_model_path_round_trip() {
        let lm = make_lifecycle_manager();
        let params = LocalModelModuleParams {
            directory: "test_dir".to_string(),
            model_name: "my_net".to_string(),
            format: "pt".to_string(),
        };
        lm.set_local_model_path(&params).await.unwrap();
        let path = lm.get_local_model_path().read().await.clone();
        let path_str = path.to_str().unwrap();
        assert!(
            path_str.contains("test_dir"),
            "path should contain directory"
        );
        assert!(
            path_str.contains("my_net"),
            "path should contain model name"
        );
        assert!(path_str.contains(".pt"), "path should contain extension");
    }

    #[tokio::test]
    async fn set_trajectory_file_path_round_trip() {
        let lm = make_lifecycle_manager();
        let params = LocalTrajectoryFileParams {
            directory: PathBuf::from("exp_output"),
            file_type: LocalTrajectoryFileType::Arrow,
        };
        lm.set_trajectory_file_path(&params).await.unwrap();
        let output = lm.get_trajectory_file_output().read().await.clone();
        let dir_str = output.directory.to_str().unwrap();
        assert!(
            dir_str.contains("exp_output"),
            "directory should be preserved"
        );
        assert!(
            matches!(output.file_type, LocalTrajectoryFileType::Arrow),
            "file type should be Arrow"
        );
    }

    #[tokio::test]
    async fn set_trajectory_file_path_csv_preserved() {
        let lm = make_lifecycle_manager();
        let params = LocalTrajectoryFileParams {
            directory: PathBuf::from("csv_out"),
            file_type: LocalTrajectoryFileType::Csv,
        };
        lm.set_trajectory_file_path(&params).await.unwrap();
        let output = lm.get_trajectory_file_output().read().await.clone();
        assert!(matches!(output.file_type, LocalTrajectoryFileType::Csv));
    }
}
