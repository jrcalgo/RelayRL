#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::HyperparameterArgs;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::TransportType;
use crate::network::client::agent::TrajectoryFileParams;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::coordination::scale_manager::AlgorithmArgs;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::prelude::config::TransportConfigParams;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::utilities::configuration::Algorithm;
use crate::utilities::configuration::{ClientConfigLoader, LocalModelModuleParams};
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::utilities::configuration::{HyperparameterConfig, NetworkParams};

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{Notify, RwLock, broadcast};

use thiserror::Error;

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
#[derive(Debug, Clone)]
pub(crate) struct ServerAddresses {
    pub(crate) inference_server_address: Arc<str>,
    pub(crate) agent_listener_address: Arc<str>,
    pub(crate) model_server_address: Arc<str>,
    pub(crate) trajectory_server_address: Arc<str>,
    pub(crate) scaling_server_address: Arc<str>,
}

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
pub(crate) fn construct_server_addresses(
    transport_config: &TransportConfigParams,
    transport_type: &TransportType,
) -> ServerAddresses {
    fn construct_address(
        transport_type: &TransportType,
        network_params: &NetworkParams,
    ) -> Arc<str> {
        match *transport_type {
            #[cfg(feature = "zmq_transport")]
            TransportType::ZMQ => Arc::<str>::from(
                network_params.prefix.clone()
                    + &network_params.host.clone()
                    + ":"
                    + &network_params.port.clone(),
            ),
        }
    }

    ServerAddresses {
        inference_server_address: construct_address(
            transport_type,
            &transport_config.inference_server_address,
        ),
        agent_listener_address: construct_address(
            transport_type,
            &transport_config.agent_listener_address,
        ),
        model_server_address: construct_address(
            transport_type,
            &transport_config.model_server_address,
        ),
        trajectory_server_address: construct_address(
            transport_type,
            &transport_config.trajectory_server_address,
        ),
        scaling_server_address: construct_address(
            transport_type,
            &transport_config.scaling_server_address,
        ),
    }
}

pub(crate) fn construct_local_model_path(local_model_module: &LocalModelModuleParams) -> PathBuf {
    let cwd: PathBuf = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cwd.join(&local_model_module.directory)
        .join(&local_model_module.model_name)
        .join(format!(".{}", &local_model_module.format))
}

pub(crate) fn construct_trajectory_file_output(
    trajectory_file_output: &TrajectoryFileParams,
) -> TrajectoryFileParams {
    let cwd: PathBuf = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let directory = cwd.join(&trajectory_file_output.directory);

    TrajectoryFileParams {
        enabled: trajectory_file_output.enabled,
        encode: trajectory_file_output.encode,
        directory,
    }
}

#[derive(Debug, Error)]
pub enum LifeCycleManagerError {
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
pub(crate) struct LifeCycleManager {
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    algorithm_args: Arc<AlgorithmArgs>,
    max_traj_length: Arc<RwLock<u128>>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    server_addresses: Arc<RwLock<ServerAddresses>>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    init_hyperparameters: Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>>,
    local_model_path: Arc<RwLock<PathBuf>>,
    trajectory_file_output: Arc<RwLock<TrajectoryFileParams>>,
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    transport_type: Arc<TransportType>,
    config_path: Arc<PathBuf>,
    config_update_polling_seconds: Arc<RwLock<f32>>,
    last_modified: Arc<RwLock<SystemTime>>,
    shutdown_tx: broadcast::Sender<()>,
    shutdown_notifier: Arc<Notify>,
}

impl LifeCycleManager {
    pub fn new(
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        algorithm_args: AlgorithmArgs,
        config: ClientConfigLoader,
        config_path: PathBuf,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        transport_type: TransportType,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(10_000);

        // Get file metadata with fallback to current time
        let last_modified: SystemTime = fs::metadata(&config_path)
            .and_then(|m| m.modified())
            .unwrap_or_else(|e| {
                eprintln!(
                    "[LifeCycleManager] Failed to read config metadata: {}, using current time",
                    e
                );
                SystemTime::now()
            });

        let config_update_polling = config.client_config.config_update_polling_seconds.clone();

        let transport_config = config.get_transport_config();
        let max_traj_length = transport_config.max_traj_length.clone();

        Self {
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            algorithm_args: Arc::new(algorithm_args),
            max_traj_length: Arc::new(RwLock::new(max_traj_length)),
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            server_addresses: Arc::new(RwLock::new(construct_server_addresses(
                &config.transport_config,
                &transport_type,
            ))),
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            init_hyperparameters: Arc::new(RwLock::new(
                config.client_config.init_hyperparameters.to_args(None),
            )),
            local_model_path: Arc::new(RwLock::new(construct_local_model_path(
                &config.transport_config.local_model_module,
            ))),
            trajectory_file_output: Arc::new(RwLock::new(construct_trajectory_file_output(
                &config.client_config.trajectory_file_output,
            ))),
            config_path: Arc::new(config_path),
            last_modified: Arc::new(RwLock::new(last_modified)),
            config_update_polling_seconds: Arc::new(RwLock::new(config_update_polling)),
            shutdown_tx,
            #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
            transport_type: Arc::new(transport_type),
            shutdown_notifier: Arc::new(Notify::new()),
        }
    }

    // Listen for shutdown signals and config changes
    pub fn spawn_loop(&self) {
        let self_clone: LifeCycleManager = self.clone();
        tokio::spawn(async move {
            if let Err(e) = self_clone._watch().await {
                eprintln!("[LifeCycleManager] Failed to spawn loop: {}", e);
            }
        });
    }

    pub fn get_config_path(&self) -> Arc<PathBuf> {
        self.config_path.clone()
    }

    pub fn get_max_traj_length(&self) -> Arc<RwLock<u128>> {
        self.max_traj_length.clone()
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn get_server_addresses(&self) -> Arc<RwLock<ServerAddresses>> {
        self.server_addresses.clone()
    }

    pub fn get_local_model_path(&self) -> Arc<RwLock<PathBuf>> {
        self.local_model_path.clone()
    }

    pub fn get_trajectory_file_output(&self) -> Arc<RwLock<TrajectoryFileParams>> {
        self.trajectory_file_output.clone()
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn get_init_hyperparameters(&self) -> Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>> {
        self.init_hyperparameters.clone()
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn get_algorithm_args(&self) -> Arc<AlgorithmArgs> {
        self.algorithm_args.clone()
    }

    pub async fn set_max_traj_length(
        &self,
        max_traj_length: &u128,
    ) -> Result<(), LifeCycleManagerError> {
        let mut max_traj_length_guard = self.max_traj_length.write().await;
        *max_traj_length_guard = max_traj_length.clone();
        Ok(())
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub async fn set_server_addresses(
        &self,
        transport_params: &TransportConfigParams,
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        transport_type: &TransportType,
    ) -> Result<(), LifeCycleManagerError> {
        let mut server_addresses_guard = self.server_addresses.write().await;
        *server_addresses_guard = construct_server_addresses(transport_params, transport_type);
        Ok(())
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub async fn set_init_hyperparameters(
        &self,
        init_hyperaparameters: &HyperparameterConfig,
    ) -> Result<(), LifeCycleManagerError> {
        let mut init_hyperparameters_guard = self.init_hyperparameters.write().await;
        *init_hyperparameters_guard =
            init_hyperaparameters.to_args(Some(&self.algorithm_args.algorithm));
        Ok(())
    }

    pub async fn set_local_model_path(
        &self,
        local_model_module: &LocalModelModuleParams,
    ) -> Result<(), LifeCycleManagerError> {
        let mut local_model_path_guard = self.local_model_path.write().await;
        *local_model_path_guard = construct_local_model_path(local_model_module);
        Ok(())
    }

    pub async fn set_trajectory_file_path(
        &self,
        trajectory_file_output: &TrajectoryFileParams,
    ) -> Result<(), LifeCycleManagerError> {
        let mut trajectory_file_output_guard = self.trajectory_file_output.write().await;
        *trajectory_file_output_guard = construct_trajectory_file_output(trajectory_file_output);
        Ok(())
    }

    pub fn _shutdown(&mut self) -> Result<(), LifeCycleManagerError> {
        self.shutdown_notifier.notify_waiters();
        self._handle_shutdown_signal()
    }

    async fn _watch(&self) -> Result<(), LifeCycleManagerError> {
        loop {
            let config_update_polling_seconds =
                self.config_update_polling_seconds.read().await.clone() as u64;
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                config_update_polling_seconds,
            ));

            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    self._handle_shutdown_signal()?;
                    break Ok(());
                }
                _ = self.shutdown_notifier.notified() => {
                    self._handle_shutdown_signal()?;
                    break Ok(());
                }
                _ = interval.tick() => {
                    if let Ok(metadata) = fs::metadata(&*self.config_path) {
                        if let Ok(modified) = metadata.modified() {
                            let mut last_modified = self.last_modified.write().await;
                            if modified > *last_modified {
                                println!("[LifeCycleManager] Config file changed, reloading...");
                                *last_modified = modified;
                                self._handle_config_change(self.config_path.as_ref().clone()).await?;
                            }
                        }
                    }
                }
            }
        }
    }

    fn _handle_shutdown_signal(&self) -> Result<(), LifeCycleManagerError> {
        if let Err(e) = self.shutdown_tx.send(()) {
            return Err(LifeCycleManagerError::SendShutdownSignalError(
                format!("Failed to send shutdown signal. No active receivers: {}", e).to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub(crate) async fn _handle_config_change(
        &self,
        path: PathBuf,
    ) -> Result<(), LifeCycleManagerError> {
        let new_config = ClientConfigLoader::load_config(&path);

        tokio::try_join!(
            self.set_max_traj_length(&new_config.transport_config.max_traj_length),
            self.set_server_addresses(&new_config.transport_config, &self.transport_type),
            self.set_local_model_path(&new_config.transport_config.local_model_module),
            self.set_trajectory_file_path(&new_config.client_config.trajectory_file_output),
            self.set_init_hyperparameters(&new_config.client_config.init_hyperparameters),
        )
        .map_err(|e| {
            LifeCycleManagerError::ConfigError(format!("Failed to reload config: {:?}", e))
        })?;

        Ok(())
    }

    #[cfg(not(any(feature = "async_transport", feature = "sync_transport")))]
    pub(crate) async fn _handle_config_change(
        &self,
        path: PathBuf,
    ) -> Result<(), LifeCycleManagerError> {
        let new_config = ClientConfigLoader::load_config(&path);

        tokio::try_join!(
            self.set_max_traj_length(&new_config.transport_config.max_traj_length),
            self.set_local_model_path(&new_config.transport_config.local_model_module),
            self.set_trajectory_file_path(&new_config.client_config.trajectory_file_output),
        )
        .map_err(|e| {
            LifeCycleManagerError::ConfigError(format!("Failed to reload config: {:?}", e))
        })?;

        Ok(())
    }

    pub fn subscribe_shutdown(&self) -> Result<broadcast::Receiver<()>, LifeCycleManagerError> {
        Ok(self.shutdown_tx.subscribe())
    }
}
