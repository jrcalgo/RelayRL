use crate::network::HyperparameterArgs;
use crate::network::TransportType;
use crate::network::client::runtime::coordination::scale_manager::AlgorithmArgs;
use crate::prelude::config::TransportConfigParams;
use crate::utilities::configuration::Algorithm;
use crate::utilities::configuration::{
    ClientConfigLoader, HyperparameterConfig, LocalModelModuleParams, NetworkParams,
    TrajectoryFileOutputParams,
};

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{Notify, RwLock, broadcast};

use thiserror::Error;

#[derive(Debug, Clone)]
pub(crate) struct FormattedTrajectoryFileParams {
    pub(crate) enabled: bool,
    pub(crate) encode: bool,
    pub(crate) path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct TransportRuntimeParams {
    pub(crate) config_update_polling: u32,
    pub(crate) grpc_idle_timeout: u32,
    pub(crate) max_traj_length: u128,
}

#[derive(Debug, Clone)]
pub(crate) struct ServerAddresses {
    pub(crate) inference_server_address: String,
    pub(crate) agent_listener_address: String,
    pub(crate) model_server_address: String,
    pub(crate) trajectory_server_address: String,
    pub(crate) scaling_server_address: String,
}

pub(crate) fn construct_server_addresses(
    transport_config: &TransportConfigParams,
    transport_type: &TransportType,
) -> ServerAddresses {
    fn construct_address(transport_type: &TransportType, network_params: &NetworkParams) -> String {
        match *transport_type {
            #[cfg(feature = "grpc_network")]
            TransportType::GRPC => network_params.host.clone() + ":" + &network_params.port.clone(),
            #[cfg(feature = "zmq_network")]
            TransportType::ZMQ => {
                network_params.prefix.clone()
                    + &network_params.host.clone()
                    + ":"
                    + &network_params.port.clone()
            }
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
    trajectory_file_output: &TrajectoryFileOutputParams,
) -> FormattedTrajectoryFileParams {
    let cwd: PathBuf = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let path = cwd
        .join(&trajectory_file_output.output.directory)
        .join(&trajectory_file_output.output.file_name)
        .join(format!(".{}", &trajectory_file_output.output.format));

    FormattedTrajectoryFileParams {
        enabled: trajectory_file_output.enabled,
        encode: trajectory_file_output.encode,
        path,
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
    algorithm_args: Arc<AlgorithmArgs>,
    transport_params: Arc<RwLock<TransportRuntimeParams>>,
    server_addresses: Arc<RwLock<ServerAddresses>>,
    init_hyperparameters: Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>>,
    local_model_path: Arc<RwLock<PathBuf>>,
    trajectory_file_output: Arc<RwLock<FormattedTrajectoryFileParams>>,
    transport_type: Arc<TransportType>,
    config_path: Arc<PathBuf>,
    last_modified: Arc<RwLock<SystemTime>>,
    shutdown_tx: broadcast::Sender<()>,
    shutdown_notifier: Arc<Notify>,
}

impl LifeCycleManager {
    pub fn new(
        algorithm_args: AlgorithmArgs,
        config: ClientConfigLoader,
        config_path: PathBuf,
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

        let transport_config = config.get_transport_config();
        let config_update_polling = transport_config.config_update_polling.clone();
        let grpc_idle_timeout = transport_config.grpc_idle_timeout.clone();
        let max_traj_length = transport_config.max_traj_length.clone();

        Self {
            algorithm_args: Arc::new(algorithm_args),
            transport_params: Arc::new(RwLock::new(TransportRuntimeParams {
                config_update_polling,
                grpc_idle_timeout,
                max_traj_length,
            })),
            server_addresses: Arc::new(RwLock::new(construct_server_addresses(
                &config.transport_config,
                &transport_type,
            ))),
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
            shutdown_tx,
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

    pub fn get_transport_params(&self) -> Arc<RwLock<TransportRuntimeParams>> {
        self.transport_params.clone()
    }

    pub fn get_server_addresses(&self) -> Arc<RwLock<ServerAddresses>> {
        self.server_addresses.clone()
    }

    pub fn get_local_model_path(&self) -> Arc<RwLock<PathBuf>> {
        self.local_model_path.clone()
    }

    pub fn get_trajectory_file_output(&self) -> Arc<RwLock<FormattedTrajectoryFileParams>> {
        self.trajectory_file_output.clone()
    }

    pub fn get_init_hyperparameters(&self) -> Arc<RwLock<HashMap<Algorithm, HyperparameterArgs>>> {
        self.init_hyperparameters.clone()
    }

    pub fn get_algorithm_args(&self) -> Arc<AlgorithmArgs> {
        self.algorithm_args.clone()
    }

    pub async fn set_transport_params(
        &self,
        transport_params: &TransportConfigParams,
    ) -> Result<(), LifeCycleManagerError> {
        let mut transport_params_guard = self.transport_params.write().await;
        *transport_params_guard = TransportRuntimeParams {
            config_update_polling: transport_params.config_update_polling,
            grpc_idle_timeout: transport_params.grpc_idle_timeout,
            max_traj_length: transport_params.max_traj_length,
        };
        Ok(())
    }

    pub async fn set_server_addresses(
        &self,
        transport_params: &TransportConfigParams,
        transport_type: &TransportType,
    ) -> Result<(), LifeCycleManagerError> {
        let mut server_addresses_guard = self.server_addresses.write().await;
        *server_addresses_guard = construct_server_addresses(transport_params, transport_type);
        Ok(())
    }

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
        trajectory_file_output: &TrajectoryFileOutputParams,
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
            let config_update_polling =
                self.transport_params.read().await.config_update_polling as u64;
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(config_update_polling));

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

    async fn _handle_config_change(&self, path: PathBuf) -> Result<(), LifeCycleManagerError> {
        let new_config = ClientConfigLoader::load_config(&path);

        tokio::try_join!(
            self.set_transport_params(&new_config.transport_config),
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

    pub fn subscribe_shutdown(&self) -> Result<broadcast::Receiver<()>, LifeCycleManagerError> {
        Ok(self.shutdown_tx.subscribe())
    }
}
