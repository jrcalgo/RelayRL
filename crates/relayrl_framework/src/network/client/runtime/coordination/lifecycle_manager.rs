use crate::utilities::configuration::ClientConfigLoader;
use crate::utilities::misc_utils::{construct_server_addresses, ServerAddresses};
use crate::network::TransportType;
use crate::prelude::config::TransportConfigParams;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{RwLock, broadcast};

use thiserror::Error;

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
    config: Arc<RwLock<ClientConfigLoader>>,
    server_addresses: Arc<RwLock<ServerAddresses>>,
    transport_type: TransportType,
    config_path: Arc<PathBuf>,
    last_modified: Arc<RwLock<SystemTime>>,
    shutdown_tx: broadcast::Sender<()>,
}

impl LifeCycleManager {
    pub fn new(config: ClientConfigLoader, config_path: PathBuf, transport_type: TransportType) -> Self {
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

        Self {
            config: Arc::new(RwLock::new(config.to_owned())),
            server_addresses: Arc::new(RwLock::new(construct_server_addresses(
                &config.transport_config,
                &transport_type,
            ))),
            config_path: Arc::new(config_path),
            last_modified: Arc::new(RwLock::new(last_modified)),
            shutdown_tx,
            transport_type,
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

    pub fn get_active_config(&self) -> Arc<RwLock<ClientConfigLoader>> {
        self.config.clone()
    }

    pub fn get_server_addresses(&self) -> Arc<RwLock<ServerAddresses>> {
        self.server_addresses.clone()
    }

    pub async fn set_active_config(
        &self,
        config: &ClientConfigLoader,
    ) -> Result<(), LifeCycleManagerError> {
        let mut config_guard = self.config.write().await;
        *config_guard = config.to_owned();
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

    pub fn _shutdown(&self) -> Result<(), LifeCycleManagerError> {
        self._handle_shutdown_signal()
    }

    async fn _watch(&self) -> Result<(), LifeCycleManagerError> {
        loop {
            let config_update_polling = self
                .config
                .read()
                .await
                .transport_config
                .config_update_polling as u64;
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(config_update_polling));

            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
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
        self.set_active_config(&new_config).await?;
        self.set_server_addresses(&new_config.transport_config, &self.transport_type)
            .await?;
        Ok(())
    }

    pub fn subscribe_shutdown(&self) -> Result<broadcast::Receiver<()>, LifeCycleManagerError> {
        Ok(self.shutdown_tx.subscribe())
    }
}
