use crate::utilities::configuration::ClientConfigLoader;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{RwLock, broadcast};

/// Orchestrates startup/shutdown signals (SIGINT, config-changes)
///
/// Spins up and tears down futures cleanly
#[derive(Clone)]
pub(crate) struct LifeCycleManager {
    config: Arc<RwLock<ClientConfigLoader>>,
    config_path: Arc<PathBuf>,
    last_modified: Arc<RwLock<SystemTime>>,
    shutdown_tx: broadcast::Sender<()>,
}

impl LifeCycleManager {
    pub fn new(config: ClientConfigLoader) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1000);
        let config_path = config.get_config_path().clone();
        let metadata = fs::metadata(&config_path).expect("Failed to read config metadata");
        let last_modified = metadata
            .modified()
            .expect("Failed to get last modified time");

        Self {
            config: Arc::new(RwLock::new(config)),
            config_path: Arc::new(config_path),
            last_modified: Arc::new(RwLock::new(last_modified)),
            shutdown_tx,
        }
    }

    // Listen for shutdown signals and config changes
    pub fn spawn_loop(&self) {
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone._watch().await;
        });
    }

    pub fn get_active_config(&self) -> Arc<RwLock<ClientConfigLoader>> {
        self.config.clone()
    }

    pub async fn set_active_config(&self, config: ClientConfigLoader) {
        let mut config_guard = self.config.write().await;
        *config_guard = config;
    }

    pub fn _shutdown(&self) {
        self._handle_shutdown_signal();
    }

    async fn _watch(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    self._handle_shutdown_signal();
                    break;
                }
                _ = interval.tick() => {
                    if let Ok(metadata) = fs::metadata(&*self.config_path) {
                        if let Ok(modified) = metadata.modified() {
                            let mut last_modified = self.last_modified.write().await;
                            if modified > *last_modified {
                                println!("[LifeCycleManager] Config file changed, reloading...");
                                *last_modified = modified;
                                self._handle_config_change(self.config_path.as_ref().clone()).await;
                            }
                        }
                    }
                }
            }
        }
    }

    fn _handle_shutdown_signal(&self) {
        if self.shutdown_tx.send(()).is_err() {
            eprintln!("[LifeCycleManager] Failed to send shutdown signal. No active receivers.");
        }
    }

    async fn _handle_config_change(&self, path: PathBuf) {
        let new_config = ClientConfigLoader::load_config(&path);
        self.set_active_config(new_config).await;
    }

    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }
}
