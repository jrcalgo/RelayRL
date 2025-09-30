use crate::utilities::config_builder::ClientConfig;
use log::{info, warn};
use std::path::PathBuf;
use std::sync::Arc;
use notify::{RecursiveMode, Watcher, recommended_watcher};
use tokio::signal;
use tokio::sync::mpsc;

/// Orchestrates startup/shutdown signals (SIGINT, config-changes)
///
/// Spins up and tears down futures cleanly
pub(crate) struct LifeCycleManager {
    config: ClientConfig,
}

impl LifeCycleManager {
    pub fn new(config: ClientConfig) -> Self {
        Self { config }
    }

    pub fn spawn(&self) {}

    pub fn get_active_config(&self) -> Arc<ClientConfig> {
        let config = self.config.clone();
        Arc::from(config)
    }

    pub fn _shutdown(&self) {}

    async fn _watch(&self) {}

    fn _handle_shutdown_signal(&self) {}

    async fn _handle_config_change(&self, path: PathBuf) {}
}
