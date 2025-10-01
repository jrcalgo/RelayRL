use crate::utilities::configuration::ClientConfigLoader;
use log::{info, warn};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::mpsc;

/// Orchestrates startup/shutdown signals (SIGINT, config-changes)
///
/// Spins up and tears down futures cleanly
pub(crate) struct LifeCycleManager {
    config: ClientConfigLoader,
}

impl LifeCycleManager {
    pub fn new(config: ClientConfigLoader) -> Self {
        Self { config }
    }

    pub fn spawn(&self) {}

    pub fn get_active_config(&self) -> Arc<ClientConfigLoader> {
        let config = self.config.clone();
        Arc::from(config)
    }

    pub fn _shutdown(&self) {}

    async fn _watch(&self) {}

    fn _handle_shutdown_signal(&self) {}

    async fn _handle_config_change(&self, path: PathBuf) {}
}
