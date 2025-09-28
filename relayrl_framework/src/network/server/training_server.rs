use std::path::PathBuf;
use tch::{CModule, Device};
use crate::network::TransportType;

struct TrainingServerStartParameters {}

/// Placeholder for server coordination logic
pub struct ServerCoordinator {
    transport_type: TransportType,
}

impl ServerCoordinator {
    pub fn new(transport_type: TransportType) -> Self {
        Self { transport_type }
    }
}

pub struct TrainingServerBuilder {
    network_type: TransportType,
    actor_count: Option<i64>,
    default_device: Option<Device>,
    default_model: Option<CModule>,
    algorithm_name: Option<String>,
    config_path: Option<PathBuf>,
}

impl TrainingServerBuilder {
    /// Create a new builder with required network type
    pub fn builder(network_type: TransportType) -> Self {
        Self {
            network_type,
            actor_count: None,
            default_device: None,
            default_model: None,
            algorithm_name: None,
            config_path: None,
        }
    }

    pub fn actor_count(mut self, count: i64) -> Self {
        self.actor_count = Some(count);
        self
    }

    pub fn default_device(mut self, device: Device) -> Self {
        self.default_device = Some(device);
        self
    }

    pub fn default_model(mut self, model: CModule) -> Self {
        self.default_model = Some(model);
        self
    }

    pub fn algorithm_name(mut self, name: String) -> Self {
        self.algorithm_name = Some(name.into());
        self
    }

    pub fn config_path(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub async fn build(self) -> Result<TrainingServer, String> {
        let actor_count = self.actor_count.unwrap_or(1);
        let default_device = self.default_device.unwrap_or(Device::Cpu);
        let default_model = self.default_model;
        let algorithm_name = self
            .algorithm_name
            .unwrap_or_else(|| "default_algorithm".to_string());
        let config_path = self
            .config_path
            .unwrap_or_else(|| PathBuf::from("config.json"));

        // Initialize the server with the provided parameters
        let server = TrainingServer::new(
            self.network_type,
            actor_count,
            default_device,
            default_model,
            algorithm_name,
            config_path,
        );

        Ok(server)
    }
}

pub struct TrainingServer {
    coordinator: ServerCoordinator,
}

impl TrainingServer {
    pub fn new(network_type: TransportType) -> Self {
        Self {
            coordinator: ServerCoordinator::new(network_type),
        }
    }

    pub async fn start(self) {}
}
