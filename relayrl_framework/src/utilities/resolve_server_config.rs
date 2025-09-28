use crate::sys_utils::config_loader::{ServerConfig, ServerParams};

/// Responsible for resolving RL4Sys server configurations
/// and storing the new parameters.
struct ServerConfigManager {
    resolved_training_server_params: ServerParams,
    resolved_trajectory_server_params: ServerParams,
    resolved_agent_listener_server_params: ServerParams,
}

/// ServerConfigManager
///     - on New(), resolves the server parameters
///
impl ServerConfigManager {
    pub fn new(config: &ServerConfig, server_type: &str) -> Self {
        let resolved_training_server_params =
            Self::resolve_server_params(config.training_server.as_ref().unwrap());
        let resolved_trajectory_server_params =
            Self::resolve_server_params(config.trajectory_server.as_ref().unwrap());
        let resolved_agent_listener_server_params =
            Self::resolve_server_params(config.agent_listener.as_ref().unwrap());

        ServerConfigManager {
            resolved_training_server_params,
            resolved_trajectory_server_params,
            resolved_agent_listener_server_params,
        }
    }

    fn resolve_server_params(params: &ServerParams) -> ServerParams {
        todo!()
    }

    fn resolve_server_prefix(prefix: &str) -> String {
        todo!()
    }

    fn resolve_server_host(host: &str) -> String {
        todo!()
    }

    fn resolve_server_port(port: &str) -> String {
        todo!()
    }
}
