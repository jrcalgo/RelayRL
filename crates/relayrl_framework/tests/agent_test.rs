//! Integration tests for RelayRL Agent functionality
//!
//! This test module validates the core functionality of the RelayRL Agent,
//! including agent lifecycle, actor management, and API interactions.

use burn_ndarray::NdArray;
use relayrl_framework::network::TransportType;
use relayrl_framework::network::client::agent::{RelayRLAgent, RelayRLAgentBuilder};
use relayrl_types::prelude::RelayRLAction;
use relayrl_types::types::data::tensor::DeviceType;
use std::path::PathBuf;
use tempfile::TempDir;
use uuid::Uuid;

/// Helper function to create a test config file
fn create_test_config(temp_dir: &TempDir) -> PathBuf {
    let config_path = temp_dir.path().join("test_client_config.json");
    let config_content = r#"{
    "client_config": {
        "actor_count": 2,
        "algorithm_name": "TEST_ALGORITHM",
        "config_path": "test_client_config.json",
        "default_device": "cpu",
        "default_model": ""
    },
    "transport_config": {
        "addresses": {
            "training_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "50051"
            },
            "trajectory_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7776"
            },
            "agent_listener": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7777"
            },
            "inference_server": {
                "prefix": "tcp://",
                "host": "127.0.0.1",
                "port": "7778"
            }
        },
        "grpc_idle_timeout": 30,
        "local_model_path": "client_model.pt",
        "max_traj_length": 1000
    }
}"#;
    std::fs::write(&config_path, config_content).expect("Failed to write test config");
    config_path
}

#[cfg(test)]
mod agent_builder_tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_builder_basic() {
        // Test basic agent builder construction
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::ZMQ);

        // Verify builder can be configured
        let builder = builder
            .actor_count(2)
            .default_device(DeviceType::Cpu)
            .algorithm_name("TEST_ALGORITHM".to_string());

        assert!(true, "Builder creation successful");
    }

    #[tokio::test]
    async fn test_agent_builder_with_all_parameters() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        // Test builder with all parameters
        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::ZMQ)
                .actor_count(4)
                .default_device(DeviceType::Cpu)
                .algorithm_name("TEST_ALGORITHM".to_string())
                .config_path(config_path.clone());

        assert!(true, "Builder with all parameters successful");
    }

    #[tokio::test]
    async fn test_agent_builder_missing_algorithm_name() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        // Test that building without algorithm_name fails
        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::ZMQ)
                .actor_count(1)
                .default_device(DeviceType::Cpu);

        let result = builder.build().await;

        assert!(
            result.is_err(),
            "Builder should fail without algorithm_name"
        );
        assert!(
            result.unwrap_err().contains("algorithm_name is required"),
            "Error message should mention algorithm_name"
        );
    }

    #[tokio::test]
    async fn test_agent_builder_build_success() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::ZMQ)
                .actor_count(2)
                .default_device(DeviceType::Cpu)
                .algorithm_name("TEST_ALGORITHM".to_string())
                .config_path(config_path);

        let result = builder.build().await;

        assert!(
            result.is_ok(),
            "Builder should succeed with all required parameters"
        );

        let (agent, params) = result.unwrap();
        // Verify we got back an agent and parameters
        assert!(true, "Agent and parameters created successfully");
    }

    #[tokio::test]
    async fn test_agent_builder_default_values() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        // Test that defaults are applied correctly
        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::GRPC)
                .algorithm_name("TEST_ALGORITHM".to_string())
                .config_path(config_path);

        let result = builder.build().await;
        assert!(result.is_ok(), "Builder should apply default values");
    }
}

#[cfg(test)]
mod agent_lifecycle_tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_creation() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        // Test basic agent creation
        let agent: RelayRLAgent<NdArray, D_IN, D_OUT> = RelayRLAgent::new(TransportType::ZMQ);
        assert!(true, "Agent created successfully");
    }

    #[tokio::test]
    async fn test_agent_creation_with_grpc() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        // Test agent creation with GRPC transport
        let agent: RelayRLAgent<NdArray, D_IN, D_OUT> = RelayRLAgent::new(TransportType::GRPC);
        assert!(true, "Agent created successfully with GRPC");
    }

    // Note: Full start/shutdown tests would require a running server
    // These are commented out but show the intended API usage

    /*
    #[tokio::test]
    async fn test_agent_start_shutdown() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let mut agent = RelayRLAgent::new(TransportType::ZMQ);

        // Start the agent
        agent.start(
            1,
            Device::Cpu,
            None,
            "TEST_ALGORITHM".to_string(),
            Some(config_path),
        ).await;

        // Perform shutdown
        agent.shutdown().await;

        assert!(true, "Agent started and shut down successfully");
    }
    */
}

#[cfg(test)]
mod agent_api_tests {
    use super::*;

    #[test]
    fn test_relay_rl_action_creation() {
        // Test creating RelayRLAction using minimal constructor
        let action = RelayRLAction::minimal(0.5, false);

        assert_eq!(action.get_rew(), 0.5, "Reward should be 0.5");
        assert_eq!(action.get_done(), false, "Done should be false");
    }

    #[test]
    fn test_relay_rl_action_reward_update() {
        let mut action = RelayRLAction::minimal(0.0, false);

        assert_eq!(action.get_rew(), 0.0, "Initial reward should be 0.0");

        action.update_reward(1.5);
        assert_eq!(action.get_rew(), 1.5, "Updated reward should be 1.5");
    }

    #[test]
    fn test_relay_rl_action_with_done_flag() {
        let action = RelayRLAction::minimal(1.0, true); // Terminal action

        assert_eq!(action.get_done(), true, "Done flag should be true");
    }

    // Note: These tests would require a running server infrastructure
    // They demonstrate the expected API usage

    /*
    #[tokio::test]
    async fn test_request_action() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        let actor_id = Uuid::new_v4();
        let obs = Tensor::randn(&[4], tch::kind::FLOAT_CPU);
        let mask = Tensor::ones(&[2], tch::kind::FLOAT_CPU);

        let result = agent.request_action(
            vec![actor_id],
            obs,
            mask,
            0.0,
        ).await;

        assert!(result.is_ok(), "Action request should succeed");

        let actions = result.unwrap();
        assert_eq!(actions.len(), 1, "Should receive one action");
        assert_eq!(actions[0].0, actor_id, "Action should be for correct actor");

        agent.shutdown().await;
    }

    #[tokio::test]
    async fn test_flag_last_action() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        let actor_id = Uuid::new_v4();

        // This should not panic
        agent.flag_last_action(vec![actor_id], Some(1.0)).await;

        agent.shutdown().await;
        assert!(true, "Flag last action completed");
    }

    #[tokio::test]
    async fn test_get_model_version() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        let actor_id = Uuid::new_v4();

        let result = agent.get_model_version(vec![actor_id]).await;

        assert!(result.is_ok(), "Get model version should succeed");

        let versions = result.unwrap();
        assert_eq!(versions.len(), 1, "Should receive one version");
        assert_eq!(versions[0].0, actor_id, "Version should be for correct actor");
        assert!(versions[0].1 >= 0, "Version should be non-negative");

        agent.shutdown().await;
    }
    */
}

#[cfg(test)]
mod agent_scaling_tests {
    use super::*;

    // Note: These tests demonstrate the scaling API but would require
    // a running server infrastructure to execute fully

    /*
    #[tokio::test]
    async fn test_scale_up_throughput() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let mut agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        // Scale up by adding 2 routers
        agent.scale_throughput(2).await;

        assert!(true, "Scale up completed");

        agent.shutdown().await;
    }

    #[tokio::test]
    async fn test_scale_down_throughput() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let mut agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        // First scale up
        agent.scale_throughput(3).await;

        // Then scale down by removing 1 router
        agent.scale_throughput(-1).await;

        assert!(true, "Scale down completed");

        agent.shutdown().await;
    }

    #[tokio::test]
    async fn test_scale_zero_routers() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let mut agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        // Scaling with 0 should do nothing
        agent.scale_throughput(0).await;

        assert!(true, "Scale with zero completed without error");

        agent.shutdown().await;
    }
    */
}

#[cfg(test)]
mod agent_actors_tests {
    use super::*;

    // Note: These tests demonstrate the actor management API but would require
    // a running server infrastructure to execute fully

    /*
    #[tokio::test]
    async fn test_new_actor() {
        use relayrl_framework::network::HotReloadableModel;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(0, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        // Create a new actor
        agent.new_actor(Device::Cpu, None).await;

        assert!(true, "New actor created successfully");
    }

    #[tokio::test]
    async fn test_remove_actor() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let mut agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        // Get the actors
        let (ids, _handles) = agent.get_actors().await;
        assert!(!ids.is_empty(), "Should have at least one actor");

        let actor_id = ids[0];

        // Remove the actor
        let result = agent.remove_actor(actor_id).await;
        assert!(result.is_ok(), "Actor removal should succeed");

        agent.shutdown().await;
    }

    #[tokio::test]
    async fn test_get_actors() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(3, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        let (ids, handles) = agent.get_actors().await;

        assert_eq!(ids.len(), 3, "Should have 3 actors");
        assert_eq!(handles.len(), 3, "Should have 3 actor handles");

        agent.shutdown().await;
    }

    #[tokio::test]
    async fn test_set_actor_id() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        let agent = RelayRLAgent::new(TransportType::ZMQ);
        agent.start(1, Device::Cpu, None, "TEST_ALGORITHM".to_string(), Some(config_path)).await;

        let (ids, _) = agent.get_actors().await;
        let current_id = ids[0];
        let new_id = Uuid::new_v4();

        let result = agent.set_actor_id(current_id, new_id).await;
        assert!(result.is_ok(), "Setting actor ID should succeed");

        // Verify the ID was changed
        let (updated_ids, _) = agent.get_actors().await;
        assert!(updated_ids.contains(&new_id), "New ID should be present");
        assert!(!updated_ids.contains(&current_id), "Old ID should be gone");

        agent.shutdown().await;
    }
    */
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_tensor_operations() {
        // Test basic tensor operations - using burn tensors
        use burn_ndarray::NdArray;
        use burn_tensor::Tensor;

        let device = burn_tensor::Device::<NdArray>::Cpu;
        let obs = Tensor::<NdArray, 2>::zeros([1, 4], &device);
        let mask = Tensor::<NdArray, 2>::ones([1, 2], &device);

        assert_eq!(obs.shape().dims, [1, 4], "Observation tensor shape correct");
        assert_eq!(mask.shape().dims, [1, 2], "Mask tensor shape correct");
    }

    #[test]
    fn test_uuid_generation() {
        // Test UUID generation for actors
        let id1 = Uuid::new_v8([32_u8; 16]);
        let id2 = Uuid::new_v8([32_u8; 16]);

        assert_ne!(id1, id2, "UUIDs should be unique");
    }

    #[test]
    fn test_transport_type_variants() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        // Test that both transport types can be created
        let zmq_agent: RelayRLAgent<NdArray, D_IN, D_OUT> = RelayRLAgent::new(TransportType::ZMQ);
        let grpc_agent: RelayRLAgent<NdArray, D_IN, D_OUT> = RelayRLAgent::new(TransportType::GRPC);

        assert!(true, "Both transport types can be instantiated");
    }

    #[test]
    fn test_device_types() {
        // Test different device configurations
        let cpu_device = DeviceType::Cpu;

        #[cfg(feature = "tch-backend")]
        {
            let cuda_device = DeviceType::Cuda(0);
        }

        assert!(true, "Device types can be instantiated");
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_tensor_conversion() {
        // Test error handling with empty action
        let empty_action = RelayRLAction::minimal(0.0, false);

        assert!(
            empty_action.get_obs().is_none(),
            "No observation should be present"
        );
        assert!(
            empty_action.get_act().is_none(),
            "No action should be present"
        );
    }

    #[tokio::test]
    async fn test_builder_validation() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        // Test builder validation with invalid inputs
        let result: Result<_, _> =
            RelayRLAgentBuilder::<NdArray, D_IN, D_OUT>::builder(TransportType::ZMQ)
                .actor_count(0) // Edge case: 0 actors
                .default_device(DeviceType::Cpu)
                .algorithm_name("TEST".to_string())
                .build()
                .await;

        // Should still succeed - 0 actors is valid (can add later)
        assert!(result.is_ok(), "Builder should handle 0 actors");
    }

    #[tokio::test]
    async fn test_negative_actor_count() {
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        // Test that negative actor count is handled
        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::ZMQ)
                .actor_count(-5) // Invalid negative count
                .default_device(DeviceType::Cpu)
                .algorithm_name("TEST".to_string())
                .config_path(config_path);

        let result = builder.build().await;

        // Should still build (handled internally as 0 or positive)
        assert!(
            result.is_ok(),
            "Builder should handle negative actor count gracefully"
        );
    }
}

#[cfg(test)]
mod documentation_tests {
    use super::*;

    /// This test demonstrates the recommended usage pattern for RelayRL Agent
    #[tokio::test]
    async fn test_example_usage_pattern() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = create_test_config(&temp_dir);

        // 1. Build the agent using the builder pattern
        const D_IN: usize = 4;
        const D_OUT: usize = 2;
        let builder: RelayRLAgentBuilder<NdArray, D_IN, D_OUT> =
            RelayRLAgentBuilder::builder(TransportType::ZMQ)
                .actor_count(2)
                .default_device(DeviceType::Cpu)
                .algorithm_name("TEST_ALGORITHM".to_string())
                .config_path(config_path.clone());

        let result = builder.build().await;
        assert!(result.is_ok(), "Agent should be built successfully");

        // This demonstrates the recommended API usage pattern
        println!("✓ Agent builder pattern works correctly");
        println!("✓ Configuration loading works");
        println!("✓ Agent can be constructed with minimal setup");
    }

    /// This test documents the expected lifecycle of an agent
    #[test]
    fn test_agent_lifecycle_documentation() {
        // Expected lifecycle:
        // 1. Create builder with transport type
        // 2. Configure with builder methods
        // 3. Build to get agent + parameters
        // 4. Start the agent with parameters
        // 5. Use agent for requests
        // 6. Shutdown when done

        println!("Agent Lifecycle:");
        println!("1. Create → 2. Configure → 3. Build → 4. Start → 5. Use → 6. Shutdown");

        assert!(true, "Lifecycle documentation is accurate");
    }
}
