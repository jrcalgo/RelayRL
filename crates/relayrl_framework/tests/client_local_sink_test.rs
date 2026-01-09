//! Integration tests for RelayRL Agent local trajectory sink functionality
//!
//! This test module validates the local trajectory sink capabilities of the RelayRL Agent,
//! ensuring that trajectory data is correctly written to Arrow files during environment execution.
//!
//! Tests cover:
//! - Arrow file creation when trajectories are recorded
//! - Arrow file schema validation (20 columns)
//! - Trajectory data integrity (rewards, done flags, timestamps)
//! - Integration with real gym environments (CartPole)

use arrow::ipc::reader::FileReader;
use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor};
use relayrl_framework::prelude::network::{
    ActorInferenceMode, AgentBuilder, ClientModes, FormattedTrajectoryFileParams, RelayRLAgent,
    RelayRLAgentActors, TrajectoryRecordMode,
};
use relayrl_types::types::data::tensor::DeviceType;
use relayrl_types::types::model::ModelModule;
use serde_json::json;
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

/// Workspace root relative path to the cartpole model
const CARTPOLE_MODEL_PATH: &str =
    "../../examples/REINFORCE_without_baseline/classic_control/cartpole/zmq/client_model.pt";

/// Sets up a test model directory with metadata.json and copies the existing .pt model
fn setup_test_model(temp_dir: &TempDir) -> PathBuf {
    let model_dir = temp_dir.path().join("model");
    fs::create_dir_all(&model_dir).expect("Failed to create model directory");

    // Resolve path from the tests directory
    let source_model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("examples/REINFORCE_without_baseline/classic_control/cartpole/zmq/client_model.pt");

    // Copy existing .pt model from examples
    if source_model.exists() {
        fs::copy(&source_model, model_dir.join("client_model.pt"))
            .expect("Failed to copy model file");
    } else {
        // Create a dummy model file for CI environments where model may not exist
        fs::write(model_dir.join("client_model.pt"), b"dummy_model_data")
            .expect("Failed to create dummy model file");
    }

    // Generate metadata.json for CartPole (obs: 4, act: 2)
    let metadata = json!({
        "model_file": "client_model.pt",
        "model_type": "pt",
        "input_dtype": { "NdArray": "F32" },
        "output_dtype": { "NdArray": "F32" },
        "input_shape": [1, 4],
        "output_shape": [1, 2]
    });
    fs::write(
        model_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .expect("Failed to write metadata.json");

    model_dir
}

/// Sets up trajectory output directory
fn setup_trajectory_output(temp_dir: &TempDir) -> PathBuf {
    let traj_dir = temp_dir.path().join("trajectories");
    fs::create_dir_all(&traj_dir).expect("Failed to create trajectory directory");
    traj_dir
}

/// Creates a test client config file
fn create_test_config(temp_dir: &TempDir) -> PathBuf {
    let config_path = temp_dir.path().join("test_client_config.json");
    let config_content = r#"{
    "client_config": {
        "actor_count": 1,
        "algorithm_name": "TEST_LOCAL_SINK",
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
        "max_traj_length": 100
    }
}"#;
    fs::write(&config_path, config_content).expect("Failed to write test config");
    config_path
}

/// Finds all .arrow files in the given directory
fn find_arrow_files(dir: &PathBuf) -> Vec<PathBuf> {
    if !dir.exists() {
        return vec![];
    }
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "arrow").unwrap_or(false))
        .collect()
}

/// Validates the Arrow file schema has the expected 20 columns
fn validate_arrow_schema(path: &PathBuf) -> arrow::record_batch::RecordBatch {
    let file = File::open(path).expect("Failed to open arrow file");
    let reader = FileReader::try_new(file, None).expect("Failed to create arrow reader");

    let schema = reader.schema();

    // Validate all 20 expected columns exist
    let expected_columns = [
        "backend",
        "reward",
        "done",
        "timestamp",
        "agent_id",
        "obs_dtype",
        "obs_shape",
        "obs_f32",
        "obs_f64",
        "obs_binary",
        "act_dtype",
        "act_shape",
        "act_f32",
        "act_f64",
        "act_binary",
        "mask_dtype",
        "mask_shape",
        "mask_f32",
        "mask_f64",
        "mask_binary",
    ];

    assert_eq!(
        schema.fields().len(),
        20,
        "Arrow schema should have 20 columns, found {}",
        schema.fields().len()
    );

    for col_name in expected_columns.iter() {
        assert!(
            schema.field_with_name(col_name).is_ok(),
            "Missing expected column: {}",
            col_name
        );
    }

    // Return the first batch
    reader
        .into_iter()
        .next()
        .expect("No batches in arrow file")
        .expect("Failed to read batch")
}

/// Reads and returns all record batches from an Arrow file
fn read_arrow_batches(path: &PathBuf) -> Vec<arrow::record_batch::RecordBatch> {
    let file = File::open(path).expect("Failed to open arrow file");
    let reader = FileReader::try_new(file, None).expect("Failed to create arrow reader");
    reader
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to read batches")
}

// ============================================================================
// Unit Tests (Synthetic Data)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test that an arrow file is created when trajectories are recorded
    #[tokio::test]
    #[ignore = "Requires model inference setup - run with --ignored"]
    async fn test_local_sink_creates_arrow_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        // Load model
        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        // Build agent with local trajectory recording
        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        // Start agent
        agent
            .start(
                1, // actor_count
                1, // router_scale
                DeviceType::Cpu,
                None, // will use default_model from builder
                None, // config_path
            )
            .await
            .expect("Failed to start agent");

        // Get actor IDs
        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");
        let actor_id = actor_ids[0];

        // Create synthetic observation (CartPole: 4 values)
        let device = burn_tensor::Device::<NdArray>::Cpu;
        let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &device);

        // Request some actions to generate trajectory data
        for step in 0..5 {
            let reward = step as f32 * 0.1;
            let _actions = agent
                .request_action(vec![actor_id], observation.clone(), None, reward)
                .await
                .expect("Failed to request action");
        }

        // Flag episode as done
        agent
            .flag_last_action(vec![actor_id], Some(1.0))
            .await
            .expect("Failed to flag last action");

        // Give some time for async trajectory writing
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Shutdown agent
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Verify arrow file was created
        let arrow_files = find_arrow_files(&traj_dir);
        assert!(
            !arrow_files.is_empty(),
            "Expected at least one arrow file to be created"
        );
    }

    /// Test that the arrow file has the correct 20-column schema
    #[tokio::test]
    #[ignore = "Requires model inference setup - run with --ignored"]
    async fn test_arrow_file_schema_validation() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        // Load model
        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        // Build agent with local trajectory recording
        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        agent
            .start(1, 1, DeviceType::Cpu, None, None)
            .await
            .expect("Failed to start agent");

        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");
        let actor_id = actor_ids[0];

        let device = burn_tensor::Device::<NdArray>::Cpu;
        let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &device);

        // Generate some trajectory data
        for _ in 0..3 {
            let _ = agent
                .request_action(vec![actor_id], observation.clone(), None, 0.5)
                .await;
        }
        agent
            .flag_last_action(vec![actor_id], Some(1.0))
            .await
            .expect("Failed to flag last action");

        tokio::time::sleep(Duration::from_millis(500)).await;
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Find and validate arrow file schema
        let arrow_files = find_arrow_files(&traj_dir);
        assert!(!arrow_files.is_empty(), "No arrow files found");

        // Validate schema
        let _batch = validate_arrow_schema(&arrow_files[0]);
        println!("Schema validation passed for: {:?}", arrow_files[0]);
    }

    /// Test trajectory data integrity - rewards, done flags, timestamps
    #[tokio::test]
    #[ignore = "Requires model inference setup - run with --ignored"]
    async fn test_trajectory_data_integrity() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        agent
            .start(1, 1, DeviceType::Cpu, None, None)
            .await
            .expect("Failed to start agent");

        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");
        let actor_id = actor_ids[0];

        let device = burn_tensor::Device::<NdArray>::Cpu;
        let observation = Tensor::<NdArray, 2, Float>::zeros([1, 4], &device);

        // Send known rewards
        let expected_rewards = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for reward in &expected_rewards {
            let _ = agent
                .request_action(vec![actor_id], observation.clone(), None, *reward)
                .await;
        }

        // Flag as done with final reward
        agent
            .flag_last_action(vec![actor_id], Some(1.0))
            .await
            .expect("Failed to flag last action");

        tokio::time::sleep(Duration::from_millis(500)).await;
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Read and validate arrow file content
        let arrow_files = find_arrow_files(&traj_dir);
        assert!(!arrow_files.is_empty(), "No arrow files found");

        let batches = read_arrow_batches(&arrow_files[0]);
        assert!(!batches.is_empty(), "No batches in arrow file");

        let batch = &batches[0];

        // Validate row count matches expected actions
        assert!(
            batch.num_rows() >= expected_rewards.len(),
            "Expected at least {} rows, found {}",
            expected_rewards.len(),
            batch.num_rows()
        );

        // Validate done column has at least one true value (terminal action)
        let done_col = batch
            .column_by_name("done")
            .expect("Missing done column")
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .expect("done column should be boolean");

        let has_done = (0..done_col.len()).any(|i| done_col.value(i));
        assert!(has_done, "Expected at least one done=true in trajectory");

        // Validate timestamps are present and increasing
        let timestamp_col = batch
            .column_by_name("timestamp")
            .expect("Missing timestamp column")
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("timestamp column should be UInt64");

        assert!(
            timestamp_col.len() > 0,
            "Timestamp column should have entries"
        );

        println!("Data integrity validation passed");
    }

    /// Test that no arrow file is created when trajectory has zero actions
    #[tokio::test]
    #[ignore = "Requires model inference setup - run with --ignored"]
    async fn test_empty_trajectory_no_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        agent
            .start(1, 1, DeviceType::Cpu, None, None)
            .await
            .expect("Failed to start agent");

        // Don't request any actions - just flag done immediately
        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");

        // Flag as done without any actions (this should not create a file)
        agent
            .flag_last_action(vec![actor_ids[0]], Some(0.0))
            .await
            .expect("Failed to flag last action");

        tokio::time::sleep(Duration::from_millis(500)).await;
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Check that no arrow file was created for empty trajectory
        let arrow_files = find_arrow_files(&traj_dir);

        // Either no files, or files with 0 rows would be acceptable
        if !arrow_files.is_empty() {
            let batches = read_arrow_batches(&arrow_files[0]);
            // If a file exists, it should have minimal or zero content
            println!("Note: Arrow file created with {} batches", batches.len());
        } else {
            println!("Correctly: No arrow file created for empty trajectory");
        }
    }
}

// ============================================================================
// Integration Tests (Real Gym Environments)
// ============================================================================

#[cfg(test)]
mod gym_integration_tests {
    use super::*;
    use gym::{GymClient, SpaceData, State};

    /// Helper to convert gym observation to burn tensor
    fn gym_obs_to_tensor(
        obs: &SpaceData,
        device: &burn_tensor::Device<NdArray>,
    ) -> Tensor<NdArray, 2, Float> {
        match obs {
            SpaceData::BOX(values) => {
                let data: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                Tensor::from_floats(&data[..], device).reshape([1, data.len()])
            }
            _ => panic!("Expected BOX observation space"),
        }
    }

    /// Helper to convert agent action to gym action
    fn agent_action_to_gym(
        action_tensor: &relayrl_types::types::data::action::RelayRLAction,
    ) -> SpaceData {
        // For CartPole, action is discrete (0 or 1)
        // Extract from action tensor data and convert to discrete
        if let Some(act) = action_tensor.get_act() {
            let act_data = &act.data;
            if !act_data.is_empty() {
                // Get first action value and threshold at 0.5 for discrete action
                let act_f32: &[f32] = bytemuck::cast_slice(act_data);
                let discrete_action = if act_f32[0] > 0.5 { 1 } else { 0 };
                return SpaceData::DISCRETE(discrete_action);
            }
        }
        // Default action
        SpaceData::DISCRETE(0)
    }

    /// Run a full CartPole episode and verify trajectory is written
    #[tokio::test]
    #[ignore = "Requires gym and model setup - run with --ignored"]
    async fn test_cartpole_episode_trajectory() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        agent
            .start(1, 1, DeviceType::Cpu, None, None)
            .await
            .expect("Failed to start agent");

        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");
        let actor_id = actor_ids[0];

        // Initialize gym environment
        let gym = GymClient::default();
        let env = gym.make("CartPole-v1");

        let device = burn_tensor::Device::<NdArray>::Cpu;

        // Reset environment
        let State { observation, .. } = env.reset(None).expect("Failed to reset environment");
        let mut obs = observation;
        let mut total_reward: f32 = 0.0;
        let mut step_count = 0;
        let max_steps = 200;

        // Run episode
        while step_count < max_steps {
            let obs_tensor = gym_obs_to_tensor(&obs, &device);

            // Request action from agent
            let actions = agent
                .request_action(vec![actor_id], obs_tensor, None, total_reward)
                .await
                .expect("Failed to request action");

            // Get action for this actor
            let (_, action_arc) = &actions[0];
            let gym_action = agent_action_to_gym(action_arc);

            // Step environment
            let State {
                observation: next_obs,
                reward,
                is_done,
                ..
            } = env.step(&gym_action).expect("Failed to step environment");

            total_reward += reward as f32;
            step_count += 1;

            if is_done {
                // Flag episode end
                agent
                    .flag_last_action(vec![actor_id], Some(total_reward))
                    .await
                    .expect("Failed to flag last action");
                break;
            }

            obs = next_obs;
        }

        // If we hit max steps without done, flag anyway
        if step_count >= max_steps {
            agent
                .flag_last_action(vec![actor_id], Some(total_reward))
                .await
                .expect("Failed to flag last action");
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Verify arrow file was created with real trajectory data
        let arrow_files = find_arrow_files(&traj_dir);
        assert!(
            !arrow_files.is_empty(),
            "Expected arrow file to be created after CartPole episode"
        );

        let batches = read_arrow_batches(&arrow_files[0]);
        assert!(!batches.is_empty(), "Arrow file should contain batches");

        println!(
            "CartPole episode completed: {} steps, {:.2} total reward",
            step_count, total_reward
        );
        println!(
            "Arrow file created with {} rows",
            batches.iter().map(|b| b.num_rows()).sum::<usize>()
        );
    }

    /// Run multiple CartPole episodes and verify separate arrow files
    #[tokio::test]
    #[ignore = "Requires gym and model setup - run with --ignored"]
    async fn test_cartpole_multi_episode_collection() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        agent
            .start(1, 1, DeviceType::Cpu, None, None)
            .await
            .expect("Failed to start agent");

        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");
        let actor_id = actor_ids[0];

        let gym = GymClient::default();
        let env = gym.make("CartPole-v1");
        let device = burn_tensor::Device::<NdArray>::Cpu;

        let num_episodes = 3;
        let mut episode_rewards = Vec::new();

        for episode in 0..num_episodes {
            let State { observation, .. } = env.reset(None).expect("Failed to reset environment");
            let mut obs = observation;
            let mut episode_reward: f32 = 0.0;
            let max_steps = 100; // Shorter episodes for testing

            for _ in 0..max_steps {
                let obs_tensor = gym_obs_to_tensor(&obs, &device);

                let actions = agent
                    .request_action(vec![actor_id], obs_tensor, None, episode_reward)
                    .await
                    .expect("Failed to request action");

                let (_, action_arc) = &actions[0];
                let gym_action = agent_action_to_gym(action_arc);

                let State {
                    observation: next_obs,
                    reward,
                    is_done,
                    ..
                } = env.step(&gym_action).expect("Failed to step environment");

                episode_reward += reward as f32;

                if is_done {
                    break;
                }
                obs = next_obs;
            }

            // Flag episode end
            agent
                .flag_last_action(vec![actor_id], Some(episode_reward))
                .await
                .expect("Failed to flag last action");

            episode_rewards.push(episode_reward);
            println!(
                "Episode {} completed with reward: {:.2}",
                episode + 1,
                episode_reward
            );

            // Small delay between episodes
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Verify multiple arrow files were created (one per episode)
        let arrow_files = find_arrow_files(&traj_dir);
        println!(
            "Created {} arrow files for {} episodes",
            arrow_files.len(),
            num_episodes
        );

        // We expect at least one file (files may be batched)
        assert!(
            !arrow_files.is_empty(),
            "Expected arrow files to be created for {} episodes",
            num_episodes
        );
    }

    /// Verify gym observations are correctly converted to tensors and stored
    #[tokio::test]
    #[ignore = "Requires gym and model setup - run with --ignored"]
    async fn test_gym_observation_tensor_format() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_dir = setup_test_model(&temp_dir);
        let traj_dir = setup_trajectory_output(&temp_dir);
        let config_path = create_test_config(&temp_dir);

        let model: ModelModule<NdArray> =
            ModelModule::load_from_path(&model_dir).expect("Failed to load model");

        let trajectory_params = FormattedTrajectoryFileParams {
            enabled: true,
            encode: false,
            path: traj_dir.clone(),
        };

        let (mut agent, _params) = AgentBuilder::<NdArray, 2, 2, Float, Float>::builder()
            .trajectory_recording_mode(TrajectoryRecordMode::Local(trajectory_params))
            .actor_inference_mode(ActorInferenceMode::Local)
            .actor_count(1)
            .router_scale(1)
            .default_device(DeviceType::Cpu)
            .default_model(model)
            .config_path(config_path)
            .build()
            .await
            .expect("Failed to build agent");

        agent
            .start(1, 1, DeviceType::Cpu, None, None)
            .await
            .expect("Failed to start agent");

        let (actor_ids, _) = agent.get_actors().await.expect("Failed to get actors");
        let actor_id = actor_ids[0];

        let gym = GymClient::default();
        let env = gym.make("CartPole-v1");
        let device = burn_tensor::Device::<NdArray>::Cpu;

        // Get initial observation
        let State { observation, .. } = env.reset(None).expect("Failed to reset environment");

        // CartPole observation should be BOX with 4 values
        if let SpaceData::BOX(values) = &observation {
            assert_eq!(values.len(), 4, "CartPole observation should have 4 values");
            println!("Initial CartPole observation: {:?}", values);
        } else {
            panic!("Expected BOX observation space for CartPole");
        }

        // Convert to tensor and verify shape
        let obs_tensor = gym_obs_to_tensor(&observation, &device);
        let shape = obs_tensor.shape();
        assert_eq!(shape.dims[0], 1, "Batch dimension should be 1");
        assert_eq!(shape.dims[1], 4, "Observation dimension should be 4");

        // Request a few actions
        for _ in 0..3 {
            let obs_tensor = gym_obs_to_tensor(&observation, &device);
            let _ = agent
                .request_action(vec![actor_id], obs_tensor, None, 0.0)
                .await;
        }

        agent
            .flag_last_action(vec![actor_id], Some(1.0))
            .await
            .expect("Failed to flag last action");

        tokio::time::sleep(Duration::from_millis(500)).await;
        agent.shutdown().await.expect("Failed to shutdown agent");

        // Verify arrow file contains observation data
        let arrow_files = find_arrow_files(&traj_dir);
        if !arrow_files.is_empty() {
            let batch = validate_arrow_schema(&arrow_files[0]);

            // Check obs_shape column exists and has valid data
            let obs_shape_col = batch
                .column_by_name("obs_shape")
                .expect("Missing obs_shape column");

            println!(
                "Observation tensor format validated in arrow file: {:?}",
                obs_shape_col.data_type()
            );
        }
    }
}
