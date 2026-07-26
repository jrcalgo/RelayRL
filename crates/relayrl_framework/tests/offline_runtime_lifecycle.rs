//! Offline runtime lifecycle integration tests: shutdown draining, restart, and live router/
//! buffer scaling through the public `RelayRLAgent` API. No transport feature is required or
//! exercised.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use burn_ndarray::NdArrayDevice;
use burn_tensor::{Float, Tensor, TensorData};
use common::{TestBackend, load_test_model_module};
use relayrl_framework::prelude::network::{
    ActorDataMode, AgentBuilder, ClientError, RelayRLActors, RelayRLStepDriven,
};
use relayrl_types::data::tensor::DeviceType;
use tempfile::tempdir;

fn zero_obs() -> Tensor<TestBackend, 1, Float> {
    Tensor::<TestBackend, 1, Float>::from_data(
        TensorData::new(vec![1.0_f32, 2.0_f32], [2]),
        &NdArrayDevice::default(),
    )
}

#[tokio::test]
async fn shutdown_returns_still_buffered_trajectories() -> Result<(), Box<dyn std::error::Error>> {
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_data_mode(ActorDataMode::OfflineWithCache(10))
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;
    agent.start(params).await?;

    let actor = agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    // Deliberately do not flag the episode boundary: the in-flight (still-open) trajectory
    // should be part of the snapshot `shutdown` returns, since it never reaches the cache via
    // the normal completed-trajectory path.
    agent
        .request_action::<1, 1, Float, Float>(
            &actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;

    let drained = agent.shutdown().await?;
    let trajectories = drained
        .as_ref()
        .and_then(|by_actor| by_actor.get(&actor.id()))
        .expect("shutdown should return the in-flight trajectory for this actor");
    assert!(!trajectories.is_empty());

    Ok(())
}

#[tokio::test]
async fn restart_tears_down_and_reinitializes_with_an_empty_actor_list()
-> Result<(), Box<dyn std::error::Error>> {
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_data_mode(ActorDataMode::Disabled)
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;
    let params_for_restart = params.clone();
    agent.start(params).await?;

    agent
        .new_actors::<1, 1>(
            2,
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;
    assert_eq!(agent.get_all_actors().await?.len(), 2);

    agent.restart(params_for_restart).await?;

    assert!(agent.get_all_actors().await?.is_empty());

    // The restarted runtime accepts new work.
    let actor = agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;
    let action = agent
        .request_action::<1, 1, Float, Float>(
            &actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;
    assert_eq!(action.get_agent_id(), Some(&actor.id()));

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn scale_data_routers_out_and_in_keeps_actors_usable()
-> Result<(), Box<dyn std::error::Error>> {
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_data_mode(ActorDataMode::Disabled)
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;
    agent.start(params).await?;

    let actors = agent
        .new_actors::<1, 1>(
            3,
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    agent.scale_data_routers(2).await?;
    let after_scale_out = agent
        .request_actions::<1, 1, Float, Float>(
            &actors,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;
    assert_eq!(after_scale_out.len(), actors.len());

    agent.scale_data_routers(-2).await?;
    let after_scale_in = agent
        .request_actions::<1, 1, Float, Float>(
            &actors,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;
    assert_eq!(after_scale_in.len(), actors.len());

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn scale_data_routers_zero_is_rejected_as_a_noop() -> Result<(), Box<dyn std::error::Error>> {
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_data_mode(ActorDataMode::Disabled)
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;
    agent.start(params).await?;

    let result = agent.scale_data_routers(0).await;
    assert!(matches!(result, Err(ClientError::NoopRouterScale(_))));

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn scale_data_buffers_resizes_without_disrupting_requests()
-> Result<(), Box<dyn std::error::Error>> {
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_data_mode(ActorDataMode::Disabled)
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;
    agent.start(params).await?;

    let actor = agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    agent.scale_data_buffers(2048).await?;

    let action = agent
        .request_action::<1, 1, Float, Float>(
            &actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;
    assert_eq!(action.get_agent_id(), Some(&actor.id()));

    let zero_size_result = agent.scale_data_buffers(0).await;
    assert!(matches!(
        zero_size_result,
        Err(ClientError::InvalidDataParams(_))
    ));

    agent.shutdown().await?;
    Ok(())
}
