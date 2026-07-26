//! Offline environment-driven execution integration tests: binding an `Environment`, scaling
//! its env-copy count, running an evaluation rollout, and the single-active-loop-per-actor
//! guard. No transport feature is required or exercised.
//!
//! `run_env_eval` drives its rollout loop via `tokio::task::block_in_place`, which panics on a
//! current-thread runtime, so every test here uses the multi-threaded flavor.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use common::{TestBackend, load_test_model_module, start_offline_agent};
use relayrl_framework::prelude::network::{
    ActorDataMode, ClientError, RelayRLActors, RelayRLBatchEnv,
};
use relayrl_framework::prelude::templates::environment::{
    Done, EnvDType, EnvNdArrayDType, Environment, EnvironmentError, EnvironmentHandle,
    EnvironmentKind, Mask, Observation, Reward, ScalarEnvReset, ScalarEnvironment, Truncated,
};
use relayrl_types::data::tensor::DeviceType;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

/// A minimal continuous scalar environment double: rank-1 `f32` observation/action of size 2,
/// matching the shared identity ONNX model's declared shape. Never terminates on its own, so
/// `loop_iters` fully controls how long a rollout runs.
#[derive(Clone)]
struct ContinuousTestEnv;

impl Environment for ContinuousTestEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }
    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(self.flat_observation_bytes()))
    }
    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(()))
    }
    fn observation_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn action_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn observation_dim(&self) -> usize {
        2
    }
    fn action_dim(&self) -> usize {
        2
    }
    fn flat_observation_bytes(&self) -> Observation {
        vec![0u8; 8]
    }
    fn flat_mask_bytes(&self) -> Mask {
        None
    }
    fn action_is_discrete(&self) -> bool {
        false
    }
    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Scalar
    }
    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(Box::new(*self))
    }
}

impl ScalarEnvironment for ContinuousTestEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        Ok(ScalarEnvReset {
            observation: self.flat_observation_bytes(),
            info: None,
        })
    }
    fn step_bytes(&self, _action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        Some((self.flat_observation_bytes(), None, 0.0, false, false))
    }
}

/// Like [`ContinuousTestEnv`], but each step blocks the (blocking-capable) worker thread for a
/// couple of milliseconds. Used only to widen the window in which a rollout is provably still
/// in flight, so the "second concurrent call is rejected" test does not race.
#[derive(Clone)]
struct SlowContinuousTestEnv;

impl Environment for SlowContinuousTestEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }
    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(self.flat_observation_bytes()))
    }
    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(()))
    }
    fn observation_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn action_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn observation_dim(&self) -> usize {
        2
    }
    fn action_dim(&self) -> usize {
        2
    }
    fn flat_observation_bytes(&self) -> Observation {
        vec![0u8; 8]
    }
    fn flat_mask_bytes(&self) -> Mask {
        None
    }
    fn action_is_discrete(&self) -> bool {
        false
    }
    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Scalar
    }
    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(Box::new(*self))
    }
}

impl ScalarEnvironment for SlowContinuousTestEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        Ok(ScalarEnvReset {
            observation: self.flat_observation_bytes(),
            info: None,
        })
    }
    fn step_bytes(&self, _action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        std::thread::sleep(Duration::from_millis(2));
        Some((self.flat_observation_bytes(), None, 0.0, false, false))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_env_reports_the_configured_copy_count() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actor = ctx
        .agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    ctx.agent
        .set_env(&actor, Box::new(ContinuousTestEnv), 4)
        .await?;

    assert_eq!(ctx.agent.get_env_count(&actor).await?, 4);

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_env_count_increases_and_decreases_live() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actor = ctx
        .agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;
    ctx.agent
        .set_env(&actor, Box::new(ContinuousTestEnv), 2)
        .await?;

    ctx.agent.set_env_count(&actor, 5).await?;
    assert_eq!(ctx.agent.get_env_count(&actor).await?, 5);

    ctx.agent.set_env_count(&actor, 1).await?;
    assert_eq!(ctx.agent.get_env_count(&actor).await?, 1);

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remove_env_makes_the_actor_report_no_bound_environment()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actor = ctx
        .agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;
    ctx.agent
        .set_env(&actor, Box::new(ContinuousTestEnv), 1)
        .await?;
    assert_eq!(ctx.agent.get_env_count(&actor).await?, 1);

    ctx.agent.remove_env(&actor).await?;

    let result = ctx.agent.get_env_count(&actor).await;
    assert!(
        result.is_err(),
        "removed environment should no longer be queryable"
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_env_eval_completes_a_small_rollout() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actor = ctx
        .agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;
    ctx.agent
        .set_env(&actor, Box::new(ContinuousTestEnv), 1)
        .await?;

    ctx.agent.run_env_eval(&actor, 5).await?;

    // The actor stays usable for a second rollout once the first completes.
    ctx.agent.run_env_eval(&actor, 5).await?;

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_env_eval_rejects_a_concurrent_call_on_the_same_actor()
-> Result<(), Box<dyn std::error::Error>> {
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempfile::tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) =
        relayrl_framework::prelude::network::AgentBuilder::<TestBackend>::builder()
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
    agent
        .set_env(&actor, Box::new(SlowContinuousTestEnv), 1)
        .await?;

    let agent = Arc::new(agent);
    let long_running_actor = actor.clone();
    let long_running_agent = agent.clone();
    let long_running = tokio::spawn(async move {
        long_running_agent
            .run_env_eval(&long_running_actor, 200)
            .await
    });
    
    tokio::time::sleep(Duration::from_millis(60)).await;

    let second_call = agent.run_env_eval(&actor, 5).await;
    assert!(
        matches!(second_call, Err(ClientError::RunEnvActive(_))),
        "a second run_env_eval on the same actor should be rejected while the first is active"
    );

    let first_call = long_running.await?;
    assert!(
        first_call.is_ok(),
        "the original rollout should still complete successfully"
    );

    let mut agent = Arc::try_unwrap(agent)
        .expect("no other Arc clones should remain after the spawned task completes");
    agent.shutdown().await?;
    Ok(())
}
