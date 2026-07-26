//! Offline trajectory data mode integration tests: in-memory cache, local file sinks
//! (CSV/Arrow), the combined mode, and `Disabled`. No transport feature is required or
//! exercised.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use burn_ndarray::NdArrayDevice;
use burn_tensor::{Float, Tensor, TensorData};
use common::{DEFAULT_POLL_TIMEOUT, TestBackend, start_offline_agent, wait_until};
use relayrl_framework::prelude::network::{
    ActorDataMode, LocalTrajectoryFileParams, LocalTrajectoryFileType, RelayRLActors,
    RelayRLStepDriven,
};
use relayrl_types::data::tensor::DeviceType;
use relayrl_types::prelude::records::{ArrowTrajectory, CsvTrajectory};
use std::path::PathBuf;
use tempfile::tempdir;

fn zero_obs() -> Tensor<TestBackend, 1, Float> {
    Tensor::<TestBackend, 1, Float>::from_data(
        TensorData::new(vec![1.0_f32, 2.0_f32], [2]),
        &NdArrayDevice::default(),
    )
}

/// Requests one action then flags the episode boundary, producing exactly one completed
/// single-action trajectory for `actor`.
async fn complete_one_episode(
    agent: &mut relayrl_framework::prelude::network::RelayRLAgent<TestBackend>,
    actor: &relayrl_framework::prelude::network::ActorInfo,
) -> Result<(), Box<dyn std::error::Error>> {
    agent
        .request_action::<1, 1, Float, Float>(
            actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;
    agent.flag_last_action(actor, Some(1.0)).await?;
    Ok(())
}

fn find_output_file(dir: &std::path::Path, extension: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some(extension))
}

#[tokio::test]
async fn cache_mode_drains_completed_trajectories() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::OfflineWithCache(10)).await? else {
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

    complete_one_episode(&mut ctx.agent, &actor).await?;

    let found = wait_until(
        || {
            ctx.agent
                .drain_trajectory_caches(std::slice::from_ref(&actor))
                .map(|cache| cache.values().any(|trajs| !trajs.is_empty()))
                .unwrap_or(false)
        },
        DEFAULT_POLL_TIMEOUT,
    )
    .await;
    assert!(found, "expected a completed trajectory to reach the cache");

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn cache_mode_evicts_oldest_trajectory_beyond_its_size()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::OfflineWithCache(2)).await? else {
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

    for _ in 0..3 {
        complete_one_episode(&mut ctx.agent, &actor).await?;
    }

    tokio::time::sleep(std::time::Duration::from_millis(800)).await;

    let drained = ctx
        .agent
        .drain_trajectory_caches(std::slice::from_ref(&actor))
        .and_then(|mut cache| cache.remove(&actor.id()))
        .unwrap_or_default();
    assert!(
        (1..=2).contains(&drained.len()),
        "expected the cache to bound itself to at most 2 entries, got {}",
        drained.len()
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn disabled_mode_never_populates_the_cache() -> Result<(), Box<dyn std::error::Error>> {
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

    complete_one_episode(&mut ctx.agent, &actor).await?;
    
    // Give the router a moment to process the flagged action, then confirm no cache exists.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let drained = ctx
        .agent
        .drain_trajectory_caches(std::slice::from_ref(&actor));
    assert!(
        drained.is_none(),
        "Disabled mode should never expose a trajectory cache"
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn file_mode_writes_a_readable_csv_trajectory() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = tempdir()?;
    let file_params = LocalTrajectoryFileParams::new(
        output_dir.path().to_path_buf(),
        LocalTrajectoryFileType::Csv,
    )?;

    let Some(mut ctx) =
        start_offline_agent(ActorDataMode::OfflineWithFiles(Some(file_params))).await?
    else {
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

    complete_one_episode(&mut ctx.agent, &actor).await?;

    let output_path = output_dir.path().to_path_buf();
    let found = wait_until(
        || find_output_file(&output_path, "csv").is_some(),
        DEFAULT_POLL_TIMEOUT,
    )
    .await;
    assert!(found, "expected a CSV trajectory file to be written");

    let csv_path = find_output_file(&output_path, "csv").expect("csv file should exist");
    let loaded = CsvTrajectory::new(None).from_csv(&csv_path, 10_000_000, None, None)?;
    let trajectory = loaded
        .trajectory
        .expect("CSV readback should reconstruct a trajectory");
    // One action from `request_action` plus the terminal marker appended by `flag_last_action`.
    assert_eq!(trajectory.len(), 2);
    assert!(
        trajectory
            .actions
            .last()
            .expect("trajectory should be non-empty")
            .get_done()
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn file_mode_writes_a_readable_arrow_trajectory() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = tempdir()?;
    let file_params = LocalTrajectoryFileParams::new(
        output_dir.path().to_path_buf(),
        LocalTrajectoryFileType::Arrow,
    )?;

    let Some(mut ctx) =
        start_offline_agent(ActorDataMode::OfflineWithFiles(Some(file_params))).await?
    else {
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

    complete_one_episode(&mut ctx.agent, &actor).await?;

    let output_path = output_dir.path().to_path_buf();
    let found = wait_until(
        || find_output_file(&output_path, "arrow").is_some(),
        DEFAULT_POLL_TIMEOUT,
    )
    .await;
    assert!(found, "expected an Arrow trajectory file to be written");

    let arrow_path = find_output_file(&output_path, "arrow").expect("arrow file should exist");
    let loaded = ArrowTrajectory::new(None).from_arrow(&arrow_path, None, None)?;
    let trajectory = loaded
        .trajectory
        .expect("Arrow readback should reconstruct a trajectory");
    // One action from `request_action` plus the terminal marker appended by `flag_last_action`.
    assert_eq!(trajectory.len(), 2);
    assert!(
        trajectory
            .actions
            .last()
            .expect("trajectory should be non-empty")
            .get_done()
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn combined_mode_populates_both_cache_and_file() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = tempdir()?;
    let file_params = LocalTrajectoryFileParams::new(
        output_dir.path().to_path_buf(),
        LocalTrajectoryFileType::Csv,
    )?;

    let Some(mut ctx) = start_offline_agent(ActorDataMode::OfflineWithFilesAndCache(
        Some(file_params),
        10,
    ))
    .await?
    else {
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

    complete_one_episode(&mut ctx.agent, &actor).await?;

    let output_path = output_dir.path().to_path_buf();
    let file_found = wait_until(
        || find_output_file(&output_path, "csv").is_some(),
        DEFAULT_POLL_TIMEOUT,
    )
    .await;
    assert!(file_found, "expected a CSV trajectory file to be written");

    let cache_found = wait_until(
        || {
            ctx.agent
                .drain_trajectory_caches(std::slice::from_ref(&actor))
                .map(|cache| cache.values().any(|trajs| !trajs.is_empty()))
                .unwrap_or(false)
        },
        DEFAULT_POLL_TIMEOUT,
    )
    .await;
    assert!(
        cache_found,
        "expected the same trajectory to also reach the cache"
    );

    ctx.agent.shutdown().await?;
    Ok(())
}
