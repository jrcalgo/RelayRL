//! Offline model hot-swap integration tests: independent per-actor targeting, shared-device
//! deduplication, version tracking, and rank-mismatch rejection. No transport feature is
//! required or exercised.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use common::{
    DEFAULT_POLL_TIMEOUT, load_test_model_module, start_offline_agent,
    start_offline_agent_with_modes, wait_for_model_version_change,
};
use relayrl_framework::prelude::network::{
    ActorDataMode, ActorInferenceMode, ClientError, ModelMode, RelayRLActors,
};
use relayrl_types::data::tensor::DeviceType;

#[tokio::test]
async fn update_models_targets_only_the_specified_actor_in_independent_mode()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let (_swap_model_dir, swap_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let actors = ctx
        .agent
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
    let (target, other) = (actors[0].clone(), actors[1].clone());

    let baseline = ctx.agent.get_model_versions(&actors).await?;
    let baseline_of = |actor: &relayrl_framework::prelude::network::ActorInfo| {
        baseline
            .iter()
            .find(|(candidate, _)| candidate == actor)
            .map(|(_, version)| *version)
            .expect("actor should report a baseline version")
    };
    let (baseline_target, baseline_other) = (baseline_of(&target), baseline_of(&other));

    ctx.agent
        .update_models::<1, 1>(Some(std::slice::from_ref(&target)), swap_model)
        .await?;

    let bumped =
        wait_for_model_version_change(&ctx.agent, &target, baseline_target, DEFAULT_POLL_TIMEOUT)
            .await;
    assert!(
        bumped.is_some(),
        "targeted actor's model version should increase"
    );

    let other_version = ctx
        .agent
        .get_model_versions(std::slice::from_ref(&other))
        .await?
        .first()
        .map(|(_, version)| *version)
        .expect("other actor should report a version");
    assert_eq!(
        other_version, baseline_other,
        "untargeted actor's model version should be unaffected"
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn update_models_shared_mode_bumps_every_actor_on_the_device()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent_with_modes(
        ActorInferenceMode::Client(ModelMode::Shared),
        ActorDataMode::Disabled,
    )
    .await?
    else {
        return Ok(());
    };

    let (_swap_model_dir, swap_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let actors = ctx
        .agent
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

    let baseline = ctx
        .agent
        .get_model_versions(std::slice::from_ref(&actors[0]))
        .await?
        .first()
        .map(|(_, version)| *version)
        .expect("actor should report a baseline version");

    ctx.agent.update_models::<1, 1>(None, swap_model).await?;

    let bumped =
        wait_for_model_version_change(&ctx.agent, &actors[0], baseline, DEFAULT_POLL_TIMEOUT).await;
    assert!(
        bumped.is_some(),
        "shared device should observe a version bump"
    );

    let second_version = ctx
        .agent
        .get_model_versions(std::slice::from_ref(&actors[1]))
        .await?
        .first()
        .map(|(_, version)| *version)
        .expect("second actor should report a version");
    assert_eq!(
        Some(second_version),
        bumped,
        "both actors sharing the device should report the same, bumped model version"
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn update_models_rejects_a_rank_mismatched_model() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let (_swap_model_dir, swap_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
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

    // The identity model's metadata declares rank-1 shapes
    let result = ctx
        .agent
        .update_models::<2, 1>(Some(std::slice::from_ref(&actor)), swap_model)
        .await;
    assert!(matches!(result, Err(ClientError::CoordinatorError(_))));

    ctx.agent.shutdown().await?;
    Ok(())
}
