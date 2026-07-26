//! Offline step-driven inference integration tests: `request_action`/`request_actions` and
//! `flag_last_action`/`flag_last_actions` through the public `RelayRLAgent` API. No transport
//! feature is required or exercised.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use burn_ndarray::NdArrayDevice;
use burn_tensor::{Float, Tensor, TensorData};
use common::{TestBackend, start_offline_agent};
use relayrl_framework::prelude::network::{ActorDataMode, RelayRLActors, RelayRLStepDriven};
use relayrl_types::data::tensor::DeviceType;

fn zero_obs() -> Tensor<TestBackend, 1, Float> {
    Tensor::<TestBackend, 1, Float>::from_data(
        TensorData::new(vec![1.0_f32, 2.0_f32], [2]),
        &NdArrayDevice::default(),
    )
}

#[tokio::test]
async fn request_action_tags_the_requesting_actor_and_reward()
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

    let action = ctx
        .agent
        .request_action::<1, 1, Float, Float>(
            &actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            2.5,
        )
        .await?;

    assert_eq!(action.get_rew(), 2.5);
    assert_eq!(action.get_agent_id(), Some(&actor.id()));
    assert!(!action.get_done());

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn request_actions_broadcasts_one_observation_to_every_actor()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actors = ctx
        .agent
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

    let results = ctx
        .agent
        .request_actions::<1, 1, Float, Float>(
            &actors,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            1.0,
        )
        .await?;

    assert_eq!(results.len(), actors.len());
    let mut returned_ids: Vec<_> = results.iter().map(|(actor, _)| actor.id()).collect();
    let mut expected_ids: Vec<_> = actors.iter().map(|actor| actor.id()).collect();
    returned_ids.sort();
    expected_ids.sort();
    assert_eq!(returned_ids, expected_ids);
    for (_, action) in &results {
        assert_eq!(action.get_rew(), 1.0);
    }

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn flag_last_action_succeeds_after_a_request() -> Result<(), Box<dyn std::error::Error>> {
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
        .request_action::<1, 1, Float, Float>(
            &actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;

    ctx.agent.flag_last_action(&actor, Some(1.0)).await?;

    // The actor keeps accepting new requests after its episode boundary is flagged.
    let next_action = ctx
        .agent
        .request_action::<1, 1, Float, Float>(
            &actor,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;
    assert_eq!(next_action.get_agent_id(), Some(&actor.id()));

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn flag_last_actions_covers_every_named_actor() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
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

    ctx.agent
        .request_actions::<1, 1, Float, Float>(
            &actors,
            zero_obs(),
            None::<Tensor<TestBackend, 1, Float>>,
            0.0,
        )
        .await?;

    ctx.agent.flag_last_actions(&actors, Some(1.0)).await?;

    ctx.agent.shutdown().await?;
    Ok(())
}
