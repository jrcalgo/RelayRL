//! Offline actor lifecycle integration tests: batch creation, lookup by rank/tag, renaming, and
//! removal through the public `RelayRLAgent` API. No transport feature is required or exercised.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use common::start_offline_agent;
use relayrl_framework::prelude::network::{ActorDataMode, RelayRLActors};
use relayrl_types::data::tensor::DeviceType;

#[tokio::test]
async fn new_actors_batch_creates_requested_count() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actors = ctx
        .agent
        .new_actors::<1, 1>(
            4,
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    assert_eq!(actors.len(), 4);

    let all_actors = ctx.agent.get_all_actors().await?;
    assert_eq!(all_actors.len(), 4);
    for actor in &actors {
        assert!(all_actors.contains(actor));
    }

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn get_actors_by_rank_filters_on_observation_and_action_rank()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let rank_1_1 = ctx
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
    let rank_2_1 = ctx
        .agent
        .new_actor::<2, 1>(
            DeviceType::Cpu,
            1_000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    let matched_1_1 = ctx.agent.get_actors_by_rank::<1, 1>().await?;
    assert_eq!(matched_1_1.len(), 2);
    for actor in &rank_1_1 {
        assert!(matched_1_1.contains(actor));
    }
    assert!(!matched_1_1.contains(&rank_2_1));

    let matched_2_1 = ctx.agent.get_actors_by_rank::<2, 1>().await?;
    assert_eq!(matched_2_1, vec![rank_2_1]);

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn get_actors_by_tag_filters_by_nametag_and_none_matches_untagged()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let tagged = ctx
        .agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1_000,
            Some("scout"),
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;
    let untagged = ctx
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

    let scouts = ctx.agent.get_actors_by_tag(Some("scout")).await?;
    assert_eq!(scouts, vec![tagged.clone()]);

    let untagged_matches = ctx.agent.get_actors_by_tag(None).await?;
    assert_eq!(untagged_matches, vec![untagged]);

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn set_actor_id_updates_lookup_and_old_id_no_longer_resolves()
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
    let old_id = actor.id();
    let new_id = active_uuid_registry::registry_uuid::Uuid::new_v4();

    ctx.agent.set_actor_id(&actor, new_id).await?;

    // `actor` observes the rename in place.
    assert_eq!(actor.id(), new_id);

    let by_new_id = ctx.agent.get_actor(new_id).await?;
    assert_eq!(by_new_id.id(), new_id);

    let old_lookup = ctx.agent.get_actor(old_id).await;
    assert!(old_lookup.is_err());

    ctx.agent.remove_actor(&actor).await?;
    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn set_actor_nametag_updates_tag_based_lookup() -> Result<(), Box<dyn std::error::Error>> {
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

    ctx.agent.set_actor_nametag(&actor, Some("renamed")).await?;
    assert_eq!(
        actor.nametag().map(|tag| tag.tag),
        Some("renamed".to_string())
    );

    let matches = ctx.agent.get_actors_by_tag(Some("renamed")).await?;
    assert_eq!(matches, vec![actor.clone()]);

    ctx.agent.set_actor_nametag(&actor, None).await?;
    assert!(actor.nametag().is_none());

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn remove_actors_shrinks_all_actors_list() -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actors: Vec<_> = ctx
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

    ctx.agent
        .remove_actors(&[actors[0].clone(), actors[1].clone()])
        .await?;

    let remaining = ctx.agent.get_all_actors().await?;
    assert_eq!(remaining, vec![actors[2].clone()]);

    ctx.agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn remove_single_actor_is_equivalent_to_remove_actors_of_one()
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

    ctx.agent.remove_actor(&actor).await?;

    let remaining = ctx.agent.get_all_actors().await?;
    assert!(remaining.is_empty());

    // The agent handle stays usable: a fresh actor can still be created after a removal.
    let _fresh = ctx
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

    ctx.agent.shutdown().await?;
    Ok(())
}
