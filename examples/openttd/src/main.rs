use openttd_relayrl_example::{
    OPENTTD_LATEST_STABLE, OPENTTD_LATEST_STABLE_PUBLISHED, OPENTTD_LATEST_STABLE_URL,
    OPENTTD_NEWEST_PRERELEASE, OPENTTD_NEWEST_PRERELEASE_URL, OPENTTD_REPOSITORY,
    host::print_system_map,
    training::{describe_training_order, run_single_actor_ppo_smoke},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RelayRL OpenTTD system-of-systems example");
    println!("OpenTTD upstream: {OPENTTD_REPOSITORY}");
    println!(
        "Latest stable release cited here: {OPENTTD_LATEST_STABLE} ({OPENTTD_LATEST_STABLE_PUBLISHED})"
    );
    println!("Stable release URL: {OPENTTD_LATEST_STABLE_URL}");
    println!("Newest prerelease observed: {OPENTTD_NEWEST_PRERELEASE}");
    println!("Prerelease URL: {OPENTTD_NEWEST_PRERELEASE_URL}");
    println!();
    print_system_map();
    println!();
    println!("Sequential PPO train/freeze order:");
    for (index, actor_name) in describe_training_order().iter().enumerate() {
        println!("  {}. {}", index + 1, actor_name);
    }
    println!();
    println!(
        "This binary is a dry-run overview by default. Use the library modules to connect model assets, create a RelayRLAgent, and run host::OpenTtdHost or training::train_then_freeze_subsystems."
    );

    if std::env::args().any(|arg| arg == "--ppo-smoke") {
        println!();
        println!("Running PPO-enabled smoke training...");
        let result = run_single_actor_ppo_smoke().await?;
        println!();
        println!("PPO smoke result:");
        println!("  actor: {}", result.actor_name);
        println!("  subsystem: {}", result.subsystem);
        println!("  env_count: {}", result.env_count);
        println!("  loop_iters: {}", result.loop_iters);
        println!("  rollout_len: {}", result.rollout_len);
        println!("  model_saved: {}", result.model_saved);
        println!("  model_dir: {}", result.model_dir.display());
        println!(
            "  environment_probe: observation_bytes={} mask_bytes={} reward={:.6} done={} truncated={}",
            result.final_observation_bytes,
            result.final_mask_bytes,
            result.probe_reward,
            result.probe_done,
            result.probe_truncated
        );
    } else {
        println!();
        println!("Pass --ppo-smoke to run a bounded PPO-enabled smoke training pass.");
    }
    Ok(())
}
