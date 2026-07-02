use openttd_relayrl_example::{
    OPENTTD_LATEST_STABLE, OPENTTD_LATEST_STABLE_PUBLISHED, OPENTTD_LATEST_STABLE_URL,
    OPENTTD_NEWEST_PRERELEASE, OPENTTD_NEWEST_PRERELEASE_URL, OPENTTD_REPOSITORY,
    host::print_system_map, training::describe_training_order,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}
