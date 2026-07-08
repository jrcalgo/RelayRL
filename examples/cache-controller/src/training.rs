use crate::ACTION_DIM;
use crate::actors::{CacheActorRole, f32_slice_to_bytes};
use crate::environment::CacheTrainingEnvironment;
use crate::host::BenchmarkConfig;
use relayrl_env_trait::{Environment, ScalarEnvironment};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSmokeResult {
    pub role: CacheActorRole,
    pub steps: u64,
    pub total_reward: f32,
    pub observation_bytes: usize,
    pub mask_bytes: usize,
    pub done: bool,
}

pub fn run_training_smoke(
    role: CacheActorRole,
    mut config: BenchmarkConfig,
) -> TrainingSmokeResult {
    config.requests = config.requests.min(1_000).max(16);
    let env = CacheTrainingEnvironment::new(role, config);
    let reset = env.reset().expect("cache training env reset should work");
    println!(
        "[training smoke] role={} reset_observation_bytes={}",
        role,
        reset.observation.len()
    );

    let mut total_reward = 0.0;
    let mut observation_bytes = reset.observation.len();
    let mut mask_bytes = env.flat_mask_bytes().map_or(0, |mask| mask.len());
    let mut done = false;
    let action = action_for_role(role);

    for step in 0..64 {
        let (obs, mask, reward, step_done, truncated) = env
            .step_bytes(&action)
            .expect("cache training env step should work");
        total_reward += reward;
        observation_bytes = obs.len();
        mask_bytes = mask.map_or(0, |mask| mask.len());
        done = step_done;
        println!(
            "[training smoke] step={:>3} role={} reward={:>8.4} done={} truncated={}",
            step + 1,
            role,
            reward,
            step_done,
            truncated
        );
        if step_done || truncated {
            break;
        }
    }

    TrainingSmokeResult {
        role,
        steps: 64,
        total_reward,
        observation_bytes,
        mask_bytes,
        done,
    }
}

fn action_for_role(role: CacheActorRole) -> Vec<u8> {
    let mut values = [0.0_f32; ACTION_DIM];
    let action = match role {
        CacheActorRole::Admission => 1,
        CacheActorRole::Eviction => 1,
        CacheActorRole::Ttl => 3,
        CacheActorRole::Resize => 2,
        CacheActorRole::Prefetch => 0,
    };
    values[action] = 1.0;
    f32_slice_to_bytes(&values)
}
