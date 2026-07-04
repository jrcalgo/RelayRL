use crate::actors::{
    CacheActorRole, action_from_bytes, action_mask_bytes, f32_slice_to_bytes, observation_for,
};
use crate::cache::CacheWorld;
use crate::heuristics::PolicyKind;
use crate::host::BenchmarkConfig;
use crate::policies::MixedPolicySet;
use crate::workload::{CacheRequest, WorkloadGenerator};
use crate::{ACTION_DIM, OBSERVATION_DIM};
use relayrl_env_trait::{
    Done, EnvDType, EnvNdArrayDType, Environment, EnvironmentError, EnvironmentHandle,
    EnvironmentKind, Mask, Observation, Reward, ScalarEnvReset, ScalarEnvironment, Truncated,
};
use std::any::Any;
use std::sync::{Arc, Mutex};

pub struct CacheTrainingEnvironment {
    role: CacheActorRole,
    config: BenchmarkConfig,
    background: MixedPolicySet,
    state: Arc<Mutex<TrainingState>>,
}

impl CacheTrainingEnvironment {
    pub fn new(role: CacheActorRole, config: BenchmarkConfig) -> Self {
        let background = MixedPolicySet::new(PolicyKind::Lru, config.seed);
        Self::with_background(role, config, background)
    }

    pub fn with_background(
        role: CacheActorRole,
        config: BenchmarkConfig,
        background: MixedPolicySet,
    ) -> Self {
        Self {
            role,
            state: Arc::new(Mutex::new(TrainingState::new(
                &config,
                background.clone(),
                role,
            ))),
            config,
            background,
        }
    }

    pub fn role(&self) -> CacheActorRole {
        self.role
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, TrainingState>, EnvironmentError> {
        self.state
            .lock()
            .map_err(|error| EnvironmentError::EnvironmentError(error.to_string()))
    }
}

impl Clone for CacheTrainingEnvironment {
    fn clone(&self) -> Self {
        let mut config = self.config.clone();
        config.seed = config.seed.wrapping_add(role_seed_offset(self.role));
        let mut background = self.background.clone();
        background.reseed(config.seed);
        Self::with_background(self.role, config, background)
    }
}

impl Environment for CacheTrainingEnvironment {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        let mut state = self.lock_state()?;
        state.advance_to_role_trigger(self.role);
        Ok(())
    }

    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(self.flat_observation_bytes()))
    }

    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(self.flat_mask_bytes()))
    }

    fn observation_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }

    fn action_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }

    fn observation_dim(&self) -> usize {
        OBSERVATION_DIM
    }

    fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    fn flat_observation_bytes(&self) -> Observation {
        self.state
            .lock()
            .map(|state| {
                f32_slice_to_bytes(&observation_for(&state.world, &state.pending, self.role))
            })
            .unwrap_or_else(|_| vec![0; OBSERVATION_DIM * 4])
    }

    fn flat_mask_bytes(&self) -> Mask {
        Some(action_mask_bytes())
    }

    fn action_is_discrete(&self) -> bool {
        true
    }

    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Scalar
    }

    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(Box::new(*self))
    }
}

impl ScalarEnvironment for CacheTrainingEnvironment {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        let mut state = self.lock_state()?;
        *state = TrainingState::new(&self.config, self.background.clone(), self.role);
        state.advance_to_role_trigger(self.role);
        Ok(ScalarEnvReset {
            observation: f32_slice_to_bytes(&observation_for(
                &state.world,
                &state.pending,
                self.role,
            )),
            info: Some(vec![
                ("role".to_string(), self.role.as_str().to_string()),
                (
                    "policy".to_string(),
                    self.config.policy.as_str().to_string(),
                ),
            ]),
        })
    }

    fn step_bytes(&self, action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        let active_action = action_from_bytes(action);
        let mut state = self.state.lock().ok()?;
        state.advance_to_role_trigger(self.role);
        let pending = state.pending.clone();
        let world_snapshot = state.world.clone();
        let mut decisions = state.background.decide_all(&world_snapshot, &pending);
        decisions.set(self.role, active_action);
        let mut active_roles = state.background.active_roles(&world_snapshot, &pending);
        if !active_roles.contains(&self.role) {
            active_roles.push(self.role);
        }
        let outcome = state
            .world
            .apply_request(&pending, decisions, &active_roles);
        state.steps += 1;
        state.role_steps += 1;
        state.pending = state.workload.next_request();
        state.advance_to_role_trigger(self.role);
        let reward = crate::actors::reward_for_role(self.role, &outcome);
        let done = state.steps >= self.config.requests
            || (self.role == CacheActorRole::Eviction && state.role_steps >= 8);
        if done && self.role == CacheActorRole::Eviction {
            state.role_steps = 0;
        }
        let observation =
            f32_slice_to_bytes(&observation_for(&state.world, &state.pending, self.role));
        Some((observation, Some(action_mask_bytes()), reward, done, false))
    }
}

struct TrainingState {
    world: CacheWorld,
    workload: WorkloadGenerator,
    background: MixedPolicySet,
    pending: CacheRequest,
    steps: u64,
    role_steps: usize,
}

impl TrainingState {
    fn new(config: &BenchmarkConfig, background: MixedPolicySet, role: CacheActorRole) -> Self {
        let mut workload = WorkloadGenerator::new(config.workload, config.seed);
        let pending = workload.next_request();
        let mut state = Self {
            world: CacheWorld::new(config.capacity_bytes),
            workload,
            background,
            pending,
            steps: 0,
            role_steps: 0,
        };
        if role == CacheActorRole::Eviction {
            state.prefill_eviction_curriculum();
        }
        state
    }

    fn advance_to_role_trigger(&mut self, role: CacheActorRole) {
        let mut guard = 0;
        while !crate::actors::should_trigger(role, &self.world, &self.pending) && guard < 128 {
            if role == CacheActorRole::Eviction && self.world.used_bytes() > 0 {
                self.world.force_capacity_pressure();
                break;
            } else if role == CacheActorRole::Eviction {
                self.prefill_eviction_curriculum();
                break;
            }
            let pending = self.pending.clone();
            let world_snapshot = self.world.clone();
            let decisions = self.background.decide_all(&world_snapshot, &pending);
            let active_roles = self.background.active_roles(&world_snapshot, &pending);
            self.world.apply_request(&pending, decisions, &active_roles);
            self.pending = self.workload.next_request();
            guard += 1;
        }
    }

    fn prefill_eviction_curriculum(&mut self) {
        let target = self.world.capacity_bytes().saturating_mul(13) / 10;
        let mut inserted = 0_u64;
        while self.world.used_bytes() <= target && inserted < 512 {
            let request = self.workload.next_request();
            let access_count = match request.popularity_class {
                0 => 16 + inserted % 16,
                1 => 4 + inserted % 8,
                _ => 1 + inserted % 3,
            };
            let ttl = match inserted % 4 {
                0 => 16,
                1 => 64,
                2 => 256,
                _ => 1024,
            };
            let priority = match request.popularity_class {
                0 => 2.0,
                1 => 1.25,
                _ => 0.75,
            };
            self.world
                .force_insert_for_training(&request, access_count, ttl, priority);
            inserted += 1;
        }
        self.world.force_capacity_pressure();
        self.pending = self.workload.next_request();
    }
}

fn role_seed_offset(role: CacheActorRole) -> u64 {
    match role {
        CacheActorRole::Admission => 11,
        CacheActorRole::Eviction => 23,
        CacheActorRole::Ttl => 37,
        CacheActorRole::Resize => 41,
        CacheActorRole::Prefetch => 53,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::BenchmarkConfig;

    #[test]
    fn scalar_env_reset_and_step_work() {
        let env =
            CacheTrainingEnvironment::new(CacheActorRole::Admission, BenchmarkConfig::default());
        let reset = env.reset().expect("reset should work");
        assert_eq!(reset.observation.len(), OBSERVATION_DIM * 4);
        let action = f32_slice_to_bytes(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let (obs, mask, _reward, done, truncated) =
            env.step_bytes(&action).expect("step should work");
        assert_eq!(obs.len(), OBSERVATION_DIM * 4);
        assert_eq!(mask.expect("mask").len(), ACTION_DIM * 4);
        assert!(!done);
        assert!(!truncated);
    }

    #[test]
    fn cloned_env_has_independent_world() {
        let env =
            CacheTrainingEnvironment::new(CacheActorRole::Admission, BenchmarkConfig::default());
        let clone = env.clone();
        let action = f32_slice_to_bytes(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let left = env.step_bytes(&action).expect("left step").0;
        let right = clone.step_bytes(&action).expect("right step").0;
        assert_eq!(left.len(), right.len());
    }
}
