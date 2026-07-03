use crate::actors::{
    CacheActorRole, action_from_bytes, action_mask_bytes, f32_slice_to_bytes, observation_for,
};
use crate::cache::CacheWorld;
use crate::heuristics::{HeuristicController, PolicyKind};
use crate::host::BenchmarkConfig;
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
    state: Arc<Mutex<TrainingState>>,
}

impl CacheTrainingEnvironment {
    pub fn new(role: CacheActorRole, config: BenchmarkConfig) -> Self {
        Self {
            role,
            state: Arc::new(Mutex::new(TrainingState::new(&config))),
            config,
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
        Self::new(self.role, config)
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
        *state = TrainingState::new(&self.config);
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
        let mut decisions = state.background.decide(&world_snapshot, &pending);
        decisions.set(self.role, active_action);
        let mut active_roles = state.background.active_roles(&world_snapshot, &pending);
        if !active_roles.contains(&self.role) {
            active_roles.push(self.role);
        }
        let outcome = state
            .world
            .apply_request(&pending, decisions, &active_roles);
        state.steps += 1;
        state.pending = state.workload.next_request();
        state.advance_to_role_trigger(self.role);
        let reward = crate::actors::reward_for_role(self.role, &outcome);
        let done = state.steps >= self.config.requests;
        let observation =
            f32_slice_to_bytes(&observation_for(&state.world, &state.pending, self.role));
        Some((observation, Some(action_mask_bytes()), reward, done, false))
    }
}

struct TrainingState {
    world: CacheWorld,
    workload: WorkloadGenerator,
    background: HeuristicController,
    pending: CacheRequest,
    steps: u64,
}

impl TrainingState {
    fn new(config: &BenchmarkConfig) -> Self {
        let mut workload = WorkloadGenerator::new(config.workload, config.seed);
        let pending = workload.next_request();
        Self {
            world: CacheWorld::new(config.capacity_bytes),
            workload,
            background: HeuristicController::new(PolicyKind::Lru, config.seed),
            pending,
            steps: 0,
        }
    }

    fn advance_to_role_trigger(&mut self, role: CacheActorRole) {
        let mut guard = 0;
        while !crate::actors::should_trigger(role, &self.world, &self.pending) && guard < 128 {
            let pending = self.pending.clone();
            let world_snapshot = self.world.clone();
            let decisions = self.background.decide(&world_snapshot, &pending);
            let active_roles = self.background.active_roles(&world_snapshot, &pending);
            self.world.apply_request(&pending, decisions, &active_roles);
            self.pending = self.workload.next_request();
            guard += 1;
        }
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
