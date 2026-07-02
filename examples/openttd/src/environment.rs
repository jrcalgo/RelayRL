use crate::openttd::{OpenTtdBridge, OpenTtdBridgeError, OpenTtdConfig, OpenTtdSnapshot};
use crate::subsystems::{
    SubsystemActorSpec, action_mask_bytes, command_from_action, f32_slice_to_bytes,
    project_observation, reward_for_transition,
};
use crate::{ACTION_DIM, OBSERVATION_DIM};
use relayrl_env_trait::{
    Done, EnvDType, EnvNdArrayDType, Environment, EnvironmentError, EnvironmentHandle,
    EnvironmentKind, Mask, Observation, Reward, ScalarEnvReset, ScalarEnvironment, Truncated,
};
use std::any::Any;
use std::sync::{Arc, Mutex};

pub struct OpenTtdEnvironment {
    config: OpenTtdConfig,
    actor: SubsystemActorSpec,
    state: Arc<Mutex<EnvironmentState>>,
}

impl OpenTtdEnvironment {
    pub fn new(
        config: OpenTtdConfig,
        actor: SubsystemActorSpec,
    ) -> Result<Self, OpenTtdBridgeError> {
        let state = EnvironmentState::new(&config)?;
        Ok(Self {
            config,
            actor,
            state: Arc::new(Mutex::new(state)),
        })
    }

    pub fn actor(&self) -> SubsystemActorSpec {
        self.actor
    }

    fn observation_for(&self, snapshot: &OpenTtdSnapshot) -> Observation {
        f32_slice_to_bytes(&project_observation(snapshot, &self.actor))
    }

    fn current_snapshot(&self) -> Result<OpenTtdSnapshot, EnvironmentError> {
        let guard = self.lock_state()?;
        guard
            .last_snapshot
            .clone()
            .ok_or_else(|| EnvironmentError::ObservationBuildingError("missing snapshot".into()))
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, EnvironmentState>, EnvironmentError> {
        self.state
            .lock()
            .map_err(|e| EnvironmentError::EnvironmentError(e.to_string()))
    }
}

impl Clone for OpenTtdEnvironment {
    fn clone(&self) -> Self {
        let state = EnvironmentState::new(&self.config)
            .unwrap_or_else(|error| EnvironmentState::failed(error.to_string()));
        Self {
            config: self.config.clone(),
            actor: self.actor,
            state: Arc::new(Mutex::new(state)),
        }
    }
}

impl Environment for OpenTtdEnvironment {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        let mut guard = self.lock_state()?;
        guard.ensure_ready()?;
        let snapshot = guard.bridge.as_mut().expect("bridge checked").step()?;
        guard.last_snapshot = Some(snapshot);
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
        self.current_snapshot()
            .map(|snapshot| self.observation_for(&snapshot))
            .unwrap_or_else(|_| vec![0; OBSERVATION_DIM * std::mem::size_of::<f32>()])
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

impl ScalarEnvironment for OpenTtdEnvironment {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        let mut guard = self.lock_state()?;
        guard.ensure_ready()?;
        let snapshot = guard.bridge.as_mut().expect("bridge checked").reset()?;
        guard.last_snapshot = Some(snapshot.clone());
        guard.last_reward = 0.0;
        Ok(ScalarEnvReset {
            observation: self.observation_for(&snapshot),
            info: Some(vec![
                ("actor".into(), self.actor.actor_name.into()),
                ("subsystem".into(), self.actor.subsystem.as_str().into()),
                ("optimizes".into(), self.actor.optimizes.into()),
            ]),
        })
    }

    fn step_bytes(
        &self,
        action: &[u8],
    ) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        let action_index = decode_action_index(action, ACTION_DIM);
        let confidence = decode_action_confidence(action, action_index);
        let command = command_from_action(&self.actor, action_index, confidence);
        let mut guard = self.state.lock().ok()?;
        if guard.ensure_ready().is_err() {
            return None;
        }
        let before = guard.last_snapshot.clone().unwrap_or_default();
        let bridge = guard.bridge.as_mut().expect("bridge checked");
        bridge.apply_command(&command).ok()?;
        let after = bridge.step().ok()?;
        let reward = reward_for_transition(&before, &after, &self.actor);
        let done = after.done(&self.config);
        let observation = self.observation_for(&after);
        guard.last_snapshot = Some(after);
        guard.last_reward = reward;
        Some((observation, Some(action_mask_bytes()), reward, done, false))
    }
}

struct EnvironmentState {
    bridge: Option<OpenTtdBridge>,
    last_snapshot: Option<OpenTtdSnapshot>,
    last_reward: f32,
    startup_error: Option<String>,
}

impl EnvironmentState {
    fn new(config: &OpenTtdConfig) -> Result<Self, OpenTtdBridgeError> {
        let mut bridge = OpenTtdBridge::new(config.clone())?;
        let last_snapshot = Some(bridge.reset()?);
        Ok(Self {
            bridge: Some(bridge),
            last_snapshot,
            last_reward: 0.0,
            startup_error: None,
        })
    }

    fn failed(error: String) -> Self {
        Self {
            bridge: None,
            last_snapshot: Some(OpenTtdSnapshot::default()),
            last_reward: 0.0,
            startup_error: Some(error),
        }
    }

    fn ensure_ready(&self) -> Result<(), EnvironmentError> {
        match &self.startup_error {
            Some(error) => Err(EnvironmentError::EnvironmentError(error.clone())),
            None => Ok(()),
        }
    }
}

impl From<OpenTtdBridgeError> for EnvironmentError {
    fn from(value: OpenTtdBridgeError) -> Self {
        EnvironmentError::EnvironmentError(value.to_string())
    }
}

pub fn decode_action_index(action: &[u8], action_dim: usize) -> usize {
    decode_action_scores(action)
        .iter()
        .take(action_dim)
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

pub fn decode_action_confidence(action: &[u8], action_index: usize) -> f32 {
    decode_action_scores(action)
        .get(action_index)
        .copied()
        .unwrap_or(1.0)
        .abs()
        .clamp(0.05, 1.0)
}

fn decode_action_scores(action: &[u8]) -> Vec<f32> {
    action
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subsystems::TRANSPORT_ROUTE_PLANNER;

    #[test]
    fn scalar_environment_steps_mock_bridge() {
        let env = OpenTtdEnvironment::new(OpenTtdConfig::default(), TRANSPORT_ROUTE_PLANNER)
            .expect("mock OpenTTD bridge should initialize");
        let reset = env.reset().expect("reset should produce an observation");
        assert_eq!(reset.observation.len(), OBSERVATION_DIM * 4);

        let action = f32_slice_to_bytes(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (observation, mask, _reward, done, truncated) =
            env.step_bytes(&action).expect("step should succeed");
        assert_eq!(observation.len(), OBSERVATION_DIM * 4);
        assert_eq!(mask.expect("mask should exist").len(), ACTION_DIM * 4);
        assert!(!done);
        assert!(!truncated);
    }
}
