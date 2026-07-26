//! Offline convergence integration tests: deterministic attractors driven by the identity
//! ONNX policy through both the step-driven and environment-driven agent APIs.
//!
//! With the identity model, action equals observation. Linear contraction uses
//! `state -= 0.5 * action`; clipped and radial regimes use saturated / direction-normalized
//! updates that still reach the origin in a small fixed step budget.
//!
//! `run_env_eval` uses `tokio::task::block_in_place`, which panics on a current-thread
//! runtime, so env-driven tests use the multi-threaded flavor.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use burn_ndarray::NdArrayDevice;
use burn_tensor::{Float, Tensor, TensorData};
use common::{TestBackend, load_batched_test_model_module, start_offline_agent};
use relayrl_framework::prelude::network::{
    ActorDataMode, ActorInfo, AgentBuilder, RelayRLActors, RelayRLAgent, RelayRLBatchEnv,
    RelayRLStepDriven,
};
use relayrl_framework::prelude::templates::environment::{
    Done, EnvDType, EnvNdArrayDType, Environment, EnvironmentError, EnvironmentHandle,
    EnvironmentKind, Mask, Observation, Reward, ScalarEnvReset, ScalarEnvironment, Truncated,
};
use relayrl_types::data::tensor::DeviceType;
use std::any::Any;
use std::sync::{Arc, Mutex};

const INITIAL_STATE: [f32; 2] = [1.0, -0.5];
const STEP_SCALE: f32 = 0.5;
const CLIP_STEP: f32 = 0.25;
const RADIAL_STEP: f32 = 0.25;
const CONVERGENCE_EPS: f32 = 1.0e-3;
const MAX_STEPS: usize = 16;
const ACTOR_MAX_TRAJ_LEN: usize = 1_000;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_pair_from_bytes(bytes: &[u8]) -> [f32; 2] {
    assert!(
        bytes.len() >= 8,
        "expected two f32 action values (8 bytes), got {} bytes",
        bytes.len()
    );
    [
        f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    ]
}

fn norm2(values: [f32; 2]) -> f32 {
    values[0] * values[0] + values[1] * values[1]
}

fn obs_tensor(values: [f32; 2]) -> Tensor<TestBackend, 1, Float> {
    Tensor::<TestBackend, 1, Float>::from_data(
        TensorData::new(values.to_vec(), [2]),
        &NdArrayDevice::default(),
    )
}

#[derive(Clone, Copy, Debug)]
enum ConvergenceRegime {
    Linear,
    ClippedStep,
    RadialStep,
}

#[derive(Clone)]
struct ConvergenceEnv {
    state: Arc<Mutex<[f32; 2]>>,
    initial: [f32; 2],
    regime: ConvergenceRegime,
}

impl ConvergenceEnv {
    fn new(initial: [f32; 2], regime: ConvergenceRegime) -> Self {
        Self {
            state: Arc::new(Mutex::new(initial)),
            initial,
            regime,
        }
    }

    fn current_state(&self) -> [f32; 2] {
        *self.state.lock().expect("state lock")
    }

    fn current_norm2(&self) -> f32 {
        norm2(self.current_state())
    }

    fn apply_action(&self, action: [f32; 2]) -> f32 {
        let mut state = self.state.lock().expect("state lock");
        match self.regime {
            ConvergenceRegime::Linear => {
                state[0] -= STEP_SCALE * action[0];
                state[1] -= STEP_SCALE * action[1];
            }
            ConvergenceRegime::ClippedStep => {
                state[0] -= action[0].clamp(-CLIP_STEP, CLIP_STEP);
                state[1] -= action[1].clamp(-CLIP_STEP, CLIP_STEP);
            }
            ConvergenceRegime::RadialStep => {
                let n = norm2(action).sqrt();
                if n <= RADIAL_STEP {
                    state[0] = 0.0;
                    state[1] = 0.0;
                } else if n > 0.0 {
                    let scale = RADIAL_STEP / n;
                    state[0] -= action[0] * scale;
                    state[1] -= action[1] * scale;
                }
            }
        }
        norm2(*state)
    }
}

impl Environment for ConvergenceEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }
    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(self.flat_observation_bytes()))
    }
    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(()))
    }
    fn observation_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn action_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn observation_dim(&self) -> usize {
        2
    }
    fn action_dim(&self) -> usize {
        2
    }
    fn flat_observation_bytes(&self) -> Observation {
        f32_bytes(&self.current_state())
    }
    fn flat_mask_bytes(&self) -> Mask {
        None
    }
    fn action_is_discrete(&self) -> bool {
        false
    }
    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Scalar
    }
    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(Box::new(*self))
    }
}

impl ScalarEnvironment for ConvergenceEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        *self.state.lock().expect("state lock") = self.initial;
        Ok(ScalarEnvReset {
            observation: self.flat_observation_bytes(),
            info: None,
        })
    }
    fn step_bytes(&self, action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        let n = self.apply_action(f32_pair_from_bytes(action));
        Some((
            self.flat_observation_bytes(),
            None,
            -n,
            n <= CONVERGENCE_EPS,
            false,
        ))
    }
}

async fn new_actor(agent: &mut RelayRLAgent<TestBackend>) -> Result<ActorInfo, Box<dyn std::error::Error>> {
    Ok(agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            ACTOR_MAX_TRAJ_LEN,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?)
}

async fn run_step_driven_case(regime: ConvergenceRegime) -> Result<(), Box<dyn std::error::Error>> {
    let Some(mut ctx) = start_offline_agent(ActorDataMode::Disabled).await? else {
        return Ok(());
    };

    let actor = new_actor(&mut ctx.agent).await?;
    let env = ConvergenceEnv::new(INITIAL_STATE, regime);
    env.reset()?;
    let initial_norm = env.current_norm2();

    let mut reward = 0.0_f32;
    let mut steps = 0usize;
    let mut done = false;
    let mut last_reward = reward;

    for _ in 0..MAX_STEPS {
        let action = ctx
            .agent
            .request_action::<1, 1, Float, Float>(
                &actor,
                obs_tensor(env.current_state()),
                None::<Tensor<TestBackend, 1, Float>>,
                reward,
            )
            .await?;

        let act_bytes = &action
            .get_act()
            .expect("identity model should produce action tensor data")
            .data;
        let (_obs, _mask, step_reward, step_done, _trunc) = env
            .step_bytes(&f32_bytes(&f32_pair_from_bytes(act_bytes)))
            .expect("convergence env should always step");

        reward = step_reward;
        last_reward = step_reward;
        steps += 1;
        if step_done {
            done = true;
            break;
        }
    }

    if done {
        ctx.agent
            .flag_last_action(&actor, Some(last_reward))
            .await?;
    }

    let final_norm = env.current_norm2();
    println!(
        "step-driven {regime:?}: final_reward={last_reward}, final_norm={final_norm}, steps={steps}"
    );
    assert!(
        final_norm < initial_norm && final_norm <= CONVERGENCE_EPS,
        "step-driven {regime:?} should converge: initial_norm={initial_norm}, final_norm={final_norm}, steps={steps}"
    );

    ctx.agent.shutdown().await?;
    Ok(())
}

async fn run_env_driven_case(regime: ConvergenceRegime) -> Result<(), Box<dyn std::error::Error>> {
    // Env-driven inference feeds `[n_envs, obs_dim]`, so the default model must accept a
    // dynamic batch axis (not the rank-1 identity used by step-driven tests).
    let (_model_dir, default_model) = match load_batched_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let config_dir = tempfile::tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_data_mode(ActorDataMode::Disabled)
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;
    agent.start(params).await?;

    let actor = new_actor(&mut agent).await?;
    let env = ConvergenceEnv::new(INITIAL_STATE, regime);
    let initial_norm = env.current_norm2();

    agent.set_env(&actor, Box::new(env.clone()), 1).await?;
    agent.run_env_eval(&actor, MAX_STEPS).await?;

    let final_norm = env.current_norm2();
    let final_reward = -final_norm;
    println!(
        "env-driven {regime:?}: final_reward={final_reward}, final_norm={final_norm}, steps={MAX_STEPS}"
    );
    assert!(
        final_norm < initial_norm && final_norm <= CONVERGENCE_EPS,
        "env-driven {regime:?} should converge: initial_norm={initial_norm}, final_norm={final_norm}, steps={MAX_STEPS}"
    );
    assert_eq!(agent.get_env_count(&actor).await?, 1);

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn step_driven_identity_policy_converges_linear_system()
-> Result<(), Box<dyn std::error::Error>> {
    run_step_driven_case(ConvergenceRegime::Linear).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn env_driven_identity_policy_converges_linear_system()
-> Result<(), Box<dyn std::error::Error>> {
    run_env_driven_case(ConvergenceRegime::Linear).await
}

#[tokio::test]
async fn step_driven_identity_policy_converges_nonlinear_systems()
-> Result<(), Box<dyn std::error::Error>> {
    for regime in [ConvergenceRegime::ClippedStep, ConvergenceRegime::RadialStep] {
        run_step_driven_case(regime).await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn env_driven_identity_policy_converges_nonlinear_systems()
-> Result<(), Box<dyn std::error::Error>> {
    for regime in [ConvergenceRegime::ClippedStep, ConvergenceRegime::RadialStep] {
        run_env_driven_case(regime).await?;
    }
    Ok(())
}
