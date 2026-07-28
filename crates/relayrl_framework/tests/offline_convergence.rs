//! Offline convergence integration tests distinguishing `run_env_eval` from `run_env_with_ppo`.
//!
//! - `run_env_eval`: rollout only (no training/update), discrete and continuous bandits.
//! - `run_env_with_ppo`: PPO collect + learn, asserting trained policy outputs.
//!
//! Both env-driven APIs use `tokio::task::block_in_place`, so every test here uses the
//! multi-threaded Tokio flavor.
#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use burn_ndarray::NdArrayDevice;
use burn_tensor::Float;
use common::{TestBackend, try_load_batched_test_model_module};
use relayrl_algorithms::TrainerArgs;
use relayrl_algorithms::algorithms::{ActivationKind, GenericMlp};
use relayrl_algorithms::prelude::ppo::algorithm::{
    ContinuousPPOPolicyHead, DiscretePPOPolicyHead, IPPOParams, PPOPolicyHead,
};
use relayrl_algorithms::prelude::ppo::trainer::{PPONetworkArgs, PPOTrainerSpec};
use relayrl_framework::prelude::network::{
    ActorDataMode, ActorInfo, AgentBuilder, RelayRLActors, RelayRLAgent, RelayRLBatchEnv,
};
use relayrl_framework::prelude::templates::environment::{
    Done, EnvDType, EnvNdArrayDType, Environment, EnvironmentError, EnvironmentHandle,
    EnvironmentKind, Mask, Observation, Reward, ScalarEnvReset, ScalarEnvironment, Truncated,
};
use relayrl_types::data::tensor::{
    DType, DeviceType, NdArrayDType, SupportedTensorBackend, TensorData,
};
use relayrl_types::model::ModelModule;
use std::any::Any;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const OBS_DIM: usize = 2;
const DISCRETE_ACT_DIM: usize = 2;
const CONTINUOUS_ACT_DIM: usize = 1;
const DISCRETE_OBS: [f32; 2] = [1.0, 0.0];
const TARGET_ACTION: f32 = 0.5;
const CONTINUOUS_OBS: [f32; 2] = [1.0, TARGET_ACTION];
const OPTIMAL_DISCRETE_ACTION: u8 = 1;

const EVAL_STEPS: usize = 16;
const ACTOR_MAX_TRAJ_LEN: usize = 1_000;

const PPO_LOOP_ITERS: usize = 128;
const PPO_MAX_TRAJ_LEN: usize = 8;
const PPO_BUFFER_SIZE: usize = 128;
const PPO_TRAJ_PER_EPOCH: u64 = 4;
const PPO_TRAIN_PI_ITERS: u64 = 12;
const PPO_TRAIN_VF_ITERS: u64 = 12;
const PPO_MINIBATCH: Option<usize> = Some(4);
const PPO_TARGET_KL: f32 = 0.05;

const DISCRETE_OPTIMAL_PROB_MIN: f32 = 0.55;
const CONTINUOUS_MEAN_ABS_ERROR_MAX: f32 = 0.75;

type TestPi = GenericMlp<TestBackend, Float, Float>;
type TestPpoSpec = PPOTrainerSpec<TestBackend, Float, Float, TestPi>;

#[derive(Clone, Copy, Debug, Default)]
struct BanditStats {
    episodes: usize,
    total_reward: f32,
    last_reward: f32,
    last_action: f32,
    optimal_action_count: usize,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk")))
        .collect()
}

fn f32_from_first_4_bytes(bytes: &[u8]) -> f32 {
    let mut buf = [0u8; 4];
    if bytes.len() >= 4 {
        buf.copy_from_slice(&bytes[..4]);
    } else if !bytes.is_empty() {
        buf[..bytes.len()].copy_from_slice(bytes);
    }
    f32::from_le_bytes(buf)
}

fn softmax2(logits: [f32; 2]) -> [f32; 2] {
    let max = logits[0].max(logits[1]);
    let e0 = (logits[0] - max).exp();
    let e1 = (logits[1] - max).exp();
    let sum = e0 + e1;
    [e0 / sum, e1 / sum]
}

fn constant_obs_tensor_data(values: &[f32]) -> TensorData {
    TensorData::new(
        vec![1, values.len()],
        DType::NdArray(NdArrayDType::F32),
        f32_bytes(values),
        SupportedTensorBackend::NdArray,
    )
}

fn infer_policy_f32(
    module: &ModelModule<TestBackend>,
    obs: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let output = module.flat_batch_inference(constant_obs_tensor_data(obs))?;
    Ok(f32s_from_bytes(&output.data))
}

fn print_eval_summary(
    kind: &str,
    episodes: usize,
    total_reward: f32,
    last_reward: f32,
    extra: &str,
) {
    let avg = if episodes == 0 {
        0.0
    } else {
        total_reward / episodes as f32
    };
    println!(
        "eval {kind}: episodes={episodes} total_reward={total_reward:.4} avg_reward={avg:.4} final_reward={last_reward:.4}{extra}"
    );
}

fn print_ppo_summary(
    kind: &str,
    episodes: usize,
    total_reward: f32,
    last_reward: f32,
    policy: &str,
) {
    let avg = if episodes == 0 {
        0.0
    } else {
        total_reward / episodes as f32
    };
    println!(
        "ppo {kind}: episodes={episodes} total_reward={total_reward:.4} avg_reward={avg:.4} final_reward={last_reward:.4} {policy}"
    );
}

fn ppo_params(discrete: bool) -> IPPOParams {
    IPPOParams {
        discrete,
        traj_per_epoch: PPO_TRAJ_PER_EPOCH,
        train_pi_iters: PPO_TRAIN_PI_ITERS,
        train_vf_iters: PPO_TRAIN_VF_ITERS,
        minibatch: PPO_MINIBATCH,
        sync_epoch_boundary: true,
        target_kl: PPO_TARGET_KL,
        ent_coef: 0.0,
        normalize_obs: false,
        normalize_returns: false,
        ..IPPOParams::default()
    }
}

fn trainer_args(act_dim: usize, env_dir: PathBuf, save_model_path: PathBuf) -> TrainerArgs {
    TrainerArgs {
        env_dir,
        save_model_path,
        obs_dim: OBS_DIM,
        obs_dtype: DType::NdArray(NdArrayDType::F32),
        act_dim,
        act_dtype: DType::NdArray(NdArrayDType::F32),
        buffer_size: PPO_BUFFER_SIZE,
        device: DeviceType::Cpu,
    }
}

fn small_mlp(input_dim: usize, output_dim: usize) -> TestPi {
    let device = NdArrayDevice::default();
    GenericMlp::new(
        input_dim,
        DType::NdArray(NdArrayDType::F32),
        &[8],
        output_dim,
        DType::NdArray(NdArrayDType::F32),
        // Avoid pulling `burn-nn` into the framework test crate; linear layers suffice here.
        ActivationKind::None,
        &device,
    )
}

fn build_discrete_ppo_spec(
    env_dir: PathBuf,
    save_model_path: PathBuf,
) -> Result<TestPpoSpec, Box<dyn std::error::Error>> {
    let networks = PPONetworkArgs {
        pi_head: PPOPolicyHead::Discrete(DiscretePPOPolicyHead::new(small_mlp(
            OBS_DIM,
            DISCRETE_ACT_DIM,
        ))?),
        vf_mlp: small_mlp(OBS_DIM, 1),
    };
    Ok(PPOTrainerSpec::ppo(
        trainer_args(DISCRETE_ACT_DIM, env_dir, save_model_path),
        Some(ppo_params(true)),
        networks,
    ))
}

fn build_continuous_ppo_spec(
    env_dir: PathBuf,
    save_model_path: PathBuf,
) -> Result<TestPpoSpec, Box<dyn std::error::Error>> {
    let networks = PPONetworkArgs {
        pi_head: PPOPolicyHead::Continuous(ContinuousPPOPolicyHead::new(small_mlp(
            OBS_DIM,
            CONTINUOUS_ACT_DIM * 2,
        ))?),
        vf_mlp: small_mlp(OBS_DIM, 1),
    };
    Ok(PPOTrainerSpec::ppo(
        trainer_args(CONTINUOUS_ACT_DIM, env_dir, save_model_path),
        Some(ppo_params(false)),
        networks,
    ))
}

#[derive(Clone)]
struct DiscreteBanditEnv {
    stats: Arc<Mutex<BanditStats>>,
}

impl DiscreteBanditEnv {
    fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(BanditStats::default())),
        }
    }

    fn stats(&self) -> BanditStats {
        *self.stats.lock().expect("discrete bandit stats lock")
    }
}

impl Environment for DiscreteBanditEnv {
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
        OBS_DIM
    }
    fn action_dim(&self) -> usize {
        DISCRETE_ACT_DIM
    }
    fn flat_observation_bytes(&self) -> Observation {
        f32_bytes(&DISCRETE_OBS)
    }
    fn flat_mask_bytes(&self) -> Mask {
        // ScalarVecEnv currently materializes missing masks as `Some(empty)`, which
        // panics discrete PPO sampling (`mask[env * act_dim + j]`). Provide an explicit
        // all-valid mask (one u8 flag per action) instead.
        Some(vec![1u8; DISCRETE_ACT_DIM])
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

impl ScalarEnvironment for DiscreteBanditEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        Ok(ScalarEnvReset {
            observation: self.flat_observation_bytes(),
            info: None,
        })
    }

    fn step_bytes(&self, action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        let action_idx = action.first().copied().unwrap_or(0);
        let reward = if action_idx == OPTIMAL_DISCRETE_ACTION {
            1.0
        } else {
            -1.0
        };
        {
            let mut stats = self.stats.lock().expect("discrete bandit stats lock");
            stats.episodes += 1;
            stats.total_reward += reward;
            stats.last_reward = reward;
            stats.last_action = action_idx as f32;
            if action_idx == OPTIMAL_DISCRETE_ACTION {
                stats.optimal_action_count += 1;
            }
        }
        Some((self.flat_observation_bytes(), None, reward, true, false))
    }
}

#[derive(Clone)]
struct ContinuousBanditEnv {
    stats: Arc<Mutex<BanditStats>>,
}

impl ContinuousBanditEnv {
    fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(BanditStats::default())),
        }
    }

    fn stats(&self) -> BanditStats {
        *self.stats.lock().expect("continuous bandit stats lock")
    }
}

impl Environment for ContinuousBanditEnv {
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
        OBS_DIM
    }
    fn action_dim(&self) -> usize {
        CONTINUOUS_ACT_DIM
    }
    fn flat_observation_bytes(&self) -> Observation {
        f32_bytes(&CONTINUOUS_OBS)
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

impl ScalarEnvironment for ContinuousBanditEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        Ok(ScalarEnvReset {
            observation: self.flat_observation_bytes(),
            info: None,
        })
    }

    fn step_bytes(&self, action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
        let action_val = f32_from_first_4_bytes(action);
        let error = action_val - TARGET_ACTION;
        let reward = -(error * error);
        {
            let mut stats = self.stats.lock().expect("continuous bandit stats lock");
            stats.episodes += 1;
            stats.total_reward += reward;
            stats.last_reward = reward;
            stats.last_action = action_val;
        }
        Some((self.flat_observation_bytes(), None, reward, true, false))
    }
}

async fn start_batched_offline_agent() -> Result<
    Option<(
        RelayRLAgent<TestBackend>,
        tempfile::TempDir,
        tempfile::TempDir,
    )>,
    Box<dyn std::error::Error>,
> {
    let Some((model_dir, default_model)) = try_load_batched_test_model_module() else {
        return Ok(None);
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
    Ok(Some((agent, model_dir, config_dir)))
}

async fn new_actor(
    agent: &mut RelayRLAgent<TestBackend>,
) -> Result<ActorInfo, Box<dyn std::error::Error>> {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discrete_run_env_eval_records_bandit_rewards_without_training()
-> Result<(), Box<dyn std::error::Error>> {
    let Some((mut agent, _model_dir, _config_dir)) = start_batched_offline_agent().await? else {
        return Ok(());
    };

    let actor = new_actor(&mut agent).await?;
    let env = DiscreteBanditEnv::new();
    agent.set_env(&actor, Box::new(env.clone()), 1).await?;
    agent.run_env_eval(&actor, EVAL_STEPS).await?;

    let stats = env.stats();
    let optimal_rate = if stats.episodes == 0 {
        0.0
    } else {
        stats.optimal_action_count as f32 / stats.episodes as f32
    };
    print_eval_summary(
        "discrete",
        stats.episodes,
        stats.total_reward,
        stats.last_reward,
        &format!(" optimal_rate={optimal_rate:.4}"),
    );

    assert_eq!(
        stats.episodes, EVAL_STEPS,
        "eval discrete: expected {EVAL_STEPS} episodes, got {}",
        stats.episodes
    );
    assert!(
        stats.last_reward.is_finite() && (stats.last_reward == -1.0 || stats.last_reward == 1.0),
        "eval discrete: final_reward must be ±1, got {}",
        stats.last_reward
    );
    assert!(
        stats.total_reward.is_finite(),
        "eval discrete: total_reward must be finite, got {}",
        stats.total_reward
    );

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn continuous_run_env_eval_records_bandit_rewards_without_training()
-> Result<(), Box<dyn std::error::Error>> {
    let Some((mut agent, _model_dir, _config_dir)) = start_batched_offline_agent().await? else {
        return Ok(());
    };

    let actor = new_actor(&mut agent).await?;
    let env = ContinuousBanditEnv::new();
    agent.set_env(&actor, Box::new(env.clone()), 1).await?;
    agent.run_env_eval(&actor, EVAL_STEPS).await?;

    let stats = env.stats();
    print_eval_summary(
        "continuous",
        stats.episodes,
        stats.total_reward,
        stats.last_reward,
        &format!(" final_action={:.4}", stats.last_action),
    );

    assert_eq!(
        stats.episodes, EVAL_STEPS,
        "eval continuous: expected {EVAL_STEPS} episodes, got {}",
        stats.episodes
    );
    assert!(
        stats.last_reward.is_finite(),
        "eval continuous: final_reward must be finite, got {}",
        stats.last_reward
    );
    assert!(
        stats.last_action.is_finite(),
        "eval continuous: final_action must be finite, got {}",
        stats.last_action
    );

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discrete_run_env_with_ppo_learns_bandit_preference()
-> Result<(), Box<dyn std::error::Error>> {
    let Some((mut agent, _model_dir, _config_dir)) = start_batched_offline_agent().await? else {
        return Ok(());
    };

    let actor = new_actor(&mut agent).await?;
    let env = DiscreteBanditEnv::new();
    agent.set_env(&actor, Box::new(env.clone()), 1).await?;

    let work = tempfile::tempdir()?;
    let spec =
        build_discrete_ppo_spec(work.path().to_path_buf(), work.path().join("discrete_ppo"))?;
    let trained = agent
        .run_env_with_ppo(&actor, PPO_LOOP_ITERS, PPO_MAX_TRAJ_LEN, spec)
        .await?;

    assert_eq!(
        trained.metadata.output_shape,
        vec![1, DISCRETE_ACT_DIM],
        "ppo discrete: expected output shape [1, {DISCRETE_ACT_DIM}], got {:?}",
        trained.metadata.output_shape
    );

    let logits = infer_policy_f32(&trained, &DISCRETE_OBS)?;
    assert_eq!(
        logits.len(),
        2,
        "ppo discrete: expected 2 logits, got {logits:?}"
    );
    let logits_arr = [logits[0], logits[1]];
    let probs = softmax2(logits_arr);
    let optimal_prob = probs[1];

    let stats = env.stats();
    let optimal_rate = if stats.episodes == 0 {
        0.0
    } else {
        stats.optimal_action_count as f32 / stats.episodes as f32
    };
    print_ppo_summary(
        "discrete",
        stats.episodes,
        stats.total_reward,
        stats.last_reward,
        &format!(
            "logits=[{:.4}, {:.4}] probs=[{:.4}, {:.4}] optimal_prob={optimal_prob:.4} optimal_rate={optimal_rate:.4}",
            logits_arr[0], logits_arr[1], probs[0], probs[1]
        ),
    );

    assert!(
        stats.episodes > 0,
        "ppo discrete: expected at least one episode"
    );
    assert!(
        optimal_prob > DISCRETE_OPTIMAL_PROB_MIN,
        "ppo discrete: optimal action prob {optimal_prob:.4} <= {DISCRETE_OPTIMAL_PROB_MIN}; \
         logits={logits_arr:?} probs={probs:?} episodes={} total_reward={} final_reward={} optimal_rate={optimal_rate:.4}",
        stats.episodes,
        stats.total_reward,
        stats.last_reward
    );
    assert!(
        stats.last_reward > 0.0 || stats.optimal_action_count > 0,
        "ppo discrete: expected at least one positive/optimal reward signal; \
         final_reward={} optimal_count={} episodes={}",
        stats.last_reward,
        stats.optimal_action_count,
        stats.episodes
    );

    agent.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn continuous_run_env_with_ppo_moves_mean_toward_target()
-> Result<(), Box<dyn std::error::Error>> {
    let Some((mut agent, _model_dir, _config_dir)) = start_batched_offline_agent().await? else {
        return Ok(());
    };

    let actor = new_actor(&mut agent).await?;
    let env = ContinuousBanditEnv::new();
    agent.set_env(&actor, Box::new(env.clone()), 1).await?;

    let work = tempfile::tempdir()?;
    let spec = build_continuous_ppo_spec(
        work.path().to_path_buf(),
        work.path().join("continuous_ppo"),
    )?;
    let trained = agent
        .run_env_with_ppo(&actor, PPO_LOOP_ITERS, PPO_MAX_TRAJ_LEN, spec)
        .await?;

    assert_eq!(
        trained.metadata.output_shape,
        vec![1, 2],
        "ppo continuous: expected output shape [1, 2] (mean, log_std), got {:?}",
        trained.metadata.output_shape
    );

    let policy_out = infer_policy_f32(&trained, &CONTINUOUS_OBS)?;
    assert_eq!(
        policy_out.len(),
        2,
        "ppo continuous: expected [mean, log_std], got {policy_out:?}"
    );
    let mean = policy_out[0];
    let log_std = policy_out[1];
    let abs_error = (mean - TARGET_ACTION).abs();

    let stats = env.stats();
    print_ppo_summary(
        "continuous",
        stats.episodes,
        stats.total_reward,
        stats.last_reward,
        &format!(
            "mean={mean:.4} log_std={log_std:.4} target={TARGET_ACTION:.4} abs_error={abs_error:.4} last_action={:.4}",
            stats.last_action
        ),
    );

    assert!(
        stats.episodes > 0,
        "ppo continuous: expected at least one episode"
    );
    assert!(
        mean.is_finite() && log_std.is_finite(),
        "ppo continuous: mean/log_std must be finite; mean={mean} log_std={log_std}"
    );
    assert!(
        abs_error < CONTINUOUS_MEAN_ABS_ERROR_MAX,
        "ppo continuous: abs(mean - target)={abs_error:.4} >= {CONTINUOUS_MEAN_ABS_ERROR_MAX}; \
         mean={mean:.4} log_std={log_std:.4} target={TARGET_ACTION} episodes={} \
         total_reward={} final_reward={} last_action={}",
        stats.episodes,
        stats.total_reward,
        stats.last_reward,
        stats.last_action
    );

    agent.shutdown().await?;
    Ok(())
}
