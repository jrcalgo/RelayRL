use crate::environment::OpenTtdEnvironment;
use crate::openttd::{GameSubsystem, OpenTtdConfig};
use crate::subsystems::{ACTOR_SPECS, SubsystemActorSpec, TRANSPORT_ROUTE_PLANNER};
use crate::{ACTION_DIM, OBSERVATION_DIM};
use burn_ndarray::NdArray;
use burn_tensor::Float;
use relayrl::algorithms::PPO::kernel::{DiscretePPOPolicyHead, PPOPolicyHead};
use relayrl::algorithms::PPO::{IPPOParams, PPONetworkArgs, PPOTrainerSpec};
use relayrl::algorithms::{ActivationKind, GenericMlp};
use relayrl::network::{AgentBuilder, RelayRLActorEnv, RelayRLAgent, RelayRLAgentActors};
use relayrl::types::model::ModelModule;
use relayrl::types::tensor::relayrl::{BackendMatcher, DType, DeviceType};
use relayrl_env_trait::ScalarEnvironment;
use relayrl_types::data::tensor::NdArrayDType;
use relayrl_types::model::{ModelFileType, ModelMetadata};
use std::error::Error;
use std::path::{Path, PathBuf};

pub type OpenTtdPpoSpec = PPOTrainerSpec<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>;
type ActorUuid = uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TrainingPhase {
    pub actor_id: ActorUuid,
    pub actor: SubsystemActorSpec,
    pub env_count: u32,
    pub loop_iters: usize,
    pub max_traj_length: usize,
}

#[derive(Debug, Clone)]
pub struct FrozenSubsystemModel {
    pub actor_id: ActorUuid,
    pub actor_name: &'static str,
    pub subsystem: GameSubsystem,
    pub model_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PpoSmokeRunResult {
    pub actor_name: &'static str,
    pub subsystem: GameSubsystem,
    pub env_count: u32,
    pub loop_iters: usize,
    pub rollout_len: usize,
    pub model_dir: PathBuf,
    pub model_saved: bool,
    pub final_observation_bytes: usize,
    pub final_mask_bytes: usize,
    pub probe_reward: f32,
    pub probe_done: bool,
    pub probe_truncated: bool,
}

pub fn build_ppo_spec(
    env_dir: PathBuf,
    save_model_path: PathBuf,
    buffer_size: usize,
) -> Result<OpenTtdPpoSpec, Box<dyn Error>> {
    Ok(PPOTrainerSpec::<
        NdArray,
        Float,
        Float,
        GenericMlp<NdArray, Float, Float>,
    >::default(
        env_dir,
        save_model_path,
        OBSERVATION_DIM,
        DType::NdArray(NdArrayDType::F32),
        ACTION_DIM,
        DType::NdArray(NdArrayDType::F32),
        buffer_size,
        DeviceType::Cpu,
    )?)
}

pub fn build_smoke_ppo_spec(
    env_dir: PathBuf,
    save_model_path: PathBuf,
    buffer_size: usize,
    rollout_len: usize,
) -> Result<OpenTtdPpoSpec, Box<dyn Error>> {
    let mut spec = build_ppo_spec(env_dir, save_model_path, buffer_size)?;
    let obs_dtype = DType::NdArray(NdArrayDType::F32);
    let act_dtype = DType::NdArray(NdArrayDType::F32);
    let device = <NdArray as BackendMatcher>::get_device(&DeviceType::Cpu)
        .map_err(|error| format!("failed to resolve ndarray CPU device: {error}"))?;
    let networks = PPONetworkArgs {
        pi_head: PPOPolicyHead::Discrete(DiscretePPOPolicyHead::new(GenericMlp::new(
            OBSERVATION_DIM,
            obs_dtype.clone(),
            &[32],
            ACTION_DIM,
            act_dtype,
            ActivationKind::None,
            &device,
        ))?),
        vf_mlp: GenericMlp::new(
            OBSERVATION_DIM,
            obs_dtype,
            &[32],
            1,
            DType::NdArray(NdArrayDType::F32),
            ActivationKind::None,
            &device,
        ),
    };
    let params = IPPOParams {
        traj_per_epoch: 1,
        train_pi_iters: 1,
        train_vf_iters: 1,
        rollout_len: Some(rollout_len),
        max_episode_steps: Some(rollout_len),
        normalize_obs: false,
        normalize_returns: false,
        ..IPPOParams::default()
    };

    match &mut spec {
        PPOTrainerSpec::PPO {
            hyperparams,
            networks: spec_networks,
            ..
        } => {
            *hyperparams = Some(params);
            *spec_networks = networks;
        }
        PPOTrainerSpec::IPPO {
            hyperparams,
            networks: spec_networks,
            ..
        } => {
            *hyperparams = Some(params);
            *spec_networks = networks;
        }
        PPOTrainerSpec::MAPPO { .. } => {}
    }

    Ok(spec)
}

pub async fn run_single_actor_ppo_smoke() -> Result<PpoSmokeRunResult, Box<dyn Error>> {
    let actor = TRANSPORT_ROUTE_PLANNER;
    let env_count = 1;
    let loop_iters = 8;
    let rollout_len = 4;
    let max_traj_length = 16;
    let output = tempfile::tempdir()?;
    let output_dir = output.path().to_path_buf();
    let env_dir = output_dir.join("rollouts");
    let model_dir = output_dir.join("frozen-model");
    std::fs::create_dir_all(&env_dir)?;
    std::fs::create_dir_all(&model_dir)?;

    println!(
        "[PPO smoke] actor={} subsystem={} objective={}",
        actor.actor_name, actor.subsystem, actor.optimizes
    );
    println!(
        "[PPO smoke] env_count={env_count} loop_iters={loop_iters} rollout_len={rollout_len} max_traj_length={max_traj_length}"
    );

    let bootstrap_model = bootstrap_policy_model()?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .router_scale(1)
        .default_model(bootstrap_model.clone())
        .build()
        .await?;
    agent.start(params).await?;

    let result = async {
        let actor_id = agent
            .new_actor::<2, 2>(DeviceType::Cpu, max_traj_length, Some(bootstrap_model))
            .await?;
        println!("[PPO smoke] created RelayRL actor {actor_id}");

        let config = OpenTtdConfig {
            max_ticks: rollout_len as u64,
            ..OpenTtdConfig::default()
        };
        let env = OpenTtdEnvironment::new(config.clone(), actor)?;
        agent.set_env(actor_id, Box::new(env), env_count).await?;
        println!(
            "[PPO smoke] bound OpenTtdEnvironment clone count={env_count}; scalar clones are independent bridge instances"
        );

        let ppo_spec = build_smoke_ppo_spec(
            env_dir.clone(),
            model_dir.clone(),
            max_traj_length,
            rollout_len,
        )?;
        println!(
            "[PPO smoke] starting PPO rollout/training with short hyperparameters; model output directory={}",
            model_dir.display()
        );
        let trained_model: ModelModule<NdArray> = agent
            .run_env_with_ppo::<Float, Float, GenericMlp<NdArray, Float, Float>>(
                actor_id,
                loop_iters,
                max_traj_length,
                ppo_spec,
            )
            .await?;
        println!("[PPO smoke] PPO run returned a trained policy module");

        trained_model.save(model_dir.clone())?;
        println!("[PPO smoke] saved trained policy module");

        let probe_env = OpenTtdEnvironment::new(config, actor)?;
        let action = crate::subsystems::f32_slice_to_bytes(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let (observation, mask, reward, done, truncated) = probe_env
            .step_bytes(&action)
            .ok_or("probe environment step failed")?;
        let mask_len = mask.as_ref().map_or(0, Vec::len);
        println!(
            "[PPO smoke] environment probe observation_bytes={} mask_bytes={} reward={reward:.6} done={done} truncated={truncated}",
            observation.len(),
            mask_len
        );

        Ok::<PpoSmokeRunResult, Box<dyn Error>>(PpoSmokeRunResult {
            actor_name: actor.actor_name,
            subsystem: actor.subsystem,
            env_count,
            loop_iters,
            rollout_len,
            model_dir,
            model_saved: true,
            final_observation_bytes: observation.len(),
            final_mask_bytes: mask_len,
            probe_reward: reward,
            probe_done: done,
            probe_truncated: truncated,
        })
    }
    .await;

    let shutdown = agent.shutdown().await;
    if let Err(error) = shutdown {
        println!("[PPO smoke] warning: agent shutdown returned {error}");
    }

    result
}

fn bootstrap_policy_model() -> Result<ModelModule<NdArray>, Box<dyn Error>> {
    let metadata = ModelMetadata {
        model_file: "openttd_ppo_smoke_bootstrap.pt".to_string(),
        model_type: ModelFileType::Pt,
        input_dtype: DType::NdArray(NdArrayDType::F32),
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: vec![OBSERVATION_DIM],
        output_shape: vec![ACTION_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<NdArray>::from_pt_bytes(Vec::new(), metadata)?)
}

pub async fn create_training_phases(
    agent: &mut RelayRLAgent<NdArray>,
    env_count: u32,
    loop_iters: usize,
    max_traj_length: usize,
) -> Result<Vec<TrainingPhase>, Box<dyn Error>> {
    let mut phases = Vec::with_capacity(ACTOR_SPECS.len());
    for actor in ACTOR_SPECS.iter().copied() {
        let actor_id = agent
            .new_actor::<2, 2>(DeviceType::Cpu, max_traj_length, None)
            .await?;
        phases.push(TrainingPhase {
            actor_id,
            actor,
            env_count,
            loop_iters,
            max_traj_length,
        });
    }
    Ok(phases)
}

pub async fn train_then_freeze_subsystems(
    agent: &mut RelayRLAgent<NdArray>,
    config: OpenTtdConfig,
    phases: &[TrainingPhase],
    output_dir: &Path,
) -> Result<Vec<FrozenSubsystemModel>, Box<dyn Error>> {
    let mut frozen = Vec::with_capacity(phases.len());

    for phase in phases {
        let env = OpenTtdEnvironment::new(config.clone(), phase.actor)?;
        agent
            .set_env(phase.actor_id, Box::new(env), phase.env_count)
            .await?;

        let actor_dir = output_dir.join(phase.actor.actor_name.replace('.', "_"));
        let env_dir = actor_dir.join("rollouts");
        let model_dir = actor_dir.join("frozen-model");
        std::fs::create_dir_all(&env_dir)?;
        std::fs::create_dir_all(&model_dir)?;

        let ppo_spec = build_ppo_spec(env_dir, model_dir.clone(), phase.max_traj_length)?;
        let trained_model: ModelModule<NdArray> = agent
            .run_env_with_ppo::<Float, Float, GenericMlp<NdArray, Float, Float>>(
                phase.actor_id,
                phase.loop_iters,
                phase.max_traj_length,
                ppo_spec,
            )
            .await?;

        trained_model.save(model_dir.clone())?;
        agent
            .update_model(Some(vec![phase.actor_id]), trained_model)
            .await?;

        frozen.push(FrozenSubsystemModel {
            actor_id: phase.actor_id,
            actor_name: phase.actor.actor_name,
            subsystem: phase.actor.subsystem,
            model_dir,
        });
    }

    Ok(frozen)
}

pub fn describe_training_order() -> Vec<&'static str> {
    ACTOR_SPECS
        .iter()
        .map(|spec| spec.actor_name)
        .collect::<Vec<_>>()
}
