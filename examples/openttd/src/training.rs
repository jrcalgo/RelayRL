use crate::environment::OpenTtdEnvironment;
use crate::openttd::{GameSubsystem, OpenTtdConfig};
use crate::subsystems::{ACTOR_SPECS, SubsystemActorSpec};
use crate::{ACTION_DIM, OBSERVATION_DIM};
use burn_ndarray::NdArray;
use burn_tensor::Float;
use relayrl::algorithms::GenericMlp;
use relayrl::algorithms::PPO::PPOTrainerSpec;
use relayrl::network::{ActorUuid, RelayRLActorEnv, RelayRLAgent, RelayRLAgentActors};
use relayrl::types::model::ModelModule;
use relayrl::types::tensor::relayrl::{DType, DeviceType, NdArrayDType};
use std::error::Error;
use std::path::{Path, PathBuf};

pub type OpenTtdPpoSpec =
    PPOTrainerSpec<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>;

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

pub fn build_ppo_spec(
    env_dir: PathBuf,
    save_model_path: PathBuf,
    buffer_size: usize,
) -> Result<OpenTtdPpoSpec, Box<dyn Error>> {
    Ok(PPOTrainerSpec::<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>::default(
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
