use crate::actors::{CacheActorRole, action_from_bytes, observation_for, reward_for_role};
use crate::cache::CacheWorld;
use crate::environment::CacheTrainingEnvironment;
use crate::heuristics::{HeuristicController, PolicyKind};
use crate::host::{BenchmarkConfig, BenchmarkResult, compare_policies};
use crate::policies::{FrozenActorPolicySet, LearnedRolePolicy, MixedPolicySet};
use crate::staged_training::PhaseReport;
use crate::workload::{WorkloadGenerator, WorkloadKind};
use crate::{ACTION_DIM, OBSERVATION_DIM};
use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData as BurnTensorData};
use relayrl::algorithms::PPO::kernel::{DiscretePPOPolicyHead, PPOPolicyHead};
use relayrl::algorithms::PPO::{IPPOParams, PPONetworkArgs, PPOTrainerSpec};
use relayrl::algorithms::{ActivationKind, GenericMlp};
use relayrl::network::{AgentBuilder, RelayRLActorEnv, RelayRLAgentActors};
use relayrl::types::model::ModelModule;
use relayrl::types::tensor::relayrl::{BackendMatcher, DType, DeviceType};
use relayrl_types::data::tensor::NdArrayDType;
use relayrl_types::model::{ModelFileType, ModelMetadata};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;

type CachePpoSpec = PPOTrainerSpec<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>;
type ActorUuid = uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorPpoConfig {
    pub rollout_len: usize,
    pub traj_per_epoch: u64,
    pub train_pi_iters: u64,
    pub train_vf_iters: u64,
    pub loop_iters: usize,
    pub max_traj_length: usize,
    pub env_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStagedTrainingConfig {
    pub baseline_policy: PolicyKind,
    pub capacity_bytes: usize,
    pub requests: u64,
    pub seed: u64,
    pub output_dir: PathBuf,
    pub min_improvement: f64,
    pub smoke: bool,
    pub ppo: ActorPpoConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrozenNeuralPolicySet {
    pub models: BTreeMap<CacheActorRole, PathBuf>,
}

impl FrozenNeuralPolicySet {
    pub fn insert(&mut self, role: CacheActorRole, model_dir: PathBuf) {
        self.models.insert(role, model_dir);
    }

    pub fn contains(&self, role: CacheActorRole) -> bool {
        self.models.contains_key(&role)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTrainingReport {
    pub phases: Vec<PhaseReport>,
    pub final_eval: Vec<BenchmarkResult>,
    pub accepted_models: FrozenNeuralPolicySet,
}

pub fn default_neural_config(smoke: bool) -> NeuralStagedTrainingConfig {
    NeuralStagedTrainingConfig {
        baseline_policy: PolicyKind::Lru,
        capacity_bytes: 512 * 1024,
        requests: if smoke { 1_500 } else { 25_000 },
        seed: 42,
        output_dir: PathBuf::from("target/cache-controller"),
        min_improvement: 0.01,
        smoke,
        ppo: if smoke {
            ActorPpoConfig {
                rollout_len: 16,
                traj_per_epoch: 1,
                train_pi_iters: 1,
                train_vf_iters: 1,
                loop_iters: 128,
                max_traj_length: 64,
                env_count: 1,
            }
        } else {
            ActorPpoConfig {
                rollout_len: 64,
                traj_per_epoch: 2,
                train_pi_iters: 4,
                train_vf_iters: 4,
                loop_iters: 20_000,
                max_traj_length: 512,
                env_count: 8,
            }
        },
    }
}

pub async fn train_actor_sequence_with_ppo(
    config: NeuralStagedTrainingConfig,
) -> Result<NeuralTrainingReport, Box<dyn Error>> {
    let mut accepted = FrozenNeuralPolicySet::default();
    let mut symbolic_background = FrozenActorPolicySet::default();
    let mut phase_reports = Vec::new();

    let bootstrap = bootstrap_policy_model()?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .router_scale(1)
        .default_model(bootstrap.clone())
        .build()
        .await?;
    agent.start(params).await?;

    let result = async {
        for role in CacheActorRole::ALL {
            let model_dir = config.output_dir.join("ppo-models").join(role.as_str());
            let env_dir = config.output_dir.join("ppo-rollouts").join(role.as_str());
            std::fs::create_dir_all(&model_dir)?;
            std::fs::create_dir_all(&env_dir)?;

            println!(
                "[PPO staged] training role={} env_count={} loop_iters={} rollout_len={}",
                role, config.ppo.env_count, config.ppo.loop_iters, config.ppo.rollout_len
            );

            let actor_id = agent
                .new_actor::<2, 2>(
                    DeviceType::Cpu,
                    config.ppo.max_traj_length,
                    Some(bootstrap.clone()),
                )
                .await?;

            let train_config = BenchmarkConfig {
                policy: config.baseline_policy,
                workload: workload_for_role(role),
                requests: config.requests,
                seed: config.seed,
                capacity_bytes: config.capacity_bytes,
            };
            let background = MixedPolicySet::with_frozen(
                config.baseline_policy,
                config.seed,
                symbolic_background.clone(),
            );
            let env = CacheTrainingEnvironment::with_background(role, train_config, background);
            agent
                .set_env(actor_id, Box::new(env), config.ppo.env_count)
                .await?;

            let spec = build_cache_ppo_spec(env_dir, model_dir.clone(), &config.ppo)?;
            let trained_model = agent
                .run_env_with_ppo::<Float, Float, GenericMlp<NdArray, Float, Float>>(
                    actor_id,
                    config.ppo.loop_iters,
                    config.ppo.max_traj_length,
                    spec,
                )
                .await?;
            trained_model.save(model_dir.clone())?;

            let baseline_eval =
                evaluate_neural_policy_set(&config, &accepted, &format!("before-{role}")).await?;
            let baseline_score = average_score(&baseline_eval);

            let mut candidate = accepted.clone();
            candidate.insert(role, model_dir.clone());
            let candidate_eval =
                evaluate_neural_policy_set(&config, &candidate, &format!("candidate-{role}"))
                    .await?;
            let candidate_score = average_score(&candidate_eval);
            let improvement = candidate_score - baseline_score;
            let accepted_role = improvement >= config.min_improvement;

            if accepted_role {
                accepted = candidate;
                symbolic_background.insert(role, LearnedRolePolicy::Adaptive);
            }

            println!(
                "[PPO staged] role={} accepted={} baseline={:.3} candidate={:.3} improvement={:.3}",
                role, accepted_role, baseline_score, candidate_score, improvement
            );

            phase_reports.push(PhaseReport {
                role,
                accepted: accepted_role,
                selected_policy: LearnedRolePolicy::Neural {
                    model_dir: model_dir.clone(),
                    role,
                },
                baseline_score,
                candidate_score,
                improvement,
                model_dir,
                eval_metrics: candidate_eval,
            });
        }

        let final_eval = final_neural_comparison(&config, &accepted).await?;
        let report = NeuralTrainingReport {
            phases: phase_reports,
            final_eval,
            accepted_models: accepted,
        };
        write_neural_report(&config, &report);
        Ok::<NeuralTrainingReport, Box<dyn Error>>(report)
    }
    .await;

    let _ = agent.shutdown().await;
    result
}

pub async fn final_neural_comparison(
    config: &NeuralStagedTrainingConfig,
    accepted: &FrozenNeuralPolicySet,
) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let eval_config = BenchmarkConfig {
        policy: config.baseline_policy,
        workload: WorkloadKind::Zipfian,
        requests: config.requests,
        seed: config.seed.wrapping_add(59),
        capacity_bytes: config.capacity_bytes,
    };
    let mut results = compare_policies(eval_config.clone());
    results.push(evaluate_neural_once(eval_config, accepted, "RelayRL-PPO-Learned").await?);
    Ok(results)
}

async fn evaluate_neural_policy_set(
    config: &NeuralStagedTrainingConfig,
    accepted: &FrozenNeuralPolicySet,
    name: &str,
) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> {
    let mut results = Vec::new();
    for workload in [
        WorkloadKind::Zipfian,
        WorkloadKind::Bursty,
        WorkloadKind::LargeObject,
        WorkloadKind::PhaseShift,
        WorkloadKind::TtlSensitive,
    ] {
        for seed in [
            config.seed,
            config.seed.wrapping_add(1),
            config.seed.wrapping_add(2),
        ] {
            let eval_config = BenchmarkConfig {
                policy: config.baseline_policy,
                workload,
                requests: config.requests,
                seed,
                capacity_bytes: config.capacity_bytes,
            };
            results.push(evaluate_neural_once(eval_config, accepted, name).await?);
        }
    }
    Ok(results)
}

async fn evaluate_neural_once(
    config: BenchmarkConfig,
    accepted: &FrozenNeuralPolicySet,
    policy_name: &str,
) -> Result<BenchmarkResult, Box<dyn Error>> {
    let bootstrap = bootstrap_policy_model()?;
    let (mut agent, params) = AgentBuilder::<NdArray>::builder()
        .router_scale(1)
        .default_model(bootstrap.clone())
        .build()
        .await?;
    agent.start(params).await?;

    let result = async {
        let mut actor_ids: BTreeMap<CacheActorRole, ActorUuid> = BTreeMap::new();
        for (role, model_dir) in &accepted.models {
            let model = ModelModule::<NdArray>::load_from_path(model_dir)?;
            let actor_id = agent
                .new_actor::<2, 2>(DeviceType::Cpu, 512, Some(model))
                .await?;
            actor_ids.insert(*role, actor_id);
        }

        let started = std::time::Instant::now();
        let mut workload = WorkloadGenerator::new(config.workload, config.seed);
        let mut world = CacheWorld::new(config.capacity_bytes);
        let mut heuristic = HeuristicController::new(config.policy, config.seed);
        let mut last_rewards: BTreeMap<CacheActorRole, f32> = BTreeMap::new();

        for _ in 0..config.requests {
            let request = workload.next_request();
            let mut decisions = heuristic.decide(&world, &request);
            let active_roles = heuristic.active_roles(&world, &request);

            for role in &active_roles {
                if let Some(actor_id) = actor_ids.get(role) {
                    let observation = observation_for(&world, &request, *role);
                    let action = agent
                        .request_action::<2, 2, Float, Float>(
                            vec![*actor_id],
                            observation_tensor(observation),
                            None,
                            *last_rewards.get(role).unwrap_or(&0.0),
                        )
                        .await?
                        .into_iter()
                        .next()
                        .and_then(|(_, action)| {
                            action
                                .get_act()
                                .map(|tensor_data| action_from_bytes(&tensor_data.data))
                        })
                        .unwrap_or_else(|| decisions.get(*role));
                    decisions.set(*role, action);
                }
            }

            let outcome = world.apply_request(&request, decisions, &active_roles);
            for role in &active_roles {
                last_rewards.insert(*role, reward_for_role(*role, &outcome));
            }
        }

        if !actor_ids.is_empty() {
            let ids: Vec<_> = actor_ids.values().copied().collect();
            let _ = agent.flag_last_action(ids, Some(0.0)).await;
        }

        Ok::<BenchmarkResult, Box<dyn Error>>(BenchmarkResult {
            policy: policy_name.to_string(),
            workload: config.workload,
            requests: config.requests,
            seed: config.seed,
            metrics: world.finalize_metrics(started.elapsed()),
        })
    }
    .await;

    let _ = agent.shutdown().await;
    result
}

fn build_cache_ppo_spec(
    env_dir: PathBuf,
    save_model_path: PathBuf,
    ppo: &ActorPpoConfig,
) -> Result<CachePpoSpec, Box<dyn Error>> {
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
            act_dtype.clone(),
            ActivationKind::None,
            &device,
        ))?),
        vf_mlp: GenericMlp::new(
            OBSERVATION_DIM,
            obs_dtype.clone(),
            &[32],
            1,
            DType::NdArray(NdArrayDType::F32),
            ActivationKind::None,
            &device,
        ),
    };
    let params = IPPOParams {
        traj_per_epoch: ppo.traj_per_epoch,
        train_pi_iters: ppo.train_pi_iters,
        train_vf_iters: ppo.train_vf_iters,
        rollout_len: Some(ppo.rollout_len),
        max_episode_steps: Some(ppo.rollout_len),
        normalize_obs: false,
        normalize_returns: false,
        ..IPPOParams::default()
    };
    let mut spec =
        PPOTrainerSpec::<NdArray, Float, Float, GenericMlp<NdArray, Float, Float>>::default(
            env_dir,
            save_model_path,
            OBSERVATION_DIM,
            DType::NdArray(NdArrayDType::F32),
            ACTION_DIM,
            DType::NdArray(NdArrayDType::F32),
            ppo.max_traj_length,
            DeviceType::Cpu,
        )?;
    if let PPOTrainerSpec::PPO {
        hyperparams,
        networks: spec_networks,
        ..
    } = &mut spec
    {
        *hyperparams = Some(params);
        *spec_networks = networks;
    }
    Ok(spec)
}

fn bootstrap_policy_model() -> Result<ModelModule<NdArray>, Box<dyn Error>> {
    let metadata = ModelMetadata {
        model_file: "cache_ppo_bootstrap.pt".to_string(),
        model_type: ModelFileType::Pt,
        input_dtype: DType::NdArray(NdArrayDType::F32),
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: vec![OBSERVATION_DIM],
        output_shape: vec![ACTION_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<NdArray>::from_pt_bytes(Vec::new(), metadata)?)
}

fn observation_tensor(observation: [f32; OBSERVATION_DIM]) -> Tensor<NdArray, 2, Float> {
    Tensor::<NdArray, 2, Float>::from_data(
        BurnTensorData::new(observation.to_vec(), [1, OBSERVATION_DIM]),
        &Default::default(),
    )
}

fn workload_for_role(role: CacheActorRole) -> WorkloadKind {
    match role {
        CacheActorRole::Admission => WorkloadKind::Zipfian,
        CacheActorRole::Eviction => WorkloadKind::LargeObject,
        CacheActorRole::Ttl => WorkloadKind::TtlSensitive,
        CacheActorRole::Resize => WorkloadKind::PhaseShift,
        CacheActorRole::Prefetch => WorkloadKind::Bursty,
    }
}

fn average_score(results: &[BenchmarkResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results
        .iter()
        .map(|result| {
            let m = &result.metrics;
            m.hit_rate() * 100.0 + m.byte_hit_rate() * 50.0
                - m.avg_latency_ms()
                - m.p95_latency_ms as f64 * 0.1
                - m.memory_utilization as f64 * 5.0
        })
        .sum::<f64>()
        / results.len() as f64
}

fn write_neural_report(config: &NeuralStagedTrainingConfig, report: &NeuralTrainingReport) {
    let _ = std::fs::create_dir_all(&config.output_dir);
    if let Ok(json) = serde_json::to_string_pretty(report) {
        let _ = std::fs::write(config.output_dir.join("ppo-training-report.json"), json);
    }
    if let Ok(json) = serde_json::to_string_pretty(&report.final_eval) {
        let _ = std::fs::write(config.output_dir.join("ppo-final-eval.json"), json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_spec_builds_for_cache_actor() {
        let tmp = std::env::temp_dir().join("cache-ppo-spec-test");
        let spec = build_cache_ppo_spec(
            tmp.join("env"),
            tmp.join("model"),
            &default_neural_config(true).ppo,
        );
        assert!(spec.is_ok());
    }

    #[test]
    fn neural_report_serializes() {
        let report = NeuralTrainingReport {
            phases: Vec::new(),
            final_eval: Vec::new(),
            accepted_models: FrozenNeuralPolicySet::default(),
        };
        assert!(serde_json::to_string(&report).is_ok());
    }
}
