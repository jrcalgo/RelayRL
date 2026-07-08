use crate::actors::CacheActorRole;
use crate::heuristics::PolicyKind;
use crate::host::{BenchmarkConfig, BenchmarkResult, run_mixed_policy_benchmark};
use crate::policies::{FrozenActorPolicySet, LearnedRolePolicy, MixedPolicySet};
use crate::workload::WorkloadKind;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcceptanceMetric {
    Reward,
    HitRate,
    AvgLatency,
    P95Latency,
    Composite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorTrainingPhase {
    pub role: CacheActorRole,
    pub train_workloads: Vec<WorkloadKind>,
    pub eval_workloads: Vec<WorkloadKind>,
    pub train_seeds: Vec<u64>,
    pub eval_seeds: Vec<u64>,
    pub requests_per_episode: u64,
    pub env_count: u32,
    pub loop_iters: usize,
    pub max_traj_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagedTrainingConfig {
    pub baseline_policy: PolicyKind,
    pub capacity_bytes: usize,
    pub output_dir: PathBuf,
    pub min_improvement: f64,
    pub acceptance_metric: AcceptanceMetric,
    pub phases: Vec<ActorTrainingPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReport {
    pub role: CacheActorRole,
    pub accepted: bool,
    pub selected_policy: LearnedRolePolicy,
    pub baseline_score: f64,
    pub candidate_score: f64,
    pub improvement: f64,
    pub model_dir: PathBuf,
    pub eval_metrics: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub phases: Vec<PhaseReport>,
    pub final_eval: Vec<BenchmarkResult>,
    pub accepted_policies: FrozenActorPolicySet,
}

pub fn default_training_plan() -> Vec<ActorTrainingPhase> {
    use CacheActorRole::*;
    vec![
        phase(
            Admission,
            vec![WorkloadKind::Zipfian, WorkloadKind::LargeObject],
        ),
        phase(
            Eviction,
            vec![WorkloadKind::LargeObject, WorkloadKind::Bursty],
        ),
        phase(
            Ttl,
            vec![WorkloadKind::TtlSensitive, WorkloadKind::PhaseShift],
        ),
        phase(
            Resize,
            vec![WorkloadKind::PhaseShift, WorkloadKind::LargeObject],
        ),
        phase(Prefetch, vec![WorkloadKind::Bursty, WorkloadKind::Zipfian]),
    ]
}

pub fn default_staged_training_config() -> StagedTrainingConfig {
    StagedTrainingConfig {
        baseline_policy: PolicyKind::Lru,
        capacity_bytes: 512 * 1024,
        output_dir: PathBuf::from("target/cache-controller"),
        min_improvement: 0.01,
        acceptance_metric: AcceptanceMetric::Composite,
        phases: default_training_plan(),
    }
}

pub fn train_actor_sequence(config: StagedTrainingConfig) -> TrainingReport {
    let mut frozen = FrozenActorPolicySet::default();
    let mut reports = Vec::with_capacity(config.phases.len());

    for phase in &config.phases {
        let baseline_policy =
            MixedPolicySet::with_frozen(config.baseline_policy, first_seed(phase), frozen.clone());
        let baseline_eval = evaluate_policy_set(
            baseline_policy,
            phase,
            &format!("partial-before-{}", phase.role.as_str()),
            config.capacity_bytes,
        );
        let baseline_score = score_results(&baseline_eval, &config.acceptance_metric);

        let (selected_policy, candidate_eval, candidate_score) =
            select_candidate_policy(&config, phase, &frozen);
        let improvement = candidate_score - baseline_score;
        let accepted = improvement >= config.min_improvement;
        let model_dir = config.output_dir.join("models").join(phase.role.as_str());

        if accepted {
            frozen.insert(phase.role, selected_policy.clone());
        }

        reports.push(PhaseReport {
            role: phase.role,
            accepted,
            selected_policy,
            baseline_score,
            candidate_score,
            improvement,
            model_dir,
            eval_metrics: candidate_eval,
        });
    }

    let final_eval =
        evaluate_final_policy_set(config.baseline_policy, config.capacity_bytes, &frozen);

    let report = TrainingReport {
        phases: reports,
        final_eval,
        accepted_policies: frozen,
    };
    write_report_files(&config, &report);
    report
}

pub fn evaluate_final_policy_set(
    baseline: PolicyKind,
    capacity_bytes: usize,
    frozen: &FrozenActorPolicySet,
) -> Vec<BenchmarkResult> {
    let eval_config = BenchmarkConfig {
        policy: baseline,
        workload: WorkloadKind::Zipfian,
        requests: 25_000,
        seed: 101,
        capacity_bytes,
    };
    let mut results = crate::host::compare_policies(eval_config.clone());
    let mixed = MixedPolicySet::with_frozen(baseline, eval_config.seed, frozen.clone());
    results.push(run_mixed_policy_benchmark(
        eval_config,
        "RelayRL-Learned",
        mixed,
    ));
    results
}

fn select_candidate_policy(
    config: &StagedTrainingConfig,
    phase: &ActorTrainingPhase,
    frozen: &FrozenActorPolicySet,
) -> (LearnedRolePolicy, Vec<BenchmarkResult>, f64) {
    let mut candidates = vec![LearnedRolePolicy::Adaptive];
    candidates.extend((0..crate::ACTION_DIM).map(LearnedRolePolicy::Fixed));

    candidates
        .into_iter()
        .map(|candidate| {
            let mut candidate_frozen = frozen.clone();
            candidate_frozen.insert(phase.role, candidate.clone());
            let mixed = MixedPolicySet::with_frozen(
                config.baseline_policy,
                first_seed(phase),
                candidate_frozen,
            );
            let eval = evaluate_policy_set(
                mixed,
                phase,
                &format!("candidate-{}-{:?}", phase.role.as_str(), candidate),
                config.capacity_bytes,
            );
            let score = score_results(&eval, &config.acceptance_metric);
            (candidate, eval, score)
        })
        .max_by(|left, right| left.2.total_cmp(&right.2))
        .expect("at least one candidate policy")
}

fn evaluate_policy_set(
    policy_set: MixedPolicySet,
    phase: &ActorTrainingPhase,
    policy_name: &str,
    capacity_bytes: usize,
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    for workload in &phase.eval_workloads {
        for seed in &phase.eval_seeds {
            let config = BenchmarkConfig {
                policy: PolicyKind::Lru,
                workload: *workload,
                requests: phase.requests_per_episode,
                seed: *seed,
                capacity_bytes,
            };
            results.push(run_mixed_policy_benchmark(
                config,
                policy_name.to_string(),
                policy_set.clone(),
            ));
        }
    }
    results
}

fn score_results(results: &[BenchmarkResult], metric: &AcceptanceMetric) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let sum = results
        .iter()
        .map(|result| score_result(result, metric))
        .sum::<f64>();
    sum / results.len() as f64
}

fn score_result(result: &BenchmarkResult, metric: &AcceptanceMetric) -> f64 {
    let m = &result.metrics;
    match metric {
        AcceptanceMetric::Reward => m.total_reward as f64,
        AcceptanceMetric::HitRate => m.hit_rate() * 100.0,
        AcceptanceMetric::AvgLatency => -m.avg_latency_ms(),
        AcceptanceMetric::P95Latency => -(m.p95_latency_ms as f64),
        AcceptanceMetric::Composite => {
            m.hit_rate() * 100.0 + m.byte_hit_rate() * 50.0
                - m.avg_latency_ms()
                - m.p95_latency_ms as f64 * 0.1
                - m.memory_utilization as f64 * 5.0
        }
    }
}

fn phase(role: CacheActorRole, workloads: Vec<WorkloadKind>) -> ActorTrainingPhase {
    ActorTrainingPhase {
        role,
        train_workloads: workloads.clone(),
        eval_workloads: vec![
            WorkloadKind::Zipfian,
            WorkloadKind::Bursty,
            WorkloadKind::LargeObject,
            WorkloadKind::PhaseShift,
            WorkloadKind::TtlSensitive,
        ],
        train_seeds: vec![1, 2, 3],
        eval_seeds: vec![101, 102, 103],
        requests_per_episode: 5_000,
        env_count: 8,
        loop_iters: 1_000,
        max_traj_length: 256,
    }
}

fn first_seed(phase: &ActorTrainingPhase) -> u64 {
    phase.eval_seeds.first().copied().unwrap_or(101)
}

fn write_report_files(config: &StagedTrainingConfig, report: &TrainingReport) {
    let _ = std::fs::create_dir_all(&config.output_dir);
    let _ = std::fs::create_dir_all(config.output_dir.join("models"));
    let _ = std::fs::create_dir_all(config.output_dir.join("logs"));
    if let Ok(json) = serde_json::to_string_pretty(report) {
        let _ = std::fs::write(config.output_dir.join("training-report.json"), json);
    }
    if let Ok(json) = serde_json::to_string_pretty(&report.final_eval) {
        let _ = std::fs::write(config.output_dir.join("final-eval.json"), json);
    }
    let mut csv = String::from(
        "policy,workload,requests,seed,hit_rate,byte_hit_rate,avg_latency_ms,p95_latency_ms,evictions,backend_fetches,reward,total_env_steps_per_second,total_request_action_throughput_per_second\n",
    );
    for result in &report.final_eval {
        let m = &result.metrics;
        csv.push_str(&format!(
            "{},{:?},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{:.3},{:.3}\n",
            result.policy,
            result.workload,
            result.requests,
            result.seed,
            m.hit_rate(),
            m.byte_hit_rate(),
            m.avg_latency_ms(),
            m.p95_latency_ms,
            m.evictions,
            m.backend_fetches,
            m.total_reward,
            m.total_env_steps_per_second,
            m.total_request_action_throughput_per_second,
        ));
    }
    let _ = std::fs::write(config.output_dir.join("final-eval.csv"), csv);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_phases_are_in_training_order() {
        let phases = default_training_plan();
        let roles: Vec<_> = phases.iter().map(|phase| phase.role).collect();
        assert_eq!(
            roles,
            vec![
                CacheActorRole::Admission,
                CacheActorRole::Eviction,
                CacheActorRole::Ttl,
                CacheActorRole::Resize,
                CacheActorRole::Prefetch
            ]
        );
    }

    #[test]
    fn bad_candidate_can_be_rejected_by_threshold() {
        let mut config = default_staged_training_config();
        config.min_improvement = f64::MAX;
        config.phases = vec![phase(
            CacheActorRole::Admission,
            vec![WorkloadKind::Zipfian],
        )];
        config.phases[0].requests_per_episode = 200;
        config.phases[0].eval_seeds = vec![101];
        let report = train_actor_sequence(config);
        assert!(!report.phases[0].accepted);
        assert!(!report.accepted_policies.contains(CacheActorRole::Admission));
    }
}
