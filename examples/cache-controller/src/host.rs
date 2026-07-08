use crate::actors::CacheActorRole;
use crate::cache::CacheWorld;
use crate::heuristics::{HeuristicController, PolicyKind};
use crate::metrics::CacheMetrics;
use crate::policies::MixedPolicySet;
use crate::workload::{WorkloadGenerator, WorkloadKind};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub policy: PolicyKind,
    pub workload: WorkloadKind,
    pub requests: u64,
    pub seed: u64,
    pub capacity_bytes: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            policy: PolicyKind::Lru,
            workload: WorkloadKind::Zipfian,
            requests: 25_000,
            seed: 42,
            capacity_bytes: 512 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub policy: String,
    pub workload: WorkloadKind,
    pub requests: u64,
    pub seed: u64,
    pub metrics: CacheMetrics,
}

pub fn run_benchmark(config: BenchmarkConfig) -> BenchmarkResult {
    let started = Instant::now();
    let mut workload = WorkloadGenerator::new(config.workload, config.seed);
    let mut world = CacheWorld::new(config.capacity_bytes);
    let mut controller = HeuristicController::new(config.policy, config.seed);

    for _ in 0..config.requests {
        let request = workload.next_request();
        let active_roles = controller.active_roles(&world, &request);
        let decisions = controller.decide(&world, &request);
        world.apply_request(&request, decisions, &active_roles);
    }

    BenchmarkResult {
        policy: controller.policy().as_str().to_string(),
        workload: config.workload,
        requests: config.requests,
        seed: config.seed,
        metrics: world.finalize_metrics(started.elapsed()),
    }
}

pub fn run_mixed_policy_benchmark(
    config: BenchmarkConfig,
    policy_name: impl Into<String>,
    mut policy_set: MixedPolicySet,
) -> BenchmarkResult {
    let started = Instant::now();
    let mut workload = WorkloadGenerator::new(config.workload, config.seed);
    let mut world = CacheWorld::new(config.capacity_bytes);

    for _ in 0..config.requests {
        let request = workload.next_request();
        let active_roles = policy_set.active_roles(&world, &request);
        let decisions = policy_set.decide_all(&world, &request);
        world.apply_request(&request, decisions, &active_roles);
    }

    BenchmarkResult {
        policy: policy_name.into(),
        workload: config.workload,
        requests: config.requests,
        seed: config.seed,
        metrics: world.finalize_metrics(started.elapsed()),
    }
}

pub fn compare_policies(mut config: BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut policies = PolicyKind::BASELINES.to_vec();
    policies.push(PolicyKind::RelayRlAdaptive);
    policies
        .into_iter()
        .map(|policy| {
            config.policy = policy;
            run_benchmark(config.clone())
        })
        .collect()
}

pub fn actor_activity_lines(metrics: &CacheMetrics) -> Vec<String> {
    CacheActorRole::ALL
        .into_iter()
        .map(|role| {
            let count = metrics
                .decision_count_by_actor
                .get(&role)
                .copied()
                .unwrap_or_default();
            format!("{:<22} {:>10}", role.as_str(), count)
        })
        .collect()
}

pub fn print_results_table(results: &[BenchmarkResult]) {
    println!(
        "{:<18} {:>8} {:>8} {:>9} {:>9} {:>10} {:>14} {:>11} {:>13} {:>13}",
        "Policy",
        "HitRate",
        "ByteHit",
        "AvgLat",
        "P95Lat",
        "Evictions",
        "BackendFetch",
        "Reward",
        "EnvSteps/s",
        "ReqAct/s",
    );
    for result in results {
        let m = &result.metrics;
        println!(
            "{:<18} {:>8.3} {:>8.3} {:>8.2}ms {:>8.2}ms {:>10} {:>14} {:>11.3} {:>13.1} {:>13.1}",
            result.policy,
            m.hit_rate(),
            m.byte_hit_rate(),
            m.avg_latency_ms(),
            m.p95_latency_ms,
            m.evictions,
            m.backend_fetches,
            m.total_reward,
            m.total_env_steps_per_second,
            m.total_request_action_throughput_per_second,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_is_seed_deterministic_for_counts() {
        let config = BenchmarkConfig {
            requests: 1_000,
            seed: 123,
            ..BenchmarkConfig::default()
        };
        let left = run_benchmark(config.clone());
        let right = run_benchmark(config);
        assert_eq!(left.metrics.hits, right.metrics.hits);
        assert_eq!(left.metrics.misses, right.metrics.misses);
        assert_eq!(left.metrics.evictions, right.metrics.evictions);
        assert_eq!(left.metrics.backend_fetches, right.metrics.backend_fetches);
    }

    #[test]
    fn throughput_metrics_are_populated() {
        let result = run_benchmark(BenchmarkConfig {
            requests: 1_000,
            seed: 7,
            ..BenchmarkConfig::default()
        });
        assert!(result.metrics.total_env_steps_per_second > 0.0);
        assert!(result.metrics.total_request_action_throughput_per_second > 0.0);
    }
}
