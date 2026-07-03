use crate::actors::CacheActorRole;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub byte_hits: u64,
    pub byte_misses: u64,
    pub evictions: u64,
    pub stale_hits: u64,
    pub backend_fetches: u64,
    pub total_latency_ms: f32,
    pub p50_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub memory_utilization: f32,
    pub total_reward: f32,
    pub reward_by_actor: BTreeMap<CacheActorRole, f32>,
    pub decision_count_by_actor: BTreeMap<CacheActorRole, u64>,
    pub benchmark_runtime_ms: f64,
    pub total_env_steps_per_second: f64,
    pub total_request_action_throughput_per_second: f64,
    #[serde(skip)]
    latencies: Vec<f32>,
}

impl CacheMetrics {
    pub fn record_hit(&mut self, size_bytes: usize, latency_ms: f32) {
        self.requests += 1;
        self.hits += 1;
        self.byte_hits += size_bytes as u64;
        self.record_latency(latency_ms);
    }

    pub fn record_miss(&mut self, size_bytes: usize, latency_ms: f32) {
        self.requests += 1;
        self.misses += 1;
        self.byte_misses += size_bytes as u64;
        self.backend_fetches += 1;
        self.record_latency(latency_ms);
    }

    pub fn record_stale_hit(&mut self) {
        self.stale_hits += 1;
    }

    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    pub fn record_actor_decision(&mut self, role: CacheActorRole) {
        *self.decision_count_by_actor.entry(role).or_default() += 1;
    }

    pub fn record_actor_reward(&mut self, role: CacheActorRole, reward: f32) {
        self.total_reward += reward;
        *self.reward_by_actor.entry(role).or_default() += reward;
    }

    pub fn finalize(&mut self, used_bytes: usize, capacity_bytes: usize, elapsed: Duration) {
        self.memory_utilization = if capacity_bytes == 0 {
            0.0
        } else {
            used_bytes as f32 / capacity_bytes as f32
        };
        self.latencies.sort_by(|left, right| left.total_cmp(right));
        self.p50_latency_ms = percentile(&self.latencies, 0.50);
        self.p95_latency_ms = percentile(&self.latencies, 0.95);
        self.p99_latency_ms = percentile(&self.latencies, 0.99);
        self.benchmark_runtime_ms = elapsed.as_secs_f64() * 1_000.0;
        let secs = elapsed.as_secs_f64().max(1e-9);
        self.total_env_steps_per_second = self.requests as f64 / secs;
        let total_actor_decisions: u64 = self.decision_count_by_actor.values().sum();
        self.total_request_action_throughput_per_second = total_actor_decisions as f64 / secs;
    }

    pub fn hit_rate(&self) -> f64 {
        ratio(self.hits, self.requests)
    }

    pub fn byte_hit_rate(&self) -> f64 {
        ratio(self.byte_hits, self.byte_hits + self.byte_misses)
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.requests as f64
        }
    }

    fn record_latency(&mut self, latency_ms: f32) {
        self.total_latency_ms += latency_ms;
        self.latencies.push(latency_ms);
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}
