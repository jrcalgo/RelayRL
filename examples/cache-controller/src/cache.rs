use crate::actors::{ActorDecisions, CacheActorRole, reward_for_role};
use crate::metrics::CacheMetrics;
use crate::workload::CacheRequest;
use std::collections::HashMap;

pub type Key = u64;

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: Key,
    pub size_bytes: usize,
    pub inserted_at: u64,
    pub last_accessed_at: u64,
    pub access_count: u64,
    pub ttl_expires_at: u64,
    pub backend_cost_ms: f32,
    pub priority: f32,
}

#[derive(Debug, Clone)]
pub struct CacheWorld {
    capacity_bytes: usize,
    initial_capacity_bytes: usize,
    used_bytes: usize,
    entries: HashMap<Key, CacheEntry>,
    historical_frequency: HashMap<Key, u64>,
    clock: u64,
    miss_burst: u64,
    resize_count: u64,
    metrics: CacheMetrics,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub clock: u64,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
    pub entry_count: usize,
    pub avg_entry_size: usize,
    pub utilization: f32,
    pub hit_rate: f32,
    pub miss_rate: f32,
    pub eviction_rate: f32,
    pub stale_rate: f32,
    pub backend_fetch_rate: f32,
    pub miss_burst: u64,
    pub resize_count: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StepOutcome {
    pub hit: bool,
    pub stale: bool,
    pub admitted: bool,
    pub prefetched: bool,
    pub evictions: u64,
    pub latency_ms: f32,
    pub memory_pressure_after: f32,
}

impl CacheWorld {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            initial_capacity_bytes: capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            historical_frequency: HashMap::new(),
            clock: 0,
            miss_burst: 0,
            resize_count: 0,
            metrics: CacheMetrics::default(),
        }
    }

    pub fn apply_request(
        &mut self,
        request: &CacheRequest,
        decisions: ActorDecisions,
        active_roles: &[CacheActorRole],
    ) -> StepOutcome {
        self.clock += 1;
        *self.historical_frequency.entry(request.key).or_default() += 1;

        if active_roles.contains(&CacheActorRole::Resize) {
            self.apply_resize(decisions.resize);
            self.metrics.record_actor_decision(CacheActorRole::Resize);
        }

        let mut outcome = StepOutcome::default();
        let fresh_hit = self.contains_fresh(request.key);
        let stale_hit = self.contains_stale(request.key);

        if fresh_hit && !request.is_write {
            self.touch(request.key);
            self.miss_burst = 0;
            outcome.hit = true;
            outcome.latency_ms = 1.0;
            self.metrics
                .record_hit(request.size_bytes, outcome.latency_ms);
        } else {
            if stale_hit {
                outcome.stale = true;
                self.remove(request.key);
                self.metrics.record_stale_hit();
            }
            self.miss_burst += 1;
            outcome.latency_ms = request.backend_cost_ms;
            self.metrics
                .record_miss(request.size_bytes, outcome.latency_ms);

            if active_roles.contains(&CacheActorRole::Admission) {
                self.metrics
                    .record_actor_decision(CacheActorRole::Admission);
            }
            if self.should_admit(decisions.admission, request) {
                if active_roles.contains(&CacheActorRole::Ttl) {
                    self.metrics.record_actor_decision(CacheActorRole::Ttl);
                }
                let ttl = self.ttl_for_action(decisions.ttl, request);
                if ttl > 0 {
                    outcome.admitted = true;
                    self.insert(request, ttl, decisions.admission);
                }
            }
        }

        if active_roles.contains(&CacheActorRole::Prefetch) {
            self.metrics.record_actor_decision(CacheActorRole::Prefetch);
            outcome.prefetched = self.apply_prefetch(decisions.prefetch, request);
        }

        while self.used_bytes > self.capacity_bytes && !self.entries.is_empty() {
            self.metrics.record_actor_decision(CacheActorRole::Eviction);
            if let Some(victim) = self.select_victim(decisions.eviction) {
                self.remove(victim);
                self.metrics.record_eviction();
                outcome.evictions += 1;
            } else {
                break;
            }
        }

        outcome.memory_pressure_after = self.utilization();
        for role in active_roles {
            self.metrics
                .record_actor_reward(*role, reward_for_role(*role, &outcome));
        }
        if outcome.evictions > 0 && !active_roles.contains(&CacheActorRole::Eviction) {
            self.metrics.record_actor_reward(
                CacheActorRole::Eviction,
                reward_for_role(CacheActorRole::Eviction, &outcome),
            );
        }
        outcome
    }

    pub fn finalize_metrics(mut self, elapsed: std::time::Duration) -> CacheMetrics {
        self.metrics
            .finalize(self.used_bytes, self.capacity_bytes, elapsed);
        self.metrics
    }

    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    pub fn stats(&self) -> CacheStats {
        let requests = self.metrics.requests.max(1);
        CacheStats {
            clock: self.clock,
            capacity_bytes: self.capacity_bytes,
            used_bytes: self.used_bytes,
            entry_count: self.entries.len(),
            avg_entry_size: if self.entries.is_empty() {
                0
            } else {
                self.used_bytes / self.entries.len()
            },
            utilization: self.utilization(),
            hit_rate: self.metrics.hits as f32 / requests as f32,
            miss_rate: self.metrics.misses as f32 / requests as f32,
            eviction_rate: self.metrics.evictions as f32 / requests as f32,
            stale_rate: self.metrics.stale_hits as f32 / requests as f32,
            backend_fetch_rate: self.metrics.backend_fetches as f32 / requests as f32,
            miss_burst: self.miss_burst,
            resize_count: self.resize_count,
        }
    }

    pub fn contains_fresh(&self, key: Key) -> bool {
        self.entries
            .get(&key)
            .is_some_and(|entry| entry.ttl_expires_at >= self.clock)
    }

    pub fn frequency_estimate(&self, key: Key) -> u64 {
        self.historical_frequency
            .get(&key)
            .copied()
            .unwrap_or_default()
    }

    pub fn recency_estimate(&self, key: Key) -> u64 {
        self.entries
            .get(&key)
            .map(|entry| self.clock.saturating_sub(entry.last_accessed_at))
            .unwrap_or(u64::MAX)
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn clock(&self) -> u64 {
        self.clock
    }

    fn contains_stale(&self, key: Key) -> bool {
        self.entries
            .get(&key)
            .is_some_and(|entry| entry.ttl_expires_at < self.clock)
    }

    fn touch(&mut self, key: Key) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_accessed_at = self.clock;
            entry.access_count += 1;
        }
    }

    fn insert(&mut self, request: &CacheRequest, ttl: u64, admission_action: usize) {
        self.remove(request.key);
        let priority = match admission_action {
            5 => 2.0,
            3 => 1.5,
            4 => 1.25,
            _ => 1.0,
        };
        self.entries.insert(
            request.key,
            CacheEntry {
                key: request.key,
                size_bytes: request.size_bytes,
                inserted_at: self.clock,
                last_accessed_at: self.clock,
                access_count: 1,
                ttl_expires_at: self.clock + ttl,
                backend_cost_ms: request.backend_cost_ms,
                priority,
            },
        );
        self.used_bytes += request.size_bytes;
    }

    fn remove(&mut self, key: Key) {
        if let Some(entry) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
        }
    }

    fn should_admit(&self, action: usize, request: &CacheRequest) -> bool {
        match action {
            0 => false,
            1 => true,
            2 => request.size_bytes <= 2_048,
            3 => request.backend_cost_ms >= 40.0,
            4 => self.frequency_estimate(request.key) >= 2 || request.popularity_class == 0,
            5 => true,
            _ => true,
        }
    }

    fn ttl_for_action(&self, action: usize, request: &CacheRequest) -> u64 {
        match action {
            0 => 0,
            1 => 16,
            2 => 64,
            3 => 256,
            4 => 1_024,
            5 => request.ttl_hint.max(1),
            _ => request.ttl_hint.max(1),
        }
    }

    fn apply_resize(&mut self, action: usize) {
        let old = self.capacity_bytes;
        let next = match action {
            0 => old.saturating_mul(90) / 100,
            1 => old.saturating_mul(95) / 100,
            2 => old,
            3 => old.saturating_mul(105) / 100,
            4 => old.saturating_mul(110) / 100,
            5 => old.saturating_mul(75) / 100,
            _ => old,
        };
        let min_capacity = self.initial_capacity_bytes / 4;
        let max_capacity = self.initial_capacity_bytes * 4;
        self.capacity_bytes = next.clamp(min_capacity.max(1), max_capacity.max(1));
        if self.capacity_bytes != old {
            self.resize_count += 1;
        }
    }

    fn apply_prefetch(&mut self, action: usize, request: &CacheRequest) -> bool {
        if !matches!(action, 1 | 2 | 5) {
            return false;
        }
        let related_key = if action == 2 {
            request.key / 16 * 16
        } else {
            request.key.wrapping_add(1)
        };
        if self.contains_fresh(related_key) {
            return false;
        }
        let prefetch = CacheRequest {
            key: related_key,
            size_bytes: (request.size_bytes / 2).max(256),
            backend_cost_ms: request.backend_cost_ms * 0.5,
            ttl_hint: request.ttl_hint / 2 + 1,
            popularity_class: request.popularity_class,
            is_write: false,
        };
        self.insert(&prefetch, self.ttl_for_action(3, &prefetch), 1);
        true
    }

    fn select_victim(&self, action: usize) -> Option<Key> {
        match action {
            0 => self
                .entries
                .values()
                .min_by_key(|entry| entry.inserted_at)
                .map(|entry| entry.key),
            1 => self
                .entries
                .values()
                .min_by_key(|entry| entry.last_accessed_at)
                .map(|entry| entry.key),
            2 => self
                .entries
                .values()
                .min_by_key(|entry| entry.access_count)
                .map(|entry| entry.key),
            3 => self
                .entries
                .values()
                .max_by_key(|entry| entry.size_bytes)
                .map(|entry| entry.key),
            4 => self
                .entries
                .values()
                .min_by(|left, right| entry_value(left).total_cmp(&entry_value(right)))
                .map(|entry| entry.key),
            5 => self
                .entries
                .values()
                .min_by_key(|entry| entry.ttl_expires_at)
                .map(|entry| entry.key),
            _ => self.entries.keys().next().copied(),
        }
    }

    fn utilization(&self) -> f32 {
        if self.capacity_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f32 / self.capacity_bytes as f32
        }
    }
}

fn entry_value(entry: &CacheEntry) -> f32 {
    entry.backend_cost_ms * entry.access_count as f32 * entry.priority / entry.size_bytes as f32
}
