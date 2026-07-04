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
    admission_credits: HashMap<Key, PendingAdmissionCredit>,
    eviction_credits: HashMap<Key, EvictionCredit>,
    ttl_credits: HashMap<Key, TtlCredit>,
    prefetch_credits: HashMap<Key, PrefetchCredit>,
    window_stats: WindowStats,
    clock: u64,
    miss_burst: u64,
    resize_count: u64,
    metrics: CacheMetrics,
}

#[derive(Debug, Clone)]
pub struct PendingAdmissionCredit {
    inserted_at: u64,
    backend_cost_ms: f32,
    size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct EvictionCredit {
    evicted_at: u64,
    backend_cost_ms: f32,
}

#[derive(Debug, Clone)]
pub struct TtlCredit {
    expires_at: u64,
}

#[derive(Debug, Clone)]
pub struct PrefetchCredit {
    prefetched_at: u64,
    backend_cost_ms: f32,
}

#[derive(Debug, Clone, Default)]
pub struct WindowStats {
    pub hits_at_window_start: u64,
    pub misses_at_window_start: u64,
    pub latency_at_window_start: f32,
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
    pub eviction_quality_reward: f32,
}

impl CacheWorld {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            initial_capacity_bytes: capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            historical_frequency: HashMap::new(),
            admission_credits: HashMap::new(),
            eviction_credits: HashMap::new(),
            ttl_credits: HashMap::new(),
            prefetch_credits: HashMap::new(),
            window_stats: WindowStats::default(),
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
        self.apply_delayed_miss_credits(request);

        if active_roles.contains(&CacheActorRole::Resize) {
            self.apply_resize(decisions.resize);
            self.metrics.record_actor_decision(CacheActorRole::Resize);
        }

        let mut outcome = StepOutcome::default();
        let fresh_hit = self.contains_fresh(request.key);
        let stale_hit = self.contains_stale(request.key);

        if fresh_hit && !request.is_write {
            self.touch(request.key);
            self.apply_delayed_hit_credits(request.key);
            self.miss_burst = 0;
            outcome.hit = true;
            outcome.latency_ms = 1.0;
            self.metrics
                .record_hit(request.size_bytes, outcome.latency_ms);
        } else {
            if stale_hit {
                outcome.stale = true;
                if let Some(ttl) = self.ttl_credits.remove(&request.key) {
                    let late_by = self.clock.saturating_sub(ttl.expires_at) as f32;
                    self.metrics.record_actor_reward(
                        CacheActorRole::Ttl,
                        -0.02 - late_by.min(128.0) * 0.0005,
                    );
                }
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
                if let Some(entry) = self.entries.get(&victim) {
                    outcome.eviction_quality_reward += eviction_victim_reward(entry, self.clock);
                    self.eviction_credits.insert(
                        victim,
                        EvictionCredit {
                            evicted_at: self.clock,
                            backend_cost_ms: entry.backend_cost_ms,
                        },
                    );
                }
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
        self.prune_expired_credits();
        self.update_window_stats();
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

    pub fn force_capacity_pressure(&mut self) {
        if self.used_bytes > 1 {
            self.capacity_bytes = (self.used_bytes.saturating_mul(8) / 10).max(1);
        }
    }

    pub fn force_insert_for_training(
        &mut self,
        request: &CacheRequest,
        access_count: u64,
        ttl: u64,
        priority: f32,
    ) {
        self.remove(request.key);
        let inserted_at = self.clock.saturating_sub(access_count.saturating_mul(3));
        let last_accessed_at = self.clock.saturating_sub(access_count.max(1));
        self.entries.insert(
            request.key,
            CacheEntry {
                key: request.key,
                size_bytes: request.size_bytes,
                inserted_at,
                last_accessed_at,
                access_count,
                ttl_expires_at: self.clock + ttl,
                backend_cost_ms: request.backend_cost_ms,
                priority,
            },
        );
        self.used_bytes += request.size_bytes;
        *self.historical_frequency.entry(request.key).or_default() += access_count;
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
        self.admission_credits.insert(
            request.key,
            PendingAdmissionCredit {
                inserted_at: self.clock,
                backend_cost_ms: request.backend_cost_ms,
                size_bytes: request.size_bytes,
            },
        );
        self.ttl_credits.insert(
            request.key,
            TtlCredit {
                expires_at: self.clock + ttl,
            },
        );
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
        self.prefetch_credits.insert(
            related_key,
            PrefetchCredit {
                prefetched_at: self.clock,
                backend_cost_ms: prefetch.backend_cost_ms,
            },
        );
        true
    }

    fn apply_delayed_hit_credits(&mut self, key: Key) {
        if let Some(credit) = self.admission_credits.remove(&key) {
            let age = self.clock.saturating_sub(credit.inserted_at).max(1) as f32;
            let size_penalty = credit.size_bytes as f32 / self.capacity_bytes.max(1) as f32;
            let reward =
                credit.backend_cost_ms / 100.0 - size_penalty * 0.05 + (1.0 / age).min(0.05);
            self.metrics
                .record_actor_reward(CacheActorRole::Admission, reward);
        }
        if let Some(credit) = self.prefetch_credits.remove(&key) {
            let age = self.clock.saturating_sub(credit.prefetched_at).max(1) as f32;
            let reward = credit.backend_cost_ms / 120.0 + (1.0 / age).min(0.05);
            self.metrics
                .record_actor_reward(CacheActorRole::Prefetch, reward);
        }
    }

    fn apply_delayed_miss_credits(&mut self, request: &CacheRequest) {
        if let Some(credit) = self.eviction_credits.remove(&request.key) {
            let age = self.clock.saturating_sub(credit.evicted_at);
            if age <= 256 {
                self.metrics.record_actor_reward(
                    CacheActorRole::Eviction,
                    -0.05 - credit.backend_cost_ms / 200.0,
                );
            }
        }
    }

    fn prune_expired_credits(&mut self) {
        let now = self.clock;
        self.admission_credits
            .retain(|_, credit| now.saturating_sub(credit.inserted_at) <= 512);
        self.eviction_credits
            .retain(|_, credit| now.saturating_sub(credit.evicted_at) <= 512);
        self.prefetch_credits
            .retain(|_, credit| now.saturating_sub(credit.prefetched_at) <= 256);
        self.ttl_credits
            .retain(|_, credit| credit.expires_at + 512 >= now);
    }

    fn update_window_stats(&mut self) {
        if self.clock.is_multiple_of(1_000) {
            self.window_stats = WindowStats {
                hits_at_window_start: self.metrics.hits,
                misses_at_window_start: self.metrics.misses,
                latency_at_window_start: self.metrics.total_latency_ms,
            };
        }
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

fn eviction_victim_reward(entry: &CacheEntry, clock: u64) -> f32 {
    let age = clock.saturating_sub(entry.inserted_at).min(1_024) as f32 / 1_024.0;
    let recency = clock.saturating_sub(entry.last_accessed_at).min(1_024) as f32 / 1_024.0;
    let ttl_remaining = entry.ttl_expires_at.saturating_sub(clock).min(1_024) as f32 / 1_024.0;
    let size_bonus = (entry.size_bytes as f32 / 65_536.0).min(1.0) * 0.04;
    let cold_bonus = recency * 0.06 + age * 0.02;
    let hot_penalty = (entry.access_count.min(32) as f32 / 32.0) * 0.08;
    let cost_penalty = (entry.backend_cost_ms / 100.0).min(1.0) * 0.04;
    let ttl_penalty = ttl_remaining * 0.02;
    size_bonus + cold_bonus - hot_penalty - cost_penalty - ttl_penalty
}
