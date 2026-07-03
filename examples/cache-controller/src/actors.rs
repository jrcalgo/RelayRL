use crate::cache::{CacheWorld, StepOutcome};
use crate::workload::CacheRequest;
use crate::{ACTION_DIM, OBSERVATION_DIM};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CacheActorRole {
    Admission,
    Eviction,
    Ttl,
    Resize,
    Prefetch,
}

impl CacheActorRole {
    pub const ALL: [Self; 5] = [
        Self::Admission,
        Self::Eviction,
        Self::Ttl,
        Self::Resize,
        Self::Prefetch,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admission => "admission",
            Self::Eviction => "eviction",
            Self::Ttl => "ttl",
            Self::Resize => "resize",
            Self::Prefetch => "prefetch-backpressure",
        }
    }
}

impl std::fmt::Display for CacheActorRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActorSpec {
    pub role: CacheActorRole,
    pub trigger: &'static str,
    pub represents: &'static str,
    pub optimizes: &'static str,
    pub action_labels: [&'static str; ACTION_DIM],
}

pub const ACTOR_SPECS: [ActorSpec; 5] = [
    ActorSpec {
        role: CacheActorRole::Admission,
        trigger: "cache miss",
        represents: "learned cache admission policy",
        optimizes: "future reuse value minus memory pressure",
        action_labels: [
            "reject",
            "admit",
            "admit-small",
            "admit-high-cost",
            "admit-frequent",
            "admit-high-priority",
        ],
    },
    ActorSpec {
        role: CacheActorRole::Eviction,
        trigger: "capacity pressure",
        represents: "learned eviction strategy",
        optimizes: "future miss avoidance minus churn",
        action_labels: [
            "evict-oldest",
            "evict-lru",
            "evict-lfu",
            "evict-largest",
            "evict-lowest-value",
            "evict-shortest-ttl",
        ],
    },
    ActorSpec {
        role: CacheActorRole::Ttl,
        trigger: "admit/refresh/write",
        represents: "learned TTL assignment policy",
        optimizes: "useful reuse minus stale and memory lockup penalties",
        action_labels: [
            "no-cache-ttl",
            "very-short",
            "short",
            "medium",
            "long",
            "refresh-existing",
        ],
    },
    ActorSpec {
        role: CacheActorRole::Resize,
        trigger: "slow request window",
        represents: "learned cache autoscaler",
        optimizes: "latency and hit-rate gain minus memory cost",
        action_labels: [
            "shrink-10",
            "shrink-5",
            "hold",
            "grow-5",
            "grow-10",
            "emergency-shrink",
        ],
    },
    ActorSpec {
        role: CacheActorRole::Prefetch,
        trigger: "miss burst/backend pressure/window",
        represents: "learned burst and backpressure controller",
        optimizes: "latency reduction minus wasted backend load",
        action_labels: [
            "no-prefetch",
            "prefetch-related",
            "prefetch-hot-group",
            "throttle-low-priority",
            "bypass-cache",
            "prefetch-and-throttle",
        ],
    },
];

#[derive(Debug, Clone, Copy, Default)]
pub struct ActorDecisions {
    pub admission: usize,
    pub eviction: usize,
    pub ttl: usize,
    pub resize: usize,
    pub prefetch: usize,
}

impl ActorDecisions {
    pub fn set(&mut self, role: CacheActorRole, action: usize) {
        let action = action.min(ACTION_DIM - 1);
        match role {
            CacheActorRole::Admission => self.admission = action,
            CacheActorRole::Eviction => self.eviction = action,
            CacheActorRole::Ttl => self.ttl = action,
            CacheActorRole::Resize => self.resize = action,
            CacheActorRole::Prefetch => self.prefetch = action,
        }
    }

    pub fn get(&self, role: CacheActorRole) -> usize {
        match role {
            CacheActorRole::Admission => self.admission,
            CacheActorRole::Eviction => self.eviction,
            CacheActorRole::Ttl => self.ttl,
            CacheActorRole::Resize => self.resize,
            CacheActorRole::Prefetch => self.prefetch,
        }
    }
}

pub fn action_from_bytes(action: &[u8]) -> usize {
    action
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index.min(ACTION_DIM - 1))
        .unwrap_or(0)
}

pub fn action_mask_bytes() -> Vec<u8> {
    f32_slice_to_bytes(&[1.0; ACTION_DIM])
}

pub fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

pub fn observation_for(
    world: &CacheWorld,
    request: &CacheRequest,
    role: CacheActorRole,
) -> [f32; OBSERVATION_DIM] {
    let stats = world.stats();
    let mut obs = [0.0; OBSERVATION_DIM];
    obs[0] = request.size_bytes as f32 / 65_536.0;
    obs[1] = request.backend_cost_ms / 100.0;
    obs[2] = request.ttl_hint as f32 / 1_000.0;
    obs[3] = request.popularity_class as f32 / 3.0;
    obs[4] = if request.is_write { 1.0 } else { 0.0 };
    obs[5] = stats.utilization;
    obs[6] = stats.hit_rate;
    obs[7] = stats.miss_rate;
    obs[8] = stats.eviction_rate;
    obs[9] = stats.backend_fetch_rate;
    obs[10] = stats.entry_count as f32 / 10_000.0;
    obs[11] = stats.avg_entry_size as f32 / 65_536.0;
    obs[12] = stats.clock as f32 / 100_000.0;
    obs[13] = role_index(role) as f32 / CacheActorRole::ALL.len() as f32;
    obs[14] = world.frequency_estimate(request.key).min(64) as f32 / 64.0;
    obs[15] = world.recency_estimate(request.key).min(10_000) as f32 / 10_000.0;
    obs[16] = stats.miss_burst as f32 / 64.0;
    obs[17] = if world.contains_fresh(request.key) {
        1.0
    } else {
        0.0
    };
    obs[18] = request.key as f32 % 997.0 / 997.0;
    obs[19] = stats.capacity_bytes as f32 / 10_000_000.0;
    obs[20] = stats.used_bytes as f32 / 10_000_000.0;
    obs[21] = stats.stale_rate;
    obs[22] = stats.resize_count as f32 / 1_000.0;
    obs[23] = 1.0;
    obs
}

pub fn reward_for_role(role: CacheActorRole, outcome: &StepOutcome) -> f32 {
    match role {
        CacheActorRole::Admission => {
            if outcome.hit {
                0.03
            } else if outcome.admitted {
                0.02 - outcome.memory_pressure_after * 0.03
            } else {
                -0.01
            }
        }
        CacheActorRole::Eviction => {
            if outcome.evictions > 0 {
                0.02 - outcome.latency_ms / 1_000.0
            } else {
                0.0
            }
        }
        CacheActorRole::Ttl => {
            if outcome.stale {
                -0.08
            } else if outcome.admitted {
                0.02
            } else {
                0.0
            }
        }
        CacheActorRole::Resize => {
            -outcome.memory_pressure_after * 0.01 + outcome.hit as u8 as f32 * 0.01
        }
        CacheActorRole::Prefetch => {
            if outcome.prefetched {
                -0.005
            } else {
                0.0
            }
        }
    }
}

pub fn should_trigger(role: CacheActorRole, world: &CacheWorld, request: &CacheRequest) -> bool {
    match role {
        CacheActorRole::Admission => !world.contains_fresh(request.key),
        CacheActorRole::Eviction => world.used_bytes() > world.capacity_bytes(),
        CacheActorRole::Ttl => !world.contains_fresh(request.key),
        CacheActorRole::Resize => world.clock() > 0 && world.clock().is_multiple_of(1_000),
        CacheActorRole::Prefetch => {
            world.stats().miss_burst >= 8 || world.clock().is_multiple_of(500)
        }
    }
}

fn role_index(role: CacheActorRole) -> usize {
    CacheActorRole::ALL
        .iter()
        .position(|candidate| *candidate == role)
        .unwrap_or_default()
}
