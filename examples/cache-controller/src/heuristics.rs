use crate::actors::{ActorDecisions, CacheActorRole, should_trigger};
use crate::cache::CacheWorld;
use crate::workload::CacheRequest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyKind {
    Fifo,
    Lru,
    Lfu,
    Random,
    StaticTtl,
    ThresholdResize,
    RelayRlAdaptive,
}

impl PolicyKind {
    pub fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "fifo" => Self::Fifo,
            "lfu" => Self::Lfu,
            "random" => Self::Random,
            "static-ttl" | "ttl" => Self::StaticTtl,
            "threshold-resize" | "resize" => Self::ThresholdResize,
            "relayrl" | "adaptive" | "relayrl-adaptive" => Self::RelayRlAdaptive,
            _ => Self::Lru,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fifo => "FIFO",
            Self::Lru => "LRU",
            Self::Lfu => "LFU",
            Self::Random => "Random",
            Self::StaticTtl => "StaticTTL",
            Self::ThresholdResize => "ThresholdResize",
            Self::RelayRlAdaptive => "RelayRL",
        }
    }

    pub const BASELINES: [Self; 6] = [
        Self::Fifo,
        Self::Lru,
        Self::Lfu,
        Self::Random,
        Self::StaticTtl,
        Self::ThresholdResize,
    ];
}

#[derive(Debug, Clone)]
pub struct HeuristicController {
    policy: PolicyKind,
    random_state: u64,
}

impl HeuristicController {
    pub fn new(policy: PolicyKind, seed: u64) -> Self {
        Self {
            policy,
            random_state: seed ^ 0xa5a5_5a5a_f00d_f00d,
        }
    }

    pub fn policy(&self) -> PolicyKind {
        self.policy
    }

    pub fn decide(&mut self, world: &CacheWorld, request: &CacheRequest) -> ActorDecisions {
        match self.policy {
            PolicyKind::Fifo => ActorDecisions {
                admission: 1,
                eviction: 0,
                ttl: 3,
                resize: 2,
                prefetch: 0,
            },
            PolicyKind::Lru => ActorDecisions {
                admission: 1,
                eviction: 1,
                ttl: 3,
                resize: 2,
                prefetch: 0,
            },
            PolicyKind::Lfu => ActorDecisions {
                admission: 4,
                eviction: 2,
                ttl: 4,
                resize: 2,
                prefetch: 0,
            },
            PolicyKind::Random => ActorDecisions {
                admission: 1,
                eviction: self.next_action(),
                ttl: 3,
                resize: 2,
                prefetch: 0,
            },
            PolicyKind::StaticTtl => ActorDecisions {
                admission: 1,
                eviction: 1,
                ttl: 2,
                resize: 2,
                prefetch: 0,
            },
            PolicyKind::ThresholdResize => {
                let utilization = world.stats().utilization;
                ActorDecisions {
                    admission: 1,
                    eviction: 1,
                    ttl: 3,
                    resize: if utilization > 0.92 {
                        4
                    } else if utilization < 0.35 {
                        1
                    } else {
                        2
                    },
                    prefetch: 0,
                }
            }
            PolicyKind::RelayRlAdaptive => self.adaptive_decisions(world, request),
        }
    }

    pub fn active_roles(&self, world: &CacheWorld, request: &CacheRequest) -> Vec<CacheActorRole> {
        CacheActorRole::ALL
            .iter()
            .copied()
            .filter(|role| should_trigger(*role, world, request))
            .collect()
    }

    fn adaptive_decisions(&mut self, world: &CacheWorld, request: &CacheRequest) -> ActorDecisions {
        let stats = world.stats();
        ActorDecisions {
            admission: if stats.utilization > 0.90 && request.size_bytes > 16_384 {
                0
            } else if request.backend_cost_ms > 50.0 {
                3
            } else if world.frequency_estimate(request.key) >= 2 || request.popularity_class == 0 {
                4
            } else if request.size_bytes <= 2_048 {
                2
            } else {
                1
            },
            eviction: if stats.utilization > 1.20 {
                3
            } else if stats.miss_burst > 16 {
                4
            } else {
                1
            },
            ttl: if stats.utilization > 0.90 {
                2
            } else if request.popularity_class == 0 {
                4
            } else {
                3
            },
            resize: if stats.utilization > 0.95 && stats.miss_rate > 0.35 {
                4
            } else if stats.utilization < 0.35 {
                1
            } else {
                2
            },
            prefetch: if stats.miss_burst > 12 && request.popularity_class == 0 {
                1
            } else {
                0
            },
        }
    }

    fn next_action(&mut self) -> usize {
        self.random_state = self
            .random_state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (self.random_state % crate::ACTION_DIM as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::{WorkloadGenerator, WorkloadKind};

    #[test]
    fn lru_policy_uses_lru_eviction_action() {
        let world = CacheWorld::new(1024);
        let request = WorkloadGenerator::new(WorkloadKind::Uniform, 1).next_request();
        let mut controller = HeuristicController::new(PolicyKind::Lru, 1);
        assert_eq!(controller.decide(&world, &request).eviction, 1);
    }

    #[test]
    fn random_policy_is_seed_deterministic() {
        let world = CacheWorld::new(1024);
        let request = WorkloadGenerator::new(WorkloadKind::Uniform, 1).next_request();
        let mut left = HeuristicController::new(PolicyKind::Random, 99);
        let mut right = HeuristicController::new(PolicyKind::Random, 99);
        let l: Vec<_> = (0..16)
            .map(|_| left.decide(&world, &request).eviction)
            .collect();
        let r: Vec<_> = (0..16)
            .map(|_| right.decide(&world, &request).eviction)
            .collect();
        assert_eq!(l, r);
    }
}
