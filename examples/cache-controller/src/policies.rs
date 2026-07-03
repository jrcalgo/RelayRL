use crate::actors::{ActorDecisions, CacheActorRole, should_trigger};
use crate::cache::CacheWorld;
use crate::heuristics::{HeuristicController, PolicyKind};
use crate::workload::CacheRequest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub trait CacheControlPolicy {
    fn decide(
        &mut self,
        role: CacheActorRole,
        world: &CacheWorld,
        request: &CacheRequest,
    ) -> Option<usize>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearnedRolePolicy {
    Adaptive,
    Fixed(usize),
    Neural {
        model_dir: PathBuf,
        role: CacheActorRole,
    },
}

impl CacheControlPolicy for LearnedRolePolicy {
    fn decide(
        &mut self,
        role: CacheActorRole,
        world: &CacheWorld,
        request: &CacheRequest,
    ) -> Option<usize> {
        match self {
            Self::Adaptive => {
                let mut controller = HeuristicController::new(PolicyKind::RelayRlAdaptive, 0);
                Some(controller.decide(world, request).get(role))
            }
            Self::Fixed(action) => Some(*action),
            Self::Neural { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeuristicPolicySet {
    controller: HeuristicController,
}

impl HeuristicPolicySet {
    pub fn new(policy: PolicyKind, seed: u64) -> Self {
        Self {
            controller: HeuristicController::new(policy, seed),
        }
    }

    pub fn decide_all(&mut self, world: &CacheWorld, request: &CacheRequest) -> ActorDecisions {
        self.controller.decide(world, request)
    }

    pub fn active_roles(&self, world: &CacheWorld, request: &CacheRequest) -> Vec<CacheActorRole> {
        self.controller.active_roles(world, request)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrozenActorPolicySet {
    policies: BTreeMap<CacheActorRole, LearnedRolePolicy>,
}

impl FrozenActorPolicySet {
    pub fn insert(&mut self, role: CacheActorRole, policy: LearnedRolePolicy) {
        self.policies.insert(role, policy);
    }

    pub fn contains(&self, role: CacheActorRole) -> bool {
        self.policies.contains_key(&role)
    }

    pub fn policies(&self) -> &BTreeMap<CacheActorRole, LearnedRolePolicy> {
        &self.policies
    }
}

#[derive(Debug, Clone)]
pub struct MixedPolicySet {
    baseline: PolicyKind,
    seed: u64,
    frozen: FrozenActorPolicySet,
}

impl MixedPolicySet {
    pub fn new(baseline: PolicyKind, seed: u64) -> Self {
        Self {
            baseline,
            seed,
            frozen: FrozenActorPolicySet::default(),
        }
    }

    pub fn with_frozen(baseline: PolicyKind, seed: u64, frozen: FrozenActorPolicySet) -> Self {
        Self {
            baseline,
            seed,
            frozen,
        }
    }

    pub fn freeze_role(&mut self, role: CacheActorRole, policy: LearnedRolePolicy) {
        self.frozen.insert(role, policy);
    }

    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn frozen(&self) -> &FrozenActorPolicySet {
        &self.frozen
    }

    pub fn decide_all(&mut self, world: &CacheWorld, request: &CacheRequest) -> ActorDecisions {
        let mut heuristic = HeuristicController::new(self.baseline, self.seed);
        let mut decisions = heuristic.decide(world, request);
        for (role, policy) in self.frozen.policies.clone() {
            let mut policy = policy;
            if let Some(action) = policy.decide(role, world, request) {
                decisions.set(role, action);
            }
        }
        decisions
    }

    pub fn active_roles(&self, world: &CacheWorld, request: &CacheRequest) -> Vec<CacheActorRole> {
        CacheActorRole::ALL
            .iter()
            .copied()
            .filter(|role| should_trigger(*role, world, request))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::{WorkloadGenerator, WorkloadKind};

    #[test]
    fn mixed_policy_overrides_only_frozen_role() {
        let world = CacheWorld::new(1024);
        let request = WorkloadGenerator::new(WorkloadKind::Uniform, 1).next_request();
        let mut mixed = MixedPolicySet::new(PolicyKind::Lru, 1);
        let baseline = mixed.decide_all(&world, &request);
        mixed.freeze_role(CacheActorRole::Admission, LearnedRolePolicy::Fixed(0));
        let overridden = mixed.decide_all(&world, &request);
        assert_eq!(overridden.admission, 0);
        assert_eq!(overridden.eviction, baseline.eviction);
    }
}
