use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadKind {
    Uniform,
    Zipfian,
    Scan,
    Bursty,
    PhaseShift,
    LargeObject,
    TtlSensitive,
}

impl WorkloadKind {
    pub fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "zipf" | "zipfian" => Self::Zipfian,
            "scan" => Self::Scan,
            "bursty" | "burst" => Self::Bursty,
            "phase" | "phase-shift" => Self::PhaseShift,
            "large" | "large-object" => Self::LargeObject,
            "ttl" | "ttl-sensitive" => Self::TtlSensitive,
            _ => Self::Uniform,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheRequest {
    pub key: u64,
    pub size_bytes: usize,
    pub backend_cost_ms: f32,
    pub ttl_hint: u64,
    pub popularity_class: u8,
    pub is_write: bool,
}

#[derive(Debug, Clone)]
pub struct WorkloadGenerator {
    kind: WorkloadKind,
    rng: Lcg,
    request_index: u64,
    key_space: u64,
}

impl WorkloadGenerator {
    pub fn new(kind: WorkloadKind, seed: u64) -> Self {
        Self {
            kind,
            rng: Lcg::new(seed),
            request_index: 0,
            key_space: 2_048,
        }
    }

    pub fn next_request(&mut self) -> CacheRequest {
        let idx = self.request_index;
        self.request_index += 1;

        let key = match self.kind {
            WorkloadKind::Uniform => self.rng.range(0, self.key_space),
            WorkloadKind::Zipfian => self.zipfian_key(),
            WorkloadKind::Scan => idx % self.key_space,
            WorkloadKind::Bursty => {
                let burst_group = (idx / 200) % 8;
                if self.rng.unit() < 0.82 {
                    burst_group * 16 + self.rng.range(0, 16)
                } else {
                    self.rng.range(0, self.key_space)
                }
            }
            WorkloadKind::PhaseShift => {
                let phase = (idx / 1_000) % 4;
                let base = phase * 128;
                if self.rng.unit() < 0.76 {
                    base + self.rng.range(0, 128)
                } else {
                    self.rng.range(0, self.key_space)
                }
            }
            WorkloadKind::LargeObject => {
                if self.rng.unit() < 0.70 {
                    self.zipfian_key()
                } else {
                    10_000 + self.rng.range(0, 128)
                }
            }
            WorkloadKind::TtlSensitive => self.zipfian_key(),
        };

        let hot = key < 128;
        let large = self.kind == WorkloadKind::LargeObject && key >= 10_000;
        let size_bytes = if large {
            32_768 + self.rng.range(0, 8_192) as usize
        } else {
            512 + self.rng.range(0, 4_096) as usize
        };
        let backend_cost_ms = if hot {
            8.0
        } else {
            20.0 + self.rng.unit() * 80.0
        };
        let ttl_hint = match self.kind {
            WorkloadKind::TtlSensitive => 16 + self.rng.range(0, 64),
            WorkloadKind::Bursty => 256,
            _ => 128 + self.rng.range(0, 512),
        };

        CacheRequest {
            key,
            size_bytes,
            backend_cost_ms,
            ttl_hint,
            popularity_class: if hot {
                0
            } else if key < 512 {
                1
            } else {
                2
            },
            is_write: self.rng.unit() < 0.04,
        }
    }

    fn zipfian_key(&mut self) -> u64 {
        let roll = self.rng.unit();
        if roll < 0.55 {
            self.rng.range(0, 64)
        } else if roll < 0.82 {
            self.rng.range(64, 256)
        } else {
            self.rng.range(256, self.key_space)
        }
    }
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn unit(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        value as f32 / (1_u64 << 24) as f32
    }

    fn range(&mut self, start: u64, end: u64) -> u64 {
        let width = end.saturating_sub(start).max(1);
        start + self.next_u64() % width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_repeats_workload() {
        let mut a = WorkloadGenerator::new(WorkloadKind::Zipfian, 42);
        let mut b = WorkloadGenerator::new(WorkloadKind::Zipfian, 42);
        let left: Vec<_> = (0..32).map(|_| a.next_request()).collect();
        let right: Vec<_> = (0..32).map(|_| b.next_request()).collect();
        assert_eq!(left, right);
    }

    #[test]
    fn zipfian_workload_has_hot_key_skew() {
        let mut workload = WorkloadGenerator::new(WorkloadKind::Zipfian, 7);
        let hot = (0..1_000)
            .map(|_| workload.next_request())
            .filter(|request| request.key < 256)
            .count();
        assert!(hot > 700, "expected hot key skew, got {hot}");
    }
}
