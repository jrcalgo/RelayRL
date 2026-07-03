# RelayRL Adaptive Cache Controller Example

This example demonstrates a benchmarkable systems-control problem for RelayRL:
a shared cache environment with multiple specialized control actors.

Unlike the OpenTTD example, this environment is fully owned by the repository,
deterministic, CI-friendly, and comparable against classic heuristics.

## Actor graph

Actors are independent control processes. They do not all act on every request.

| Actor | Trigger frequency | Represents |
| --- | --- | --- |
| Admission | cache miss | learned cache admission policy |
| Eviction | capacity pressure | learned eviction strategy |
| TTL | admission / refresh / write | learned TTL assignment |
| Resize | slow request window | learned cache autoscaler |
| Prefetch / backpressure | miss burst / backend pressure / window | learned burst controller |

The host owns the cache world, requests decisions only from actors whose trigger
fires, validates actions, applies cache updates, and records per-actor rewards.

## Baselines

The example includes:

- FIFO
- LRU
- LFU
- Random
- Static TTL
- Threshold resize
- RelayRL-style adaptive controller

## Metrics

Benchmarks report:

- hit rate
- byte hit rate
- average / p95 / p99 latency
- evictions
- backend fetches
- reward
- `total_env_steps_per_second`
- `total_request_action_throughput_per_second`
- per-actor decision counts

## Commands

Compare all policies:

```bash
cargo run -p cache-controller-example -- --compare --requests 100000 --seed 42
```

Run one baseline:

```bash
cargo run -p cache-controller-example -- --bench --policy lru --workload zipfian
```

Exercise the RelayRL-compatible training environment for one actor role:

```bash
cargo run -p cache-controller-example -- --train --role admission
```

Run sequential train/freeze/rollback over all actor roles:

```bash
cargo run -p cache-controller-example -- --train-and-compare --requests 25000 --seed 42
```

This trains candidate role policies in order:

```text
Admission -> Eviction -> TTL -> Resize -> Prefetch
```

Each candidate is evaluated against the current frozen policy set plus heuristic
background policies. Candidates that do not clear `--min-improvement` are
rejected and the heuristic remains active for that role.

Reports are written to:

```text
target/cache-controller/
├── training-report.json
├── final-eval.json
├── final-eval.csv
├── models/
└── logs/
```

Emit JSON:

```bash
cargo run -p cache-controller-example -- --compare --output-json
```

## Design note

This example intentionally separates the shared benchmark host from
`CacheTrainingEnvironment`. The host demonstrates irregular multi-actor control
frequencies over one shared system. The training environment wraps that same
cache simulator for one active actor role at a time, matching RelayRL's current
strongest PPO path: sequential train-and-freeze of specialized actors.

## Sample staged result

On a short smoke run:

```bash
cargo run -p cache-controller-example -- --train-and-compare --requests 750 --seed 42
```

the learned composed policy improved over LRU on the final Zipfian evaluation:

```text
LRU              HitRate 0.442  AvgLat 21.11ms  Reward   -4.168
RelayRL-Learned HitRate 0.532  AvgLat 19.33ms  Reward  597.946
```

Actor activity for that run:

```text
admission                  11186
eviction                   10983
ttl                        11186
resize                        24
prefetch-backpressure        202
```
