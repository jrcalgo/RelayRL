use cache_controller_example::actors::{ACTOR_SPECS, CacheActorRole};
use cache_controller_example::benchmark::{print_actor_activity, print_json};
use cache_controller_example::heuristics::PolicyKind;
use cache_controller_example::host::{
    BenchmarkConfig, compare_policies, print_results_table, run_benchmark,
};
use cache_controller_example::staged_training::{
    StagedTrainingConfig, default_staged_training_config, evaluate_final_policy_set,
    train_actor_sequence,
};
use cache_controller_example::training::run_training_smoke;
use cache_controller_example::workload::WorkloadKind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.mode.as_str() {
        "compare" => {
            let results = compare_policies(args.config());
            print_results_table(&results);
            if let Some(relayrl) = results.iter().find(|result| result.policy == "RelayRL") {
                print_actor_activity(relayrl);
            }
            if args.output_json {
                print_json(&results)?;
            }
        }
        "train-sequential" | "train-and-compare" => {
            let training_config = args.staged_config();
            println!("Sequential cache actor training");
            println!(
                "baseline={:?} min_improvement={:.4} requests_per_phase={}",
                training_config.baseline_policy,
                training_config.min_improvement,
                training_config
                    .phases
                    .first()
                    .map(|phase| phase.requests_per_episode)
                    .unwrap_or_default()
            );
            let report = train_actor_sequence(training_config);
            println!();
            println!(
                "{:<22} {:>10} {:>12} {:>12} {:>12} {:>10}",
                "Phase", "Accepted", "Baseline", "Candidate", "Improve", "Policy"
            );
            for phase in &report.phases {
                println!(
                    "{:<22} {:>10} {:>12.3} {:>12.3} {:>12.3} {:>10?}",
                    phase.role,
                    phase.accepted,
                    phase.baseline_score,
                    phase.candidate_score,
                    phase.improvement,
                    phase.selected_policy
                );
            }
            println!();
            println!("Final evaluation:");
            print_results_table(&report.final_eval);
            if let Some(learned) = report
                .final_eval
                .iter()
                .find(|result| result.policy == "RelayRL-Learned")
            {
                print_actor_activity(learned);
            }
            if args.output_json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        }
        "evaluate-learned" => {
            let training_config = args.staged_config();
            let frozen = Default::default();
            let results = evaluate_final_policy_set(
                training_config.baseline_policy,
                training_config.capacity_bytes,
                &frozen,
            );
            print_results_table(&results);
        }
        "train" => {
            println!("RelayRL cache-controller training environment smoke");
            println!("Actors use different trigger frequencies:");
            for spec in ACTOR_SPECS {
                println!(
                    "  {:<22} trigger={:<30} optimizes={}",
                    spec.role, spec.trigger, spec.optimizes
                );
            }
            let role = args.role.unwrap_or(CacheActorRole::Admission);
            let result = run_training_smoke(role, args.config());
            println!();
            println!("Training smoke result:");
            println!("  role: {}", result.role);
            println!("  steps_attempted: {}", result.steps);
            println!("  total_reward: {:.4}", result.total_reward);
            println!("  observation_bytes: {}", result.observation_bytes);
            println!("  mask_bytes: {}", result.mask_bytes);
            println!("  done: {}", result.done);
        }
        _ => {
            let result = run_benchmark(args.config());
            print_results_table(std::slice::from_ref(&result));
            print_actor_activity(&result);
            if args.output_json {
                print_json(&[result])?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    mode: String,
    policy: PolicyKind,
    workload: WorkloadKind,
    requests: u64,
    seed: u64,
    capacity_bytes: usize,
    output_json: bool,
    role: Option<CacheActorRole>,
    output_dir: String,
    min_improvement: f64,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            mode: "bench".to_string(),
            policy: PolicyKind::Lru,
            workload: WorkloadKind::Zipfian,
            requests: 25_000,
            seed: 42,
            capacity_bytes: 512 * 1024,
            output_json: false,
            role: None,
            output_dir: "target/cache-controller".to_string(),
            min_improvement: 0.01,
        };

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--bench" => args.mode = "bench".to_string(),
                "--compare" => args.mode = "compare".to_string(),
                "--train" => args.mode = "train".to_string(),
                "--train-sequential" => args.mode = "train-sequential".to_string(),
                "--evaluate-learned" => args.mode = "evaluate-learned".to_string(),
                "--train-and-compare" => args.mode = "train-and-compare".to_string(),
                "--policy" => {
                    if let Some(value) = iter.next() {
                        args.policy = PolicyKind::parse(&value);
                    }
                }
                "--workload" => {
                    if let Some(value) = iter.next() {
                        args.workload = WorkloadKind::parse(&value);
                    }
                }
                "--requests" => {
                    if let Some(value) = iter.next() {
                        args.requests = value.parse().unwrap_or(args.requests);
                    }
                }
                "--seed" => {
                    if let Some(value) = iter.next() {
                        args.seed = value.parse().unwrap_or(args.seed);
                    }
                }
                "--capacity-bytes" => {
                    if let Some(value) = iter.next() {
                        args.capacity_bytes = value.parse().unwrap_or(args.capacity_bytes);
                    }
                }
                "--output-json" => args.output_json = true,
                "--output-dir" => {
                    if let Some(value) = iter.next() {
                        args.output_dir = value;
                    }
                }
                "--min-improvement" => {
                    if let Some(value) = iter.next() {
                        args.min_improvement = value.parse().unwrap_or(args.min_improvement);
                    }
                }
                "--role" => {
                    if let Some(value) = iter.next() {
                        args.role = parse_role(&value);
                    }
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {}
            }
        }

        args
    }

    fn config(&self) -> BenchmarkConfig {
        BenchmarkConfig {
            policy: self.policy,
            workload: self.workload,
            requests: self.requests,
            seed: self.seed,
            capacity_bytes: self.capacity_bytes,
        }
    }

    fn staged_config(&self) -> StagedTrainingConfig {
        let mut config = default_staged_training_config();
        config.baseline_policy = self.policy;
        config.capacity_bytes = self.capacity_bytes;
        config.output_dir = self.output_dir.clone().into();
        config.min_improvement = self.min_improvement;
        for phase in &mut config.phases {
            phase.requests_per_episode = self.requests;
            phase.eval_seeds = vec![
                self.seed,
                self.seed.wrapping_add(1),
                self.seed.wrapping_add(2),
            ];
            phase.train_seeds = vec![self.seed];
        }
        config
    }
}

fn parse_role(value: &str) -> Option<CacheActorRole> {
    match value.to_ascii_lowercase().as_str() {
        "admission" => Some(CacheActorRole::Admission),
        "eviction" => Some(CacheActorRole::Eviction),
        "ttl" => Some(CacheActorRole::Ttl),
        "resize" => Some(CacheActorRole::Resize),
        "prefetch" | "prefetch-backpressure" => Some(CacheActorRole::Prefetch),
        _ => None,
    }
}

fn print_help() {
    println!("cache-controller-example");
    println!("  --bench --policy <lru|lfu|fifo|random|relayrl>");
    println!("  --compare");
    println!("  --train --role <admission|eviction|ttl|resize|prefetch>");
    println!("  --train-sequential");
    println!("  --evaluate-learned");
    println!("  --train-and-compare");
    println!("  --workload <uniform|zipfian|scan|bursty|phase|large|ttl>");
    println!("  --requests <n>");
    println!("  --seed <n>");
    println!("  --capacity-bytes <n>");
    println!("  --output-json");
    println!("  --output-dir <path>");
    println!("  --min-improvement <score>");
}
