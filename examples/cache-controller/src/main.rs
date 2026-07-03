use cache_controller_example::actors::{ACTOR_SPECS, CacheActorRole};
use cache_controller_example::benchmark::{print_actor_activity, print_json};
use cache_controller_example::heuristics::PolicyKind;
use cache_controller_example::host::{
    BenchmarkConfig, compare_policies, print_results_table, run_benchmark,
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
        };

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--bench" => args.mode = "bench".to_string(),
                "--compare" => args.mode = "compare".to_string(),
                "--train" => args.mode = "train".to_string(),
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
    println!("  --workload <uniform|zipfian|scan|bursty|phase|large|ttl>");
    println!("  --requests <n>");
    println!("  --seed <n>");
    println!("  --capacity-bytes <n>");
    println!("  --output-json");
}
