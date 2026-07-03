use crate::host::{BenchmarkResult, actor_activity_lines};

pub fn print_actor_activity(result: &BenchmarkResult) {
    println!();
    println!("Actor activity for {}:", result.policy);
    for line in actor_activity_lines(&result.metrics) {
        println!("  {line}");
    }
}

pub fn print_json(results: &[BenchmarkResult]) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(results)?);
    Ok(())
}
