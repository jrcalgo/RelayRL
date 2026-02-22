#[allow(non_snake_case)]
pub mod DDPG;
#[allow(non_snake_case)]
pub mod REINFORCE;
#[allow(non_snake_case)]
pub mod PPO;
#[allow(non_snake_case)]
pub mod TD3;

pub(crate) fn discounted_cumsum(x: &[f32], discount: f32) -> Vec<f32> {
    let n = x.len();
    let mut result = vec![0.0f32; n];
    let mut running = 0.0f32;
    for i in (0..n).rev() {
        running = x[i] + discount * running;
        result[i] = running;
    }
    result
}

pub(crate) fn scalar_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, variance.sqrt())
}

pub(crate) fn compute_normed_advantages(advantages: &[f32], mean: f32, std: f32) -> Vec<f32> {
    advantages.iter().map(|a| (a - mean) / std).collect()
}