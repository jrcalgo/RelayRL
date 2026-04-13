// #[allow(non_snake_case)]
// pub mod DDPG;
pub mod onnx_builder;
#[allow(non_snake_case)]
pub mod PPO;
#[allow(non_snake_case)]
pub mod REINFORCE;
// #[allow(non_snake_case)]
// pub mod TD3;

// pub use DDPG::DDPGAlgorithm;
// pub use TD3::TD3Algorithm;

pub use PPO::IPPOAlgorithm;
pub use PPO::IndependentPPOAlgorithm;
pub use PPO::MAPPOAlgorithm;
pub use PPO::MultiagentPPOAlgorithm;
pub use PPO::PPOAlgorithm;

pub use REINFORCE::IREINFORCEAlgorithm;
pub use REINFORCE::IndependentReinforceAlgorithm;
pub use REINFORCE::MAREINFORCEAlgorithm;
pub use REINFORCE::MultiagentReinforceAlgorithm;
pub use REINFORCE::ReinforceAlgorithm;

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
