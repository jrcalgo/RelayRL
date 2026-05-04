use crate::network::client::agent::LocalTrajectoryFileParams;
pub use crate::utilities::config_json::*;

use relayrl_algorithms::{
    DDPGParams, IDDPGParams, IPPOParams, IREINFORCEParams, ITD3Params, MADDPGParams, MAPPOParams,
    MAREINFORCEParams, MATD3Params, PPOParams, REINFORCEParams, TD3Params,
};
use relayrl_types::HyperparameterArgs;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{fs::File, io::Read, path::PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub enum Algorithm {
    PPO,
    IPPO,
    MAPPO,
    REINFORCE,
    IREINFORCE,
    MAREINFORCE,
    DDPG,
    IDDPG,
    MADDPG,
    TD3,
    ITD3,
    MATD3,
    // CUSTOM(String),
    ConfigInit,
}

impl Algorithm {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "DDPG" => Some(Algorithm::DDPG),
            "IDDPG" => Some(Algorithm::IDDPG),
            "MADDPG" => Some(Algorithm::MADDPG),
            "PPO" => Some(Algorithm::PPO),
            "IPPO" => Some(Algorithm::IPPO),
            "MAPPO" => Some(Algorithm::MAPPO),
            "REINFORCE" => Some(Algorithm::REINFORCE),
            "IREINFORCE" => Some(Algorithm::IREINFORCE),
            "MAREINFORCE" => Some(Algorithm::MAREINFORCE),
            "TD3" => Some(Algorithm::TD3),
            "ITD3" => Some(Algorithm::ITD3),
            "MATD3" => Some(Algorithm::MATD3),
            "CONFIG_INIT" => Some(Algorithm::ConfigInit),
            // _ => Some(Algorithm::CUSTOM(s.to_string())),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Algorithm::DDPG => "DDPG",
            Algorithm::IDDPG => "IDDPG",
            Algorithm::MADDPG => "MADDPG",
            Algorithm::PPO => "PPO",
            Algorithm::IPPO => "IPPO",
            Algorithm::MAPPO => "MAPPO",
            Algorithm::REINFORCE => "REINFORCE",
            Algorithm::IREINFORCE => "IREINFORCE",
            Algorithm::MAREINFORCE => "MAREINFORCE",
            Algorithm::TD3 => "TD3",
            Algorithm::ITD3 => "ITD3",
            Algorithm::MATD3 => "MATD3",
            // Algorithm::CUSTOM(custom) => custom,
            Algorithm::ConfigInit => "CONFIG_INIT",
        }
    }
}

/// Configuration parameters for various algorithms.
///
/// Each field is optional and holds algorithm-specific parameters.
///
/// In a future edition, this struct will be useful when multiple algorithm init is supported.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HyperparameterConfig {
    #[serde(rename = "DDPG")]
    pub ddpg: Option<relayrl_algorithms::DDPGParams>,
    #[serde(rename = "IDDPG")]
    pub iddpg: Option<relayrl_algorithms::IDDPGParams>,
    #[serde(rename = "MADDPG")]
    pub maddpg: Option<relayrl_algorithms::MADDPGParams>,
    #[serde(rename = "PPO")]
    pub ppo: Option<relayrl_algorithms::PPOParams>,
    #[serde(rename = "IPPO")]
    pub ippo: Option<relayrl_algorithms::IPPOParams>,
    #[serde(rename = "MAPPO")]
    pub mappo: Option<relayrl_algorithms::MAPPOParams>,
    #[serde(rename = "REINFORCE")]
    pub reinforce: Option<relayrl_algorithms::REINFORCEParams>,
    #[serde(rename = "IREINFORCE")]
    pub ireinforce: Option<relayrl_algorithms::IREINFORCEParams>,
    #[serde(rename = "MAREINFORCE")]
    pub mareinforce: Option<relayrl_algorithms::MAREINFORCEParams>,
    #[serde(rename = "TD3")]
    pub td3: Option<relayrl_algorithms::TD3Params>,
    #[serde(rename = "ITD3")]
    pub itd3: Option<relayrl_algorithms::ITD3Params>,
    #[serde(rename = "MATD3")]
    pub matd3: Option<relayrl_algorithms::MATD3Params>,
    // Add other fields later for in-house algorithms
    // #[serde(rename = "custom")]
    // pub custom: Option<CustomAlgorithmParams>,
}

// TODO: desperate need for refactor to align with AlgorithmCfg and proper Custom setup instructions"
impl HyperparameterConfig {
    /// Converts the hyperparameter config to a map of algorithm names to hyperparameter arguments.
    ///
    /// If a specific algorithm is provided, only the hyperparameters for that algorithm are returned.
    ///
    /// Otherwise, all hyperparameters for all loaded algorithms are returned.
    pub fn to_args(&self, algorithm: Option<&Algorithm>) -> HashMap<Algorithm, HyperparameterArgs> {
        match algorithm {
            Some(algo) => match algo {
                Algorithm::DDPG => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(ddpg) = &self.ddpg {
                        let mut map = HashMap::new();
                        map.insert("gamma".to_string(), ddpg.gamma.to_string());
                        map.insert("tau".to_string(), ddpg.tau.to_string());
                        map.insert("actor_lr".to_string(), ddpg.actor_lr.to_string());
                        map.insert("critic_lr".to_string(), ddpg.critic_lr.to_string());
                        map.insert("batch_size".to_string(), ddpg.batch_size.to_string());
                        map.insert("buffer_size".to_string(), ddpg.buffer_size.to_string());
                        map.insert(
                            "learning_starts".to_string(),
                            ddpg.learning_starts.to_string(),
                        );
                        map.insert(
                            "policy_frequency".to_string(),
                            ddpg.policy_frequency.to_string(),
                        );
                        map.insert("noise_scale".to_string(), ddpg.noise_scale.to_string());
                        map.insert("train_iters".to_string(), ddpg.train_iters.to_string());
                        args.insert(Algorithm::DDPG, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::PPO => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(ppo) = &self.ppo {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), ppo.discrete.to_string());
                        map.insert("traj_per_epoch".to_string(), ppo.traj_per_epoch.to_string());
                        map.insert("clip_ratio".to_string(), ppo.clip_ratio.to_string());
                        map.insert("gamma".to_string(), ppo.gamma.to_string());
                        map.insert("lam".to_string(), ppo.lam.to_string());
                        map.insert("pi_lr".to_string(), ppo.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), ppo.vf_lr.to_string());
                        map.insert("train_pi_iters".to_string(), ppo.train_pi_iters.to_string());
                        map.insert("train_vf_iters".to_string(), ppo.train_vf_iters.to_string());
                        map.insert("target_kl".to_string(), ppo.target_kl.to_string());
                        args.insert(Algorithm::PPO, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::REINFORCE => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(reinforce) = &self.reinforce {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), reinforce.discrete.to_string());
                        map.insert(
                            "with_vf_baseline".to_string(),
                            reinforce.with_vf_baseline.to_string(),
                        );
                        map.insert("seed".to_string(), reinforce.seed.to_string());
                        map.insert(
                            "traj_per_epoch".to_string(),
                            reinforce.traj_per_epoch.to_string(),
                        );
                        map.insert("gamma".to_string(), reinforce.gamma.to_string());
                        map.insert("lambda".to_string(), reinforce.lambda.to_string());
                        map.insert("pi_lr".to_string(), reinforce.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), reinforce.vf_lr.to_string());
                        map.insert(
                            "train_vf_iters".to_string(),
                            reinforce.train_vf_iters.to_string(),
                        );
                        args.insert(Algorithm::REINFORCE, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::TD3 => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(td3) = &self.td3 {
                        let mut map = HashMap::new();
                        map.insert("gamma".to_string(), td3.gamma.to_string());
                        map.insert("tau".to_string(), td3.tau.to_string());
                        map.insert("actor_lr".to_string(), td3.actor_lr.to_string());
                        map.insert("critic_lr".to_string(), td3.critic_lr.to_string());
                        map.insert("batch_size".to_string(), td3.batch_size.to_string());
                        map.insert("buffer_size".to_string(), td3.buffer_size.to_string());
                        map.insert(
                            "exploration_noise".to_string(),
                            td3.exploration_noise.to_string(),
                        );
                        map.insert("policy_noise".to_string(), td3.policy_noise.to_string());
                        map.insert("noise_clip".to_string(), td3.noise_clip.to_string());
                        map.insert(
                            "learning_starts".to_string(),
                            td3.learning_starts.to_string(),
                        );
                        map.insert(
                            "policy_frequency".to_string(),
                            td3.policy_frequency.to_string(),
                        );
                        map.insert("train_iters".to_string(), td3.train_iters.to_string());
                        args.insert(Algorithm::TD3, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::IPPO => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(ippo) = &self.ippo {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), ippo.discrete.to_string());
                        map.insert("gamma".to_string(), ippo.gamma.to_string());
                        map.insert("lam".to_string(), ippo.lam.to_string());
                        map.insert("clip_ratio".to_string(), ippo.clip_ratio.to_string());
                        map.insert("pi_lr".to_string(), ippo.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), ippo.vf_lr.to_string());
                        map.insert(
                            "train_pi_iters".to_string(),
                            ippo.train_pi_iters.to_string(),
                        );
                        map.insert(
                            "train_vf_iters".to_string(),
                            ippo.train_vf_iters.to_string(),
                        );
                        map.insert("target_kl".to_string(), ippo.target_kl.to_string());
                        map.insert(
                            "traj_per_epoch".to_string(),
                            ippo.traj_per_epoch.to_string(),
                        );
                        args.insert(Algorithm::IPPO, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::MAPPO => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(mappo) = &self.mappo {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), mappo.discrete.to_string());
                        map.insert("gamma".to_string(), mappo.gamma.to_string());
                        map.insert("lam".to_string(), mappo.lam.to_string());
                        map.insert("clip_ratio".to_string(), mappo.clip_ratio.to_string());
                        map.insert("pi_lr".to_string(), mappo.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), mappo.vf_lr.to_string());
                        map.insert(
                            "train_pi_iters".to_string(),
                            mappo.train_pi_iters.to_string(),
                        );
                        map.insert(
                            "train_vf_iters".to_string(),
                            mappo.train_vf_iters.to_string(),
                        );
                        map.insert("target_kl".to_string(), mappo.target_kl.to_string());
                        map.insert(
                            "traj_per_epoch".to_string(),
                            mappo.traj_per_epoch.to_string(),
                        );
                        args.insert(Algorithm::MAPPO, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::IREINFORCE => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(ireinforce) = &self.ireinforce {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), ireinforce.discrete.to_string());
                        map.insert(
                            "with_vf_baseline".to_string(),
                            ireinforce.with_vf_baseline.to_string(),
                        );
                        map.insert("gamma".to_string(), ireinforce.gamma.to_string());
                        map.insert("lambda".to_string(), ireinforce.lambda.to_string());
                        map.insert(
                            "traj_per_epoch".to_string(),
                            ireinforce.traj_per_epoch.to_string(),
                        );
                        map.insert("seed".to_string(), ireinforce.seed.to_string());
                        map.insert("pi_lr".to_string(), ireinforce.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), ireinforce.vf_lr.to_string());
                        map.insert(
                            "train_vf_iters".to_string(),
                            ireinforce.train_vf_iters.to_string(),
                        );
                        args.insert(Algorithm::IREINFORCE, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::MAREINFORCE => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(mareinforce) = &self.mareinforce {
                        let mut map = HashMap::new();
                        map.insert("discrete".to_string(), mareinforce.discrete.to_string());
                        map.insert("gamma".to_string(), mareinforce.gamma.to_string());
                        map.insert("lambda".to_string(), mareinforce.lambda.to_string());
                        map.insert(
                            "traj_per_epoch".to_string(),
                            mareinforce.traj_per_epoch.to_string(),
                        );
                        map.insert("seed".to_string(), mareinforce.seed.to_string());
                        map.insert("pi_lr".to_string(), mareinforce.pi_lr.to_string());
                        map.insert("vf_lr".to_string(), mareinforce.vf_lr.to_string());
                        args.insert(Algorithm::MAREINFORCE, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::IDDPG => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(iddpg) = &self.iddpg {
                        let mut map = HashMap::new();
                        map.insert("gamma".to_string(), iddpg.gamma.to_string());
                        map.insert("tau".to_string(), iddpg.tau.to_string());
                        map.insert("actor_lr".to_string(), iddpg.actor_lr.to_string());
                        map.insert("critic_lr".to_string(), iddpg.critic_lr.to_string());
                        map.insert("batch_size".to_string(), iddpg.batch_size.to_string());
                        map.insert("buffer_size".to_string(), iddpg.buffer_size.to_string());
                        map.insert(
                            "learning_starts".to_string(),
                            iddpg.learning_starts.to_string(),
                        );
                        map.insert(
                            "policy_frequency".to_string(),
                            iddpg.policy_frequency.to_string(),
                        );
                        map.insert("noise_scale".to_string(), iddpg.noise_scale.to_string());
                        map.insert("train_iters".to_string(), iddpg.train_iters.to_string());
                        args.insert(Algorithm::IDDPG, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::MADDPG => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(maddpg) = &self.maddpg {
                        let mut map = HashMap::new();
                        map.insert("gamma".to_string(), maddpg.gamma.to_string());
                        map.insert("tau".to_string(), maddpg.tau.to_string());
                        map.insert("actor_lr".to_string(), maddpg.actor_lr.to_string());
                        map.insert("critic_lr".to_string(), maddpg.critic_lr.to_string());
                        map.insert("batch_size".to_string(), maddpg.batch_size.to_string());
                        map.insert("buffer_size".to_string(), maddpg.buffer_size.to_string());
                        map.insert(
                            "policy_frequency".to_string(),
                            maddpg.policy_frequency.to_string(),
                        );
                        map.insert(
                            "traj_per_epoch".to_string(),
                            maddpg.traj_per_epoch.to_string(),
                        );
                        map.insert("train_iters".to_string(), maddpg.train_iters.to_string());
                        map.insert("noise_scale".to_string(), maddpg.noise_scale.to_string());
                        args.insert(Algorithm::MADDPG, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::ITD3 => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(itd3) = &self.itd3 {
                        let mut map = HashMap::new();
                        map.insert("gamma".to_string(), itd3.gamma.to_string());
                        map.insert("tau".to_string(), itd3.tau.to_string());
                        map.insert("actor_lr".to_string(), itd3.actor_lr.to_string());
                        map.insert("critic_lr".to_string(), itd3.critic_lr.to_string());
                        map.insert("batch_size".to_string(), itd3.batch_size.to_string());
                        map.insert("buffer_size".to_string(), itd3.buffer_size.to_string());
                        map.insert(
                            "exploration_noise".to_string(),
                            itd3.exploration_noise.to_string(),
                        );
                        map.insert("policy_noise".to_string(), itd3.policy_noise.to_string());
                        map.insert("noise_clip".to_string(), itd3.noise_clip.to_string());
                        map.insert(
                            "learning_starts".to_string(),
                            itd3.learning_starts.to_string(),
                        );
                        map.insert(
                            "policy_frequency".to_string(),
                            itd3.policy_frequency.to_string(),
                        );
                        map.insert("train_iters".to_string(), itd3.train_iters.to_string());
                        args.insert(Algorithm::ITD3, HyperparameterArgs::Map(map));
                    }
                    args
                }
                Algorithm::MATD3 => {
                    let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                    if let Some(matd3) = &self.matd3 {
                        let mut map = HashMap::new();
                        map.insert("gamma".to_string(), matd3.gamma.to_string());
                        map.insert("tau".to_string(), matd3.tau.to_string());
                        map.insert("actor_lr".to_string(), matd3.actor_lr.to_string());
                        map.insert("critic_lr".to_string(), matd3.critic_lr.to_string());
                        map.insert("batch_size".to_string(), matd3.batch_size.to_string());
                        map.insert("buffer_size".to_string(), matd3.buffer_size.to_string());
                        map.insert(
                            "policy_frequency".to_string(),
                            matd3.policy_frequency.to_string(),
                        );
                        map.insert("policy_noise".to_string(), matd3.policy_noise.to_string());
                        map.insert("noise_clip".to_string(), matd3.noise_clip.to_string());
                        map.insert(
                            "exploration_noise".to_string(),
                            matd3.exploration_noise.to_string(),
                        );
                        map.insert(
                            "traj_per_epoch".to_string(),
                            matd3.traj_per_epoch.to_string(),
                        );
                        map.insert("train_iters".to_string(), matd3.train_iters.to_string());
                        args.insert(Algorithm::MATD3, HyperparameterArgs::Map(map));
                    }
                    args
                }
                // Algorithm::CUSTOM(custom_name) => {
                //     let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                //     if let Some(custom) = &self.custom {
                //         let mut map = HashMap::new();
                //         map.insert(
                //             "custom_algorithm_name".to_string(),
                //             custom.algorithm_name.as_str().to_string(),
                //         );
                //         for (key, value) in custom.hyperparams.iter() {
                //             map.insert(key.to_string(), value.to_string());
                //         }
                //         args.insert(
                //             Algorithm::CUSTOM(custom_name.clone()),
                //             HyperparameterArgs::Map(map),
                //         );
                //     }
                //     args
                // }
                Algorithm::ConfigInit => HashMap::new(),
            },
            None => {
                let mut args = HashMap::<Algorithm, HyperparameterArgs>::new();
                if let Some(ddpg) = &self.ddpg {
                    let mut map = HashMap::new();
                    map.insert("gamma".to_string(), ddpg.gamma.to_string());
                    map.insert("tau".to_string(), ddpg.tau.to_string());
                    map.insert("actor_lr".to_string(), ddpg.actor_lr.to_string());
                    map.insert("critic_lr".to_string(), ddpg.critic_lr.to_string());
                    map.insert("batch_size".to_string(), ddpg.batch_size.to_string());
                    map.insert("buffer_size".to_string(), ddpg.buffer_size.to_string());
                    map.insert(
                        "learning_starts".to_string(),
                        ddpg.learning_starts.to_string(),
                    );
                    map.insert(
                        "policy_frequency".to_string(),
                        ddpg.policy_frequency.to_string(),
                    );
                    map.insert("noise_scale".to_string(), ddpg.noise_scale.to_string());
                    map.insert("train_iters".to_string(), ddpg.train_iters.to_string());
                    args.insert(Algorithm::DDPG, HyperparameterArgs::Map(map));
                }

                if let Some(ppo) = &self.ppo {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), ppo.discrete.to_string());
                    map.insert("traj_per_epoch".to_string(), ppo.traj_per_epoch.to_string());
                    map.insert("clip_ratio".to_string(), ppo.clip_ratio.to_string());
                    map.insert("gamma".to_string(), ppo.gamma.to_string());
                    map.insert("lam".to_string(), ppo.lam.to_string());
                    map.insert("pi_lr".to_string(), ppo.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), ppo.vf_lr.to_string());
                    map.insert("train_pi_iters".to_string(), ppo.train_pi_iters.to_string());
                    map.insert("train_vf_iters".to_string(), ppo.train_vf_iters.to_string());
                    map.insert("target_kl".to_string(), ppo.target_kl.to_string());
                    args.insert(Algorithm::PPO, HyperparameterArgs::Map(map));
                }

                if let Some(reinforce) = &self.reinforce {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), reinforce.discrete.to_string());
                    map.insert(
                        "with_vf_baseline".to_string(),
                        reinforce.with_vf_baseline.to_string(),
                    );
                    map.insert("seed".to_string(), reinforce.seed.to_string());
                    map.insert(
                        "traj_per_epoch".to_string(),
                        reinforce.traj_per_epoch.to_string(),
                    );
                    map.insert("gamma".to_string(), reinforce.gamma.to_string());
                    map.insert("lambda".to_string(), reinforce.lambda.to_string());
                    map.insert("pi_lr".to_string(), reinforce.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), reinforce.vf_lr.to_string());
                    map.insert(
                        "train_vf_iters".to_string(),
                        reinforce.train_vf_iters.to_string(),
                    );
                    args.insert(Algorithm::REINFORCE, HyperparameterArgs::Map(map));
                }

                if let Some(td3) = &self.td3 {
                    let mut map = HashMap::new();
                    map.insert("gamma".to_string(), td3.gamma.to_string());
                    map.insert("tau".to_string(), td3.tau.to_string());
                    map.insert("actor_lr".to_string(), td3.actor_lr.to_string());
                    map.insert("critic_lr".to_string(), td3.critic_lr.to_string());
                    map.insert("batch_size".to_string(), td3.batch_size.to_string());
                    map.insert("buffer_size".to_string(), td3.buffer_size.to_string());
                    map.insert(
                        "exploration_noise".to_string(),
                        td3.exploration_noise.to_string(),
                    );
                    map.insert("policy_noise".to_string(), td3.policy_noise.to_string());
                    map.insert("noise_clip".to_string(), td3.noise_clip.to_string());
                    map.insert(
                        "learning_starts".to_string(),
                        td3.learning_starts.to_string(),
                    );
                    map.insert(
                        "policy_frequency".to_string(),
                        td3.policy_frequency.to_string(),
                    );
                    map.insert("train_iters".to_string(), td3.train_iters.to_string());
                    args.insert(Algorithm::TD3, HyperparameterArgs::Map(map));
                }

                if let Some(ippo) = &self.ippo {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), ippo.discrete.to_string());
                    map.insert("gamma".to_string(), ippo.gamma.to_string());
                    map.insert("lam".to_string(), ippo.lam.to_string());
                    map.insert("clip_ratio".to_string(), ippo.clip_ratio.to_string());
                    map.insert("pi_lr".to_string(), ippo.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), ippo.vf_lr.to_string());
                    map.insert(
                        "train_pi_iters".to_string(),
                        ippo.train_pi_iters.to_string(),
                    );
                    map.insert(
                        "train_vf_iters".to_string(),
                        ippo.train_vf_iters.to_string(),
                    );
                    map.insert("target_kl".to_string(), ippo.target_kl.to_string());
                    map.insert(
                        "traj_per_epoch".to_string(),
                        ippo.traj_per_epoch.to_string(),
                    );
                    args.insert(Algorithm::IPPO, HyperparameterArgs::Map(map));
                }

                if let Some(mappo) = &self.mappo {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), mappo.discrete.to_string());
                    map.insert("gamma".to_string(), mappo.gamma.to_string());
                    map.insert("lam".to_string(), mappo.lam.to_string());
                    map.insert("clip_ratio".to_string(), mappo.clip_ratio.to_string());
                    map.insert("pi_lr".to_string(), mappo.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), mappo.vf_lr.to_string());
                    map.insert(
                        "train_pi_iters".to_string(),
                        mappo.train_pi_iters.to_string(),
                    );
                    map.insert(
                        "train_vf_iters".to_string(),
                        mappo.train_vf_iters.to_string(),
                    );
                    map.insert("target_kl".to_string(), mappo.target_kl.to_string());
                    map.insert(
                        "traj_per_epoch".to_string(),
                        mappo.traj_per_epoch.to_string(),
                    );
                    args.insert(Algorithm::MAPPO, HyperparameterArgs::Map(map));
                }

                if let Some(ireinforce) = &self.ireinforce {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), ireinforce.discrete.to_string());
                    map.insert(
                        "with_vf_baseline".to_string(),
                        ireinforce.with_vf_baseline.to_string(),
                    );
                    map.insert("gamma".to_string(), ireinforce.gamma.to_string());
                    map.insert("lambda".to_string(), ireinforce.lambda.to_string());
                    map.insert(
                        "traj_per_epoch".to_string(),
                        ireinforce.traj_per_epoch.to_string(),
                    );
                    map.insert("seed".to_string(), ireinforce.seed.to_string());
                    map.insert("pi_lr".to_string(), ireinforce.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), ireinforce.vf_lr.to_string());
                    map.insert(
                        "train_vf_iters".to_string(),
                        ireinforce.train_vf_iters.to_string(),
                    );
                    args.insert(Algorithm::IREINFORCE, HyperparameterArgs::Map(map));
                }

                if let Some(mareinforce) = &self.mareinforce {
                    let mut map = HashMap::new();
                    map.insert("discrete".to_string(), mareinforce.discrete.to_string());
                    map.insert("gamma".to_string(), mareinforce.gamma.to_string());
                    map.insert("lambda".to_string(), mareinforce.lambda.to_string());
                    map.insert(
                        "traj_per_epoch".to_string(),
                        mareinforce.traj_per_epoch.to_string(),
                    );
                    map.insert("seed".to_string(), mareinforce.seed.to_string());
                    map.insert("pi_lr".to_string(), mareinforce.pi_lr.to_string());
                    map.insert("vf_lr".to_string(), mareinforce.vf_lr.to_string());
                    args.insert(Algorithm::MAREINFORCE, HyperparameterArgs::Map(map));
                }

                if let Some(iddpg) = &self.iddpg {
                    let mut map = HashMap::new();
                    map.insert("gamma".to_string(), iddpg.gamma.to_string());
                    map.insert("tau".to_string(), iddpg.tau.to_string());
                    map.insert("actor_lr".to_string(), iddpg.actor_lr.to_string());
                    map.insert("critic_lr".to_string(), iddpg.critic_lr.to_string());
                    map.insert("batch_size".to_string(), iddpg.batch_size.to_string());
                    map.insert("buffer_size".to_string(), iddpg.buffer_size.to_string());
                    map.insert(
                        "learning_starts".to_string(),
                        iddpg.learning_starts.to_string(),
                    );
                    map.insert(
                        "policy_frequency".to_string(),
                        iddpg.policy_frequency.to_string(),
                    );
                    map.insert("noise_scale".to_string(), iddpg.noise_scale.to_string());
                    map.insert("train_iters".to_string(), iddpg.train_iters.to_string());
                    args.insert(Algorithm::IDDPG, HyperparameterArgs::Map(map));
                }

                if let Some(maddpg) = &self.maddpg {
                    let mut map = HashMap::new();
                    map.insert("gamma".to_string(), maddpg.gamma.to_string());
                    map.insert("tau".to_string(), maddpg.tau.to_string());
                    map.insert("actor_lr".to_string(), maddpg.actor_lr.to_string());
                    map.insert("critic_lr".to_string(), maddpg.critic_lr.to_string());
                    map.insert("batch_size".to_string(), maddpg.batch_size.to_string());
                    map.insert("buffer_size".to_string(), maddpg.buffer_size.to_string());
                    map.insert(
                        "policy_frequency".to_string(),
                        maddpg.policy_frequency.to_string(),
                    );
                    map.insert(
                        "traj_per_epoch".to_string(),
                        maddpg.traj_per_epoch.to_string(),
                    );
                    map.insert("train_iters".to_string(), maddpg.train_iters.to_string());
                    map.insert("noise_scale".to_string(), maddpg.noise_scale.to_string());
                    args.insert(Algorithm::MADDPG, HyperparameterArgs::Map(map));
                }

                if let Some(itd3) = &self.itd3 {
                    let mut map = HashMap::new();
                    map.insert("gamma".to_string(), itd3.gamma.to_string());
                    map.insert("tau".to_string(), itd3.tau.to_string());
                    map.insert("actor_lr".to_string(), itd3.actor_lr.to_string());
                    map.insert("critic_lr".to_string(), itd3.critic_lr.to_string());
                    map.insert("batch_size".to_string(), itd3.batch_size.to_string());
                    map.insert("buffer_size".to_string(), itd3.buffer_size.to_string());
                    map.insert(
                        "exploration_noise".to_string(),
                        itd3.exploration_noise.to_string(),
                    );
                    map.insert("policy_noise".to_string(), itd3.policy_noise.to_string());
                    map.insert("noise_clip".to_string(), itd3.noise_clip.to_string());
                    map.insert(
                        "learning_starts".to_string(),
                        itd3.learning_starts.to_string(),
                    );
                    map.insert(
                        "policy_frequency".to_string(),
                        itd3.policy_frequency.to_string(),
                    );
                    map.insert("train_iters".to_string(), itd3.train_iters.to_string());
                    args.insert(Algorithm::ITD3, HyperparameterArgs::Map(map));
                }

                if let Some(matd3) = &self.matd3 {
                    let mut map = HashMap::new();
                    map.insert("gamma".to_string(), matd3.gamma.to_string());
                    map.insert("tau".to_string(), matd3.tau.to_string());
                    map.insert("actor_lr".to_string(), matd3.actor_lr.to_string());
                    map.insert("critic_lr".to_string(), matd3.critic_lr.to_string());
                    map.insert("batch_size".to_string(), matd3.batch_size.to_string());
                    map.insert("buffer_size".to_string(), matd3.buffer_size.to_string());
                    map.insert(
                        "policy_frequency".to_string(),
                        matd3.policy_frequency.to_string(),
                    );
                    map.insert("policy_noise".to_string(), matd3.policy_noise.to_string());
                    map.insert("noise_clip".to_string(), matd3.noise_clip.to_string());
                    map.insert(
                        "exploration_noise".to_string(),
                        matd3.exploration_noise.to_string(),
                    );
                    map.insert(
                        "traj_per_epoch".to_string(),
                        matd3.traj_per_epoch.to_string(),
                    );
                    map.insert("train_iters".to_string(), matd3.train_iters.to_string());
                    args.insert(Algorithm::MATD3, HyperparameterArgs::Map(map));
                }

                // if let Some(custom) = &self.custom {
                //     let mut map = HashMap::new();
                //     map.insert(
                //         "custom_algorithm_name".to_string(),
                //         custom.algorithm_name.as_str().to_string(),
                //     );
                //     for (key, value) in custom.hyperparams.iter() {
                //         map.insert(key.to_string(), value.to_string());
                //     }
                //     args.insert(
                //         Algorithm::CUSTOM(custom.algorithm_name.as_str().to_string()),
                //         HyperparameterArgs::Map(map),
                //     );
                // }

                args
            }
        }
    }
}

impl Default for HyperparameterConfig {
    fn default() -> Self {
        Self {
            ddpg: None,
            iddpg: None,
            maddpg: None,
            ppo: None,
            ippo: None,
            mappo: None,
            reinforce: Some(REINFORCEParams::default()),
            ireinforce: None,
            mareinforce: None,
            td3: None,
            itd3: None,
            matd3: None,
            // custom: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CustomAlgorithmParams {
    pub algorithm_name: Algorithm,
    pub hyperparams: HashMap<String, String>,
}

impl Default for CustomAlgorithmParams {
    fn default() -> Self {
        Self {
            algorithm_name: Algorithm::ConfigInit,
            hyperparams: HashMap::new(),
        }
    }
}

/// Server address parameters.
///
/// Each server parameter includes a prefix, host, and port.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct NetworkParams {
    pub host: String,
    pub port: String,
}

/// Tensorboard configuration structure.
///
/// Contains optional tensorboard writer parameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorboardConfig {
    pub training_tensorboard: Option<TensorboardParams>,
}

/// Parameters for Training Tensorboard Writer, used for real-time plotting.
///
/// The scalar_tags field is deserialized from a semicolon-separated string.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorboardParams {
    pub launch_tb_on_startup: bool,
    #[serde(deserialize_with = "vec_scalar_tags")]
    pub scalar_tags: Vec<String>,
    pub global_step_tag: String,
}

/// Helper function to deserialize a semicolon-separated string into a vector of strings.
///
/// # Arguments
///
/// * `deserializer` - A serde deserializer.
///
/// # Returns
///
/// A [Result] containing a vector of strings on success.
fn vec_scalar_tags<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(s.split(';').map(|s| s.to_string()).collect())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OtlpEndpointParams {
    pub prefix: String,
    pub host: String,
    pub port: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientConfigParams {
    pub algorithm_name: String,
    pub config_update_polling_seconds: f32,
    pub init_hyperparameters: HyperparameterConfig,
    pub trajectory_file_output: LocalTrajectoryFileParams,
    pub metrics_meter_name: String,
    pub metrics_otlp_endpoint: OtlpEndpointParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ClientConfigFile {
    client_config: ClientConfigParams,
    transport_config: TransportConfigParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientConfigLoader {
    pub config_path: PathBuf,
    pub client_config: ClientConfigParams,
    pub transport_config: TransportConfigParams,
}

impl ClientConfigLoader {
    pub fn new_config(algorithm_name: Option<String>, config_path: Option<PathBuf>) -> Self {
        let _config_path: PathBuf = if let Some(config_path_value) = config_path {
            config_path_value
        } else {
            DEFAULT_CLIENT_CONFIG_PATH
                .clone()
                .expect("[ClientConfigParams - new] Invalid config path")
        };

        let config: ClientConfigLoader = Self::load_config(&_config_path);

        let client_config: ClientConfigParams = config.client_config;
        let transport_config: TransportConfigParams = config.transport_config;

        let algorithm_name_: String = match algorithm_name {
            Some(algorithm_name) => algorithm_name,
            None => client_config.algorithm_name,
        };

        Self {
            config_path: _config_path,
            client_config: ClientConfigParams {
                algorithm_name: algorithm_name_,
                config_update_polling_seconds: client_config.config_update_polling_seconds,
                init_hyperparameters: client_config.init_hyperparameters,
                trajectory_file_output: client_config.trajectory_file_output,
                metrics_meter_name: client_config.metrics_meter_name,
                metrics_otlp_endpoint: client_config.metrics_otlp_endpoint,
            },
            transport_config,
        }
    }

    fn from_file(config_path: PathBuf, file: ClientConfigFile) -> Self {
        Self {
            config_path,
            client_config: file.client_config,
            transport_config: file.transport_config,
        }
    }

    pub fn load_config(config_path: &PathBuf) -> Self {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents)
                    .expect("[ClientConfigParams - load_config] Failed to read configuration file");
                let file_config: ClientConfigFile = serde_json::from_str(&contents).unwrap_or_else(|_| {
                    log::error!("[ClientConfigParams - load_config] Failed to parse configuration, loading empty defaults...");
                    ClientConfigFile {
                        client_config: ClientConfigParams {
                            algorithm_name: "REINFORCE".to_string(),
                            config_update_polling_seconds: 10.0,
                            init_hyperparameters: HyperparameterConfig::default(),
                            trajectory_file_output: LocalTrajectoryFileParams::default(),
                            metrics_meter_name: "relayrl-client".to_string(),
                            metrics_otlp_endpoint: OtlpEndpointParams {
                                prefix: "http://".to_string(),
                                host: "127.0.0.1".to_string(),
                                port: "4317".to_string(),
                            },
                        },
                        transport_config: TransportConfigBuilder::build_default(),
                    }
                });

                Self::from_file(config_path.clone(), file_config)
            }
            Err(e) => {
                panic!(
                    "[ClientConfigParams - load_config] Failed to open configuration file: {}",
                    e
                );
            }
        }
    }

    pub fn get_algorithm_name(&self) -> &str {
        &self.client_config.algorithm_name
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.config_path
    }

    pub fn get_init_hyperparameters(&self) -> &HyperparameterConfig {
        &self.client_config.init_hyperparameters
    }

    pub fn get_init_hyperparameter_args(
        &self,
        algorithm: Option<&Algorithm>,
    ) -> HashMap<Algorithm, HyperparameterArgs> {
        self.client_config.init_hyperparameters.to_args(algorithm)
    }

    pub fn get_trajectory_file_output(&self) -> &LocalTrajectoryFileParams {
        &self.client_config.trajectory_file_output
    }

    pub fn get_metrics_meter_name(&self) -> &str {
        &self.client_config.metrics_meter_name
    }

    pub fn get_metrics_otlp_endpoint(&self) -> &OtlpEndpointParams {
        &self.client_config.metrics_otlp_endpoint
    }

    pub fn get_transport_config(&self) -> &TransportConfigParams {
        &self.transport_config
    }
}

pub trait ClientConfigBuildParams {
    fn set_algorithm_name(&mut self, algorithm_name: &str) -> &mut Self;
    fn set_init_hyperparameters(&mut self, init_hyperparameters: HyperparameterConfig)
    -> &mut Self;
    fn set_metrics_name(&mut self, metrics_name: &str) -> &mut Self;
    fn set_otlp_endpoint(&mut self, otlp_endpoint: OtlpEndpointParams) -> &mut Self;
    fn set_trajectory_file_output(
        &mut self,
        trajectory_file_output: LocalTrajectoryFileParams,
    ) -> &mut Self;
    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self;
    fn build(&self) -> ClientConfigLoader;
    fn build_default() -> ClientConfigLoader;
}

pub struct ClientConfigBuilder {
    algorithm_name: Option<String>,
    config_update_polling_seconds: Option<f32>,
    init_hyperparameters: Option<HyperparameterConfig>,
    transport_config: Option<TransportConfigParams>,
    trajectory_file_output: Option<LocalTrajectoryFileParams>,
    metrics_name: Option<String>,
    otlp_endpoint: Option<OtlpEndpointParams>,
}

impl ClientConfigBuildParams for ClientConfigBuilder {
    fn set_algorithm_name(&mut self, algorithm_name: &str) -> &mut Self {
        self.algorithm_name = Some(algorithm_name.to_string());
        self
    }

    fn set_init_hyperparameters(
        &mut self,
        init_hyperparameters: HyperparameterConfig,
    ) -> &mut Self {
        self.init_hyperparameters = Some(init_hyperparameters);
        self
    }

    fn set_trajectory_file_output(
        &mut self,
        trajectory_file_output: LocalTrajectoryFileParams,
    ) -> &mut Self {
        self.trajectory_file_output = Some(trajectory_file_output);
        self
    }

    fn set_metrics_name(&mut self, metrics_name: &str) -> &mut Self {
        self.metrics_name = Some(metrics_name.to_string());
        self
    }

    fn set_otlp_endpoint(&mut self, otlp_endpoint: OtlpEndpointParams) -> &mut Self {
        self.otlp_endpoint = Some(otlp_endpoint);
        self
    }

    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self {
        self.transport_config = Some(transport_config);
        self
    }

    fn build(&self) -> ClientConfigLoader {
        let client_config: ClientConfigParams = ClientConfigParams {
            algorithm_name: self
                .algorithm_name
                .clone()
                .unwrap_or_else(|| "REINFORCE".to_string()),
            config_update_polling_seconds: self.config_update_polling_seconds.unwrap_or(10.0),
            init_hyperparameters: self.init_hyperparameters.clone().unwrap_or_default(),
            trajectory_file_output: self.trajectory_file_output.clone().unwrap_or_default(),
            metrics_meter_name: self
                .metrics_name
                .clone()
                .unwrap_or_else(|| "relayrl-client".to_string()),
            metrics_otlp_endpoint: self.otlp_endpoint.clone().unwrap_or_else(|| {
                OtlpEndpointParams {
                    prefix: "http://".to_string(),
                    host: "127.0.0.1".to_string(),
                    port: "4317".to_string(),
                }
            }),
        };

        let transport_config: TransportConfigParams = match &self.transport_config {
            Some(transport_config) => TransportConfigParams {
                nats_addresses: transport_config.nats_addresses.clone(),
                zmq_addresses: transport_config.zmq_addresses.clone(),
                max_traj_length: transport_config.max_traj_length,
                local_model_module: transport_config.local_model_module.clone(),
            },
            None => TransportConfigBuilder::build_default(),
        };

        ClientConfigLoader {
            config_path: PathBuf::from("client_config.json"),
            client_config,
            transport_config,
        }
    }

    fn build_default() -> ClientConfigLoader {
        ClientConfigLoader {
            config_path: PathBuf::from("client_config.json"),
            client_config: ClientConfigParams {
                algorithm_name: "REINFORCE".to_string(),
                config_update_polling_seconds: 10.0,
                init_hyperparameters: HyperparameterConfig::default(),
                trajectory_file_output: LocalTrajectoryFileParams::default(),
                metrics_meter_name: "relayrl-client".to_string(),
                metrics_otlp_endpoint: OtlpEndpointParams {
                    prefix: "http://".to_string(),
                    host: "127.0.0.1".to_string(),
                    port: "4317".to_string(),
                },
            },
            transport_config: TransportConfigBuilder::build_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingServerConfigParams {
    pub config_update_polling_seconds: f32,
    pub default_hyperparameters: Option<HyperparameterConfig>,
    pub training_tensorboard: TensorboardParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TrainingServerConfigFile {
    training_server_config: TrainingServerConfigParams,
    transport_config: TransportConfigParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingServerConfigLoader {
    pub config_path: PathBuf,
    pub training_server_config: TrainingServerConfigParams,
    pub transport_config: TransportConfigParams,
}

impl TrainingServerConfigLoader {
    pub fn new_config(config_path: Option<PathBuf>) -> Self {
        let _config_path: PathBuf = if let Some(config_path_value) = config_path {
            config_path_value
        } else {
            DEFAULT_TRAINING_SERVER_CONFIG_PATH
                .clone()
                .expect("[TrainingServerConfigParams - new] Invalid config path")
        };

        let config: TrainingServerConfigLoader = Self::load_config(&_config_path);

        let training_server_config: TrainingServerConfigParams = config.training_server_config;
        let transport_config: TransportConfigParams = config.transport_config;

        Self {
            config_path: _config_path,
            training_server_config,
            transport_config,
        }
    }

    fn from_file(config_path: PathBuf, file: TrainingServerConfigFile) -> Self {
        Self {
            config_path,
            training_server_config: file.training_server_config,
            transport_config: file.transport_config,
        }
    }

    pub fn load_config(config_path: &PathBuf) -> Self {
        match File::open(config_path) {
            Ok(mut file) => {
                let mut contents: String = String::new();
                file.read_to_string(&mut contents).expect(
                    "[TrainingServerConfigParams - load_config] Failed to read configuration file",
                );
                let file_config: TrainingServerConfigFile = serde_json::from_str(&contents).unwrap_or_else(|_| {
                    log::error!("[TrainingServerConfigParams - load_config] Failed to parse configuration, loading empty defaults...");
                    TrainingServerConfigFile {
                        training_server_config: TrainingServerConfigParams {
                            config_update_polling_seconds: 10.0,
                            default_hyperparameters: None,
                            training_tensorboard: TensorboardParams {
                                launch_tb_on_startup: false,
                                scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                                global_step_tag: "Epoch".to_string(),
                            },
                        },
                        transport_config: TransportConfigBuilder::build_default(),
                    }
                });

                Self::from_file(config_path.clone(), file_config)
            }
            Err(e) => {
                panic!(
                    "[TrainingServerConfigParams - load_config] Failed to open configuration file: {}",
                    e
                );
            }
        }
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.config_path
    }

    pub fn get_config_update_polling_seconds(&self) -> f32 {
        self.training_server_config.config_update_polling_seconds
    }

    pub fn get_hyperparameters(&self) -> &Option<HyperparameterConfig> {
        &self.training_server_config.default_hyperparameters
    }

    pub fn get_training_tensorboard(&self) -> &TensorboardParams {
        &self.training_server_config.training_tensorboard
    }

    pub fn get_transport_config(&self) -> &TransportConfigParams {
        &self.transport_config
    }
}

pub trait TrainingServerConfigBuildParams {
    fn set_config_update_polling_seconds(
        &mut self,
        config_update_polling_seconds: f32,
    ) -> &mut Self;
    fn set_hyperparameters(
        &mut self,
        algorithm_name: &str,
        hyperparameter_args: HyperparameterArgs,
    ) -> &mut Self;
    fn set_training_tensorboard_params(
        &mut self,
        launch_tb_on_startup: bool,
        scalar_tags: &str,
        global_step_tag: &str,
    ) -> &mut Self;
    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self;
    fn build(&self) -> TrainingServerConfigLoader;
    fn build_default() -> TrainingServerConfigLoader;
}

pub struct TrainingServerConfigBuilder {
    config_update_polling_seconds: Option<f32>,
    default_hyperparameters: Option<HyperparameterConfig>,
    training_tensorboard: Option<TensorboardParams>,
    transport_config: Option<TransportConfigParams>,
}

impl TrainingServerConfigBuildParams for TrainingServerConfigBuilder {
    fn set_config_update_polling_seconds(
        &mut self,
        config_update_polling_seconds: f32,
    ) -> &mut Self {
        self.config_update_polling_seconds = Some(config_update_polling_seconds);
        self
    }

    fn set_hyperparameters(
        &mut self,
        algorithm_name: &str,
        hyperparameter_args: HyperparameterArgs,
    ) -> &mut Self {
        let hp_map: HashMap<String, String> =
            crate::network::parse_args(&Some(hyperparameter_args.clone()));

        // Start from defaults for all supported algorithms.
        let mut all_cfg = HyperparameterConfig {
            ddpg: Some(DDPGParams::default()),
            iddpg: Some(IDDPGParams::default()),
            maddpg: Some(MADDPGParams::default()),
            ppo: Some(PPOParams::default()),
            ippo: Some(IPPOParams::default()),
            mappo: Some(MAPPOParams::default()),
            reinforce: Some(REINFORCEParams::default()),
            ireinforce: Some(IREINFORCEParams::default()),
            mareinforce: Some(MAREINFORCEParams::default()),
            td3: Some(TD3Params::default()),
            itd3: Some(ITD3Params::default()),
            matd3: Some(MATD3Params::default()),
            // custom: None,
        };

        let algo_upper = algorithm_name.to_uppercase();
        let algorithm = Algorithm::from_str(algo_upper.as_str()).unwrap_or(Algorithm::REINFORCE);

        match algorithm {
            Algorithm::DDPG => {
                if let Some(params) = &mut all_cfg.ddpg {
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map.get("actor_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.actor_lr = v;
                    } else if let Some(v) = hp_map
                        .get("learning_rate")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.actor_lr = v;
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("critic_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("batch_size").and_then(|s| s.parse::<u32>().ok()) {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_starts")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.learning_starts = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("noise_scale")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.noise_scale = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_iters = v;
                    }
                }
            }
            Algorithm::PPO => {
                if let Some(params) = &mut all_cfg.ppo {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map.get("clip_ratio").and_then(|s| s.parse::<f32>().ok()) {
                        params.clip_ratio = v;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lam = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_pi_iters")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_pi_iters = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_vf_iters")
                        .or_else(|| hp_map.get("train_v_iters"))
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_vf_iters = v;
                    }
                    if let Some(v) = hp_map.get("target_kl").and_then(|s| s.parse::<f32>().ok()) {
                        params.target_kl = v;
                    }
                }
            }
            Algorithm::REINFORCE => {
                if let Some(params) = &mut all_cfg.reinforce {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("with_vf_baseline") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.with_vf_baseline = vv;
                    }
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u64>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lambda").and_then(|s| s.parse::<f32>().ok()) {
                        params.lambda = v;
                    } else if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lambda = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_vf_iters")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_vf_iters = v;
                    }
                }
            }
            Algorithm::TD3 => {
                if let Some(params) = &mut all_cfg.td3 {
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map.get("actor_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.actor_lr = v;
                    } else if let Some(v) = hp_map
                        .get("learning_rate")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.actor_lr = v;
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("critic_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("batch_size").and_then(|s| s.parse::<u32>().ok()) {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("exploration_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.exploration_noise = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.policy_noise = v;
                    }
                    if let Some(v) = hp_map.get("noise_clip").and_then(|s| s.parse::<f32>().ok()) {
                        params.noise_clip = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_starts")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.learning_starts = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_iters = v;
                    }
                }
            }
            Algorithm::IPPO => {
                if let Some(params) = &mut all_cfg.ippo {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lam = v;
                    }
                    if let Some(v) = hp_map.get("clip_ratio").and_then(|s| s.parse::<f32>().ok()) {
                        params.clip_ratio = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_pi_iters")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_pi_iters = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_vf_iters")
                        .or_else(|| hp_map.get("train_v_iters"))
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_vf_iters = v;
                    }
                    if let Some(v) = hp_map.get("target_kl").and_then(|s| s.parse::<f32>().ok()) {
                        params.target_kl = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                }
            }
            Algorithm::MAPPO => {
                if let Some(params) = &mut all_cfg.mappo {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lam = v;
                    }
                    if let Some(v) = hp_map.get("clip_ratio").and_then(|s| s.parse::<f32>().ok()) {
                        params.clip_ratio = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_pi_iters")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_pi_iters = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_vf_iters")
                        .or_else(|| hp_map.get("train_v_iters"))
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_vf_iters = v;
                    }
                    if let Some(v) = hp_map.get("target_kl").and_then(|s| s.parse::<f32>().ok()) {
                        params.target_kl = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                }
            }
            Algorithm::IREINFORCE => {
                if let Some(params) = &mut all_cfg.ireinforce {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("with_vf_baseline") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.with_vf_baseline = vv;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lambda").and_then(|s| s.parse::<f32>().ok()) {
                        params.lambda = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lambda = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u64>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_vf_iters")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.train_vf_iters = v;
                    }
                }
            }
            Algorithm::MAREINFORCE => {
                if let Some(params) = &mut all_cfg.mareinforce {
                    if let Some(v) = hp_map.get("discrete") {
                        let vv = matches!(v.to_lowercase().as_str(), "true" | "1" | "yes");
                        params.discrete = vv;
                    }
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("lambda").and_then(|s| s.parse::<f32>().ok()) {
                        params.lambda = v;
                    }
                    if let Some(v) = hp_map.get("lam").and_then(|s| s.parse::<f32>().ok()) {
                        params.lambda = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map.get("seed").and_then(|s| s.parse::<u64>().ok()) {
                        params.seed = v;
                    }
                    if let Some(v) = hp_map.get("pi_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.pi_lr = v;
                    }
                    if let Some(v) = hp_map.get("vf_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.vf_lr = v;
                    }
                }
            }
            Algorithm::IDDPG => {
                if let Some(params) = &mut all_cfg.iddpg {
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map.get("actor_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.actor_lr = v;
                    } else if let Some(v) = hp_map
                        .get("learning_rate")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.actor_lr = v;
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("critic_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("batch_size").and_then(|s| s.parse::<u32>().ok()) {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_starts")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.learning_starts = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("noise_scale")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.noise_scale = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_iters = v;
                    }
                }
            }
            Algorithm::MADDPG => {
                if let Some(params) = &mut all_cfg.maddpg {
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map.get("actor_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.actor_lr = v;
                    }
                    if let Some(v) = hp_map.get("critic_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("batch_size")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.train_iters = v;
                    }
                    if let Some(v) = hp_map
                        .get("noise_scale")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.noise_scale = v;
                    }
                }
            }
            Algorithm::ITD3 => {
                if let Some(params) = &mut all_cfg.itd3 {
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map.get("actor_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.actor_lr = v;
                    } else if let Some(v) = hp_map
                        .get("learning_rate")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.actor_lr = v;
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("critic_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map.get("batch_size").and_then(|s| s.parse::<u32>().ok()) {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("exploration_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.exploration_noise = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.policy_noise = v;
                    }
                    if let Some(v) = hp_map.get("noise_clip").and_then(|s| s.parse::<f32>().ok()) {
                        params.noise_clip = v;
                    }
                    if let Some(v) = hp_map
                        .get("learning_starts")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.learning_starts = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        params.train_iters = v;
                    }
                }
            }
            Algorithm::MATD3 => {
                if let Some(params) = &mut all_cfg.matd3 {
                    if let Some(v) = hp_map.get("gamma").and_then(|s| s.parse::<f32>().ok()) {
                        params.gamma = v;
                    }
                    if let Some(v) = hp_map.get("tau").and_then(|s| s.parse::<f32>().ok()) {
                        params.tau = v;
                    }
                    if let Some(v) = hp_map.get("actor_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.actor_lr = v;
                    }
                    if let Some(v) = hp_map.get("critic_lr").and_then(|s| s.parse::<f32>().ok()) {
                        params.critic_lr = v;
                    }
                    if let Some(v) = hp_map
                        .get("batch_size")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.batch_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("buffer_size")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.buffer_size = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_frequency")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.policy_frequency = v;
                    }
                    if let Some(v) = hp_map
                        .get("policy_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.policy_noise = v;
                    }
                    if let Some(v) = hp_map.get("noise_clip").and_then(|s| s.parse::<f32>().ok()) {
                        params.noise_clip = v;
                    }
                    if let Some(v) = hp_map
                        .get("exploration_noise")
                        .and_then(|s| s.parse::<f32>().ok())
                    {
                        params.exploration_noise = v;
                    }
                    if let Some(v) = hp_map
                        .get("traj_per_epoch")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        params.traj_per_epoch = v;
                    }
                    if let Some(v) = hp_map
                        .get("train_iters")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        params.train_iters = v;
                    }
                }
            }
            // Algorithm::CUSTOM(custom_algorithm) => {
            //     let mut custom = all_cfg.custom.take().unwrap_or_default();
            //     custom.algorithm_name = Algorithm::CUSTOM(custom_algorithm.clone());
            //     custom.hyperparams = hp_map.clone();
            //     all_cfg.custom = Some(custom);
            // }
            Algorithm::ConfigInit => {
                log::error!(
                    "[ServerConfigBuilder] ConfigInit is not a valid algorithm for hyperparameters"
                );
                return self;
            }
        }

        self.default_hyperparameters = Some(all_cfg);
        self
    }

    fn set_training_tensorboard_params(
        &mut self,
        launch_tb_on_startup: bool,
        scalar_tags: &str,
        global_step_tag: &str,
    ) -> &mut Self {
        self.training_tensorboard = Some(TensorboardParams {
            launch_tb_on_startup,
            scalar_tags: scalar_tags.split(';').map(|s| s.to_string()).collect(),
            global_step_tag: global_step_tag.to_string(),
        });
        self
    }

    fn set_transport_config(&mut self, transport_config: TransportConfigParams) -> &mut Self {
        self.transport_config = Some(transport_config);
        self
    }

    fn build(&self) -> TrainingServerConfigLoader {
        let training_server_config: TrainingServerConfigParams = TrainingServerConfigParams {
            config_update_polling_seconds: self.config_update_polling_seconds.unwrap_or(10.0),
            default_hyperparameters: self.default_hyperparameters.clone(),
            training_tensorboard: self.training_tensorboard.clone().unwrap_or_else(|| {
                TensorboardParams {
                    launch_tb_on_startup: false,
                    scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                    global_step_tag: "Epoch".to_string(),
                }
            }),
        };

        let transport_config: TransportConfigParams = match &self.transport_config {
            Some(transport_config) => TransportConfigParams {
                nats_addresses: transport_config.nats_addresses.clone(),
                zmq_addresses: transport_config.zmq_addresses.clone(),
                max_traj_length: transport_config.max_traj_length,
                local_model_module: transport_config.local_model_module.clone(),
            },
            None => TransportConfigBuilder::build_default(),
        };

        TrainingServerConfigLoader {
            config_path: PathBuf::from("training_server_config.json"),
            training_server_config,
            transport_config,
        }
    }

    fn build_default() -> TrainingServerConfigLoader {
        TrainingServerConfigLoader {
            config_path: PathBuf::from("training_server_config.json"),
            training_server_config: TrainingServerConfigParams {
                config_update_polling_seconds: 10.0,
                default_hyperparameters: None,
                training_tensorboard: TensorboardParams {
                    launch_tb_on_startup: false,
                    scalar_tags: vec!["AverageEpRet".to_string(), "StdEpRet".to_string()],
                    global_step_tag: "Epoch".to_string(),
                },
            },
            transport_config: TransportConfigBuilder::build_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZmqTransportInferenceAddresses {
    pub inference_server_address: NetworkParams,
    pub inference_scaling_server_address: NetworkParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZmqTransportTrainingAddresses {
    pub agent_listener_address: NetworkParams,
    pub model_server_address: NetworkParams,
    pub trajectory_server_address: NetworkParams,
    pub training_scaling_server_address: NetworkParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZmqTransportAddresses {
    pub inference_addresses: ZmqTransportInferenceAddresses,
    pub training_addresses: ZmqTransportTrainingAddresses,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NatsTransportAddresses {
    pub inference_server_address: NetworkParams,
    pub training_server_address: NetworkParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransportConfigParams {
    pub nats_addresses: NatsTransportAddresses,
    pub zmq_addresses: ZmqTransportAddresses,
    pub max_traj_length: usize,
    pub local_model_module: LocalModelModuleParams,
}

impl TransportConfigParams {
    pub fn get_nats_inference_server_address(&self) -> &NetworkParams {
        &self.nats_addresses.inference_server_address
    }

    pub fn get_nats_training_server_address(&self) -> &NetworkParams {
        &self.nats_addresses.training_server_address
    }

    pub fn get_zmq_inference_server_address(&self) -> &NetworkParams {
        &self
            .zmq_addresses
            .inference_addresses
            .inference_server_address
    }

    pub fn get_zmq_agent_listener_address(&self) -> &NetworkParams {
        &self.zmq_addresses.training_addresses.agent_listener_address
    }

    pub fn get_zmq_model_server_address(&self) -> &NetworkParams {
        &self.zmq_addresses.training_addresses.model_server_address
    }

    pub fn get_zmq_trajectory_server_address(&self) -> &NetworkParams {
        &self
            .zmq_addresses
            .training_addresses
            .trajectory_server_address
    }

    pub fn get_zmq_inference_scaling_server_address(&self) -> &NetworkParams {
        &self
            .zmq_addresses
            .inference_addresses
            .inference_scaling_server_address
    }

    pub fn get_zmq_training_scaling_server_address(&self) -> &NetworkParams {
        &self
            .zmq_addresses
            .training_addresses
            .training_scaling_server_address
    }
}

pub trait TransportConfigBuildParams {
    fn set_nats_inference_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_nats_training_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_zmq_inference_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_zmq_agent_listener_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_zmq_model_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_zmq_trajectory_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_zmq_inference_scaling_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_zmq_training_scaling_server_address(&mut self, host: &str, port: &str) -> &mut Self;
    fn set_max_traj_length(&mut self, max_traj_length: usize) -> &mut Self;
    fn set_local_model_module(&mut self, directory_name: &str, model_name: &str) -> &mut Self;
    fn build(&self) -> TransportConfigParams;
    fn build_default() -> TransportConfigParams;
}

pub struct TransportConfigBuilder {
    nats_inference_server_address: Option<NetworkParams>,
    nats_training_server_address: Option<NetworkParams>,
    zmq_inference_server_address: Option<NetworkParams>,
    zmq_agent_listener_address: Option<NetworkParams>,
    zmq_model_server_address: Option<NetworkParams>,
    zmq_trajectory_server_address: Option<NetworkParams>,
    zmq_inference_scaling_server_address: Option<NetworkParams>,
    zmq_training_scaling_server_address: Option<NetworkParams>,
    max_traj_length: Option<usize>,
    local_model_module: Option<LocalModelModuleParams>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocalModelModuleParams {
    pub directory: String,
    pub model_name: String,
    pub format: String,
}

impl TransportConfigBuildParams for TransportConfigBuilder {
    fn set_nats_inference_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.nats_inference_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_nats_training_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.nats_training_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_zmq_inference_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.zmq_inference_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_zmq_agent_listener_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.zmq_agent_listener_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_zmq_model_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.zmq_model_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_zmq_trajectory_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.zmq_trajectory_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_zmq_inference_scaling_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.zmq_inference_scaling_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_zmq_training_scaling_server_address(&mut self, host: &str, port: &str) -> &mut Self {
        self.zmq_training_scaling_server_address = Some(NetworkParams {
            host: host.to_string(),
            port: port.to_string(),
        });
        self
    }

    fn set_max_traj_length(&mut self, max_traj_length: usize) -> &mut Self {
        self.max_traj_length = Some(max_traj_length);
        self
    }

    fn set_local_model_module(&mut self, directory_name: &str, model_name: &str) -> &mut Self {
        self.local_model_module = Some(LocalModelModuleParams {
            directory: directory_name.to_string(),
            format: "pt".to_string(),
            model_name: model_name.to_string(),
        });
        self
    }

    fn build(&self) -> TransportConfigParams {
        let nats_inference_server_address: NetworkParams = match &self.nats_inference_server_address
        {
            Some(address) => address.clone(),
            None => NetworkParams {
                host: "127.0.0.1".to_string(),
                port: "50050".to_string(),
            },
        };

        let nats_training_server_address: NetworkParams = match &self.nats_training_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                host: "127.0.0.1".to_string(),
                port: "50051".to_string(),
            },
        };

        let zmq_inference_server_address: NetworkParams = match &self.zmq_inference_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                host: "127.0.0.1".to_string(),
                port: "50050".to_string(),
            },
        };

        let zmq_agent_listener_address: NetworkParams = match &self.zmq_agent_listener_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                host: "127.0.0.1".to_string(),
                port: "7778".to_string(),
            },
        };

        let zmq_model_server_address: NetworkParams = match &self.zmq_model_server_address {
            Some(address) => address.clone(),
            None => NetworkParams {
                host: "127.0.0.1".to_string(),
                port: "50051".to_string(),
            },
        };

        let zmq_trajectory_server_address: NetworkParams = match &self.zmq_trajectory_server_address
        {
            Some(address) => address.clone(),
            None => NetworkParams {
                host: "127.0.0.1".to_string(),
                port: "7776".to_string(),
            },
        };

        let zmq_inference_scaling_server_address: NetworkParams =
            match &self.zmq_inference_scaling_server_address {
                Some(address) => address.clone(),
                None => NetworkParams {
                    host: "127.0.0.1".to_string(),
                    port: "7777".to_string(),
                },
            };

        let zmq_training_scaling_server_address: NetworkParams =
            match &self.zmq_training_scaling_server_address {
                Some(address) => address.clone(),
                None => NetworkParams {
                    host: "127.0.0.1".to_string(),
                    port: "7778".to_string(),
                },
            };

        let max_traj_length: usize = match &self.max_traj_length {
            Some(length) => *length,
            None => 1000,
        };

        let local_model_module: LocalModelModuleParams = match &self.local_model_module {
            Some(module) => module.clone(),
            None => LocalModelModuleParams {
                directory: "model_module".to_string(),
                format: "pt".to_string(),
                model_name: "model".to_string(),
            },
        };

        TransportConfigParams {
            nats_addresses: NatsTransportAddresses {
                inference_server_address: nats_inference_server_address,
                training_server_address: nats_training_server_address,
            },
            zmq_addresses: ZmqTransportAddresses {
                inference_addresses: ZmqTransportInferenceAddresses {
                    inference_server_address: zmq_inference_server_address,
                    inference_scaling_server_address: zmq_inference_scaling_server_address,
                },
                training_addresses: ZmqTransportTrainingAddresses {
                    agent_listener_address: zmq_agent_listener_address,
                    model_server_address: zmq_model_server_address,
                    trajectory_server_address: zmq_trajectory_server_address,
                    training_scaling_server_address: zmq_training_scaling_server_address,
                },
            },
            max_traj_length,
            local_model_module,
        }
    }

    fn build_default() -> TransportConfigParams {
        TransportConfigParams {
            nats_addresses: NatsTransportAddresses {
                inference_server_address: NetworkParams {
                    host: "127.0.0.1".to_string(),
                    port: "50050".to_string(),
                },
                training_server_address: NetworkParams {
                    host: "127.0.0.1".to_string(),
                    port: "50051".to_string(),
                },
            },
            zmq_addresses: ZmqTransportAddresses {
                inference_addresses: ZmqTransportInferenceAddresses {
                    inference_server_address: NetworkParams {
                        host: "127.0.0.1".to_string(),
                        port: "50050".to_string(),
                    },
                    inference_scaling_server_address: NetworkParams {
                        host: "127.0.0.1".to_string(),
                        port: "7777".to_string(),
                    },
                },
                training_addresses: ZmqTransportTrainingAddresses {
                    agent_listener_address: NetworkParams {
                        host: "127.0.0.1".to_string(),
                        port: "7779".to_string(),
                    },
                    model_server_address: NetworkParams {
                        host: "127.0.0.1".to_string(),
                        port: "50051".to_string(),
                    },
                    trajectory_server_address: NetworkParams {
                        host: "127.0.0.1".to_string(),
                        port: "7776".to_string(),
                    },
                    training_scaling_server_address: NetworkParams {
                        host: "127.0.0.1".to_string(),
                        port: "7778".to_string(),
                    },
                },
            },
            max_traj_length: 1000,
            local_model_module: LocalModelModuleParams {
                directory: "model_module".to_string(),
                format: "pt".to_string(),
                model_name: "model".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const VALID_CLIENT_CONFIG_JSON: &str = r#"{
    "client_config": {
        "algorithm_name": "TD3",
        "config_update_polling_seconds": 5.0,
        "init_hyperparameters": {
            "DDPG": {
                "seed": 1,
                "gamma": 0.99,
                "tau": 1e-2,
                "learning_rate": 3e-3,
                "batch_size": 128,
                "buffer_size": 50000,
                "learning_starts": 128,
                "policy_frequency": 1,  
                "noise_scale": 0.1,
                "train_iters": 50
            },
            "PPO": {
                "discrete": true,
                "seed": 0,
                "traj_per_epoch": 1,
                "clip_ratio": 0.1,
                "gamma": 0.99,
                "lam": 0.97,
                "pi_lr": 3e-4,
                "vf_lr": 3e-4,
                "train_pi_iters": 40,
                "train_v_iters": 40,
                "target_kl": 0.01
            },
            "REINFORCE": {
                "discrete": true,
                "with_vf_baseline": true,
                "seed": 1,
                "traj_per_epoch": 8,
                "gamma": 0.98,
                "lam": 0.97,
                "pi_lr": 3e-4,
                "vf_lr": 1e-3,
                "train_vf_iters": 80
            },
            "TD3": {
                "seed": 1,
                "gamma": 0.99,
                "tau": 0.005,
                "learning_rate": 3e-4,
                "batch_size": 128,
                "buffer_size": 50000,
                "exploration_noise": 0.1,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "learning_starts": 25000,
                "policy_frequency": 2
            },
            "CUSTOM": {
                "_comment": "Add custom algorithm hyperparams here formatted just like the other algorithms. i.e. \"MAPPO\": {...}",
                "_comment2": "Make sure to add the algorithm name to the algorithm_name field",
                "_comment3": "These key-values will be sent to the server for initialization"
            }
        },
        "trajectory_file_output": {
            "directory": "experiment_data",
            "_comment": "use `Csv` or `Arrow`",
            "file_type": "Csv"
        },
        "metrics_meter_name": "my-custom-metric",
        "metrics_otlp_endpoint": {
            "prefix": "https://",
            "host": "0.0.0.0",
            "port": "9317"
        }
    },
    "transport_config": {
        "nats_addresses": {
            "inference_server_address": {
                "host": "127.0.0.1",
                "port": "50050"
            },
            "training_server_address": {
                "host": "127.0.0.1",
                "port": "50051"
            }
        },
        "zmq_addresses": {
            "inference_addresses": {
                "inference_server_address": {
                    "host": "127.0.0.1",
                    "port": "7800"
                },
                "inference_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7801"
                }
            },
            "training_addresses": {
                "model_server_address": {
                    "host": "127.0.0.1",
                    "port": "50051"
                },
                "trajectory_server_address": {
                    "host": "127.0.0.1",
                    "port": "7776"
                },
                "agent_listener_address": {
                    "host": "127.0.0.1",
                    "port": "7777"
                },
                "training_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7778"
                }
            }
        },
        "local_model_module": {
            "directory": "model_module",
            "model_name": "client_model",
            "format": "pt"
        },
        "max_traj_length": 1000
    }
}"#;

    const VALID_TRAINING_SERVER_CONFIG_JSON: &str = r#"{
    "training_server_config": {
        "config_update_polling_seconds": 10.0,
        "default_hyperparameters": {
            "DDPG": {
                "seed": 1,
                "gamma": 0.99,
                "tau": 1e-2,
                "learning_rate": 3e-3,
                "batch_size": 128,
                "buffer_size": 50000,
                "learning_starts": 128,
                "policy_frequency": 1,  
                "noise_scale": 0.1,
                "train_iters": 50
            },
            "PPO": {
                "discrete": true,
                "seed": 0,
                "traj_per_epoch": 1,
                "clip_ratio": 0.1,
                "gamma": 0.99,
                "lam": 0.97,
                "pi_lr": 3e-4,
                "vf_lr": 3e-4,
                "train_pi_iters": 40,
                "train_v_iters": 40,
                "target_kl": 0.01
            },
            "REINFORCE": {
                "discrete": true,
                "with_vf_baseline": true,
                "seed": 1,
                "traj_per_epoch": 8,
                "gamma": 0.98,
                "lam": 0.97,
                "pi_lr": 3e-4,
                "vf_lr": 1e-3,
                "train_vf_iters": 80
            },
            "TD3": {
                "seed": 1,
                "gamma": 0.99,
                "tau": 0.005,
                "learning_rate": 3e-4,
                "batch_size": 128,
                "buffer_size": 50000,
                "exploration_noise": 0.1,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "learning_starts": 25000,
                "policy_frequency": 2
            }
        },
        "training_tensorboard": {
            "_comment1": "Runs `tensorboard --logdir /logs` in cwd on start up of server.",
            "launch_tb_on_startup": true,
            "_comment2": "scalar tags can be any column header from `progress.txt` files.",
            "_comment3": "For more than one tag, separate by semi-colon (;)",
            "scalar_tags": "AverageEpRet;LossQ",
            "global_step_tag": "Epoch"
        }
    },
    "transport_config": {
        "nats_addresses": {
            "inference_server_address": {
                "host": "127.0.0.1",
                "port": "50050"
            },
            "training_server_address": {
                "host": "127.0.0.1",
                "port": "50051"
            }
        },
        "zmq_addresses": {
            "inference_addresses": {
                "inference_server_address": {
                    "host": "127.0.0.1",
                    "port": "7800"
                },
                "inference_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7801"
                }
            },
            "training_addresses": {
                "model_server_address": {
                    "host": "127.0.0.1",
                    "port": "50051"
                },
                "trajectory_server_address": {
                    "host": "127.0.0.1",
                    "port": "7776"
                },
                "agent_listener_address": {
                    "host": "127.0.0.1",
                    "port": "7777"
                },
                "training_scaling_server_address": {
                    "host": "127.0.0.1",
                    "port": "7778"
                }
            }
        },
        "local_model_module": {
            "directory": "model_module",
            "model_name": "some_server_model",
            "format": "pt"
        },
        "max_traj_length": 1000
    }
}"#;

    fn write_temp_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("failed to create temp file");
        file.write_all(content.as_bytes())
            .expect("failed to write temp file");
        file
    }

    #[test]
    fn algorithm_from_str_known_variants() {
        assert_eq!(Algorithm::from_str("DDPG"), Some(Algorithm::DDPG));
        assert_eq!(Algorithm::from_str("PPO"), Some(Algorithm::PPO));
        assert_eq!(Algorithm::from_str("REINFORCE"), Some(Algorithm::REINFORCE));
        assert_eq!(Algorithm::from_str("TD3"), Some(Algorithm::TD3));
        assert_eq!(
            Algorithm::from_str("CONFIG_INIT"),
            Some(Algorithm::ConfigInit)
        );
    }

    #[test]
    fn algorithm_from_str_unknown_returns_none() {
        assert_eq!(Algorithm::from_str("NOT_A_REAL_ALGORITHM"), None);
    }

    #[test]
    fn algorithm_from_str_mappo() {
        assert_eq!(Algorithm::from_str("MAPPO"), Some(Algorithm::MAPPO));
    }

    #[test]
    fn algorithm_as_str_round_trips_known() {
        assert_eq!(Algorithm::DDPG.as_str(), "DDPG");
        assert_eq!(Algorithm::PPO.as_str(), "PPO");
        assert_eq!(Algorithm::REINFORCE.as_str(), "REINFORCE");
        assert_eq!(Algorithm::TD3.as_str(), "TD3");
        assert_eq!(Algorithm::ConfigInit.as_str(), "CONFIG_INIT");
    }

    #[test]
    fn hyperparameter_default_reinforce_only() {
        let hp = HyperparameterConfig::default();
        assert!(hp.ddpg.is_none());
        assert!(hp.ppo.is_none());
        assert!(hp.reinforce.is_some());
        assert!(hp.td3.is_none());
    }

    #[test]
    fn hyperparameter_to_args_specific_algorithm_returns_expected_keys() {
        let hp = HyperparameterConfig::default();
        let args = hp.to_args(Some(&Algorithm::REINFORCE));
        assert_eq!(args.len(), 1);
        let entry = args
            .get(&Algorithm::REINFORCE)
            .expect("REINFORCE entry missing");
        if let HyperparameterArgs::Map(map) = entry {
            assert!(map.contains_key("seed"));
            assert!(map.contains_key("gamma"));
            assert!(map.contains_key("lambda"));
            assert!(map.contains_key("pi_lr"));
            assert!(map.contains_key("vf_lr"));
        } else {
            panic!("Expected HyperparameterArgs::Map");
        }
    }

    #[test]
    fn hyperparameter_to_args_none_returns_all_present_algorithms() {
        // Default only populates REINFORCE, so to_args(None) should return 1 entry.
        let hp = HyperparameterConfig::default();
        let args = hp.to_args(None);
        assert_eq!(args.len(), 1);
        assert!(args.contains_key(&Algorithm::REINFORCE));
    }

    #[test]
    fn transport_build_default_nats_addresses() {
        let transport = TransportConfigBuilder::build_default();
        let nats_inf = transport.get_nats_inference_server_address();
        let nats_train = transport.get_nats_training_server_address();
        assert_eq!(nats_inf.host, "127.0.0.1");
        assert_eq!(nats_inf.port, "50050");
        assert_eq!(nats_train.host, "127.0.0.1");
        assert_eq!(nats_train.port, "50051");
    }

    #[test]
    fn transport_build_default_zmq_addresses() {
        let transport = TransportConfigBuilder::build_default();
        let model_server = transport.get_zmq_model_server_address();
        let traj_server = transport.get_zmq_trajectory_server_address();
        assert_eq!(model_server.host, "127.0.0.1");
        assert_eq!(model_server.port, "50051");
        assert_eq!(traj_server.host, "127.0.0.1");
        assert_eq!(traj_server.port, "7776");
    }

    #[test]
    fn transport_builder_custom_nats_inference_address() {
        let mut builder = TransportConfigBuilder {
            nats_inference_server_address: None,
            nats_training_server_address: None,
            zmq_inference_server_address: None,
            zmq_agent_listener_address: None,
            zmq_model_server_address: None,
            zmq_trajectory_server_address: None,
            zmq_inference_scaling_server_address: None,
            zmq_training_scaling_server_address: None,
            max_traj_length: None,
            local_model_module: None,
        };
        builder.set_nats_inference_server_address("10.0.0.1", "9999");
        let transport = builder.build();
        let addr = transport.get_nats_inference_server_address();
        assert_eq!(addr.host, "10.0.0.1");
        assert_eq!(addr.port, "9999");
    }

    #[test]
    fn transport_builder_custom_max_traj_length() {
        let mut builder = TransportConfigBuilder {
            nats_inference_server_address: None,
            nats_training_server_address: None,
            zmq_inference_server_address: None,
            zmq_agent_listener_address: None,
            zmq_model_server_address: None,
            zmq_trajectory_server_address: None,
            zmq_inference_scaling_server_address: None,
            zmq_training_scaling_server_address: None,
            max_traj_length: None,
            local_model_module: None,
        };
        builder.set_max_traj_length(500);
        let transport = builder.build();
        assert_eq!(transport.max_traj_length, 500);
    }

    #[test]
    fn client_build_default_algorithm_is_reinforce() {
        let loader = ClientConfigBuilder::build_default();
        assert_eq!(loader.get_algorithm_name(), "REINFORCE");
    }

    #[test]
    fn client_build_default_config_path() {
        let loader = ClientConfigBuilder::build_default();
        assert_eq!(
            loader.get_config_path(),
            &PathBuf::from("client_config.json")
        );
    }

    #[test]
    fn client_build_default_polling_seconds() {
        let loader = ClientConfigBuilder::build_default();
        assert_eq!(loader.client_config.config_update_polling_seconds, 10.0_f32);
    }

    #[test]
    fn client_build_default_metrics_name() {
        let loader = ClientConfigBuilder::build_default();
        assert_eq!(loader.get_metrics_meter_name(), "relayrl-client");
    }

    #[test]
    fn client_build_default_otlp_endpoint() {
        let loader = ClientConfigBuilder::build_default();
        assert_eq!(
            loader.get_metrics_otlp_endpoint().prefix,
            "http://".to_string()
        );
        assert_eq!(
            loader.get_metrics_otlp_endpoint().host,
            "127.0.0.1".to_string()
        );
        assert_eq!(loader.get_metrics_otlp_endpoint().port, "4317".to_string());
    }

    #[test]
    fn client_build_default_hyperparameters_has_reinforce_only() {
        let loader = ClientConfigBuilder::build_default();
        let hp = loader.get_init_hyperparameters();
        assert!(hp.ddpg.is_none());
        assert!(hp.ppo.is_none());
        assert!(hp.reinforce.is_some());
        assert!(hp.td3.is_none());
    }

    #[test]
    fn client_builder_overrides_algorithm_name() {
        let mut builder = ClientConfigBuilder {
            algorithm_name: None,
            config_update_polling_seconds: None,
            init_hyperparameters: None,
            transport_config: None,
            trajectory_file_output: None,
            metrics_name: None,
            otlp_endpoint: None,
        };
        builder.set_algorithm_name("PPO");
        let loader = builder.build();
        assert_eq!(loader.get_algorithm_name(), "PPO");
    }

    #[test]
    fn client_builder_overrides_metrics_name() {
        let mut builder = ClientConfigBuilder {
            algorithm_name: None,
            config_update_polling_seconds: None,
            init_hyperparameters: None,
            transport_config: None,
            trajectory_file_output: None,
            metrics_name: None,
            otlp_endpoint: None,
        };
        builder.set_metrics_name("my-custom-metric");
        let loader = builder.build();
        assert_eq!(loader.get_metrics_meter_name(), "my-custom-metric");
    }

    #[test]
    fn client_builder_overrides_otlp_endpoint() {
        let mut builder = ClientConfigBuilder {
            algorithm_name: None,
            config_update_polling_seconds: None,
            init_hyperparameters: None,
            transport_config: None,
            trajectory_file_output: None,
            metrics_name: None,
            otlp_endpoint: None,
        };
        builder.set_otlp_endpoint(OtlpEndpointParams {
            prefix: "http://".to_string(),
            host: "0.0.0.0".to_string(),
            port: "9317".to_string(),
        });
        let loader = builder.build();
        assert_eq!(
            loader.get_metrics_otlp_endpoint().prefix,
            "http://".to_string()
        );
        assert_eq!(
            loader.get_metrics_otlp_endpoint().host,
            "0.0.0.0".to_string()
        );
        assert_eq!(loader.get_metrics_otlp_endpoint().port, "9317".to_string());
    }

    #[test]
    fn client_load_config_from_valid_json_file() {
        let temp = write_temp_file(VALID_CLIENT_CONFIG_JSON);
        let path = temp.path().to_path_buf();
        let loader = ClientConfigLoader::load_config(&path);
        assert_eq!(loader.get_algorithm_name(), "TD3");
        assert_eq!(loader.client_config.config_update_polling_seconds, 5.0_f32);
        assert_eq!(loader.get_metrics_meter_name(), "my-custom-metric");
        assert_eq!(
            loader.get_metrics_otlp_endpoint().prefix,
            "https://".to_string()
        );
        assert_eq!(
            loader.get_metrics_otlp_endpoint().host,
            "0.0.0.0".to_string()
        );
        assert_eq!(loader.get_metrics_otlp_endpoint().port, "9317".to_string());
    }

    #[test]
    fn client_load_config_transport_parsed_from_valid_json() {
        let temp = write_temp_file(VALID_CLIENT_CONFIG_JSON);
        let path = temp.path().to_path_buf();
        let loader = ClientConfigLoader::load_config(&path);
        let transport = loader.get_transport_config();
        assert_eq!(transport.get_nats_inference_server_address().port, "50050");
        assert_eq!(transport.get_zmq_trajectory_server_address().port, "7776");
        assert_eq!(transport.max_traj_length, 1000);
    }

    #[test]
    fn client_load_config_fallback_on_malformed_json() {
        let temp = write_temp_file("NOT VALID JSON {{{{");
        let path = temp.path().to_path_buf();
        let loader = ClientConfigLoader::load_config(&path);
        // Hardcoded fallback values (from unwrap_or_else closure)
        assert_eq!(loader.get_algorithm_name(), "REINFORCE");
        assert_eq!(loader.client_config.config_update_polling_seconds, 10.0_f32);
        assert_eq!(loader.get_metrics_meter_name(), "relayrl-client");
    }

    #[test]
    fn client_new_config_algorithm_override() {
        // JSON has algorithm_name "PPO"; passing Some("TD3") must win.
        let temp = write_temp_file(VALID_CLIENT_CONFIG_JSON);
        let path = temp.path().to_path_buf();
        let loader = ClientConfigLoader::new_config(Some("TD3".to_string()), Some(path));
        assert_eq!(loader.get_algorithm_name(), "TD3");
    }

    #[test]
    fn client_new_config_uses_file_algorithm_when_none_passed() {
        // JSON has algorithm_name "PPO"; passing None should keep it.
        let temp = write_temp_file(VALID_CLIENT_CONFIG_JSON);
        let path = temp.path().to_path_buf();
        let loader = ClientConfigLoader::new_config(None, Some(path.clone()));
        assert_eq!(loader.get_algorithm_name(), "TD3");
        // config_path in the struct is the path we provided, not the one inside the JSON.
        assert_eq!(loader.get_config_path(), &path);
    }

    #[test]
    fn training_server_build_default_config_path() {
        let loader = TrainingServerConfigBuilder::build_default();
        assert_eq!(
            loader.get_config_path(),
            &PathBuf::from("training_server_config.json")
        );
    }

    #[test]
    fn training_server_build_default_no_hyperparameters() {
        let loader = TrainingServerConfigBuilder::build_default();
        assert!(loader.get_hyperparameters().is_none());
    }

    #[test]
    fn training_server_build_default_tensorboard_not_launched() {
        let loader = TrainingServerConfigBuilder::build_default();
        assert!(!loader.get_training_tensorboard().launch_tb_on_startup);
    }

    #[test]
    fn training_server_build_default_tensorboard_scalar_tags() {
        let loader = TrainingServerConfigBuilder::build_default();
        let tags = &loader.get_training_tensorboard().scalar_tags;
        assert_eq!(
            tags,
            &vec!["AverageEpRet".to_string(), "StdEpRet".to_string()]
        );
    }

    #[test]
    fn training_server_builder_build_default_config_path() {
        let builder = TrainingServerConfigBuilder {
            config_update_polling_seconds: None,
            default_hyperparameters: None,
            training_tensorboard: None,
            transport_config: None,
        };
        let loader = builder.build();
        assert_eq!(
            loader.get_config_path(),
            &PathBuf::from("training_server_config.json")
        );
        assert_eq!(loader.get_config_update_polling_seconds(), 10.0);
        assert!(loader.get_hyperparameters().is_none());
        assert!(!loader.get_training_tensorboard().launch_tb_on_startup);
        assert_eq!(
            loader
                .get_transport_config()
                .get_nats_inference_server_address()
                .port,
            "50050"
        );
        assert_eq!(
            loader
                .get_transport_config()
                .get_zmq_trajectory_server_address()
                .port,
            "7776"
        );
        assert_eq!(loader.get_transport_config().max_traj_length, 1000);
        assert_eq!(
            loader.get_transport_config().local_model_module.directory,
            "model_module"
        );
        assert_eq!(
            loader.get_transport_config().local_model_module.model_name,
            "model"
        );
        assert_eq!(
            loader.get_transport_config().local_model_module.format,
            "pt"
        );
    }

    #[test]
    fn training_server_builder_overrides_tensorboard_params() {
        let mut builder = TrainingServerConfigBuilder {
            config_update_polling_seconds: None,
            default_hyperparameters: None,
            training_tensorboard: None,
            transport_config: None,
        };
        builder.set_training_tensorboard_params(true, "MetricA;MetricB", "Step");
        let loader = builder.build();
        let tb = loader.get_training_tensorboard();
        assert!(tb.launch_tb_on_startup);
        assert_eq!(
            tb.scalar_tags,
            vec!["MetricA".to_string(), "MetricB".to_string()]
        );
        assert_eq!(tb.global_step_tag, "Step");
    }

    #[test]
    fn training_server_load_config_from_valid_json_file() {
        let temp = write_temp_file(VALID_TRAINING_SERVER_CONFIG_JSON);
        let path = temp.path().to_path_buf();
        let loader = TrainingServerConfigLoader::load_config(&path);
        assert_eq!(loader.get_config_path(), &path);
        assert!(loader.get_hyperparameters().is_some());
    }

    #[test]
    fn training_server_load_config_tensorboard_tags_parsed_from_semicolon() {
        let temp = write_temp_file(VALID_TRAINING_SERVER_CONFIG_JSON);
        let path = temp.path().to_path_buf();
        let loader = TrainingServerConfigLoader::load_config(&path);
        let tb = loader.get_training_tensorboard();
        assert!(tb.launch_tb_on_startup);
        assert_eq!(
            tb.scalar_tags,
            vec!["AverageEpRet".to_string(), "LossQ".to_string()]
        );
        assert_eq!(tb.global_step_tag, "Epoch");
    }

    #[test]
    fn training_server_load_config_fallback_on_malformed_json() {
        let temp = write_temp_file("NOT VALID JSON {{{{");
        let path = temp.path().to_path_buf();
        let loader = TrainingServerConfigLoader::load_config(&path);
        // Hardcoded fallback values
        assert_eq!(loader.get_config_path(), &path);
        assert!(loader.get_hyperparameters().is_none());
        assert!(!loader.get_training_tensorboard().launch_tb_on_startup);
    }
}
