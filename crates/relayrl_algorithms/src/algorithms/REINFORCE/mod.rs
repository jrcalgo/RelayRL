pub mod kernel;
pub mod replay_buffer;

pub use kernel::*;
pub use replay_buffer::*;

struct RuntimeArgs {
    env_dir: PathBuf,
    config_path: PathBuf,
    obs_dim: u32,
    act_dim: u32,
    buffer_size: usize
}

struct RuntimeComponents<B: Backend + BackendMatcher, K: StepKernelTrait<B>> {
    save_model_path: PathBuf,
    epoch_logger: EpochLogger,
    traj_count: u64,
    epoch_count: u64,
    kernel: K,
    replay_buffer: ReinforceReplayBuffer,
    policy_optimizer: OptimizerAdaptor<Adam, M, B>,
    value_optimizer: Option<OptimizerAdaptor<Adam, M, B>>,
}

struct RuntimeParams<B: Backend + BackendMatcher> {
    args: RuntimeArgs,
    components: RuntimeComponents<B, K>
}

pub struct ReinforceAlgorithm<B: Backend + BackendMatcher> {
    runtime: RuntimeParams<B>,
    hyperparams: REINFORCEParams,
}

impl ReinforceAlgorithm {
    pub fn new(hyperparams: Option<REINFORCEParams>, env_dir: Path, save_model_path: Path, obs_dim: u32, act_dim: u32, buffer_size: usize) -> Self {
        let hyperparams = match hyperparams {
            Some(params) => params,
            None => REINFORCEParams::default()
        };

        let trajectory_count: u64 = 0;
        let epoch_count: u64 = 0;

        let replay_buffer = ReinforceReplayBuffer::new(buffer_size, hyperparams.gamma, hyperparams.lambda, hyperparams.with_vf_baseline);

        let (kernel, value_optimizer): (dyn StepKernelTrait, Option<Adam>) = if hyperparams.with_vf_baseline {
            let optimizer_init: OptimizerAdaptor<Adam, M, B> = AdamConfig::new().init<M, B>();
            (PolicyWithBaseline(obs_dim, act_dim, hyperparams.discrete), Some(optimizer_init))
        } else {
            (PolicyWithoutBaseline(obs_dim, act_dim, hyperparams.discrete), None)
        };

        let policy_optimizer: OptimizerAdaptor<Adam, M, B> = AdamConfig::new().init<M, B>();

        Self {
            runtime: {
                args: RuntimeArgs {
                    env_dir,
                    save_model_path,
                    obs_dim,
                    act_dim,
                    buffer_size
                },
                components: RuntimeComponents {
                    epoch_logger,
                    trajectory_count,
                    epoch_count,
                    kernel,
                    replay_buffer,
                    policy_optimizer,
                    value_optimizer
                }
            },
            hyperparams
        }
    }

    fn compute_policy_loss(&self, batch: &Batch) {
        let obs = batch.get(BatchKey::Obs);
        let act = batch.get(BatchKey::Act);
        let mask = batch.get(BatchKey::Mask);
        let adv = batch.get(BatchKey::Custom("Adv".to_string()));
        let old_log = batch.get(BatchKey::Custom("LogP".to_string()));

        let (policy_tensor, policy_logits, logp_act) = self.kernel.policy.forward(obs, mask, act);
        let loss_pi = Math::mean(-(logp_act * adv));

        let approximate_kl = Math::mean(old_logp - logp);

        // entropy calc
        let min_real =
        let logits =
    }

    fn compute_value_loss(&self, batch: &Batch) {
        let obs = batch.get(BatchKey::Obs);
        let mask = batch.get(BatchKey::Act);
        let ret = batch.get(BatchKey::Custom("Ret"));

        Math::mean(self.kernel.baseline.forward(obs, mask) ** 2)
    }
}

impl<B: Backend + BackendMatcher> AlgorithmTrait<B> for ReinforceAlgorithm<B> {
    fn save(&self, filename: &str) {

    }

    async fn receive_trajectoy<T: TrajectoryData>(&self, trajectory: T) -> Result<bool, AlgorithmError> {
        self.runtime.components.trajectory_count += 1;

        let extracted_traj: RelayRLTrajectory = match trajectory::get_trajectory() {
            TrajectoryType::RelayRL(relayrl_traj) => relayrl_traj,
            TrajectoryType::Csv(csv_traj) => {
                csv_traj.trajectory
            },
            TrajectoryType::Arrow(arrow_traj) => {
                arrow_traj.trajectory
            }
        };

        let (episode_return, episode_length) = self.runtime.components.replay_buffer.insert_trajectory(extracted_traj).await.map_error(AlgorithmError::from)?;

        if self.runtime.components.trajectory_count > 0 && self.runtime.components.trajectory_count % self.hyperparams.traj_per_epoch {
            self.runtime.components.epoch_count += 1;
            self.train_model();
            self.log_epoch();
            return Ok(true);
        }

        return Ok(false);
    }

    fn train_model(&self) {
        let batch: Batch = self.runtime.components.replay_buffer.sample_buffer();

        let (old_policy_loss, old_policy_info) = self.compute_policy_loss(&batch);

        let old_value_loss = if self.hyperparams.with_vf_baseline {
            self.compute_value_loss(&batch)
        } else {
            None
        };

        // zero the pi optimizer gradients
        let (policy_loss, policy_info) = self.compute_policy_loss(&batch);
        // back propogate loss through kernel
        // then take a pi optimizer step

        let value_loss = if self.hyperparams.with_vf_baseline {
            let mut loss =
            for i in self.hyperparams.train_vf_iters.iter() {
                // zero the vf optimizer gradients
                loss = self.compute_value_loss(&batch);
                // back propogate loss through kernel
                // then take a vf optimizer step
            }
            Some(loss)
        } else {
            None
        };

        let (kl_divergence, entropy) = (policy_info.get("kl_divergence"), policy_info.get("entropy"));

        let policy_loss_delta = policy_loss - old_policy_loss;
        let value_loss_delta = if self.hyperparams.with_vf_baseline {
            value_loss - old_value_loss
        } else {
            0
        };

        // store pi, pi_delta, vf, and vf_delta loss in logger

    }

    fn log_epoch(&self) {

    }
}
