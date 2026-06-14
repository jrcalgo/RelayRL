use crate::network::client::runtime::actor::ErasedActorRuntime;
use crate::network::client::runtime::coordination::state_manager::{ActorUuid, env_dtype_to_dtype};
use crate::network::client::runtime::data::environments::{
    EnvironmentInterface, EnvironmentInterfaceError,
};

use relayrl_algorithms::algorithms::convert_byte_dtype_to_f32;
use relayrl_algorithms::prelude::nn::{NeuralNetwork, NeuralNetworkError};
use relayrl_algorithms::prelude::ppo::algorithm::{EpochTrainOutput, PPOParams};
use relayrl_algorithms::prelude::ppo::trainer::{PPOTrainer, PPOTrainerSpec};
use relayrl_algorithms::prelude::templates::AlgorithmError;
use relayrl_types::data::action::{RelayRLAction, RelayRLData};
use relayrl_types::data::tensor::{DType, NdArrayDType, SupportedTensorBackend, TensorData};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::ModelModule;
use relayrl_types::prelude::tensor::burn::{BasicOps, Numeric, TensorKind, backend::Backend};
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, DeviceType};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum TrainingError {
    #[error("Distribution error: {0}")]
    Distribution(String),
    #[error("Unsupported environment dtype: {0}")]
    UnsupportedEnvDType(String),
    #[error("Algorithm configuration error: {0}")]
    AlgorithmConfig(String),
    #[error("Temporary directory creation failed: {0}")]
    TempDirCreationFailed(std::io::Error),
    #[error("Trainer error: {0}")]
    TrainerError(String),
    #[error("Inference request error: {0}")]
    InferenceRequestError(String),
    #[error("Environment interface not found for actor {0}")]
    EnvironmentInterfaceNotFound(ActorUuid),
    #[error(transparent)]
    EnvironmentInterface(#[from] EnvironmentInterfaceError),
    #[error(transparent)]
    Actor(#[from] crate::network::client::runtime::actor::ActorError),
    #[error(transparent)]
    Algorithm(#[from] AlgorithmError),
    #[error(transparent)]
    NeuralNetwork(#[from] NeuralNetworkError),
}

pub(crate) struct TrainingInterface<B: Backend + BackendMatcher<Backend = B>> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrainingInterface<B> {
    pub(crate) fn train_ppo<KindIn, KindOut, Pi>(
        actor_id: ActorUuid,
        mut shutdown_rx: Option<tokio::sync::broadcast::Receiver<()>>,
        runtime: Arc<dyn ErasedActorRuntime<B>>,
        env_map: Arc<DashMap<ActorUuid, EnvironmentInterface>>,
        loop_iters: usize,
        max_traj_length: usize,
        trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, TrainingError>
    where
        KindIn: TensorKind<B> + burn_tensor::BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + Numeric<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Clone + Send + 'static,
        B: Default + Send + Sync + 'static,
    {
        #[inline(always)]
        async fn refresh_models<B2, KindIn2, KindOut2, Pi2>(
            runtime: &Arc<dyn ErasedActorRuntime<B2>>,
            actor_id: &ActorUuid,
            trainer: &mut PPOTrainer<B2, KindIn2, KindOut2, Pi2>,
            device: &DeviceType,
        ) -> Result<(), TrainingError>
        where
            B2: Backend + BackendMatcher<Backend = B2> + Default + Send + 'static,
            KindIn2: TensorKind<B2> + BasicOps<B2> + Send + 'static,
            KindOut2: TensorKind<B2> + BasicOps<B2> + Send + 'static,
            Pi2: NeuralNetwork<B2, KindIn2, KindOut2> + Send + 'static,
        {
            if let Some(pi_module) = trainer.acquire_pi_module() {
                runtime
                    .perform_env_refresh_model_erased("ppo_pi", pi_module, device.clone())
                    .await
                    .map_err(|e| {
                        TrainingError::TrainerError(format!(
                            "[TrainingInterface] {} - PPO Policy ModelModule refresh failed: {}",
                            actor_id, e
                        ))
                    })?;
            }
            if let Some(vf_module) = trainer.acquire_vf_module() {
                runtime
                    .perform_env_refresh_model_erased("ppo_vf", vf_module, device.clone())
                    .await
                    .map_err(|e| {
                        TrainingError::TrainerError(format!(
                            "[TrainingInterface] {} - PPO Value ModelModule refresh failed: {}",
                            actor_id, e
                        ))
                    })?;
            }
            Ok(())
        }

        #[inline(always)]
        fn log_epoch<B2, KindIn2, KindOut2, Pi2>(
            trainer: &mut PPOTrainer<B2, KindIn2, KindOut2, Pi2>,
            output: EpochTrainOutput<B2, KindIn2, KindOut2, Pi2>,
            epoch_count: &std::sync::atomic::AtomicU64,
        ) where
            B2: Backend + BackendMatcher<Backend = B2> + Default + Send + 'static,
            KindIn2: TensorKind<B2> + BasicOps<B2> + Send + 'static,
            KindOut2: TensorKind<B2> + BasicOps<B2> + Send + 'static,
            Pi2: NeuralNetwork<B2, KindIn2, KindOut2> + Send + 'static,
        {
            trainer.apply_epoch_result(output);
            trainer.log_epoch();
            epoch_count.fetch_add(1, std::sync::atomic::Ordering::Release);
        }

        let (max_episode_steps, rollout_len, traj_per_epoch, normalize_obs, device) =
            match &trainer_spec {
                PPOTrainerSpec::PPO {
                    args, hyperparams, ..
                }
                | PPOTrainerSpec::IPPO {
                    args, hyperparams, ..
                } => {
                    let p = hyperparams
                        .as_ref()
                        .map_or(PPOParams::default(), |hp| hp.clone());

                    (
                        p.max_episode_steps,
                        p.rollout_len,
                        p.traj_per_epoch as usize,
                        p.normalize_obs,
                        args.device.clone(),
                    )
                }
                _ => {
                    return Err(TrainingError::AlgorithmConfig(
                        "[TrainingInterface] Expected PPO/IPPO, got MAPPO".to_string(),
                    ));
                }
            };

        tokio::task::block_in_place(|| {
            let local_runtime = tokio::task::LocalSet::new();

            tokio::runtime::Handle::current().block_on(local_runtime.run_until(async {
                let mut env_interface = env_map.get_mut(&actor_id).ok_or(TrainingError::EnvironmentInterfaceNotFound(actor_id))?;

                env_interface.ensure_ready()?;

                let (n_envs, obs_dim, act_dim) = env_interface.n_envs_dims().ok_or(TrainingError::EnvironmentInterfaceNotFound(actor_id))?;

                let (traj_tx, mut traj_rx) = tokio::sync::mpsc::channel::<RelayRLTrajectory>(traj_per_epoch);

                let shared_epoch_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

                let mut trainer: PPOTrainer<B, KindIn, KindOut, Pi> = PPOTrainer::new(trainer_spec).map_err(TrainingError::Algorithm)?;
                trainer.register_first_slot_with_key(actor_id.to_string())?;
                refresh_models(&Arc::clone(&runtime), &actor_id, &mut trainer, &device).await?;
                let initial_kernel_snapshot = trainer
                    .get_ppo_actor_kernel()
                    .map_err(|e| TrainingError::TrainerError(e.to_string()))?
                    .to_arc_snapshot();
                let kernel_snapshot = Arc::new(ArcSwap::from_pointee(initial_kernel_snapshot));

                let learner_epoch_count = Arc::clone(&shared_epoch_count);
                let learner_runtime = Arc::clone(&runtime);
                let learner_device = device.clone();
                let learner_kernel_snapshot = Arc::clone(&kernel_snapshot);

                let learner_handle = tokio::task::spawn_local(async move {
                    let mut trainer = trainer;
                    let mut pending_train: Option<tokio::task::JoinHandle<EpochTrainOutput<B, KindIn, KindOut, Pi>>> = None;

                    loop {
                        if let Some(ref mut handle) = pending_train {
                            tokio::select! {
                                _ = async {
                                    if let Some(ref mut rx) = shutdown_rx {
                                        let _ = rx.recv().await;
                                    } else {
                                        std::future::pending::<()>().await;
                                    }
                                } => {
                                    break;
                                }

                                result = handle => {
                                    let output = result.map_err(|e| TrainingError::TrainerError(format!("[TrainingInterface] {} - PPO EpochTrainOutput join failed: {}", actor_id, e)))?;

                                    log_epoch::<B, KindIn, KindOut, Pi>(&mut trainer, output, &learner_epoch_count);
                                    refresh_models(&learner_runtime, &actor_id, &mut trainer, &learner_device).await?;
                                    let next_kernel_snapshot = trainer
                                        .get_ppo_actor_kernel()
                                        .map_err(|e| TrainingError::TrainerError(e.to_string()))?
                                        .to_arc_snapshot();
                                    learner_kernel_snapshot.store(Arc::new(next_kernel_snapshot));

                                    pending_train = trainer.start_epoch_training();
                                }

                                maybe_traj = traj_rx.recv() => {
                                    match maybe_traj {
                                        Some(traj) => {
                                            trainer.receive_trajectory(traj).await.map_err(|e| TrainingError::TrainerError(e.to_string()))?;
                                        }
                                        None => {
                                            if let Some(handle) = pending_train.take() &&
                                                let Ok(output) = handle.await {
                                                    log_epoch::<B, KindIn, KindOut, Pi>(&mut trainer, output, &learner_epoch_count);
                                                    refresh_models(&learner_runtime, &actor_id, &mut trainer, &learner_device).await?;
                                                    let next_kernel_snapshot = trainer
                                                        .get_ppo_actor_kernel()
                                                        .map_err(|e| TrainingError::TrainerError(e.to_string()))?
                                                        .to_arc_snapshot();
                                                    learner_kernel_snapshot.store(Arc::new(next_kernel_snapshot));
                                                }

                                            break;
                                        }
                                    }
                                }
                            }
                        } else {
                            match traj_rx.recv().await {
                                Some(traj) => {
                                    let epoch_started = trainer.receive_trajectory(traj).await.map_err(|e| TrainingError::TrainerError(e.to_string()))?;
                                    if epoch_started {
                                        pending_train = trainer.start_epoch_training();
                                    }
                                }
                                None => {
                                    break;
                                }
                            }
                        }

                    }

                    Ok::<ModelModule<B>, TrainingError>(trainer.acquire_pi_module().ok_or(TrainingError::TrainerError("Policy model not found".to_string()))?)
                });

                let mut per_env_trajs: Vec<RelayRLTrajectory> = (0..n_envs).map(|_| RelayRLTrajectory::new(max_traj_length)).collect();
                let mut per_env_episode: Vec<u64> = vec![0u64; n_envs];
                let mut per_env_step_count: Vec<usize> = vec![0usize; n_envs];
                let mut per_env_rollout_step: Vec<usize> = vec![0usize; n_envs];
                let mut per_env_episode_return: Vec<f32> = vec![0.0f32; n_envs];

                let mut completed_episodes: u64 = 0;
                let mut return_window: Vec<f32> = Vec::with_capacity(100);
                let mut last_printed_epoch: u64 = 0;

                let obs_dtype: DType = env_dtype_to_dtype(match &env_interface.obs_dtype() {
                    Some(d) => d,
                    None => return Err(TrainingError::EnvironmentInterfaceNotFound(actor_id)),
                })?;
                let act_dtype: DType = env_dtype_to_dtype(match &env_interface.act_dtype() {
                    Some(d) => d,
                    None => return Err(TrainingError::EnvironmentInterfaceNotFound(actor_id)),
                })?;

                let (obs_bytes_per_env, act_bytes_per_env) = {
                    #[inline(always)]
                    fn dtype_bytes_per_elem(dtype: &DType) -> usize {
                        match dtype {
                            DType::NdArray(nd) => match nd {
                                NdArrayDType::F16 | NdArrayDType::I16 => 2,
                                NdArrayDType::F32 | NdArrayDType::I32 => 4,
                                NdArrayDType::F64 | NdArrayDType::I64 => 8,
                                NdArrayDType::I8 | NdArrayDType::Bool => 1,
                            }
                            #[cfg(feature = "tch-backend")]
                            DType::Tch(tch) => match tch {
                                TchDType::F16 | TchDType::Bf16 | TchDType::I16 => 2,
                                TchDType::F32 | TchDType::I32 => 4,
                                TchDType::F64 | TchDType::I64 => 8,
                                TchDType::I8 | TchDType::U8 | TchDType::Bool => 1,
                            }
                        }
                    }

                    let obs_bytes = dtype_bytes_per_elem(&obs_dtype);
                    let act_bytes = dtype_bytes_per_elem(&act_dtype);

                    (obs_bytes, act_bytes)
                };

                let mut obs_normalizer = ObsNormalizer::new(obs_dim, obs_dtype.clone());

                let backend = B::get_supported_backend();
                let backend_f32_dtype = match &backend {
                    SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
                    #[cfg(feature = "tch-backend")]
                    SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
                    _ => panic!("Unsupported backend: {:?}", backend.clone()),
                };

                let (inference_obs_tx, mut inference_act_rx, inference_handle) = {
                    let (inf_obs_tx, mut inf_obs_rx) = tokio::sync::mpsc::channel::<(Vec<u8>, Option<Vec<u8>>)>(2);
                    let (inf_act_tx, inf_act_rx) = tokio::sync::mpsc::channel::<((Vec<u8>, Vec<u8>), Vec<f32>)>(2);
                    let inf_runtime = Arc::clone(&runtime);
                    let inf_obs_dtype = obs_dtype.clone();
                    let inf_act_dtype = act_dtype.clone();
                    let inf_backend_f32 = backend_f32_dtype.clone();
                    let inference_kernel_snapshot = Arc::clone(&kernel_snapshot);

                    let inference_handle = tokio::task::spawn_local(async move {
                        while let Some((obs_bytes, mask_bytes)) = inf_obs_rx.recv().await {
                            let kernel_snapshot = inference_kernel_snapshot.load_full();
                            let (raw_pi_output, raw_vf_output) = tokio::join!(
                                inf_runtime.perform_env_byte_inference_erased("ppo_pi", &obs_bytes, n_envs, obs_dim, &inf_obs_dtype),
                                inf_runtime.perform_env_byte_inference_erased("ppo_vf", &obs_bytes, n_envs, obs_dim, &inf_obs_dtype),
                            );

                            let pi_result = kernel_snapshot.policy_forward_bytes(&raw_pi_output?, mask_bytes.as_deref(), n_envs, &inf_act_dtype);
                            let vf_f32 = convert_byte_dtype_to_f32(raw_vf_output?.data, inf_backend_f32.clone())?;

                            match pi_result {
                                Ok((action_bytes, logp_bytes)) => {
                                    let msg = ((action_bytes, logp_bytes), vf_f32);
                                    if inf_act_tx.send(msg).await.is_err() {
                                        break;
                                    }
                                }
                                Err(_) => {
                                    break;
                                }
                            }
                        }

                        Ok::<(), TrainingError>(())
                    });

                    (inf_obs_tx, inf_act_rx, inference_handle)
                };

                let (mut current_obs_bytes, (mut current_act_bytes, mut current_logp_bytes), mut current_values) = {
                    let mut initial_obs_bytes = env_interface.flat_observation_bytes().ok_or({
                        TrainingError::EnvironmentInterfaceNotFound(actor_id)
                    })?;

                    if normalize_obs {
                        obs_normalizer.update(&initial_obs_bytes)?;
                        obs_normalizer.normalize(&mut initial_obs_bytes)?;
                    }
                    let initial_mask_bytes = env_interface.flat_mask_bytes();

                    inference_obs_tx.send((initial_obs_bytes.clone(), initial_mask_bytes.clone())).await.map_err(|e| TrainingError::InferenceRequestError(format!("[TrainingInterface] {} - PPO inference worker died at bootstrap values: {}", actor_id, e)))?;
                    let (initial_obs_bytes, (initial_act_bytes, initial_logp_bytes), initial_values) = match inference_act_rx.recv().await {
                        Some(result) => (initial_obs_bytes, (result.0.0, result.0.1), result.1),
                        None => return Err(TrainingError::InferenceRequestError(format!("[TrainingInterface] {} - PPO inference worker closed at bootstrap values", actor_id))),
                    };

                    (initial_obs_bytes, (initial_act_bytes, initial_logp_bytes), initial_values)
                };

                let loop_epoch_count = Arc::clone(&shared_epoch_count);
                for _ in 0..loop_iters {
                    let (mut new_obs_bytes, new_mask_bytes, rewards, dones, truncateds) = env_interface.step_bytes(&current_act_bytes).ok_or(TrainingError::EnvironmentInterfaceNotFound(actor_id))?;

                    if normalize_obs {
                        obs_normalizer.update(&new_obs_bytes)?;
                        obs_normalizer.normalize(&mut new_obs_bytes)?;
                    }

                    inference_obs_tx.send((new_obs_bytes.clone(), new_mask_bytes.clone())).await.map_err(|e| TrainingError::InferenceRequestError(format!("[TrainingInterface] {} - PPO inference worker died during collection: {}", actor_id, e)))?;

                    for i in 0..n_envs {
                        let obs_i = {
                            let start = i * obs_bytes_per_env;
                            TensorData::new(vec![obs_dim], obs_dtype.clone(), current_obs_bytes[start..start + obs_bytes_per_env].to_vec(), backend.clone())
                        };

                        let action_i = {
                            let start = i * act_bytes_per_env;
                            TensorData::new(vec![act_dim], act_dtype.clone(), current_act_bytes[start..start + act_bytes_per_env].to_vec(), backend.clone())
                        };

                        let mask_i = match new_mask_bytes {
                            Some(ref mask_bytes) => {
                                let start = i * 4;
                                Some(TensorData::new(vec![act_dim], backend_f32_dtype.clone(), mask_bytes[start..start + 4].to_vec(), backend.clone()))
                            }
                            None => None,
                        };

                        let data_map = {
                            let logp_i = {
                                let start = i * 4;
                                TensorData::new(vec![1], backend_f32_dtype.clone(), current_logp_bytes[start..start + 4].to_vec(), backend.clone())
                            };

                            let value_i = {
                                let bytes = current_values.get(i).copied().unwrap_or(0.0);
                                TensorData::new(vec![1], backend_f32_dtype.clone(), bytes.to_le_bytes().to_vec(), backend.clone())
                            };

                            let mut map: HashMap<String, RelayRLData> = HashMap::new();
                            map.insert("logp_a".to_string(), RelayRLData::Tensor(logp_i));
                            map.insert("val".to_string(), RelayRLData::Tensor(value_i));
                            map
                        };

                        per_env_episode_return[i] += rewards[i];
                        per_env_step_count[i] += 1;
                        if rollout_len.is_some() {
                            per_env_rollout_step[i] += 1;
                        }

                        let action_obj = RelayRLAction::new(
                            Some(obs_i),
                            Some(action_i),
                            mask_i,
                            rewards[i],
                            dones[i],
                            Some(data_map),
                            Some(actor_id),
                        );
                        per_env_trajs[i].add_action(action_obj);

                        if dones[i] || truncateds[i] {
                            let episode_return = per_env_episode_return[i];
                            per_env_episode_return[i] = 0.0;
                            return_window.push(episode_return);
                            if return_window.len() > 100 {
                                return_window.remove(0);
                            }
                            completed_episodes += 1;

                            let mut traj = std::mem::replace(
                                &mut per_env_trajs[i],
                                RelayRLTrajectory::new(max_traj_length),
                            );
                            traj.set_episode(per_env_episode[i]);
                            per_env_episode[i] += 1;

                            if truncateds[i] {
                                traj.set_truncated();
                            } else if let Some(max_steps) = max_episode_steps &&
                                per_env_step_count[i] >= max_steps {
                                    traj.set_truncated();
                                }


                            per_env_step_count[i] = 0;
                            per_env_rollout_step[i] = 0;

                            let _ = traj_tx.send(traj).await;

                            let current_epoch = loop_epoch_count.load(std::sync::atomic::Ordering::Acquire);
                            if current_epoch > last_printed_epoch {
                                last_printed_epoch = current_epoch;
                                let mean_ret = if return_window.is_empty() {
                                    0.0
                                } else {
                                    return_window.iter().sum::<f32>() / return_window.len() as f32
                                };
                                println!("[TrainingInterface] {} - Epoch {:>4} - Episodes={:>5} - MeanReturn={:>8.1} - LastEpisode={:>8.1}", actor_id, current_epoch, completed_episodes, mean_ret, episode_return);
                            }
                        } else if let Some(rl) = rollout_len &&
                            per_env_rollout_step[i] >= rl {
                                let mut traj = std::mem::replace(
                                    &mut per_env_trajs[i],
                                    RelayRLTrajectory::new(max_traj_length),
                                );
                                traj.set_episode(per_env_episode[i]);
                                traj.set_truncated();
                                per_env_rollout_step[i] = 0;
                                let _ = traj_tx.send(traj).await;
                                continue;
                            }
                    }
                    let ((next_action_bytes, next_logp_bytes), next_values) = inference_act_rx.recv().await.ok_or_else(|| TrainingError::InferenceRequestError(format!("[TrainingInterface] {} - PPO inference worker closed at loop iteration: {}", actor_id, loop_iters)))?;

                    current_obs_bytes = new_obs_bytes.clone();
                    current_act_bytes = next_action_bytes;
                    current_logp_bytes = next_logp_bytes;
                    current_values = next_values;
                }

                drop(inference_obs_tx);
                drop(traj_tx);
                let model_module = learner_handle.await.map_err(|e| TrainingError::TrainerError(e.to_string()))??;
                inference_handle.await.map_err(|e| TrainingError::InferenceRequestError(e.to_string()))??;

                Ok(model_module)
            }))
        })
    }

    pub(crate) fn train_ippo<KindIn, KindOut, Pi>(
        _actor_id: ActorUuid,
        _shutdown_rx: tokio::sync::broadcast::Receiver<()>,
        _runtime: Arc<dyn ErasedActorRuntime<B>>,
        _env_map: Arc<DashMap<ActorUuid, EnvironmentInterface>>,
        _loop_iters: usize,
        _max_traj_length: usize,
        _trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, TrainingError>
    where
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    {
        unimplemented!()
    }

    pub(crate) fn train_mappo<KindIn, KindOut, Pi>(
        _actor_id: ActorUuid,
        _shutdown_rx: tokio::sync::broadcast::Receiver<()>,
        _runtime: Arc<dyn ErasedActorRuntime<B>>,
        _env_map: Arc<DashMap<ActorUuid, EnvironmentInterface>>,
        _loop_iters: usize,
        _max_traj_length: usize,
        _trainer_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<ModelModule<B>, TrainingError>
    where
        KindIn: TensorKind<B> + BasicOps<B> + Send + 'static,
        KindOut: TensorKind<B> + BasicOps<B> + Send + 'static,
        Pi: NeuralNetwork<B, KindIn, KindOut> + Send + 'static,
    {
        unimplemented!()
    }
}

struct ObsNormalizer {
    mean: Vec<f64>,
    var: Vec<f64>,
    count: u64,
    obs_dtype: DType,
}
impl ObsNormalizer {
    fn new(obs_dim: usize, obs_dtype: DType) -> Self {
        Self {
            mean: vec![0.0; obs_dim],
            var: vec![1.0; obs_dim],
            count: 0,
            obs_dtype,
        }
    }

    fn update(&mut self, obs_bytes: &[u8]) -> Result<(), TrainingError> {
        enum DTypeSlice<'a> {
            F16(&'a [half::f16]),
            #[allow(unused)]
            Bf16(&'a [half::bf16]),
            F32(&'a [f32]),
            F64(&'a [f64]),
        }

        impl<'a> DTypeSlice<'a> {
            fn len(&self) -> usize {
                match self {
                    DTypeSlice::F16(slice) => slice.len(),
                    DTypeSlice::Bf16(slice) => slice.len(),
                    DTypeSlice::F32(slice) => slice.len(),
                    DTypeSlice::F64(slice) => slice.len(),
                }
            }

            fn get_f64(&self, index: usize) -> f64 {
                match self {
                    DTypeSlice::F16(slice) => f64::from(slice[index]),
                    DTypeSlice::Bf16(slice) => f64::from(slice[index]),
                    DTypeSlice::F32(slice) => slice[index] as f64,
                    DTypeSlice::F64(slice) => slice[index],
                }
            }
        }

        #[inline(always)]
        fn byte_slice_to_dtype(
            bytes: &[u8],
            byte_dtype: DType,
        ) -> Result<DTypeSlice<'_>, TrainingError> {
            let dtype_vec: DTypeSlice<'_> = match byte_dtype {
                DType::NdArray(nd) => match nd {
                    NdArrayDType::F16 => {
                        DTypeSlice::F16(bytemuck::cast_slice::<u8, half::f16>(bytes))
                    }
                    NdArrayDType::F32 => DTypeSlice::F32(bytemuck::cast_slice::<u8, f32>(bytes)),
                    NdArrayDType::F64 => DTypeSlice::F64(bytemuck::cast_slice::<u8, f64>(bytes)),
                    _ => {
                        return Err(TrainingError::AlgorithmConfig(format!(
                            "Unsupported byte dtype for NdArray Obs Normalizer: {:?}",
                            nd
                        )));
                    }
                },
                #[cfg(feature = "tch-backend")]
                DType::Tch(tch) => match tch {
                    TchDType::F16 => DTypeSlice::F16(bytemuck::cast_slice::<u8, half::f16>(bytes)),
                    TchDType::Bf16 => {
                        DTypeSlice::Bf16(bytemuck::cast_slice::<u8, half::bf16>(bytes))
                    }
                    TchDType::F32 => DTypeSlice::F32(bytemuck::cast_slice::<u8, f32>(bytes)),
                    TchDType::F64 => DTypeSlice::F64(bytemuck::cast_slice::<u8, f64>(bytes)),
                    _ => {
                        return Err(TrainingError::AlgorithmConfig(format!(
                            "Unsupported byte dtype for Tch Obs Normalizer: {:?}",
                            tch
                        )));
                    }
                },
                _ => {
                    return Err(TrainingError::AlgorithmConfig(
                        "Unsupported byte backend for Obs Normalizer".to_string(),
                    ));
                }
            };

            Ok(dtype_vec)
        }

        let obs_slice: DTypeSlice = byte_slice_to_dtype(obs_bytes, self.obs_dtype.clone())?;
        let obs_dim = self.mean.len(); // mean same size as obs_dim var
        let n_envs = obs_slice.len() / obs_dim;

        for i in 0..n_envs {
            self.count += 1;
            let start = i * obs_dim;

            for j in 0..obs_dim {
                let x = obs_slice.get_f64(start + j);
                let delta = x - self.mean[j];
                self.mean[j] += delta / self.count as f64;
                let delta2 = x - self.mean[j];
                self.var[j] += delta * delta2;
            }
        }

        Ok(())
    }

    fn normalize(&self, obs_bytes: &mut [u8]) -> Result<(), TrainingError> {
        enum DTypeSliceMut<'a> {
            F16(&'a mut [half::f16]),
            Bf16(&'a mut [half::bf16]),
            F32(&'a mut [f32]),
            F64(&'a mut [f64]),
        }

        impl<'a> DTypeSliceMut<'a> {
            fn len(&self) -> usize {
                match self {
                    DTypeSliceMut::F16(slice) => slice.len(),
                    DTypeSliceMut::Bf16(slice) => slice.len(),
                    DTypeSliceMut::F32(slice) => slice.len(),
                    DTypeSliceMut::F64(slice) => slice.len(),
                }
            }

            fn get_f64(&self, index: usize) -> f64 {
                match self {
                    DTypeSliceMut::F16(slice) => f64::from(slice[index]),
                    DTypeSliceMut::Bf16(slice) => f64::from(slice[index]),
                    DTypeSliceMut::F32(slice) => slice[index] as f64,
                    DTypeSliceMut::F64(slice) => slice[index],
                }
            }

            fn set_f64(&mut self, index: usize, value: f64) -> Result<(), TrainingError> {
                match self {
                    DTypeSliceMut::F16(slice) => slice[index] = half::f16::from_f64(value),
                    DTypeSliceMut::Bf16(slice) => slice[index] = half::bf16::from_f64(value),
                    DTypeSliceMut::F32(slice) => slice[index] = value as f32,
                    DTypeSliceMut::F64(slice) => slice[index] = value,
                }

                Ok(())
            }
        }

        #[inline(always)]
        fn byte_slice_to_dtype_mut(
            bytes: &mut [u8],
            byte_dtype: DType,
        ) -> Result<DTypeSliceMut<'_>, TrainingError> {
            let dtype_vec = match byte_dtype {
                DType::NdArray(nd) => match nd {
                    NdArrayDType::F16 => {
                        DTypeSliceMut::F16(bytemuck::cast_slice_mut::<u8, half::f16>(bytes))
                    }
                    NdArrayDType::F32 => {
                        DTypeSliceMut::F32(bytemuck::cast_slice_mut::<u8, f32>(bytes))
                    }
                    NdArrayDType::F64 => {
                        DTypeSliceMut::F64(bytemuck::cast_slice_mut::<u8, f64>(bytes))
                    }
                    _ => {
                        return Err(TrainingError::AlgorithmConfig(format!(
                            "Unsupported byte dtype for NdArray Obs Normalizer: {:?}",
                            nd
                        )));
                    }
                },
                #[cfg(feature = "tch-backend")]
                DType::Tch(tch) => match tch {
                    TchDType::F16 => {
                        DTypeSliceMut::F16(bytemuck::cast_slice_mut::<u8, half::f16>(bytes))
                    }
                    TchDType::Bf16 => {
                        DTypeSliceMut::Bf16(bytemuck::cast_slice_mut::<u8, half::bf16>(bytes))
                    }
                    TchDType::F32 => DTypeSliceMut::F32(bytemuck::cast_slice_mut::<u8, f32>(bytes)),
                    TchDType::F64 => DTypeSliceMut::F64(bytemuck::cast_slice_mut::<u8, f64>(bytes)),
                    _ => {
                        return Err(TrainingError::AlgorithmConfig(format!(
                            "Unsupported byte dtype for Tch Obs Normalizer: {:?}",
                            tch
                        )));
                    }
                },
                _ => {
                    return Err(TrainingError::AlgorithmConfig(
                        "Unsupported byte backend for Obs Normalizer".to_string(),
                    ));
                }
            };

            Ok(dtype_vec)
        }

        let mut obs_slice = byte_slice_to_dtype_mut(obs_bytes, self.obs_dtype.clone())?;
        let obs_dim = self.mean.len(); // mean same size as obs_dim var
        let n_envs = obs_slice.len() / obs_dim;

        for i in 0..n_envs {
            for j in 0..obs_dim {
                let std = (self.var[j] / self.count.max(1) as f64).sqrt().max(1e-4);
                let norm = (obs_slice.get_f64(i * obs_dim + j) - self.mean[j]) / std;
                obs_slice.set_f64(i * obs_dim + j, norm)?;
            }
        }

        Ok(())
    }
}
