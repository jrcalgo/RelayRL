use crate::network::client::agent::{AlgorithmCfg, ReplayBufferSize, SaveModelPath};
use crate::network::client::agent::PPONetworks;
use crate::network::client::runtime::actor::ActorRuntime;
use crate::network::client::runtime::coordination::state_manager::{ActorUuid, env_dtype_to_dtype};
use crate::network::client::runtime::data::environments::EnvironmentInterface;

use relayrl_env_trait::{EnvDType, EnvNdArrayDType, EnvTchDType};
use relayrl_types::prelude::tensor::burn::{TensorKind, backend::Backend};
use relayrl_types::prelude::tensor::relayrl::{BackendMatcher, DeviceType};

use dashmap::DashMap;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum TrainingError {
    #[error("Distribution error: {0}")]
    DistributionError(String),
    #[error("Unsupported environment dtype: {0}")]
    UnsupportedEnvDType(String),
    #[error("Algorithm configuration error: {0}")]
    AlgorithmConfigError(String),
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
    Algorithm(#[from] relayrl_algorithms::AlgorithmError),
}

#[derive(thiserror::Error, Debug)]
pub enum PPOTrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum REINFORCETrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum DDPGTrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum TD3TrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum MATD3TrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum MAPPOTrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum MAREINFORCETrainingError {}

#[derive(thiserror::Error, Debug)]
pub enum MADDPGTrainingError {}

const PARALLEL_ITER_MIN_ENVS: usize = 8;

pub(crate) struct TrainingInterface<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    _phantom: PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    TrainingInterface<B, D_IN, D_OUT>
{
    pub(crate) fn train_ppo<KindIn, KindOut, Pi>(
        actor_id: ActorUuid,
        shutdown_rx: Option<tokio::sync::broadcast::Receiver<()>>,
        runtime: Arc<ActorRuntime<B, D_IN, D_OUT>>,
        env_map: Arc<DashMap<ActorUuid, EnvironmentInterface>>,
        step_count: usize,
        max_traj_length: usize,
        training_spec: PPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), TrainingError> where KindIn: TensorKind<B> + burn_tensor::BasicOps<B> + Send + 'static, KindOut: TensorKind<B> + burn_tensor::Numeric<B> + Send + 'static, Pi: NeuralNetworkSpec<B, KindIn, KindOut> + NeuralNetworkForward<B, KindIn, KindOut>
    {
        use relayrl_algorithms::templates::base_algorithm::AlgorithmTrait;
        use relayrl_algorithms::prelude::ppo::trainer::{PpoTrainer, PPOTrainerSpec};
        use relayrl_types::data::trajectory::RelayRLTrajectory;
        use relayrl_types::data::action::{RelayRLAction, RelayRLData};
        use relayrl_types::data::tensor::{
            DType, NdArrayDType, SupportedTensorBackend, TensorData,
        };
        use std::collections::HashMap;

        #[inline(always)]
        fn build_ppo_action(
            raw_model_output: &TensorData,
            mask_bytes: Option<&[u8]>,
            n_envs: usize,
            act_dim: usize,
            act_dtype: &EnvDType,
            discrete: bool,
        ) -> Result<(Vec<u8>, Vec<u8>), TrainingError> {
            let (logits, act_dtype_byte_count) = match raw_model_output.dtype {
                DType::NdArray(nd) => match nd {
                    NdArrayDType::F16 => (
                        bytemuck::cast_slice::<u8, half::f16>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x.to_f32())
                            .collect(),
                        2 as usize,
                    ),
                    NdArrayDType::F32 => (
                        bytemuck::cast_slice::<u8, f32>(&raw_model_output.data).to_vec(),
                        4 as usize,
                    ),
                    NdArrayDType::F64 => (
                        bytemuck::cast_slice::<u8, f64>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        8 as usize,
                    ),
                    NdArrayDType::I8 => (
                        bytemuck::cast_slice::<u8, i8>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        1 as usize,
                    ),
                    NdArrayDType::I16 => (
                        bytemuck::cast_slice::<u8, i16>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        2 as usize,
                    ),
                    NdArrayDType::I32 => (
                        bytemuck::cast_slice::<u8, i32>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        4 as usize,
                    ),
                    NdArrayDType::I64 => (
                        bytemuck::cast_slice::<u8, i64>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        8 as usize,
                    ),
                    NdArrayDType::Bool => (
                        raw_model_output
                            .data
                            .iter()
                            .map(|&x| if x != 0 { 1.0f32 } else { 0.0f32 })
                            .collect(),
                        1 as usize,
                    ),
                },
                #[cfg(feature = "tch-backend")]
                DType::Tch(tch) => match tch {
                    TchDType::F16 => (
                        bytemuck::cast_slice::<u8, half::f16>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x.to_f32())
                            .collect(),
                        2 as usize,
                    ),
                    TchDType::Bf16 => (
                        bytemuck::cast_slice::<u8, half::bf16>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x.to_f32())
                            .collect(),
                        2 as usize,
                    ),
                    TchDType::F32 => (
                        bytemuck::cast_slice::<u8, f32>(&raw_model_output.data).to_vec(),
                        4 as usize,
                    ),
                    TchDType::F64 => (
                        bytemuck::cast_slice::<u8, f64>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        8 as usize,
                    ),
                    TchDType::I8 => (
                        bytemuck::cast_slice::<u8, i8>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        1 as usize,
                    ),
                    TchDType::I16 => (
                        bytemuck::cast_slice::<u8, i16>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        2 as usize,
                    ),
                    TchDType::I32 => (
                        bytemuck::cast_slice::<u8, i32>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        4 as usize,
                    ),
                    TchDType::I64 => (
                        bytemuck::cast_slice::<u8, i64>(&raw_model_output.data)
                            .iter()
                            .map(|&x| x as f32)
                            .collect(),
                        8 as usize,
                    ),
                    TchDType::U8 => (
                        bytemuck::cast_slice::<u8, u8>(&raw_model_output.data).to_vec(),
                        1 as usize,
                    ),
                    TchDType::Bool => (
                        raw_model_output
                            .data
                            .iter()
                            .map(|&x| if x != 0 { 1.0f32 } else { 0.0f32 })
                            .collect(),
                        1 as usize,
                    ),
                },
            };

            let logp_bytes = Vec::<u8>::with_capacity(n_envs * 4); // 4 == f32
            let mut rng: rand::prelude::ThreadRng = rand::rng();

            let (action_bytes, act_idx, logps) = match discrete {
                true => {
                    let action_bytes = Vec::<u8>::with_capacity(n_envs * act_dtype_byte_count);

                    let (act_idx, logps) = match n_envs {
                        _ if n_envs < PARALLEL_ITER_MIN_ENVS => (0..n_envs)
                            .into_iter()
                            .map(|i| discrete_ppo_action(i, &logits, mask_bytes, act_dim, &mut rng))
                            .unzip(),
                        _ => (0..n_envs)
                            .into_par_iter()
                            .map(|i| discrete_ppo_action(i, &logits, mask_bytes, act_dim, &mut rng))
                            .unzip(),
                    };

                    (action_bytes, act_idx, logps)
                }
                false => {
                    let action_bytes =
                        Vec::<u8>::with_capacity(n_envs * act_dim * act_dtype_byte_count);

                    let stride = act_dim.saturating_mul(2);

                    let (act_idx, logps) = match n_envs {
                        _ if n_envs < PARALLEL_ITER_MIN_ENVS => (0..n_envs)
                            .into_iter()
                            .map(|i| continuous_ppo_action(i, &logits, act_dim, stride, &mut rng))
                            .unzip(),
                        _ => (0..n_envs)
                            .into_par_iter()
                            .map(|i| continuous_ppo_action(i, &logits, act_dim, stride, &mut rng))
                            .unzip(),
                    };

                    (action_bytes, act_idx, logps)
                }
            };

            logp_bytes.extend_from_slice(&logp.to_le_bytes());

            match act_dtype {
                EnvDType::NdArray(nd) => match nd {
                    EnvNdArrayDType::I8 => {
                        action_bytes.extend_from_slice(&(act_idx as i8).to_le_bytes())
                    }
                    EnvNdArrayDType::I16 => {
                        action_bytes.extend_from_slice(&(act_idx as i16).to_le_bytes())
                    }
                    EnvNdArrayDType::I32 => {
                        action_bytes.extend_from_slice(&(act_idx as i32).to_le_bytes())
                    }
                    EnvNdArrayDType::I64 => action_bytes.extend_from_slice(&act_idx.to_le_bytes()),
                    EnvNdArrayDType::F16 => action_bytes
                        .extend_from_slice(&half::f16::from_bits(act_idx as u16).to_le_bytes()),
                    EnvNdArrayDType::F32 => {
                        action_bytes.extend_from_slice(&(act_idx as f32).to_le_bytes())
                    }
                    EnvNdArrayDType::F64 => {
                        action_bytes.extend_from_slice(&(act_idx as f64).to_le_bytes())
                    }
                    EnvNdArrayDType::Bool => {
                        action_bytes.extend_from_slice(&if act_idx != 0 { [1u8] } else { [0u8] })
                    }
                },
                #[cfg(feature = "tch-backend")]
                EnvDType::Tch(tch) => match tch {
                    EnvTchDType::I8 => {
                        action_bytes.extend_from_slice(&(act_idx as i8).to_le_bytes())
                    }
                    EnvTchDType::I16 => {
                        action_bytes.extend_from_slice(&(act_idx as i16).to_le_bytes())
                    }
                    EnvTchDType::I32 => {
                        action_bytes.extend_from_slice(&(act_idx as i32).to_le_bytes())
                    }
                    EnvTchDType::I64 => action_bytes.extend_from_slice(&act_idx.to_le_bytes()),
                    EnvTchDType::F16 => action_bytes
                        .extend_from_slice(&half::f16::from_bits(act_idx as u16).to_le_bytes()),
                    EnvTchDType::Bf16 => action_bytes
                        .extend_from_slice(&half::bf16::from_bits(act_idx as u16).to_le_bytes()),
                    EnvTchDType::F32 => {
                        action_bytes.extend_from_slice(&(act_idx as f32).to_le_bytes())
                    }
                    EnvTchDType::F64 => {
                        action_bytes.extend_from_slice(&(act_idx as f64).to_le_bytes())
                    }
                    EnvTchDType::U8 => {
                        action_bytes.extend_from_slice(&(act_idx as u8).to_le_bytes())
                    }
                    EnvTchDType::Bool => {
                        action_bytes.extend_from_slice(&if act_idx != 0 { [1u8] } else { [0u8] })
                    }
                    _ => return Err(TrainingError::UnsupportedEnvDType(act_dtype.to_string())),
                },
                #[cfg(not(feature = "tch-backend"))]
                _ => return Err(TrainingError::UnsupportedEnvDType(act_dtype.to_string())),
            }

            Ok((action_bytes, logp_bytes))
        }

        fn trajectory_send_procedure() {}

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut env_interface = env_map.get_mut(&actor_id).ok_or_else(|| {
                    TrainingError::EnvironmentInterfaceNotFound(actor_id)
                })?;

                env_interface.ensure_ready()?;

                let (n_envs, obs_dim, act_dim) = env_interface.n_envs_dims().ok_or_else(|| {
                    TrainingError::EnvironmentInterfaceNotFound(actor_id)
                })?;

                let env_context = env_interface.get_env_context().unwrap_or_else(|| {
                    format!("{}:ppo:{}", ENVIRONMENT_CONTEXT_PREFIX, actor_id)
                });

                let obs_dim = env_interface.obs_dim().unwrap_or(0);
                let act_dim = env_interface.act_dim().unwrap_or(0);;
                
                let (_temp_env_dir, trainer_args) = {
                    let temp_env_dir = tempfile::tempdir().map_err(|e| TrainingError::TempDirCreationFailed(e))?;
                    let temp_env_dir_path = temp_env_dir.path().to_path_buf();
                    let trainer_args = TrainerArgs {
                        env_dir: temp_env_dir_path,
                        save_model_path: save_model_path.clone(),
                        obs_dim,
                        act_dim,
                        buffer_size: replay_buffer_size,
                    };
                    (temp_env_dir, trainer_args)
                };

                let discrete = env_interface.action_is_discrete().ok_or_else(|| {
                    TrainingError::EnvironmentInterfaceNotFound(actor_id)
                })?;

                let max_episode_steps: Option<usize> = match &algorithm_cfg {
                    AlgorithmCfg::PPO(Some(p)) | AlgorithmCfg::IPPO(Some(p)) => p.max_episode_steps,
                    _ => None,
                };
                let spec = match algorithm_cfg {
                    AlgorithmCfg::PPO(params) => PpoTrainerSpec::ppo(trainer_args, params),
                    AlgorithmCfg::IPPO(params) => PpoTrainerSpec::ippo(trainer_args, params),
                    other => return Err(TrainingError::AlgorithmConfigError(format!("[StateManager] Expected PPO/IPPO, got {:?}", other))),
                };

                let mut trainer: PpoTrainer<B, KindIn, KindOut, KN> = PpoTrainer::new(spec, kernel).map_err(TrainingError::from)?;

                trainer.register_first_slot_with_key(actor_id.to_string());

                if let Some(model_module) = trainer.acquire_model_module() {
                    runtime.perform_refresh_model(model_module, device.clone())
                        .await
                        .map_err(|e| TrainingError::TrainerError(
                            format!("[StateManager] ORT policy prime failed: {}", e)
                        ))?;
                }

                if let Some(vf_module) = trainer.acquire_value_module() {
                    runtime.perform_refresh_value_model(vf_module, device.clone())
                        .await
                        .map_err(|e| TrainingError::TrainerError(
                            format!("[StateManager] ORT value prime failed: {}", e)
                        ))?;
                }

                let (traj_tx, mut traj_rx) = tokio::sync::mpsc::channel::<RelayRLTrajectory>(2);
                let atomic_epoch_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
                let shared_epoch_count = Arc::clone(&atomic_epoch_count);
                let shared_runtime = Arc::clone(&runtime);
                let device = device.clone();

                let learner_task = tokio::task::spawn(async move {
                    let mut trainer = trainer;
                    
                    loop {
                        tokio::select! {
                            biased;

                            _ = async {
                                if let Some(rx) = &mut shutdown_rx {
                                    let _ = rx.recv().await;
                                } else {
                                    std::future::pending::<()>().await;
                                }
                            } => {
                                break;
                            }
    
                            _ = Some(traj) = traj_rx.recv() => {
                                let trained = AlgorithmTrait::<RelayRLTrajectory>::receive_trajectory(&mut trainer, traj).await.map_err(|e| TrainingError::TrainerError(e.to_string()))?;
    
                                if trained {
                                    shared_epoch_count.fetch_add(1, std::sync::atomic::Ordering::Release);
                                    
                                    if let Some(pi_module) = trainer.acquire_model_module() {
                                        let _ = shared_runtime.perform_env_refresh_model("ppo_pi", pi_module, device).await;
                                    }
                                    if let Some(vf_module) = trainer.acquire_value_module() {
                                        let _ = shared_runtime.perform_env_refresh_model("ppo_vf", vf_module, device).await;
                                    }
                                }
                            }
                        }
                    }
                });

                let mut per_env_trajs: Vec<RelayRLTrajectory> = (0..n_envs).map(|_| RelayRLTrajectory::new(max_traj_length)).collect();
                let mut per_env_episode: Vec<u64> = vec![0u64; n_envs];
                let mut per_env_step_count: Vec<usize> = vec![0usize; n_envs];
                
                let mut per_env_episode_return: Vec<f32> = vec![0.0f32; n_envs];
                let mut completed_episodes: u64 = 0;
                let mut return_window: Vec<f32> = Vec::with_capacity(100);
                let mut last_printed_epoch: u64 = 0;

                let obs_dtype = env_interface.obs_dtype().unwrap_or(EnvDType::NdArray(EnvNdArrayDType::F32));
                let act_dtype = env_interface.act_dtype().unwrap_or(EnvDType::NdArray(EnvNdArrayDType::F32));

                let (obs_bytes_per_env, act_bytes_per_env) = {
                    fn dtype_bytes_per_elem(dtype: &EnvDType) -> usize {
                        match dtype {
                            EnvDType::NdArray(nd) => match nd {
                                EnvNdArrayDType::F16 | EnvNdArrayDType::I16 => 2,
                                EnvNdArrayDType::F32 | EnvNdArrayDType::I32 => 4,
                                EnvNdArrayDType::F64 | EnvNdArrayDType::I64 => 8,
                                EnvNdArrayDType::I8 | EnvNdArrayDType::Bool => 1,
                            }
                            #[cfg(feature = "tch-backend")]
                            EnvDType::Tch(tch) => match tch {
                                EnvTchDType::F16 | EnvTchDType::Bf16 | EnvTchDType::I16 => 2,
                                EnvTchDType::F32 | EnvTchDType::I32 => 4,
                                EnvTchDType::F64 | EnvTchDType::I64 => 8,
                                EnvTchDType::I8 | EnvTchDType::U8 | EnvTchDType::Bool => 1,
                            }
                        }
                    }

                    let obs_bytes = dtype_bytes_per_elem(&obs_dtype);
                    let act_bytes = dtype_bytes_per_elem(&act_dtype);

                    (obs_bytes, act_bytes)
                };

                let mut obs_bytes = env_interface.flat_observation_bytes().ok_or_else(|| {
                    TrainingError::EnvironmentInterfaceNotFound(actor_id)
                })?;
                let mask_bytes = env_interface.flat_mask_bytes().ok_or_else(|| {
                    TrainingError::EnvironmentInterfaceNotFound(actor_id)
                })?;

                for _ in 0..step_count {
                    let (action_bytes, logp_bytes) = {
                        let raw_pi_output = runtime.perform_env_byte_inference("policy",
                            &obs_bytes, n_envs, obs_dim, &obs_dtype,
                        )
                        .await
                        .map_err(|e| TrainingError::InferenceRequestError(e.to_string()))?;

                        build_ppo_action(&raw_pi_output, None, n_envs, act_dim, &act_dtype, discrete)?
                    };

                    let values_f32 = {
                        let raw_vf_output = runtime.perform_env_byte_inference("value", &obs_bytes, n_envs, obs_dim, &obs_dtype,
                        )
                        .await
                        .map_err(|e| TrainingError::InferenceRequestError(e.to_string()))?;

                        bytemuck::cast_slice::<u8, f32>(&raw_vf_output.data).to_vec()
                    };

                    let (new_obs_bytes, new_mask_bytes, rewards, dones, truncateds) = env_interface.step_bytes(&action_bytes).ok_or_else(|| {
                        TrainingError::EnvironmentInterfaceNotFound(actor_id)
                    })?;

                    obs_bytes = new_obs_bytes;

                    match n_envs {
                        _ if n_envs <= PARALLEL_ITER_MIN_ENVS => {  // serial processing
                            for i in 0..n_envs {
                                trajectory_send_procedure(i);
                            }
                        }
                        _ => {
                            (0..n_envs).into_par_iter().map(|i| {  // parallel processing
                                trajectory_send_procedure(i);
                            })
                        }
                    }

                    (0..n_envs).into_par_iter().map(|i| {
                        let obs_i = {
                            let start = i * obs_bytes_per_env;
                            TensorData::new(vec![obs_dim], env_dtype_to_dtype(&obs_dtype)?, obs_bytes[start..start + obs_bytes_per_env].to_vec(), B::get_supported_backend())
                        };

                        let action_i = {
                            let start = i * obs_bytes_per_env;
                            TensorData::new(vec![act_dim], env_dtype_to_dtype(&act_dtype)?, action_bytes[start..start + act_bytes_per_env].to_vec(), B::get_supported_backend())
                        };

                        let data_map = {
                            let logp_i = {
                                let start = i * 4;
                                let logp_dtype = match B::get_supported_backend() {
                                    SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
                                    SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
                                };
                                TensorData::new(vec![1], logp_dtype, logp_bytes[start..start + 4].to_vec(), B::get_supported_backend())
                            };

                            let value_i = {
                                let bytes = values_f32[i].to_le_bytes().to_vec();
                                let value_dtype = match B::get_supported_backend() {
                                    SupportedTensorBackend::NdArray => DType::NdArray(NdArrayDType::F32),
                                    SupportedTensorBackend::Tch => DType::Tch(TchDType::F32),
                                };
                                TensorData::new(vec![1], value_dtype, bytes, B::get_supported_backend())
                            };

                            let map = HashMap::new();
                            map.insert("logp_a".to_string(), RelayRLData::Tensor(logp_i));
                            map.insert("val".to_string(), RelayRLData::Tensor(value_i));
                            map
                        };

                        per_env_ep_return[i] += rewards[i];
                        per_env_step_count[i] += 1;

                        let action_obj = RelayRLAction::new(
                            Some(obs_i),
                            Some(action_i),
                            None,
                            rewards[i],
                            dones[i],
                            Some(data_map),
                            Some(actor_id),
                        );
                        per_env_trajs[i].add_action(action_obj);

                        if dones[i] {
                            let episode_return = per_env_ep_return[i];
                            per_env_ep_return[i] = 0.0;
                            match n_envs {
                                _ if n_envs <= PARALLEL_ITER_MIN_ENVS => {  // serial processing
                                    return_window.push(episode_return);
                                    if return_window.len() > 100 { return_window.remove(0); }
                                    completed_episodes += 1;
                                }
                                _ => {  // parallel processing

                                }
                            }

                            let mut traj = std::mem::replace(
                                &mut per_env_trajs[i],
                                RelayRLTrajectory::new(max_traj_length),
                            );
                            traj.set_episode(per_env_episode[i]);
                            per_env_episode[i] += 1;

                        }
                    }))
                }

            })
        })
    }

    pub(crate) fn train_mappo<KindIn: TensorKind<B> + BasicOps<B>, KindOut: TensorKind<B> + BasicOps<B>, Pi: NeuralNetwork<B, KindIn, KindOut>>(
        actor_id: ActorUuid,
        shutdown_rx: Option<tokio::sync::broadcast::Receiver<()>>,
        runtime: Arc<ActorRuntime<B, D_IN, D_OUT>>,
        env_map: Arc<DashMap<ActorUuid, EnvironmentInterface>>,
        step_count: usize,
        max_traj_length: usize,
        training_spec: MAPPOTrainerSpec<B, KindIn, KindOut, Pi>,
    ) -> Result<(), TrainingError>
    {
        unimplemented!()
    }
}
