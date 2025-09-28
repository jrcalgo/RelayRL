//! This module implements the ZMQ-based training server for the RelayRL framework.
//! The server is responsible for handling model distribution, receiving trajectories,
//! coordinating with a python command request, and managing multiple agents.

use crate::network::server::python_subprocesses::python_algorithm_request::{
    PythonAlgorithmCommand, PythonAlgorithmRequest,
};
use crate::network::server::python_subprocesses::python_training_tensorboard::PythonTrainingTensorboard;
use crate::network::server::training_server_wrapper::{
    MultiactorParams, PythonSubprocesses, resolve_new_training_server_address,
};
use crate::sys_utils::config_loader::{
    ConfigLoader, DEFAULT_CONFIG_CONTENT, DEFAULT_CONFIG_PATH, ServerParams,
};
use crate::types::action::RelayRLAction;
use crate::types::trajectory::RelayRLTrajectory;

use rand::Rng;
use serde_pickle as pickle;

use rand::prelude::ThreadRng;
use std::collections::HashMap;
use std::fs;
use std::fs::{File, Metadata};
use std::io::{BufReader, Read};
use std::ops::DerefMut;
use std::path::PathBuf;
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::runtime::Handle;
use tokio::sync::mpsc::{Receiver as TokioReceiver, Sender as TokioSender};
use tokio::sync::{Mutex as TokioMutex, RwLock, RwLockWriteGuard};
use tokio::sync::{OwnedRwLockWriteGuard, RwLock as TokioRwLock};
use tokio::task::{JoinHandle as TokioJoinHandle, JoinHandle};

use zmq::{Context, Sendable, Socket};

#[cfg(all(feature = "console-subscriber", not(feature = "python_bindings")))]
#[cfg(feature = "console-subscriber")]
use crate::orchestration::tokio::utils::get_or_init_console_subscriber;
#[cfg(not(feature = "python_bindings"))]
use crate::orchestration::tokio::utils::get_or_init_tokio_runtime;
#[cfg(not(feature = "python_bindings"))]
use tokio::runtime::Runtime as TokioRuntime;

use crate::{get_or_create_config_json_path, resolve_config_json_path};

/// Struct that holds the ZMQ server address parameters.
/// These parameters define where the training server, trajectory server, and agent listener are located.
pub struct ZmqServerParams {
    /// Unique identifier for the training server instance.
    training_server_id: String,
    /// Address for the training server (used for sending updated models).
    training_server: String,
    /// Address for the trajectory server (used for receiving trajectories).
    trajectory_server: String,
    /// Address for the agent listener (used for receiving requests from agents).
    agent_listening_server: String,
}

/// Main struct representing the ZMQ-based training server for RelayRL.
///
/// This server handles:
/// - Model distribution to agents upon request,
/// - Reception of training trajectories from agents,
/// - Communication with the python command request for saving models and processing trajectories,
/// - And coordination of multiactor training if enabled.
pub struct TrainingServerZmq {
    /// Status of network
    active: Arc<AtomicBool>,
    /// Path where the model is saved and loaded.
    server_model_path: PathBuf,
    /// Max trajectory length.
    max_traj_length: u32,
    /// Optional hyperparameters for training.
    hyperparams: Option<HashMap<String, String>>,
    /// Handle for the agent listener thread (for receiving agent requests).
    agent_listener_thread: TokioMutex<Option<TokioJoinHandle<()>>>,
    /// Handle for the training thread (for processing received trajectories).
    training_thread: TokioMutex<Option<TokioJoinHandle<()>>>,
    /// Parameters for multiactor training including current count and agent IDs.
    actors: MultiactorParams,
    /// Wrapper struct around arguments for and instances of Python subprocess interfaces.
    python_subprocesses: PythonSubprocesses,
    /// ZMQ-specific server parameters containing various server addresses.
    zmq_params: ZmqServerParams,
    /// Tokio runtime (assuming bindings aren't compiled)
    #[cfg(not(feature = "python_bindings"))]
    tokio_runtime: Arc<TokioRuntime>,
}

impl TrainingServerZmq {
    /// Creates a new instance of the ZMQ-based TrainingServer.
    ///
    /// This function performs the following steps:
    /// - Loads the server configuration,
    /// - Resolves and prints the addresses for training, trajectory, and agent listener servers,
    /// - Initializes communication channels with the python command request,
    /// - And starts the ZMQ threads for agent listening and training.
    ///
    /// # Arguments
    /// * `algorithm_name` - Name of the learning algorithm.
    /// * `algorithm_dir` - Directory where the algorithm's scripts are located.
    /// * `tensorboard` - Flag indicating if Tensorboard integration is enabled.
    /// * `multiactor` - Flag indicating if multiactor training is enabled.
    /// * `env_dir` - Optional environment directory.
    /// * `config_path` - Optional path to the configuration file.
    /// * `hyperparams` - Optional hyperparameters as a HashMap.
    /// * `training_server_address` - Optional override for the training server address.
    ///
    /// # Returns
    /// * An Arc-wrapped TokioRwLock containing the TrainingServerZmq instance.
    pub async fn init_server(
        algorithm_name: String,
        algorithm_dir: String,
        tensorboard: bool,
        multiactor: bool,
        env_dir: Option<String>,
        config_path: Option<PathBuf>,
        hyperparams: Option<HashMap<String, String>>,
        training_server_address: Option<String>,
    ) -> Arc<TokioRwLock<TrainingServerZmq>> {
        #[cfg(all(feature = "console-subscriber", not(feature = "python_bindings")))]
        get_or_init_console_subscriber();
        #[cfg(not(feature = "python_bindings"))]
        let tokio_runtime: Arc<TokioRuntime> = get_or_init_tokio_runtime();

        println!("[Instantiating RelayRL-Framework ZMQ TrainingServer...]");

        // Resolve config path
        let config_path: Option<PathBuf> = resolve_config_json_path!(config_path);

        // Load the configuration settings.
        let config: Arc<ConfigLoader> = Arc::new(ConfigLoader::new(
            Some(algorithm_name.to_uppercase()),
            config_path.clone(),
        ));
        println!(
            "[TrainingServer - new] Resolved configuration path: {}",
            config_path
                .clone()
                .unwrap_or("default_config_path".parse().unwrap())
                .display()
        );

        // Generate a unique training server ID using the process ID and a random number.
        let mut rng: ThreadRng = rand::thread_rng();
        let training_server_id: String =
            format!("TS_ID-{:?}{:?}", process::id(), rng.gen_range(0..=99)).to_string();

        // Resolve the training server address.
        let training_server_address: String = match training_server_address {
            Some(_) => training_server_address.expect("Training server address is None"),
            None => {
                let train_server: &ServerParams = config.get_train_server();
                format!(
                    "{}{}:{}",
                    train_server.prefix, train_server.host, train_server.port
                )
            }
        };
        println!(
            "[TrainingServer - new] Training server address: {:}",
            training_server_address
        );

        // Resolve the trajectory server address.
        let traj_server: &ServerParams = config.get_traj_server();
        let trajectory_server_address: String = format!(
            "{}{}:{}",
            traj_server.prefix, traj_server.host, traj_server.port
        );
        println!(
            "[TrainingServer - new] Trajectory server address: {}",
            trajectory_server_address
        );

        // Resolve the agent listener address.
        let agent_listener: &ServerParams = config.get_agent_listener();
        let agent_listener_address: String = format!(
            "{}{}:{}",
            agent_listener.prefix, agent_listener.host, agent_listener.port
        );
        println!(
            "[TrainingServer - new] Agent listener address: {}",
            agent_listener_address
        );

        // Serialize hyperparameters to JSON string.
        let hyperparam_args: String =
            serde_json::to_string(&hyperparams).expect("Failed to serialize hyperparams");

        // Create a channel for sending commands to the PythonAlgorithmRequest.
        let (ts_tx, par_rx): (
            TokioSender<PythonAlgorithmCommand>,
            TokioReceiver<PythonAlgorithmCommand>,
        ) = tokio::sync::mpsc::channel(100000);

        // Build the argument list for the PythonAlgorithmRequest.
        let par_args: [String; 5] = [
            format!("--algorithm_name={}", algorithm_name),
            format!(
                "--env_dir={}",
                env_dir.clone().unwrap_or_else(|| "./".to_string())
            ),
            format!("--resolved_algorithm_dir={}", algorithm_dir),
            format!(
                "--config_path={}",
                config_path
                    .clone()
                    .expect("config_path is None")
                    .to_str()
                    .expect("Failed to convert config_path to str")
            ),
            format!("--hyperparams={}", hyperparam_args),
        ];

        // Convert the argument list to a vector of strings.
        let par_args: Vec<String> = par_args
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        // Initialize the PythonAlgorithmRequest and wait for the algorithm pyscript to initialize.
        let par_obj: Arc<PythonAlgorithmRequest> =
            Arc::new(PythonAlgorithmRequest::init_pyscript(par_rx, par_args.clone()).await);

        par_obj.notify_algorithm_status.notified().await;
        let algorithm_pyscript_status: Arc<AtomicBool> = par_obj.algorithm_pyscript_status.clone();
        println!(
            "[TrainingServer - new] Learning algorithm status acquired: {}",
            algorithm_pyscript_status.load(Ordering::SeqCst)
        );

        let ptt_args: Vec<String> = vec![
            format!(
                "--env_dir={}",
                env_dir.clone().unwrap_or_else(|| "./".to_string())
            ),
            format!(
                "--config_path={}",
                config_path
                    .clone()
                    .expect("config_path is None")
                    .to_str()
                    .expect("Failed to convert config_path to str")
            ),
            format!("--algorithm_name={}", algorithm_name),
        ];

        let ptt_args_clone: Vec<String> = ptt_args.clone();
        let ptt_obj: Option<Arc<PythonTrainingTensorboard>> = if tensorboard {
            Some(Arc::new(PythonTrainingTensorboard::init_pyscript(
                ptt_args_clone,
            )))
        } else {
            None
        };

        // Construct the TrainingServerZmq instance.
        let training_server = TrainingServerZmq {
            active: Arc::new(AtomicBool::new(false)),
            server_model_path: config.server_model_path.to_path_buf(),
            max_traj_length: config.max_traj_length,
            hyperparams,
            agent_listener_thread: TokioMutex::new(None),
            training_thread: TokioMutex::new(None),
            actors: MultiactorParams {
                multiactor,
                current_actor_count: 0,
                agent_ids: Vec::new(),
            },
            python_subprocesses: PythonSubprocesses {
                ptt_args,
                ptt_obj,
                par_args,
                par_obj: Some(par_obj),
                command_sender: Some(ts_tx),
                algorithm_pyscript_status,
            },
            zmq_params: ZmqServerParams {
                training_server_id,
                training_server: training_server_address,
                trajectory_server: trajectory_server_address,
                agent_listening_server: agent_listener_address,
            },
            #[cfg(not(feature = "python_bindings"))]
            tokio_runtime,
        };

        // Wrap the training server in an Arc and TokioRwLock.
        let zmq_server_arc: Arc<TokioRwLock<TrainingServerZmq>> =
            Arc::new(TokioRwLock::new(training_server));
        let zmq_server_arc_clone: Arc<TokioRwLock<TrainingServerZmq>> = Arc::clone(&zmq_server_arc);

        {
            let mut server = zmq_server_arc.write().await;

            // Initialize the main ZMQ threads for agent listening and training.
            let (agent_listener_thread, training_thread): (
                TokioJoinHandle<()>,
                TokioJoinHandle<()>,
            ) = Self::initialize_main_zmq_threads(Arc::clone(&zmq_server_arc_clone)).await;
            server.agent_listener_thread = TokioMutex::new(Some(agent_listener_thread));
            server.training_thread = TokioMutex::new(Some(training_thread));
            server.active.store(true, Ordering::SeqCst);
        }

        zmq_server_arc
    }

    /// Restarts the current instance of the ZMQ server.
    ///
    /// This is equivalent to the following function operations:
    ///
    /// 1. `disable_server()`
    /// 2. `enable_server()`
    ///
    pub async fn restart_server(
        &mut self,
        training_server_address: Option<String>,
    ) -> Vec<Result<(), Box<dyn std::error::Error>>> {
        println!("[TrainingServer - restart] Restarting ZMQ server...");
        let disable_result: Result<(), Box<dyn std::error::Error>> = self.disable_server().await;
        let enable_result: Result<(), Box<dyn std::error::Error>> =
            self.enable_server(training_server_address).await;
        vec![disable_result, enable_result]
    }

    /// Disables the server's training loop but keeps the agent instance alive.
    pub async fn disable_server(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.active.load(Ordering::SeqCst) {
            println!("[TrainingServer - disable] Disabling ZMQ server...");

            {
                // Join active threads
                self.joins().await;
            }

            {
                // Kill the algorithm python channel
                let par_obj: Option<Arc<PythonAlgorithmRequest>> =
                    self.python_subprocesses.par_obj.clone();
                if let Some(object) = &self.python_subprocesses.par_obj {
                    object
                        .algorithm_pyscript_status
                        .store(false, Ordering::SeqCst);
                    drop(par_obj);
                }
            }

            {
                // Kill the tensorboard writer instance
                let ptt_obj: Option<Arc<PythonTrainingTensorboard>> =
                    self.python_subprocesses.ptt_obj.clone();
                if let Some(object) = &self.python_subprocesses.ptt_obj {
                    object
                        .tensorboard_pyscript_status
                        .store(false, Ordering::Release);
                    drop(ptt_obj);
                }
            }

            self.active.store(false, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Re-activates networking operations for ZMQ server.
    ///
    /// For initial instantiation, use `init_server()`
    pub async fn enable_server(
        &mut self,
        training_server_address: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("[TrainingServer - enable] Reactivating ZMQ server...");

        if !self.active.load(Ordering::SeqCst) {
            // Create a channel for sending commands to the PythonAlgorithmRequest.
            let (ts_tx, par_rx): (
                TokioSender<PythonAlgorithmCommand>,
                TokioReceiver<PythonAlgorithmCommand>,
            ) = tokio::sync::mpsc::channel(100000);

            let par_args: Vec<String> = self.python_subprocesses.par_args.clone();
            let par_obj: Arc<PythonAlgorithmRequest> =
                Arc::new(PythonAlgorithmRequest::init_pyscript(par_rx, par_args).await);

            par_obj.notify_algorithm_status.notified().await;

            let ptt_args: Vec<String> = self.python_subprocesses.ptt_args.clone();
            let ptt_obj_some: bool = self.python_subprocesses.ptt_obj.is_some();

            let ptt_obj: Option<Arc<PythonTrainingTensorboard>> = if ptt_obj_some {
                Some(Arc::new(PythonTrainingTensorboard::init_pyscript(ptt_args)))
            } else {
                None
            };

            self.python_subprocesses.command_sender = Some(ts_tx);
            self.python_subprocesses.par_obj = Some(par_obj);
            self.python_subprocesses.ptt_obj = ptt_obj;
            self.python_subprocesses
                .algorithm_pyscript_status
                .store(true, Ordering::SeqCst);

            let old_training_server_address: &str = self.zmq_params.training_server.as_str();
            self.zmq_params.training_server = resolve_new_training_server_address(
                old_training_server_address,
                training_server_address,
            )
            .await;

            // Create a proper Arc<TokioRwLock<TrainingServerZmq>> for the threads
            let self_arc = Arc::new(TokioRwLock::new(TrainingServerZmq {
                active: self.active.clone(),
                server_model_path: self.server_model_path.clone(),
                max_traj_length: self.max_traj_length,
                hyperparams: self.hyperparams.clone(),
                agent_listener_thread: TokioMutex::new(None),
                training_thread: TokioMutex::new(None),
                actors: MultiactorParams {
                    multiactor: self.actors.multiactor,
                    current_actor_count: self.actors.current_actor_count,
                    agent_ids: self.actors.agent_ids.clone(),
                },
                python_subprocesses: PythonSubprocesses {
                    ptt_args: self.python_subprocesses.ptt_args.clone(),
                    ptt_obj: self.python_subprocesses.ptt_obj.clone(),
                    par_args: self.python_subprocesses.par_args.clone(),
                    par_obj: self.python_subprocesses.par_obj.clone(),
                    command_sender: self.python_subprocesses.command_sender.clone(),
                    algorithm_pyscript_status: self
                        .python_subprocesses
                        .algorithm_pyscript_status
                        .clone(),
                },
                zmq_params: ZmqServerParams {
                    training_server_id: self.zmq_params.training_server_id.clone(),
                    training_server: self.zmq_params.training_server.clone(),
                    trajectory_server: self.zmq_params.trajectory_server.clone(),
                    agent_listening_server: self.zmq_params.agent_listening_server.clone(),
                },
                #[cfg(not(feature = "python_bindings"))]
                tokio_runtime: self.tokio_runtime.clone(),
            }));

            // Initialize the main ZMQ threads for agent listening and training.
            let (agent_listener_thread, training_thread): (
                TokioJoinHandle<()>,
                TokioJoinHandle<()>,
            ) = Self::initialize_main_zmq_threads(self_arc).await;

            self.agent_listener_thread = TokioMutex::new(Some(agent_listener_thread));
            self.training_thread = TokioMutex::new(Some(training_thread));

            self.active.store(true, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Sends a save model command request to the PythonAlgorithmRequest.
    ///
    /// The command instructs the algorithm pyscript to save the current model.
    /// A oneshot channel is used to receive a boolean outcome indicating success or failure.
    ///
    /// # Returns
    /// * `true` if the save model command was successfully processed; otherwise, `false`.
    pub async fn par_send_save_model(&self) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let command = PythonAlgorithmCommand::SaveModel(tx);

        if self
            .python_subprocesses
            .command_sender
            .clone()
            .expect("Command sender is None")
            .send(command)
            .await
            .is_err()
        {
            eprintln!("[TrainingServer - send_save_model] Failed to send save model");
            return false;
        }

        rx.await.unwrap_or(false)
    }

    /// Sends a receive trajectory command request to the PythonAlgorithmRequest.
    ///
    /// The command includes a trajectory that the algorithm pyscript should process.
    /// A oneshot channel is used to receive a boolean outcome indicating whether the processing led to a model update.
    ///
    /// # Arguments
    /// * `trajectory` - The RelayRLTrajectory to be sent.
    ///
    /// # Returns
    /// * `true` if the trajectory was successfully processed and triggered an update; otherwise, `false`.
    pub async fn par_send_receive_trajectory(&self, trajectory: RelayRLTrajectory) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let command = PythonAlgorithmCommand::ReceiveTrajectory(tx, trajectory);

        if self
            .python_subprocesses
            .command_sender
            .clone()
            .expect("Command sender is None")
            .send(command)
            .await
            .is_err()
        {
            eprintln!(
                "[TrainingServer - send_receive_trajectory] Failed to send receive trajectory"
            );
            return false;
        }

        rx.await.unwrap_or(false)
    }

    /// Waits for both the agent listener thread and the training thread to finish.
    ///
    /// This function joins both threads by aborting them and waiting for their termination.
    pub(crate) async fn joins(&mut self) {
        if self.agent_listener_thread.lock().await.is_some() {
            self.join_agent_listener_thread().await;
            self.join_training_thread().await;
        } else {
            self.join_training_thread().await;
        }
    }

    /// Initializes the main ZMQ threads: the agent listener thread and the training loop thread.
    ///
    /// This function spawns two asynchronous tasks:
    /// - One for listening to incoming agent requests via a ROUTER socket.
    /// - Another for processing incoming training trajectories via a PULL socket.
    ///
    /// # Arguments
    /// * `server` - An Arc-wrapped reference to the TrainingServerZmq.
    ///
    /// # Returns
    /// * A tuple containing the join handles of the agent listener thread and the training thread.
    async fn initialize_main_zmq_threads(
        server: Arc<TokioRwLock<TrainingServerZmq>>,
    ) -> (TokioJoinHandle<()>, TokioJoinHandle<()>) {
        let runtime_handle = Handle::current();

        let (agent_listener_thread, training_thread) = (
            Self::start_agent_listener_thread(&server, runtime_handle.clone()).await,
            Self::start_training_thread(&server, runtime_handle.clone()).await,
        );

        (agent_listener_thread, training_thread)
    }

    /// Starts the agent listener thread.
    ///
    /// This thread listens for incoming agent requests (e.g., GET_MODEL, MODEL_SET)
    /// using a ROUTER socket, and processes each request accordingly.
    ///
    /// # Arguments
    /// * `server` - An Arc reference to the TrainingServerZmq.
    /// * `runtime_handle1` - A handle to the current Tokio runtime.
    ///
    /// # Returns
    /// * A join handle for the spawned agent listener thread.
    async fn start_agent_listener_thread(
        server: &Arc<TokioRwLock<TrainingServerZmq>>,
        runtime_handle1: Handle,
    ) -> tokio::task::JoinHandle<()> {
        let agent_listener_thread: TokioJoinHandle<()> = {
            let server: Arc<TokioRwLock<TrainingServerZmq>> = Arc::clone(&server);
            runtime_handle1.spawn(async move {
                if let Err(e) = Self::_listen_for_agents(server).await {
                    eprintln!(
                        "[TrainingServer - initialize_threads] Failed to listen for agents: {}",
                        e
                    );
                }
            })
        };
        agent_listener_thread
    }

    /// Starts the training loop thread.
    ///
    /// This thread waits for trajectories via a PULL socket, processes them,
    /// sends them to the algorithm pyscript, and if necessary, sends updated models.
    ///
    /// # Arguments
    /// * `server` - An Arc reference to the TrainingServerZmq.
    /// * `runtime_handle2` - A handle to the current Tokio runtime.
    ///
    /// # Returns
    /// * A join handle for the spawned training thread.
    async fn start_training_thread(
        server: &Arc<TokioRwLock<TrainingServerZmq>>,
        runtime_handle2: Handle,
    ) -> tokio::task::JoinHandle<()> {
        let training_thread: TokioJoinHandle<()> = {
            let server: Arc<TokioRwLock<TrainingServerZmq>> = Arc::clone(server);
            runtime_handle2.spawn(async move {
                if let Err(e) = Self::_start_training_loop(server).await {
                    eprintln!(
                        "[TrainingServer - initialize_threads] Failed to handle trajectory: {}",
                        e
                    );
                }
            })
        };
        training_thread
    }

    /// Aborts and waits for the agent listener thread to terminate.
    async fn join_agent_listener_thread(&mut self) {
        if let Some(handle) = self.agent_listener_thread.lock().await.take() {
            handle.abort();
            if let Err(e) = handle.await {
                if e.is_cancelled() {
                    println!(
                        "[TrainingServer - join_agent_listener_thread] Agent listener thread was cancelled"
                    );
                } else {
                    panic!(
                        "[TrainingServer - join_agent_listener_thread] Failed to join agent listener thread: {:?}",
                        e
                    );
                }
            }
        }
    }

    /// Aborts and waits for the training thread to terminate.
    async fn join_training_thread(&mut self) {
        if let Some(handle) = self.training_thread.lock().await.take() {
            handle.abort();
            if let Err(e) = handle.await {
                if e.is_cancelled() {
                    println!(
                        "[TrainingServer - join_training_thread] Training thread was cancelled"
                    );
                } else {
                    panic!(
                        "[TrainingServer - join_training_thread] Failed to join training thread: {:?}",
                        e
                    );
                }
            }
        }
    }

    /// Listens for agent requests via a ROUTER socket.
    ///
    /// The agent listener receives multipart messages from agents (consisting of agent identity,
    /// an empty frame, and the request data). Depending on the request (e.g., GET_MODEL or MODEL_SET),
    /// the server responds by sending the model file or acknowledging the agent's registration.
    ///
    /// # Arguments
    /// * `self_arc` - An Arc reference to the TrainingServerZmq instance.
    ///
    /// # Returns
    /// * `Result<(), zmq::Error>` - Ok(()) if listening continues successfully, or an error otherwise.
    async fn _listen_for_agents(self_arc: Arc<TokioRwLock<Self>>) -> Result<(), zmq::Error> {
        println!(
            "[TrainingServer - listen_for_agents] Listening for agent requests via ROUTER-DEALER..."
        );

        // Retrieve necessary configuration values from the server.
        let (agent_listener_address, server_model_path, algorithm_pyscript_status, multiactor): (
            String,
            PathBuf,
            bool,
            bool,
        ) = {
            let server: Arc<TokioRwLock<TrainingServerZmq>> = self_arc.clone();
            let returned = (
                server
                    .read()
                    .await
                    .zmq_params
                    .agent_listening_server
                    .clone(),
                server.read().await.server_model_path.clone(),
                server
                    .read()
                    .await
                    .python_subprocesses
                    .algorithm_pyscript_status
                    .load(Ordering::Acquire),
                server.read().await.actors.multiactor,
            );
            returned
        };
        if !algorithm_pyscript_status {
            panic!("[TrainingServer - listen_for_agents] Algorithm script not initialized");
        }

        // Set up a ZMQ ROUTER socket to listen for incoming agent messages.
        let context: Context = Context::new();
        let router_socket: Socket = context.socket(zmq::ROUTER)?;
        router_socket.set_rcvtimeo(0)?;

        println!(
            "[TrainingServer - listen_for_agents] Binding to {}",
            agent_listener_address
        );
        router_socket.bind(agent_listener_address.as_str())?;

        let empty_frame: &Vec<u8> = &vec![];

        let mut loop_iter: u32 = 0;
        const PRINT_LOOP_MAX: u32 = 1000000;
        loop {
            loop_iter += 1;
            if loop_iter % PRINT_LOOP_MAX == 0 {
                println!(
                    "[TrainingServer - listen_for_agents] Waiting for requests from agents..."
                );
            }

            // The ROUTER socket receives a multipart message: [agent identity, empty frame, request]
            match router_socket.recv_multipart(0) {
                Ok(message_parts) => {
                    if message_parts.len() < 3 {
                        eprintln!(
                            "[TrainingServer - listen_for_agents] Received malformed request"
                        );
                        continue;
                    }

                    let agent_id: &Vec<u8> = &message_parts[0]; // Agent identity
                    let request: &Vec<u8> = &message_parts[2]; // Request payload

                    println!(
                        "[TrainingServer - listen_for_agents] Received request: ({}::{})",
                        String::from_utf8_lossy(agent_id),
                        String::from_utf8_lossy(request)
                    );

                    // Process GET_MODEL requests: send the current model file back to the agent.
                    if request == b"GET_MODEL" {
                        println!(
                            "[TrainingServer - listen_for_agents] Responding to model request..."
                        );

                        let outcome: bool = self_arc.read().await.par_send_save_model().await;

                        if !outcome {
                            eprintln!(
                                "[TrainingServer - listen_for_agents] Failed to send save model request"
                            );
                            let error_message: &Vec<u8> =
                                &Vec::from("ERROR: Failed to send save model request");
                            router_socket
                                .send_multipart(&[agent_id, empty_frame, error_message], 0)?;
                            continue;
                        }

                        // Read the model file metadata and send its contents.
                        match fs::metadata(&server_model_path) {
                            Ok(metadata) => {
                                let file_size: u64 = metadata.len();

                                let model_file: std::io::Result<File> =
                                    File::open(&server_model_path);
                                if let Ok(file) = model_file {
                                    let mut reader: BufReader<File> = BufReader::new(file);
                                    let mut buffer: Vec<u8> =
                                        Vec::with_capacity(file_size as usize);
                                    reader
                                        .read_to_end(&mut buffer)
                                        .expect("Failed to read model file");

                                    // Send the model file (agent identity, empty frame, model data)
                                    println!(
                                        "[TrainingServer - listen_for_agents] Sending model file..."
                                    );
                                    router_socket
                                        .send_multipart([agent_id, empty_frame, &buffer], 0)?;
                                } else {
                                    eprintln!(
                                        "[TrainingServer - listen_for_agents] Failed to open model file"
                                    );
                                    let error_message: &Vec<u8> =
                                        &Vec::from("ERROR: Failed to open model file");
                                    router_socket.send_multipart(
                                        [agent_id, empty_frame, error_message],
                                        0,
                                    )?;
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "[TrainingServer - listen_for_agents] Failed to get metadata: {}",
                                    e
                                );
                                let error_message =
                                    &Vec::from("ERROR: Failed to get model metadata");
                                router_socket
                                    .send_multipart([agent_id, empty_frame, error_message], 0)?;
                            }
                        }
                    }
                    // Process MODEL_SET requests: register the agent with the training server.
                    else if request == b"MODEL_SET" {
                        {
                            let mut self_arc2: RwLockWriteGuard<TrainingServerZmq> =
                                self_arc.write().await;
                            self_arc2
                                .actors
                                .agent_ids
                                .push(String::from_utf8_lossy(agent_id).to_string());
                            self_arc2.actors.current_actor_count += 1;
                        }

                        let id_logged_response: Vec<u8> = Vec::from("ID_LOGGED");
                        router_socket
                            .send_multipart(&[agent_id, empty_frame, &id_logged_response], 0)?;

                        // For non-multiactor setups, break after one registration.
                        if !multiactor {
                            break;
                        }
                    } else {
                        eprintln!(
                            "[TrainingServer - listen_for_agents] Unknown request: {}",
                            String::from_utf8_lossy(request)
                        );
                        let error_message = &Vec::from("ERROR: Unknown request");
                        router_socket.send_multipart(&[agent_id, empty_frame, error_message], 0)?;
                    }
                }
                Err(e) => {
                    if e == zmq::Error::EAGAIN {
                        if loop_iter % PRINT_LOOP_MAX == 0 {
                            println!("[TrainingServer - listen_for_agents] No request received");
                        }
                        continue;
                    } else {
                        eprintln!(
                            "[TrainingServer - listen_for_agents] Failed to receive request: {}",
                            e
                        );
                        break;
                    }
                }
            }

            if loop_iter % PRINT_LOOP_MAX == 0 {
                loop_iter = 0;
            }

            // Pause briefly to prevent busy waiting.
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        Ok(())
    }

    /// Sends the current model (from the algorithm pyscript) to the RelayRLAgent.
    ///
    /// This function reads the model file from disk and sends it to the training server using a ZMQ PUSH socket.
    /// It ensures that the algorithm pyscript is initialized before sending the model.
    ///
    /// # Arguments
    /// * `self_arc` - An Arc reference to the TrainingServerZmq.
    ///
    /// # Returns
    /// * `Result<(), zmq::Error>` - Ok(()) if the model was sent successfully; an error otherwise.
    async fn _send_ongoing_model(self_arc: Arc<TokioRwLock<Self>>) -> Result<(), zmq::Error> {
        println!(
            "[TrainingServer - send_ongoing_model] Preparing to send new model to RelayRLAgent"
        );

        // Retrieve necessary information from the server.
        let (server_model_path, training_server, algorithm_pyscript_status) = {
            let server: Arc<TokioRwLock<TrainingServerZmq>> = self_arc.clone();
            (
                server.read().await.server_model_path.clone(),
                server.read().await.zmq_params.training_server.clone(),
                server
                    .read()
                    .await
                    .python_subprocesses
                    .algorithm_pyscript_status
                    .load(Ordering::Acquire),
            )
        };
        if !algorithm_pyscript_status {
            panic!("[TrainingServer - send_ongoing_model] Algorithm script not initialized");
        }

        {
            // Trigger the algorithm pyscript to save the current model.
            let outcome = self_arc.read().await.par_send_save_model().await;
            ()
        }

        let mut buffer: Vec<u8>;
        {
            // Read the model file from disk.
            let metadata: Metadata = fs::metadata(&server_model_path)
                .expect("[TrainingServer - send_ongoing_model] Failed to get metadata");
            let file_size: u64 = metadata.len();

            let model_file: File = File::open(server_model_path)
                .expect("[TrainingServer - send_ongoing_model] Failed to open model file");
            let mut reader: BufReader<File> = BufReader::new(model_file);
            buffer = Vec::with_capacity(file_size as usize);
            reader
                .read_to_end(&mut buffer)
                .expect("[TrainingServer - send_ongoing_model] Failed to read model file");
        }

        // Set up a ZMQ PUSH socket to send the model.
        let context: Context = Context::new();
        let socket: Socket = context.socket(zmq::PUSH)?;

        socket.set_sndhwm(100)?;
        socket.set_maxmsgsize(-1)?;
        socket.connect(training_server.as_str())?;

        // Send the model file bytes.
        socket.send(&buffer, 0)?;
        println!("[TrainingServer - send_ongoing_model] Model sent");

        Ok(())
    }

    /// Starts the training loop that receives trajectories and processes them.
    ///
    /// This function binds a ZMQ PULL socket to the trajectory server address,
    /// waits for incoming serialized trajectories, deserializes them, and sends them
    /// to the algorithm pyscript for processing. If processing indicates a model update,
    /// the new model is sent to the agents.
    ///
    /// # Arguments
    /// * `self_arc` - An Arc reference to the TrainingServerZmq.
    ///
    /// # Returns
    /// * `Result<(), zmq::Error>` - Ok(()) if the loop exits gracefully; an error otherwise.
    async fn _start_training_loop(self_arc: Arc<TokioRwLock<Self>>) -> Result<(), zmq::Error> {
        println!("[TrainingServer - training_loop] Starting training loop");

        let context: Context = Context::new();
        let (max_traj_length, traj_server, algorithm_pyscript_status): (u32, String, bool) = {
            let server: Arc<TokioRwLock<TrainingServerZmq>> = self_arc.clone();
            let returned: (u32, String, bool) = (
                server.read().await.max_traj_length,
                server.read().await.zmq_params.trajectory_server.clone(),
                server
                    .read()
                    .await
                    .python_subprocesses
                    .algorithm_pyscript_status
                    .load(Ordering::Acquire),
            );
            returned
        };
        if !algorithm_pyscript_status {
            panic!("[TrainingServer - training_loop] Algorithm script not initialized");
        }

        // Bind the ZMQ PULL socket to receive trajectory data.
        let socket: Socket = context
            .socket(zmq::PULL)
            .expect("[TrainingServer - training_loop] Failed to create socket");

        println!(
            "[TrainingServer - training_loop] Binding to trajectory server: {}",
            traj_server
        );
        socket
            .bind(traj_server.as_str())
            .expect("[TrainingServer - training_loop] Failed to bind socket");
        socket.set_rcvtimeo(0)?;

        let mut trajectory_count: u32 = 0;
        let mut loop_iter: u32 = 0;
        const PRINT_LOOP_MAX: u32 = 1000000;
        loop {
            loop_iter += 1;
            if loop_iter % PRINT_LOOP_MAX == 0 {
                println!("[TrainingServer - training_loop] Waiting for new trajectory...");
            }

            // Wait for trajectory data from the agents.
            match socket.recv_bytes(0) {
                Ok(trajectory_bytes) => {
                    trajectory_count += 1;
                    // Deserialize the received trajectory using pickle.
                    let actions: Vec<RelayRLAction> =
                        pickle::from_slice(&trajectory_bytes, Default::default()).expect(
                            "[TrainingServer - training_loop] Failed to deserialize trajectory",
                        );
                    println!(
                        "[TrainingServer - training_loop] Received trajectory #{:?}",
                        trajectory_count
                    );

                    // Reconstruct the trajectory from the list of actions.
                    let mut trajectory: RelayRLTrajectory =
                        RelayRLTrajectory::new(Some(max_traj_length), None);
                    for action in actions {
                        trajectory.actions.push(action);
                    }

                    // If the algorithm pyscript is active, send the trajectory for processing.
                    if algorithm_pyscript_status {
                        let updated_model_result: bool = self_arc
                            .read()
                            .await
                            .par_send_receive_trajectory(trajectory)
                            .await;

                        // If processing indicates an update, send the updated model.
                        if updated_model_result {
                            Self::_send_ongoing_model(Arc::clone(&self_arc))
                                .await
                                .expect(
                                    "[TrainingServer - training_loop] Failed to send ongoing model",
                                );
                        }
                    }
                }
                Err(e) => {
                    if e == zmq::Error::EAGAIN {
                        if loop_iter % PRINT_LOOP_MAX == 0 {
                            println!("[TrainingServer - training_loop] No trajectory received");
                        }
                        continue;
                    } else {
                        eprintln!(
                            "[TrainingServer - training_loop] Failed to receive trajectory: {}",
                            e
                        );
                        break;
                    }
                }
            }

            if loop_iter % PRINT_LOOP_MAX == 0 {
                loop_iter = 0;
            }

            // Sleep briefly to reduce CPU usage.
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        #[allow(unreachable_code)]
        Ok(())
    }
}
