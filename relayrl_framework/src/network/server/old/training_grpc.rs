//! This module implements the gRPC-based training server for the RelayRL framework.
//! The server is responsible for handling model distribution, receiving trajectories,
//! coordinating with the Python command request, and managing multi-actor training if enabled.

use crate::network::server::python_subprocesses::python_algorithm_request::{
    PythonAlgorithmCommand, PythonAlgorithmRequest,
};
use crate::network::server::python_subprocesses::python_training_tensorboard::PythonTrainingTensorboard;
use crate::sys_utils::grpc_utils::{
    grpc_trajectory_to_relayrl_trajectory, serialize_model,
};
use crate::proto::{
    ActionResponse, RelayRlAction as grpc_RelayRLAction, RelayRlModel as RelayRLModel, RequestModel,
    Trajectory as grpc_Trajectory, relay_rl_route_server::RelayRlRouteServer as RelayRLRouteServer,
};
use crate::sys_utils::config_loader::{
    ConfigLoader, DEFAULT_CONFIG_CONTENT, DEFAULT_CONFIG_PATH, LoadedAlgorithmParams,
};
use crate::types::trajectory::RelayRLTrajectory;

use std::collections::{HashMap, VecDeque};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tch::{CModule, Device};

use crate::network::server::training_server_wrapper::{
    MultiactorParams, PythonSubprocesses, resolve_new_training_server_address,
};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc::{Receiver as TokioReceiver, Sender as TokioSender};
use tokio::sync::{Mutex, RwLock as TokioRwLock, RwLockReadGuard};
use tokio::sync::{RwLock, RwLockWriteGuard, watch};

use crate::proto::relay_rl_route_server::RelayRlRoute as RelayRLRoute;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[cfg(all(feature = "console-subscriber", not(feature = "python_bindings")))]
use crate::orchestration::tokio::utils::get_or_init_console_subscriber;
#[cfg(not(feature = "python_bindings"))]
use crate::orchestration::tokio::utils::get_or_init_tokio_runtime;
use crate::{get_or_create_config_json_path, resolve_config_json_path};
#[cfg(not(feature = "python_bindings"))]
use tokio::runtime::Runtime as TokioRuntime;
use tokio::sync::watch::Receiver;
use tokio::task::JoinHandle;
use zmq::Sendable;

pub async fn grpc_serve(
    training_server: Arc<TokioRwLock<TrainingServerGrpc>>,
    address: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket_address = address.parse().map_err(|e| {
        eprintln!(
            "Failed to parse address: {}. Ensure address is in the format IP:PORT",
            e
        );
        e
    })?;

    Server::builder()
        .add_service(RelayRLRouteServer::new(training_server))
        .serve(socket_address)
        .await
        .expect("Failed to start gRPC server");

    Ok(())
}

type State = Arc<TokioRwLock<u64>>;

/// RPC parameters for the gRPC service.
///
/// This struct holds configuration and state for the gRPC training service including:
/// - A buffer for incoming RelayRL actions,
/// - The batch size for training,
/// - The idle timeout for polling clients,
/// - Flags and fields to indicate if a model is ready or if there was an error,
/// - And a place to store the trained model.
pub struct GrpcServiceParams {
    servicing_task: TokioMutex<Option<JoinHandle<()>>>,
    training_server_address: String,
    trajectory_buffer: VecDeque<grpc_RelayRLAction>,
    request_state: State,
    reply_state: State,
    batch_size: u32,
    idle_timeout: u32,
    model_ready: bool,
    model_ready_tx: Option<watch::Sender<bool>>,
    trained_model: TokioRwLock<Option<CModule>>,
    error_message: Option<String>,
}

/// Represents the gRPC training server for the RelayRL framework.
///
/// The server is responsible for:
/// - Distributing models to clients via gRPC,
/// - Receiving training trajectories from agents,
/// - Coordinating with a Python-based command request system,
/// - And managing multi-actor parameters if enabled.
pub struct TrainingServerGrpc {
    /// Status of network
    active: Arc<AtomicBool>,
    /// Path where the trained model is saved and loaded.
    server_model_path: PathBuf,
    /// Maximum trajectory length.
    max_traj_length: u32,
    /// Optional hyperparameters for training.
    hyperparams: Option<HashMap<String, String>>,
    /// Multi-actor training parameters including the current actor count and registered agent IDs.
    actors: MultiactorParams,
    /// Wrapper struct around arguments for and instances of Python subprocess interfaces.
    python_subprocesses: PythonSubprocesses,
    /// Shared gRPC service parameters wrapped in a Tokio Mutex for thread-safe access.
    grpc_params: Arc<TokioRwLock<GrpcServiceParams>>,
    /// Tokio runtime (assuming bindings aren't compiled)
    #[cfg(not(feature = "python_bindings"))]
    tokio_runtime: Arc<TokioRuntime>,
}

impl TrainingServerGrpc {
    /// Creates and initializes a new gRPC TrainingServer instance.
    ///
    /// This function performs the following steps:
    /// - Loads the configuration using the provided path or a default value.
    /// - Serializes hyperparameters into a JSON string.
    /// - Creates a communication channel for Python commands.
    /// - Builds an argument list for the Python channel request.
    /// - Starts the Python channel request and waits for the algorithm script to initialize.
    /// - Determines the training batch size from the configuration.
    /// - Returns the fully initialized server wrapped in an Arc and Tokio RwLock.
    ///
    /// # Arguments
    /// * `algorithm_name` - Name of the learning algorithm.
    /// * `algorithm_dir` - Directory where algorithm resources reside.
    /// * `tensorboard` - Flag indicating if Tensorboard integration is enabled.
    /// * `multiactor` - Flag indicating if multi-actor training is enabled.
    /// * `hyperparams` - Optional hyperparameters provided as a HashMap.
    /// * `env_dir` - Optional environment directory.
    /// * `config_path` - Optional path to the configuration file.
    ///
    /// # Returns
    /// * An Arc-wrapped Tokio RwLock containing the TrainingServerGrpc instance.
    pub async fn init_server(
        training_server_address: String,
        algorithm_name: String,
        algorithm_dir: String,
        tensorboard: bool,
        multiactor: bool,
        hyperparams: Option<HashMap<String, String>>,
        env_dir: Option<String>,
        config_path: Option<PathBuf>,
    ) -> Arc<TokioRwLock<TrainingServerGrpc>> {
        #[cfg(all(feature = "console-subscriber", not(feature = "python_bindings")))]
        get_or_init_console_subscriber();
        #[cfg(not(feature = "python_bindings"))]
        let tokio_runtime: Arc<TokioRuntime> = get_or_init_tokio_runtime();

        println!("[Instantiating RelayRL-Framework gRPC TrainingServer...]");

        // Resolve config path
        let config_path: Option<PathBuf> = resolve_config_json_path!(config_path);

        // Load configuration settings using the provided configuration path (or a default if none is provided).
        let config: Arc<ConfigLoader> = Arc::new(ConfigLoader::new(
            Some(algorithm_name.to_uppercase()),
            config_path.clone(),
        ));
        println!(
            "[TrainingServer - new] Resolved configuration path: {:?}",
            config_path.clone()
        );

        // get max trajectory
        let max_traj_length: u32 = hyperparams
            .clone()
            .expect("hyperparams is None")
            .get("buf_size")
            .expect("`buf_size` key not found in hyperparams")
            .parse()
            .unwrap_or(config.max_traj_length);

        // Serialize the optional hyperparameters to a JSON string.
        let hyperparam_args: String =
            serde_json::to_string(&hyperparams).expect("Failed to serialize hyperparams");

        // Create a Tokio channel for communication with the Python command center.
        let (ts_tx, par_rx): (
            TokioSender<PythonAlgorithmCommand>,
            TokioReceiver<PythonAlgorithmCommand>,
        ) = tokio::sync::mpsc::channel(100000);

        // Build the argument list required for initializing the PythonAlgorithmRequest.
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

        let par_args: Vec<String> = par_args
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        // Start the Python channel request and await notification that the algorithm script is ready.
        let par_args_clone: Vec<String> = par_args.clone();
        let par_obj: Arc<PythonAlgorithmRequest> =
            Arc::new(PythonAlgorithmRequest::init_pyscript(par_rx, par_args_clone).await);
        par_obj.notify_algorithm_status.notified().await;
        let algorithm_pyscript_status: Arc<AtomicBool> = par_obj.algorithm_pyscript_status.clone();
        println!(
            "[TrainingServer - new] Learning algorithm status acquired: {}",
            algorithm_pyscript_status.load(Ordering::SeqCst)
        );

        let ptt_args: Vec<String> = vec![
            format!("--env_dir={}", env_dir.unwrap_or_else(|| "./".to_string())),
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

        // Retrieve the idle timeout parameter from the configuration.
        let idle_timeout: u32 = config.grpc_idle_timeout;

        // Determine the batch size from the loaded algorithm parameters (default to 1 if none are set).
        let batch_size: u32 = if let Some(loaded_params) = config.get_algorithm_params() {
            match loaded_params {
                LoadedAlgorithmParams::REINFORCE(_) => 1,
            }
        } else {
            1
        };

        // Construct and return the new TrainingServerGrpc instance wrapped in an Arc and Tokio RwLock.
        let grpc_service: Arc<RwLock<TrainingServerGrpc>> =
            Arc::new(TokioRwLock::new(TrainingServerGrpc {
                active: Arc::new(AtomicBool::new(false)),
                server_model_path: config.server_model_path.to_path_buf(),
                max_traj_length,
                hyperparams,
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
                grpc_params: Arc::new(TokioRwLock::new(GrpcServiceParams {
                    servicing_task: TokioMutex::new(None),
                    training_server_address: training_server_address.to_string(),
                    trajectory_buffer: VecDeque::new(),
                    request_state: Arc::new(TokioRwLock::new(0)),
                    reply_state: Arc::new(TokioRwLock::new(0)),
                    batch_size,
                    idle_timeout,
                    model_ready: false,
                    model_ready_tx: Option::from(watch::channel(false).0),
                    trained_model: None.into(),
                    error_message: None,
                })),
                #[cfg(not(feature = "python_bindings"))]
                tokio_runtime,
            }));

        let grpc_service_clone: Arc<RwLock<TrainingServerGrpc>> = Arc::clone(&grpc_service);

        let servicing_task: Mutex<Option<JoinHandle<()>>> =
            TokioMutex::new(Some(tokio::task::spawn(async move {
                if let Err(e) =
                    grpc_serve(Arc::clone(&grpc_service_clone), training_server_address).await
                {
                    eprintln!("[TrainingServer - new] Failed to start gRPC server: {}", e);
                }
            })));

        grpc_service
            .write()
            .await
            .active
            .store(true, Ordering::SeqCst);
        grpc_service
            .write()
            .await
            .grpc_params
            .write()
            .await
            .servicing_task = servicing_task;

        grpc_service
    }

    /// Restarts the current instance of the gRPC server.
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
        println!("[TrainingServer - restart] Restarting gRPC server...");
        let disable_result: Result<(), Box<dyn std::error::Error>> = self.disable_server().await;
        let enable_result: Result<(), Box<dyn std::error::Error>> =
            self.enable_server(training_server_address).await;
        vec![disable_result, enable_result]
    }

    /// Disables the server's training/listening loop but keeps the server instance alive.
    pub async fn disable_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("[TrainingServer - disable] Disabling gRPC server...");
        let result: Result<(), Box<dyn std::error::Error>> = if self.active.load(Ordering::SeqCst) {
            let params: RwLockWriteGuard<GrpcServiceParams> = self.grpc_params.write().await;

            if let Some(handle) = params.servicing_task.lock().await.take() {
                handle.abort();
                if let Err(err) = handle.await {
                    if err.is_cancelled() {
                        println!("TrainingServer - disable] gRPC servicing task was cancelled");
                    } else {
                        return Err(err.into());
                    }
                }
            };

            let par_obj: Option<Arc<PythonAlgorithmRequest>> =
                self.python_subprocesses.par_obj.clone();
            if let Some(object) = &self.python_subprocesses.par_obj {
                object
                    .algorithm_pyscript_status
                    .store(false, Ordering::SeqCst);
                drop(par_obj);
            }

            let ptt_obj: Option<Arc<PythonTrainingTensorboard>> =
                self.python_subprocesses.ptt_obj.clone();
            if let Some(object) = &self.python_subprocesses.ptt_obj {
                object
                    .tensorboard_pyscript_status
                    .store(false, Ordering::SeqCst);
                drop(ptt_obj);
            }

            self.active.store(false, Ordering::SeqCst);

            Ok(())
        } else {
            Ok(())
        };

        result
    }

    /// Re-activates networking operations for gRPC server.
    ///
    /// For initial instantiation, use `init_server()`
    pub async fn enable_server(
        &mut self,
        training_server_address: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("[TrainingServer - enable] Reactivating gRPC server...");

        if !self.active.load(Ordering::SeqCst) {
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

            let old_training_server_address = {
                let grpc_params: RwLockReadGuard<GrpcServiceParams> = self.grpc_params.read().await;
                grpc_params.training_server_address.clone()
            };
            self.grpc_params.write().await.training_server_address =
                resolve_new_training_server_address(
                    old_training_server_address.as_str(),
                    training_server_address,
                )
                .await;

            let self_arc: Arc<TokioRwLock<TrainingServerGrpc>> =
                Arc::new(TokioRwLock::new(TrainingServerGrpc {
                    active: self.active.clone(),
                    server_model_path: self.server_model_path.clone(),
                    max_traj_length: self.max_traj_length,
                    hyperparams: self.hyperparams.clone(),
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
                    grpc_params: self.grpc_params.clone(),
                    #[cfg(not(feature = "python_bindings"))]
                    tokio_runtime: self.tokio_runtime.clone(),
                }));

            let training_server_address: String = self
                .grpc_params
                .read()
                .await
                .training_server_address
                .clone();
            let servicing_task: TokioMutex<Option<JoinHandle<()>>> =
                TokioMutex::new(Some(tokio::task::spawn(async move {
                    if let Err(e) = grpc_serve(self_arc.clone(), training_server_address).await {
                        eprintln!(
                            "[TrainingServer - enable] Failed to start gRPC server: {}",
                            e
                        );
                    }
                })));

            self.grpc_params.write().await.servicing_task = servicing_task;
            self.active.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Sends a save model command request to the PythonAlgorithmRequest.
    ///
    /// This instructs the algorithm Python script to save the current model.
    /// A oneshot channel is used to receive the outcome as a boolean.
    ///
    /// # Returns
    /// * `true` if the model was successfully saved; `false` otherwise.
    pub async fn par_send_save_model(&self) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let command = PythonAlgorithmCommand::SaveModel(tx);

        // Attempt to send the save model command via the command channel.
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

        // Await and return the command's outcome.
        rx.await.unwrap_or(false)
    }

    /// Sends a trajectory command request to the PythonAlgorithmRequest.
    ///
    /// This command delivers a training trajectory to the algorithm Python script for processing.
    /// A oneshot channel is used to receive a boolean outcome indicating whether processing triggered a model update.
    ///
    /// # Arguments
    /// * `trajectory` - The RelayRLTrajectory representing the training data.
    ///
    /// # Returns
    /// * `true` if the trajectory was processed and resulted in a model update; `false` otherwise.
    pub async fn par_send_receive_trajectory(&self, trajectory: RelayRLTrajectory) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let command = PythonAlgorithmCommand::ReceiveTrajectory(tx, trajectory);

        // Attempt to send the trajectory command.
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

        // Await and return the outcome of the trajectory processing.
        rx.await.unwrap_or(false)
    }

    pub(crate) async fn increment_req_state(self: &Self) {
        let mut grpc_params_write = self.grpc_params.write().await;
        let mut state = grpc_params_write.request_state.write().await;
        *state += 1;
    }

    pub(crate) async fn increment_rep_state(self: &Self) {
        let mut grpc_params_write = self.grpc_params.write().await;
        let mut state = grpc_params_write.reply_state.write().await;
        *state += 1;
    }
}

#[tonic::async_trait]
impl RelayRLRoute for Arc<TokioRwLock<TrainingServerGrpc>> {
    /// Handles incoming training actions from clients.
    ///
    /// This method performs the following:
    /// - Receives a client's trajectory submission,
    /// - Converts the trajectory into the internal RelayRLTrajectory format,
    /// - Spawns an asynchronous task to process the trajectory without blocking the current thread,
    /// - If processing succeeds, triggers a model save and updates shared gRPC parameters.
    ///
    /// # Arguments
    /// * `request` - A gRPC request containing a Trajectory message.
    ///
    /// # Returns
    /// * A gRPC response with an ActionResponse message indicating that training has started.
    async fn send_actions(
        &self,
        request: Request<grpc_Trajectory>,
    ) -> Result<Response<ActionResponse>, Status> {
        println!("[TrainingServer - send_actions] Received trajectory from client");
        let trajectory: grpc_Trajectory = request.into_inner();
        let server_model_path: PathBuf = self.read().await.server_model_path.clone();
        let grpc_params: Arc<TokioRwLock<GrpcServiceParams>> =
            Arc::clone(&self.read().await.grpc_params);
        let relayrl_trajectory: RelayRLTrajectory =
            grpc_trajectory_to_relayrl_trajectory(trajectory, self.read().await.max_traj_length);

        // Spawn an asynchronous task to process the trajectory without blocking.
        let self_arc: Arc<RwLock<TrainingServerGrpc>> = Arc::clone(self);
        tokio::task::spawn(async move {
            let updated_model_result: bool = self_arc
                .read()
                .await
                .par_send_receive_trajectory(relayrl_trajectory)
                .await;
            if updated_model_result {
                // If trajectory processing resulted in a model update, trigger saving the new model.
                let saved_model: bool = self_arc.read().await.par_send_save_model().await;

                if !saved_model {
                    eprintln!("[TrainingServer - send_actions] Failed to send save model request");
                }

                // Lock the shared gRPC parameters and update them with the new model.
                let mut grpc_state: RwLockWriteGuard<GrpcServiceParams> = grpc_params.write().await;
                // Optionally reset the model_ready flag if needed.
                grpc_state.model_ready = false;
                // Update the trained model by loading it from disk.
                {
                    let mut model_lock = grpc_state.trained_model.write().await;
                    *model_lock = Some(
                        CModule::load_on_device(&server_model_path, Device::Cpu)
                            .expect("Failed to load received model into runtime CModule"),
                    );
                }
                // Set model_ready based on the save outcome.
                grpc_state.model_ready = saved_model;
                grpc_state.trajectory_buffer.clear();

                // **(Solution 2)**: Send an update on the watch channel.
                if let Some(tx) = &grpc_state.model_ready_tx {
                    let _ = tx.send(grpc_state.model_ready);
                }
            } else {
                // If no model update occurred, log an error message in the shared state.
                let mut grpc_state = grpc_params.write().await;
                grpc_state.model_ready = false;
                grpc_state.error_message = Some(String::from("No new model trained"));
            }
        });

        // Respond to the client indicating that training has been initiated.
        self.write().await.increment_rep_state().await;
        Ok(Response::new(ActionResponse {
            code: 1,
            message: "Training started successfully for client".into(),
        }))
    }

    /// Listens for handshake requests from agents and responds with the current model status.
    ///
    /// This method handles initial polling (handshake) requests from clients. If it is the client's
    /// first request, a handshake is performed. Otherwise, if a new model is ready, it is sent to the client.
    /// If the model is still training or an error exists, an appropriate message is returned.
    ///
    /// # Arguments
    /// * `request` - A gRPC request containing a RequestModel message.
    ///
    /// # Returns
    /// * A gRPC response with an RelayRLModel message containing the model data or status information.
    async fn client_poll(
        &self,
        request: Request<RequestModel>,
    ) -> Result<Response<RelayRLModel>, Status> {
        println!("[TrainingServer - client_poll] Received poll request from client...");
        let req: RequestModel = request.into_inner();

        // For the initial handshake
        if req.first_time != 0 {
            println!(
                "[TrainingServer - client_poll] Handshake initiated by client. Client version: {}",
                req.version
            );

            let self_state = self.read().await;

            // Check if a model load is needed
            let need_model_load = {
                let grpc_params_read = self_state.grpc_params.read().await;
                grpc_params_read.trained_model.read().await.is_none()
            };

            if need_model_load {
                let outcome = self_state.par_send_save_model().await;
                println!(
                    "[TrainingServer - client_poll] Attempted to save model during handshake: {:?}",
                    outcome
                );

                if !outcome {
                    eprintln!("[TrainingServer - client_poll] Failed to save model during handshake");
                    return Ok(Response::new(RelayRLModel {
                        code: 0,
                        model: vec![],
                        version: 0,
                        error: String::from("Failed to save model during handshake"),
                    }));
                }

                // Load the model from disk
                match CModule::load_on_device(&self_state.server_model_path, Device::Cpu) {
                    Ok(model) => {
                        println!("[TrainingServer - client_poll] Loaded model during handshake");
                        let mut grpc_params_write = self_state.grpc_params.write().await;
                        let mut trained = grpc_params_write.trained_model.write().await;
                        *trained = Some(model);
                    }
                    Err(e) => {
                        eprintln!("Failed to load model during handshake: {}", e);
                    }
                }
                self_state.grpc_params.write().await.model_ready = true;
            }

            // Respond with the model if ready
            let grpc_params_read = self_state.grpc_params.read().await;
            if grpc_params_read.model_ready {
                let model_bytes = {
                    let lock = grpc_params_read.trained_model.read().await;
                    if let Some(ref model) = *lock {
                        serialize_model(model, std::env::current_dir()?)
                    } else {
                        vec![]
                    }
                };
                println!("[TrainingServer - client_poll] Sending model to client...");
                return Ok(Response::new(RelayRLModel {
                    code: 1,
                    model: model_bytes,
                    version: 0,
                    error: String::from("Handshake successful."),
                }));
            }
        }

        // For subsequent polling requests:
        let self_state = self.read().await;
        let grpc_params_read = self_state.grpc_params.read().await;
        if grpc_params_read.model_ready {
            let model_bytes: Vec<u8> = {
                let lock = grpc_params_read.trained_model.read().await;
                if let Some(ref model) = *lock {
                    serialize_model(model, env::current_dir()?)
                } else {
                    vec![]
                }
            };
            println!("[TrainingServer - client_poll] Sending model to client...");
            return Ok(Response::new(RelayRLModel {
                code: 1,
                model: model_bytes,
                version: 0,
                error: String::new(),
            }));
        }

        // Wait for a notification that the model is ready, with a timeout.
        let mut rx = grpc_params_read
            .model_ready_tx
            .as_ref()
            .expect("Failed to get watch channel")
            .subscribe();
        let timeout_duration = Duration::from_millis(grpc_params_read.idle_timeout as u64);
        drop(grpc_params_read);
        match tokio::time::timeout(timeout_duration, rx.changed()).await {
            Ok(Ok(())) => {
                let self_state = self.read().await;
                let grpc_params = self_state.grpc_params.read().await;
                if grpc_params.model_ready {
                    let model_bytes: Vec<u8> = {
                        let lock = grpc_params.trained_model.read().await;
                        if let Some(ref model) = *lock {
                            serialize_model(model, env::current_dir()?)
                        } else {
                            vec![]
                        }
                    };
                    Ok(Response::new(RelayRLModel {
                        code: 1,
                        model: model_bytes,
                        version: 0,
                        error: String::new(),
                    }))
                } else {
                    Ok(Response::new(RelayRLModel {
                        code: 0,
                        model: vec![],
                        version: 0,
                        error: String::from("Model is still training"),
                    }))
                }
            }
            Ok(Err(_)) => Err(Status::internal("Model readiness channel closed")),
            Err(_) => {
                Ok(Response::new(RelayRLModel {
                    code: 0,
                    model: vec![],
                    version: 0,
                    error: String::from("Timeout: Model is still training"),
                }))
            }
        }
    }
}
