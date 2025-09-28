//! This module implements the gRPC-based RelayRL agent interface for communicating with the training server.
//! It defines a trait for asynchronous agent operations and provides an implementation that manages the
//! model, handles action requests, collects trajectories, and updates the model when new versions are available.

use crate::proto::relay_rl_route_client::RelayRlRouteClient as RelayRLRouteClient;
use crate::proto::{
    ActionResponse, RelayRlAction as grpc_RelayRLAction, RelayRlModel as RelayRLModel, RequestModel,
    Trajectory as grpc_Trajectory,
};
use crate::sys_utils::config_loader::{DEFAULT_CONFIG_CONTENT, DEFAULT_CONFIG_PATH};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;
use tch::{CModule, Device, IValue, Kind, Tensor, no_grad};
use tokio::fs;
use tokio::sync::{RwLock as TokioRwLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tonic::Request;
use tonic::transport::{Channel, Endpoint, Error};

use crate::network::client::agent_wrapper::{convert_generic_dict, validate_model};
use crate::sys_utils::grpc_utils::{deserialize_model, serialize_action};
use crate::sys_utils::config_loader::ConfigLoader;
use crate::types::action::{RelayRLAction, RelayRLData, TensorData};
use crate::types::trajectory::{RelayRLTrajectory, RelayRLTrajectoryTrait};

#[cfg(all(feature = "console-subscriber", not(feature = "python_bindings")))]
use crate::orchestration::tokio::utils::get_or_init_console_subscriber;
#[cfg(not(feature = "python_bindings"))]
use crate::orchestration::tokio::utils::get_or_init_tokio_runtime;
#[cfg(not(feature = "python_bindings"))]
use tokio::runtime::Runtime as TokioRuntime;

use crate::{get_or_create_config_json_path, resolve_config_json_path};

/// Trait defining the asynchronous gRPC agent interface for the RelayRL framework.
///
/// This trait includes methods for:
/// - Performing an initial model handshake,
/// - Requesting actions from the current model,
/// - Flagging the final action of a trajectory,
/// - Sending trajectories to the training server,
/// - Polling for model updates,
/// - And closing the gRPC channel.
#[tonic::async_trait]
pub trait RelayRLAgentGrpcTrait {
    /// Initiates a one-time handshake with the training server to retrieve the initial model.
    async fn initial_model_handshake(&mut self);

    /// Requests an action from the model given the current observation and mask.
    ///
    /// # Arguments
    /// * `obs` - The observation tensor.
    /// * `mask` - The mask tensor.
    /// * `reward` - The immediate reward.
    ///
    /// # Returns
    /// * `Result<Arc<RelayRLAction>, String>` - On success, returns the generated action (wrapped in an Arc);
    ///   on failure, returns an error string.
    async fn request_for_action(
        &mut self,
        obs: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Arc<RelayRLAction>, String>;

    /// Finalizes the current trajectory by appending a terminal action, sends the trajectory to the server,
    /// polls for any updated model, and then resets the local trajectory buffer.
    ///
    /// # Arguments
    /// * `reward` - The final reward associated with the terminal action.
    async fn flag_last_action(&mut self, reward: f32);

    /// Serializes and sends the current trajectory to the training server.
    ///
    /// # Returns
    /// * `i32` - Returns 1 if the server successfully accepts the trajectory, or 0 otherwise.
    async fn send_trajectory_to_server(&mut self) -> i32;

    /// Polls the training server for an updated model.
    ///
    /// If a new model is available (i.e. non-empty model bytes are received and the version is updated),
    /// the new model is deserialized, loaded, and the local version is updated.
    async fn poll_for_model_update(&mut self);
}

/// gRPC-based RelayRL Agent implementation.
///
/// This struct implements the RelayRLAgentGrpcTrait and provides a gRPC interface
/// for communicating with the training server. It manages the model, handles action requests,
/// collects trajectories, and updates the model as new versions become available.
pub struct RelayRLAgentGrpc {
    /// Status flag indicating whether the agent is active.
    active: Arc<AtomicBool>,
    /// The current model, stored as a TorchScript module.
    /// Wrapped in an Arc<TokioRwLock<>> for thread-safe shared access.
    model: Arc<TokioRwLock<Option<CModule>>>,
    /// File path for saving and loading the model.
    client_model_path: PathBuf,
    /// gRPC client stub used for communication with the training server.
    stub: Option<RelayRLRouteClient<Channel>>,
    /// Local model version number, used to check for updates.
    local_version: AtomicI64,
    /// Buffer holding the sequence of actions (trajectory) collected by the agent.
    current_traj: RelayRLTrajectory,
    /// Tokio runtime (assuming bindings aren't compiled)
    #[cfg(not(feature = "python_bindings"))]
    tokio_runtime: Arc<TokioRuntime>,
}

impl RelayRLAgentGrpc {
    /// Creates a new instance of the gRPC-based RelayRL agent.
    ///
    /// This function loads configuration settings, establishes a gRPC channel to the training server,
    /// and initializes the agent's state including the model (if provided) and trajectory buffer.
    ///
    /// # Arguments
    /// * `model` - An optional initial TorchScript model.
    /// * `config_path` - An optional configuration file path.
    /// * `training_server_address` - An optional address for the training server (defaults to "http://localhost:50051").
    ///
    /// # Returns
    /// * A new instance of RelayRLAgentGrpc.
    pub async fn init_agent(
        model: Option<CModule>,
        config_path: Option<PathBuf>,
        training_server_address: Option<String>,
    ) -> Self {
        #[cfg(all(feature = "console-subscriber", not(feature = "python_bindings")))]
        get_or_init_console_subscriber();
        #[cfg(not(feature = "python_bindings"))]
        let tokio_runtime: Arc<TokioRuntime> = get_or_init_tokio_runtime();

        // Load configloader parameters
        let client_model_path: PathBuf;
        let max_traj_length: u32;
        {
            let config_path = resolve_config_json_path!(config_path);
            let config: ConfigLoader = ConfigLoader::new(None, config_path);
            client_model_path = config.client_model_path;
            max_traj_length = config.max_traj_length;
        }

        // Create and connect the gRPC channel using the specified server address.
        let server_addr: String =
            training_server_address.unwrap_or_else(|| "127.0.0.1:50051".to_string());
        let grpc_endpoint: Endpoint =
            Channel::from_shared(server_addr).expect("Invalid server address");
        let grpc_channel: Option<Channel> = match grpc_endpoint.connect().await {
            Ok(channel) => Some(channel),
            Err(_) => {
                // retry 60 times
                let mut counter: i32 = 60;
                'retry: loop {
                    if counter > 0 {
                        match grpc_endpoint.connect().await {
                            Ok(channel) => Some(channel),
                            Err(_) => {
                                continue;
                            }
                        };
                    } else {
                        break 'retry;
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                None
            }
        };

        let stub = match grpc_channel {
            Some(channel) => Some(RelayRLRouteClient::new(channel)),
            None => {
                panic!("grpc_channel not provided");
            }
        };

        // acquire shared access to the model
        let model_arc: Arc<RwLock<Option<CModule>>> = Arc::new(TokioRwLock::new(model));

        // Initialize the agent's state.
        let mut agent: RelayRLAgentGrpc = Self {
            active: Arc::new(AtomicBool::new(true)),
            model: Arc::clone(&model_arc),
            client_model_path,
            current_traj: RelayRLTrajectory::new(Some(max_traj_length), None),
            stub: stub,
            local_version: AtomicI64::new(0),
            #[cfg(not(feature = "python_bindings"))]
            tokio_runtime,
        };

        let model_is_none = agent.model.read().await.is_none();

        // Handle initial model handshake with server
        if model_is_none {
            agent.initial_model_handshake().await;
        } else {
            validate_model(
                agent
                    .model
                    .read()
                    .await
                    .as_ref()
                    .expect("Failed to read runtime model"),
            );
        }

        agent
    }

    /// Restarts the current instance of the gRPC agent.
    ///
    /// This is an abstraction of the following function operations:
    ///
    /// 1. `disable_agent()`
    /// 2. `enable_agent()`
    ///
    pub async fn restart_agent(
        &mut self,
        training_server_address: Option<String>,
    ) -> Vec<Result<(), Box<dyn std::error::Error>>> {
        let disable_result: Result<(), Box<dyn std::error::Error>> = self.disable_agent().await;
        let enable_result: Result<(), Box<dyn std::error::Error>> =
            self.enable_agent(training_server_address).await;
        vec![disable_result, enable_result]
    }

    /// Disables networking operations for gRPC agent.
    pub async fn disable_agent(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.active.load(Ordering::SeqCst) {
            self.stub.take().ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "gRPC stub not found",
                )) as Box<dyn std::error::Error>
            })?;
            self.active.store(false, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Re-enables networking operations for gRPC agent.
    ///
    /// For initial instantiation, use init_agent()
    pub async fn enable_agent(
        &mut self,
        training_server_address: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let result: Result<(), Box<dyn std::error::Error>> = if !self.active.load(Ordering::SeqCst)
        {
            let mut inner_results: Result<(), Box<dyn std::error::Error>> = Ok(());

            let server_addr: String =
                training_server_address.unwrap_or_else(|| "127.0.0.1:50051".to_string());
            let grpc_endpoint: Endpoint =
                Channel::from_shared(server_addr).expect("Invalid server address");
            let grpc_channel: Option<Channel> = match grpc_endpoint.connect().await {
                Ok(channel) => Some(channel),
                Err(err) => {
                    let mut counter: i32 = 60;
                    // retry 60 times
                    'retry: loop {
                        if counter > 0 {
                            match grpc_endpoint.connect().await {
                                Ok(channel) => Some(channel),
                                Err(_) => {
                                    continue;
                                }
                            };
                        } else {
                            inner_results = Err(Box::new(err));
                            break 'retry;
                        };
                        counter -= 1;
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                    None
                }
            };

            self.stub = match grpc_channel {
                Some(channel) => Some(RelayRLRouteClient::new(channel)),
                None => {
                    panic!("grpc_channel not provided");
                }
            };

            let model_is_none: bool = self.model.read().await.is_none();
            if model_is_none {
                self.initial_model_handshake().await;
            }

            inner_results
        } else {
            Ok(())
        };

        result
    }

    pub async fn get_model_version(&self) -> i64 {
        let version: i64 = self.local_version.load(Ordering::SeqCst);
        version
    }
}

#[tonic::async_trait]
impl RelayRLAgentGrpcTrait for RelayRLAgentGrpc {
    /// Initiates a one-time handshake with the training server to retrieve the initial model.
    ///
    /// Sends a RequestModel message with `first_time` set to 1.
    /// If the server responds with a valid model (code == 1 and non-empty model bytes),
    /// the model is deserialized and loaded, and the local version is updated.
    async fn initial_model_handshake(&mut self) {
        println!("[RelayRLAgent - initial_handshake] requesting initial model from server...");
        let req_msg = RequestModel {
            first_time: 1,
            version: 0, // Use simple versioning
        };
        loop {
            match self
                .stub
                .as_mut()
                .expect("Failed to get gRPC stub")
                .client_poll(Request::new(req_msg))
                .await
            {
                Ok(response) => {
                    let resp: RelayRLModel = response.into_inner();
                    if resp.code == 1 && !resp.model.is_empty() {
                        println!("[RelayRLAgent - initial_handshake] Received initial model from server...");
                        match deserialize_model(resp.model) {
                            Ok(loaded_model) => {
                                validate_model(&loaded_model);
                                loaded_model.save(&self.client_model_path).expect("Failed to save runtime model to path");
                                let mut model_lock = self.model.write().await;
                                *model_lock = Some(loaded_model);
                                println!("[RelayRLAgent - initial_handshake] Model loaded, proceeding to RL loop.");
                                break;
                            }
                            Err(e) => {
                                eprintln!("[RelayRLAgent - initial_handshake] Failed to load model: {}", e);
                            }
                        }
                    } else {
                        println!("[RelayRLAgent - initial_handshake] No initial model available. Waiting...");
                    }
                }
                Err(e) => {
                    eprintln!("[RelayRLAgent - initial_handshake] gRPC error during handshake: {}", e);
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        // After this, always proceed to RL loop!
    }

    /// Requests an action from the current model using the provided observation and mask.
    ///
    /// This function offloads all TorchScript inference to a blocking thread using `spawn_blocking`.
    /// It then constructs an `RelayRLAction` (which contains only safe, serializable data) inside the
    /// blocking thread and returns it back to the async context.
    ///
    /// # Arguments
    /// * `obs` - The observation tensor (moved).
    /// * `mask` - The mask tensor (moved).
    /// * `reward` - The immediate reward.
    async fn request_for_action(
        &mut self,
        obs: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<Arc<RelayRLAction>, String> {
        let action_result = {
            // Lock the model
            let model_guard: RwLockReadGuard<Option<CModule>> = self.model.read().await;
            let model = match &*model_guard {
                Some(m) => m,
                None => return Err("No model available yet!".to_string()),
            };

            // Move Tensors to CPU contiguously
            let obs: Tensor = obs.to_device(Device::Cpu).contiguous();
            let mask: Tensor = mask.to_device(Device::Cpu).contiguous();

            // Convert Tensors -> IValue
            let obs_ivalue = IValue::Tensor(obs.to_kind(Kind::Float));
            let mask_ivalue = IValue::Tensor(mask.to_kind(Kind::Float));
            let inputs: Vec<IValue> = vec![obs_ivalue, mask_ivalue];

            // Execute step(...) in a blocking context
            no_grad(|| {
                let output_ivalue: IValue = model
                    .method_is("step", &inputs)
                    .map_err(|e| format!("Failed to call model.step: {}", e))?;

                // Expect output to be a 2-tuple: (action_tensor, data_dict)
                let outputs: &Vec<IValue> = match output_ivalue {
                    IValue::Tuple(ref tup) if tup.len() == 2 => tup,
                    _ => return Err("step() did not return (action, data_dict) tuple".to_string()),
                };

                // Extract action
                let action_tensor: Tensor = if let IValue::Tensor(t) = &outputs[0] {
                    t.to_kind(Kind::Float)
                } else {
                    Tensor::zeros([], (Kind::Float, Device::Cpu))
                };

                // Convert data
                let data_dict: Option<HashMap<String, RelayRLData>> = match &outputs[1] {
                    IValue::GenericDict(dict) => {
                        Some(convert_generic_dict(dict).expect("Failed to convert data dict"))
                    }
                    _ => Some(HashMap::new()),
                };

                // Build RelayRLAction with Tensors turned into `TensorData`
                let obs_td: TensorData =
                    TensorData::try_from(&obs).expect("Failed to convert obs to TensorData");
                let act_td: TensorData = TensorData::try_from(&action_tensor)
                    .expect("Failed to convert act to TensorData");
                let mask_td: TensorData =
                    TensorData::try_from(&mask).expect("Failed to convert mask to TensorData");

                let r4sa: RelayRLAction = RelayRLAction::new(
                    Some(obs_td),
                    Some(act_td),
                    Some(mask_td),
                    reward,
                    data_dict,
                    false,
                    false,
                );

                Ok(r4sa)
            })
        };

        // Unwrap any error from inside the blocking code
        let r4sa: RelayRLAction = match action_result {
            Ok(r4sa) => r4sa,
            Err(msg) => return Err(msg),
        };

        // Add the action to the current trajectory
        self.current_traj.add_action(&r4sa, false);

        // Return the Arc
        Ok(Arc::new(r4sa))
    }

    /// Finalizes the current trajectory by appending a terminal action, sending the trajectory to the server,
    /// polling for model updates, and resetting the local trajectory.
    ///
    /// A terminal action is created with `done` set to true and appended to the trajectory.
    /// Then, the trajectory is serialized and sent to the server. If the server accepts the trajectory,
    /// the agent polls for an updated model before clearing the trajectory buffer.
    ///
    /// # Arguments
    /// * `reward` - The final reward associated with the terminal action.
    async fn flag_last_action(&mut self, reward: f32) {
        // Create a terminal action (indicating the end of an episode) with the given reward.
        let mut last_action: RelayRLAction =
            RelayRLAction::new(None, None, None, reward, None, true, false);
        last_action.update_reward(reward);
        self.current_traj.add_action(&last_action, false);

        // Send the trajectory to the training server.
        let response_code: i32 = self.send_trajectory_to_server().await;

        if response_code == 0 {
            println!("[RelayRLAgent - flag_last_action] Keep collecting trajectory");
            return;
        }

        // Poll the server for a model update.
        self.poll_for_model_update().await;
    }

    /// Serializes the current trajectory and sends it to the training server.
    ///
    /// Each stored action is converted to its gRPC representation using the `serialize_action` utility.
    /// The actions are then wrapped into a `grpc_Trajectory` message and sent using the gRPC client stub.
    ///
    /// # Returns
    /// * `i32` - Returns 1 if the trajectory was successfully sent, or 0 if it was rejected.
    async fn send_trajectory_to_server(&mut self) -> i32 {
        let mut action_msgs: Vec<grpc_RelayRLAction> = Vec::new();
        // Convert each action in the trajectory into its proto representation.
        for action in &self.current_traj.actions {
            let proto_action: grpc_RelayRLAction = serialize_action(action); // return == grpc_RelayRLAction.
            action_msgs.push(proto_action);
        }
        // Wrap the actions into a grpc_Trajectory message.
        let traj = grpc_Trajectory {
            actions: action_msgs,
        };
        println!("[RelayRLAgent - send_trajectory] Sending trajectory to server...");
        let request: Request<grpc_Trajectory> = Request::new(traj);
        match self
            .stub
            .as_mut()
            .expect("Failed to get gRPC stub")
            .send_actions(request)
            .await
        {
            Ok(response) => {
                let resp: ActionResponse = response.into_inner();
                if resp.code == 1 {
                    println!(
                        "[RelayRLAgent - send_trajectory] Trajectory sent: {}",
                        resp.message
                    );
                    1
                } else {
                    println!(
                        "[RelayRLAgent - send_trajectory] Trajectory rejected: {}",
                        resp.message
                    );
                    0
                }
            }
            Err(e) => {
                eprintln!("[RelayRLAgent - send_trajectory] Trajectory error: {}", e);
                std::process::exit(1);
            }
        }
    }

    /// Polls the training server for an updated model.
    ///
    /// A RequestModel message is sent with `first_time` set to 0 and the current local version.
    /// If a new model is available (i.e. the returned model bytes are non-empty), the model is deserialized,
    /// loaded, and the local version is updated.
    async fn poll_for_model_update(&mut self) {
        println!("[RelayRLAgent - poll_for_model] Polling for model update...");
        let current_version: i64 = self.local_version.load(Ordering::SeqCst);
        let poll_req = RequestModel {
            first_time: 0,
            version: current_version,
        };
        let request: Request<RequestModel> = Request::new(poll_req);
        match self
            .stub
            .as_mut()
            .expect("Failed to get gRPC stub")
            .client_poll(request)
            .await
        {
            Ok(response) => {
                println!("[RelayRLAgent - poll_for_model] Received poll response from server...");
                let resp: RelayRLModel = response.into_inner();
                if resp.code == 1 {
                    if !resp.model.is_empty() {
                        let mut model_lock: RwLockWriteGuard<Option<CModule>> =
                            self.model.write().await;
                        match deserialize_model(resp.model) {
                            Ok(loaded_model) => {
                                // validate and then load new model into memory
                                validate_model(&loaded_model);
                                loaded_model
                                    .save(&self.client_model_path)
                                    .expect("Failed to save runtime model to path");
                                *model_lock = Some(loaded_model);
                                self.local_version = AtomicI64::from(resp.version);
                                println!(
                                    "[RelayRLAgent - poll_for_model] Updated local model from server (poll)."
                                );
                            }
                            Err(e) => {
                                eprintln!(
                                    "[RelayRLAgent - poll_for_model] Failed to load updated model: {}",
                                    e
                                );
                            }
                        }
                    }
                } else if resp.code == 0 {
                    // No update available.
                } else if resp.code == -1 {
                    println!(
                        "[RelayRLAgent - poll_for_model] Server reported error: {}",
                        resp.error
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[RelayRLAgent - poll_for_model] gRPC error while polling for model: {}",
                    e
                );
            }
        }
    }
}
