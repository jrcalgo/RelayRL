//! This module implements a ZeroMQ (ZMQ) based agent for the RelayRL framework.
//! It handles model initialization, action requests, trajectory recording, and continuous model updates via ZMQ sockets.

use rand::Rng;
use rand::prelude::ThreadRng;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;
use std::{process, thread};
use tch::IValue::GenericDict;
use tch::{CModule, Device, IValue, Kind, Tensor, no_grad};
use zmq::{Context, Socket};

use crate::network::client::agent_wrapper::{convert_generic_dict, validate_model};
use crate::types::action::{RelayRLAction, RelayRLData, TensorData};
use crate::types::trajectory::{RelayRLTrajectory, RelayRLTrajectoryTrait};

use crate::sys_utils::config_loader::{ConfigLoader, DEFAULT_CONFIG_CONTENT, DEFAULT_CONFIG_PATH};
use crate::{get_or_create_config_json_path, resolve_config_json_path};

/// Trait defining the public interface for a ZMQ-based RelayRL agent.
///
/// This trait provides functions for:
/// - Requesting an action based on current observations.
/// - Recording an action into the agent's trajectory.
/// - Flagging the final action in an episode/trajectory.
/// - Continuously listening for updated models from the training server.
pub trait RelayRLAgentZmqTrait {
    ///
    fn initial_model_handshake(
        &mut self,
        model_arc: Arc<Mutex<Option<CModule>>>,
        agent_id: &[u8],
        agent_listening_server: &str,
        client_model_path: &PathBuf,
        training_server: &str,
    );

    /// Request an action from the model using the provided observation, mask, and reward.
    ///
    /// # Arguments
    /// * `obs` - A tensor representing the current observation.
    /// * `mask` - A tensor representing any applicable mask on the observation.
    /// * `reward` - A float representing the immediate reward.
    ///
    /// # Returns
    /// * `Result<Arc<RelayRLAction>, &str>` - An Arc-wrapped action on success or an error string on failure.
    fn request_for_action(
        &mut self,
        obs: &Tensor,
        mask: &Tensor,
        reward: f32,
    ) -> Result<Arc<RelayRLAction>, &str>;

    /// Record an action into the agent's trajectory.
    ///
    /// This method stores detailed information about the action taken including:
    /// - The observation,
    /// - The executed action,
    /// - The corresponding mask,
    /// - The reward received,
    /// - Additional auxiliary data,
    /// - A flag indicating if the episode is done,
    /// - A flag indicating if the reward was updated.
    ///
    /// # Arguments
    /// * `obs` - The observation tensor.
    /// * `act` - The tensor representing the action.
    /// * `mask` - The mask tensor.
    /// * `reward` - The reward received.
    /// * `data` - Optional auxiliary data.
    /// * `done` - A boolean flag indicating episode termination.
    /// * `reward_update_flag` - A boolean flag to indicate reward updates.
    fn record_action(
        &mut self,
        obs: &Tensor,
        act: &Tensor,
        mask: &Tensor,
        reward: &f32,
        data: &Option<HashMap<String, RelayRLData>>,
        done: bool,
        reward_update_flag: bool,
    );

    /// Flag the final action in the trajectory with the given reward.
    ///
    /// This method creates a terminal action that marks the end of an episode
    /// and triggers the sending of the trajectory to the training server.
    ///
    /// # Arguments
    /// * `reward` - The reward to be assigned to the final action.
    fn flag_last_action(&mut self, reward: f32);

    /// Continuously poll for an updated model from the training server.
    ///
    /// This function sets up a loop that:
    /// - Uses a ZMQ PULL socket to receive model bytes,
    /// - Writes the received bytes to a file,
    /// - Loads the updated TorchScript model,
    /// - And updates the agent's current model.
    ///
    /// # Arguments
    /// * `model` - Shared reference to the model wrapped in an Arc and Mutex.
    /// * `training_server` - The training server address to bind the socket.
    /// * `client_model_path` - The file path to save the received model.
    fn _loop_for_updated_model(
        model: Arc<Mutex<Option<CModule>>>,
        training_server: String,
        client_model_path: PathBuf,
    );
}

/// Struct representing a ZMQ-based RelayRL agent.
///
/// The agent is responsible for:
/// - Initializing by performing a handshake with the training server to obtain the initial model.
/// - Requesting actions using the model's inference method.
/// - Recording the trajectory of actions.
/// - Continuously updating its model from the training server via a background thread.
pub struct RelayRLAgentZmq {
    /// Status of network
    active: Arc<AtomicBool>,
    /// Unique identifier for this agent.
    agent_id: String,
    /// The current model wrapped in an Arc<Mutex<>> for thread-safe sharing.
    model: Arc<Mutex<Option<CModule>>>,
    /// Address of the training server used for receiving model updates.
    training_server: String,
    /// Path to the local file where the model is stored.
    client_model_path: PathBuf,
    /// Local model version number, used to check for updates.
    local_version: AtomicI64,
    /// Trajectory buffer for storing actions taken by the agent.
    current_traj: RelayRLTrajectory,
    /// Thread handle for listening for model from training server.
    model_listener_thread: Mutex<Option<JoinHandle<()>>>,
}

impl RelayRLAgentZmq {
    /// Creates a new instance of the RelayRLAgentZmq.
    ///
    /// This method initializes the agent by:
    /// - Loading configuration parameters (such as server addresses and file paths).
    /// - Setting up a ZMQ DEALER socket for initial model handshake.
    /// - Waiting for and loading the initial model.
    /// - Spawning a background thread for continuous model updates.
    ///
    /// # Arguments
    /// * `model` - An optional initial TorchScript model.
    /// * `config_path` - Optional path to the configuration file.
    /// * `training_prefix` - Optional prefix for the training server address.
    /// * `training_host` - Optional host for the training server.
    /// * `training_port` - Optional port for the training server.
    ///
    /// # Returns
    /// * `Self` - A fully initialized RelayRLAgentZmq instance.
    pub fn init_agent(
        model: Option<CModule>,
        config_path: Option<PathBuf>,
        training_server_address: Option<String>,
    ) -> Self {
        println!("[Instantiating RelayRL-Framework Agent...]");

        // Generate a unique agent identifier using the process ID and a random number.
        let mut rng: ThreadRng = rand::thread_rng();
        let agent_id: Vec<u8> = format!("AGENT_ID-{:?}{:?}", process::id(), rng.gen_range(0..=99))
            .as_bytes()
            .to_vec();

        // Variables for server addresses and configuration settings.
        let trajectory_server: String;
        let agent_listening_server: String;
        let client_model_path: PathBuf;
        let max_traj_length: u32;
        {
            let config_path: Option<PathBuf> = resolve_config_json_path!(config_path);
            // Load configuration settings.
            let config: ConfigLoader = ConfigLoader::new(None, config_path);

            // Construct the trajectory server address.
            let mut prefix: String = config.traj_server.prefix;
            let mut host: String = config.traj_server.host;
            let mut port: String = config.traj_server.port;
            trajectory_server = format!("{}{}:{}", prefix, host, port);

            // Construct the agent listener server address.
            prefix = config.agent_listener.prefix;
            host = config.agent_listener.host;
            port = config.agent_listener.port;
            agent_listening_server = format!("{}{}:{}", prefix, host, port);

            // Retrieve the local model file path and maximum trajectory length.
            client_model_path = config.client_model_path;
            max_traj_length = config.max_traj_length;
        }

        // Wrap the optional initial model in an Arc and Mutex for safe sharing.
        let model_arc: Arc<Mutex<Option<CModule>>> = Arc::new(Mutex::new(model));

        let training_server_clone: String = training_server_address
            .clone()
            .expect("training server address is None");
        let client_model_path_clone: PathBuf = client_model_path.clone();

        let mut zmq_agent = RelayRLAgentZmq {
            active: Arc::new(AtomicBool::new(true)),
            agent_id: String::from_utf8_lossy(&agent_id).into_owned(),
            model: Arc::clone(&model_arc),
            training_server: training_server_address.expect("Training server address unavailable"),
            client_model_path,
            local_version: AtomicI64::new(0),
            current_traj: RelayRLTrajectory::new(Some(max_traj_length), Some(trajectory_server)),
            model_listener_thread: Mutex::new(None),
        };

        let handshake_model_arc: Arc<Mutex<Option<CModule>>> = Arc::clone(&model_arc);
        Self::initial_model_handshake(
            &mut zmq_agent,
            handshake_model_arc,
            agent_id.as_slice(),
            agent_listening_server.as_str(),
            &client_model_path_clone,
            training_server_clone.as_str(),
        );

        let listening_model_arc: Arc<Mutex<Option<CModule>>> = Arc::clone(&model_arc);
        // Spawn a background thread to continuously listen for updated models from the training server.
        println!("[RelayRLAgent - new] Starting thread to listen for updated models");
        let model_listener_thread: JoinHandle<()> = thread::spawn(move || {
            RelayRLAgentZmq::_loop_for_updated_model(
                listening_model_arc,
                training_server_clone,
                client_model_path_clone,
            )
        });
        zmq_agent.model_listener_thread = Mutex::new(Some(model_listener_thread));

        zmq_agent
    }

    /// Restarts the current instance of the ZMQ agent.
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

    /// Disables networking operations for ZMQ agent.
    pub async fn disable_agent(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.active.load(Ordering::SeqCst) {
            if let Ok(mut handle_guard) = self.model_listener_thread.lock() {
                if let Some(handle) = handle_guard.take() {
                    handle.thread().unpark();
                    handle.join().expect("Failed to join model listener thread");
                } else {
                    eprintln!("[RelayRLAgent - disable_agent] No model listener thread to join");
                }
            } else {
                eprintln!("[RelayRLAgent - disable_agent] Failed to lock model listener thread");
            }

            self.active.store(false, Ordering::SeqCst);
        } else {
            eprintln!("[RelayRLAgent - disable_agent] Agent is already inactive");
        }

        Ok(())
    }

    /// Re-enables networking operations for ZMQ agent.
    ///
    /// For initial instantiation, use init_agent()
    pub async fn enable_agent(
        &self,
        training_server_address: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.active.load(Ordering::SeqCst) {
            self.active.store(true, Ordering::SeqCst);
        } else {
            eprintln!("[RelayRLAgent - enable_agent] Agent is already active");
        }

        Ok(())
    }

    /// Returns the agent's current model version.
    pub fn get_model_version(&self) -> i64 {
        let version: i64 = self.local_version.load(Ordering::SeqCst);
        version
    }
}

/// Implementation of the RelayRLAgentZmqTrait for RelayRLAgentZmq.
///
/// This section contains the core logic for:
/// - Requesting actions using model inference.
/// - Recording and finalizing actions i2n the trajectory.
/// - Handling updated model retrieval via a background loop.
impl RelayRLAgentZmqTrait for RelayRLAgentZmq {
    fn initial_model_handshake(
        &mut self,
        model_arc: Arc<Mutex<Option<CModule>>>,
        agent_id: &[u8],
        agent_listening_server: &str,
        client_model_path: &PathBuf,
        training_server: &str,
    ) {
        // Initialize the model asynchronously using a ZMQ DEALER socket.
        let context: Context = Context::new();
        let socket: Socket = context
            .socket(zmq::DEALER)
            .expect("failed to create DEALER socket");

        // Set socket options: identity, send high-water mark, and maximum message size.
        socket
            .set_identity(agent_id)
            .expect("Socket failed to set identity");
        socket
            .set_sndhwm(100)
            .expect("Socket failed to set high-water mark");
        socket
            .set_maxmsgsize(-1)
            .expect("Socket failed to set max message size");
        socket
            .connect(agent_listening_server)
            .expect("Failed to connect socket");

        println!(
            "[RelayRLAgent - new] Waiting for initial model at {:?}",
            training_server
        );

        // An empty frame used to initiate the handshake.
        let empty_frame: &Vec<u8> = &vec![];

        let mut locked_model: MutexGuard<Option<CModule>> =
            model_arc.lock().expect("runtime model cannot be locked");
        if locked_model.is_some() {
            validate_model(
                locked_model
                    .as_ref()
                    .expect("runtime model cannot be locked"),
            );
            println!("[RelayRLAgent - new] Model already initialized");

            let model_set: &[u8; 9] = b"MODEL_SET";
            if let Err(e) = socket.send_multipart([empty_frame, model_set.as_ref()], 0) {
                eprintln!("[RelayRLAgent - new] Failed to send MODEL_SET: {}", e);
            }
        }

        // Loop until an initial model is successfully received and loaded.
        while locked_model.is_none() {
            println!("[RelayRLAgent - new] Requesting initial model...");

            let get_model: &[u8; 9] = b"GET_MODEL";
            if let Err(e) = socket.send_multipart([empty_frame, get_model.as_ref()], 0) {
                eprintln!("[RelayRLAgent - new] Failed to send GET_MODEL: {}", e);
            }

            match socket.recv_multipart(0) {
                Ok(message_parts) => {
                    if message_parts.len() > 2 {
                        eprintln!("[RelayRLAgent - new] Malformed response received");
                        continue;
                    }

                    // The second frame should contain the serialized model bytes.
                    let model_bytes: &Vec<u8> = &message_parts[1];
                    println!("[RelayRLAgent - new] Received the initial model");

                    // Write the received model bytes to a local file.
                    let mut file: File = std::fs::File::create(&client_model_path)
                        .expect("[RelayRLAgent - new] Failed to create initial model file");
                    file.write_all(model_bytes)
                        .expect("[RelayRLAgent - new] Failed to write to initial model file");

                    // Load the TorchScript model from the file.
                    let loaded_model: CModule = CModule::load(&client_model_path)
                        .expect("[RelayRLAgent - new] Failed to load initial model file");

                    // validate and update the shared model.
                    validate_model(&loaded_model);
                    *locked_model = Some(loaded_model);

                    // Notify the server that the model is set.
                    let model_set: &[u8; 9] = b"MODEL_SET";
                    if let Err(e) = socket.send_multipart([empty_frame, model_set.as_ref()], 0) {
                        eprintln!("[RelayRLAgent - new] Failed to send MODEL_SET: {}", e);
                    } else {
                        // Wait for confirmation reply from the training server.
                        match socket.recv_multipart(0) {
                            Ok(reply_parts) => {
                                if reply_parts.len() == 2 {
                                    let reply: String = String::from_utf8(reply_parts[1].clone())
                                        .expect("Failed to convert UTF-8 reply to string");
                                    if reply == "ID_LOGGED" {
                                        println!(
                                            "[RelayRLAgent - new] Received reply: (TrainingServer::ID_LOGGED)",
                                        );
                                        break; // Exit loop once handshake is confirmed.
                                    } else {
                                        eprintln!("[RelayRLAgent - new] Invalid reply: {}", reply);
                                        break; // Exit loop to avoid infinite looping.
                                    }
                                } else {
                                    eprintln!("[RelayRLAgent - new] Malformed reply");
                                }
                            }
                            Err(e) => {
                                eprintln!("[RelayRLAgent - new] Failed to receive reply: {}", e);
                            }
                        }
                    }

                    println!("[RelayRLAgent - new] Model updated");
                }
                Err(e) => {
                    eprintln!("[RelayRLAgent - new] Failed to receive model: {}", e);
                }
            }

            // Pause briefly before retrying.
            thread::sleep(Duration::from_secs(1));
        }
    }

    /// Requests an action by running inference on the current model.
    ///
    /// The function converts the observation and mask to tensors,
    /// calls the model's `step` method, extracts the resulting action and auxiliary data,
    /// and then creates an RelayRLAction which is appended to the trajectory.
    ///
    /// # Arguments
    /// * `obs` - The observation tensor.
    /// * `mask` - The mask tensor.
    /// * `reward` - The immediate reward.
    ///
    /// # Returns
    /// * `Result<Arc<RelayRLAction>, &str>` - On success, returns an Arc-wrapped RelayRLAction.
    ///   On failure, returns an error message.
    fn request_for_action(
        &mut self,
        obs: &Tensor,
        mask: &Tensor,
        reward: f32,
    ) -> Result<Arc<RelayRLAction>, &str> {
        {
            // Acquire a lock on the current model.
            let model_lock: MutexGuard<Option<CModule>> = self
                .model
                .lock()
                .map_err(|_| "[RelayRLAgent - request_for_action] Failed to lock model")?;
            // Ensure that the model is initialized.
            let model: &CModule = model_lock
                .as_ref()
                .ok_or("[RelayRLAgent - request_for_action] Model not initialized")?;

            // Prepare the inputs by converting the observation and mask to float type.
            let obs_ivalue = IValue::Tensor(obs.to_kind(Kind::Float));
            let mask_ivalue = IValue::Tensor(mask.to_kind(Kind::Float));
            let inputs: Vec<IValue> = vec![obs_ivalue, mask_ivalue];
            // Run inference in a no_grad context to avoid gradient computations.
            let (action, data): (Tensor, Option<HashMap<String, RelayRLData>>) = no_grad(|| {
                match model.method_is("step", &inputs) {
                    Ok(output_ivalue) => {
                        if let IValue::Tuple(ref outputs) = output_ivalue {
                            if outputs.len() == 2 {
                                // Extract the action tensor from the first element of the tuple.
                                let action: Tensor = match &outputs[0] {
                                    IValue::Tensor(tensor) => tensor.to_kind(Kind::Float),
                                    _ => Tensor::zeros([], (Kind::Uint8, Device::Cpu)),
                                };
                                // Convert the auxiliary output into a HashMap.
                                let data: Option<HashMap<String, RelayRLData>> = match &outputs[1] {
                                    GenericDict(dict) => Some(
                                        convert_generic_dict(dict)
                                            .expect("Failed to convert GenericDict"),
                                    ),
                                    _ => {
                                        eprintln!(
                                            "[RelayRLAgent - request_for_action] Failed to convert output[1] to GenericDict"
                                        );
                                        Some(HashMap::new())
                                    }
                                };

                                (action, data)
                            } else {
                                eprintln!(
                                    "[RelayRLAgent - request_for_action] Output length is less than 2"
                                );
                                (
                                    Tensor::zeros([], (Kind::Uint8, Device::Cpu)),
                                    Some(HashMap::new()),
                                )
                            }
                        } else {
                            eprintln!("[RelayRLAgent - request_for_action] Output is not a Tuple");
                            (
                                Tensor::zeros([], (Kind::Uint8, Device::Cpu)),
                                Some(HashMap::new()),
                            )
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[RelayRLAgent - request_for_action] Failed to call model.step: {}",
                            e
                        );
                        (
                            Tensor::zeros([], (Kind::Uint8, Device::Cpu)),
                            Some(HashMap::new()),
                        )
                    }
                }
            });

            // Create an RelayRLAction from the observed data, the resulting action tensor, and the mask.
            let r4sa: RelayRLAction = RelayRLAction::new(
                Some(
                    TensorData::try_from(obs).expect("Failed to convert obs Tensor to TensorData"),
                ),
                Some(
                    TensorData::try_from(&action)
                        .expect("Failed to convert act Tensor to TensorData"),
                ),
                Some(
                    TensorData::try_from(mask)
                        .expect("Failed to convert mask Tensor to TensorData"),
                ),
                reward,
                data,
                false,
                false,
            );

            // Append the newly created action to the current trajectory.
            self.current_traj.add_action(&r4sa, true);
        }

        // Wrap the last action in an Arc pointer and return it.
        let r4sa_arc: Option<Arc<RelayRLAction>> = Some(Arc::new(
            self.current_traj
                .actions
                .last()
                .expect("Failed to get last action")
                .clone(),
        ));

        match r4sa_arc {
            Some(r4sa) => Ok(r4sa),
            None => Err("[RelayRLAgent - request_for_action] Failed to create RelayRLAction"),
        }
    }

    /// Records an action into the agent's trajectory.
    ///
    /// Currently, this function is a placeholder and remains unimplemented.
    ///
    /// # Arguments
    /// * `obs` - The observation tensor.
    /// * `act` - The action tensor.
    /// * `mask` - The mask tensor.
    /// * `reward` - The reward received.
    /// * `data` - Optional auxiliary data.
    /// * `done` - Boolean flag indicating if the episode has ended.
    /// * `reward_update_flag` - Boolean flag indicating if the reward was updated.
    fn record_action(
        &mut self,
        obs: &Tensor,
        act: &Tensor,
        mask: &Tensor,
        reward: &f32,
        data: &Option<HashMap<String, RelayRLData>>,
        done: bool,
        reward_update_flag: bool,
    ) {
        todo!(); // Functionality to record an action is currently not implemented.
    }

    /// Flags the final action in the trajectory.
    ///
    /// This method creates a terminal RelayRLAction (with `done` set to true)
    /// to signal the end of an episode, and appends it to the trajectory.
    ///
    /// # Arguments
    /// * `reward` - The final reward for the episode.
    fn flag_last_action(&mut self, reward: f32) {
        // Create a terminal action with the specified reward.
        let last_action: RelayRLAction =
            RelayRLAction::new(None, None, None, reward, None, true, false);
        self.current_traj.add_action(&last_action, true);
    }

    /// Continuously polls for updated models from the training server.
    ///
    /// This method creates a ZMQ PULL socket bound to the training server address,
    /// then enters a loop that:
    /// - Waits for new model bytes,
    /// - Writes the received bytes to a file,
    /// - Loads the model from the file onto the CPU,
    /// - And updates the agent's shared model.
    ///
    /// # Arguments
    /// * `model` - Shared reference to the current model.
    /// * `training_server` - Address of the training server to bind for receiving models.
    /// * `client_model_path` - File path for saving and loading the model.
    fn _loop_for_updated_model(
        model: Arc<Mutex<Option<CModule>>>,
        training_server: String,
        client_model_path: PathBuf,
    ) {
        println!("[RelayRLAgent - loop_for_updated_model] Starting loop for updated model");

        let context: Context = Context::new();
        let socket: Socket = context
            .socket(zmq::PULL)
            .expect("Failed to create PULL socket");
        socket
            .bind(training_server.as_str())
            .expect("[RelayRLAgent - loop_for_updated_model] Failed to bind socket");

        // Set non-blocking mode (rcvtimeo set to 0) to poll continuously.
        socket
            .set_rcvtimeo(0)
            .expect("Socket failed to set non-blocking mode");

        loop {
            match socket.recv_bytes(0) {
                Ok(model_bytes) => {
                    println!("[RelayRLAgent - loop_for_updated_model] Receives the model");
                    // Write the received model bytes to the specified file.
                    let mut file = std::fs::File::create(&client_model_path).expect(
                        "[RelayRLAgent - loop_for_updated_model] Failed to create model file",
                    );
                    file.write_all(&model_bytes)
                        .expect("[RelayRLAgent - loop_for_updated_model] Failed to write model");

                    // Load the TorchScript model from the file onto the CPU.
                    let loaded_model: CModule = match CModule::load_on_device(
                        &client_model_path,
                        Device::Cpu,
                    ) {
                        Ok(model) => model,
                        Err(e) => {
                            panic!(
                                "[RelayRLAgent - loop_for_updated_model] Failed to load model: {}",
                                e
                            );
                        }
                    };

                    {
                        // validate and then load new model into memory
                        validate_model(&loaded_model);
                        let mut model_lock: MutexGuard<Option<CModule>> = model
                            .lock()
                            .expect("[RelayRLAgent - loop_for_updated_model] Failed to lock model");
                        *model_lock = Some(loaded_model);
                    }

                    println!("[RelayRLAgent - loop_for_updated_model] Model updated");
                }
                Err(e) => {
                    if e == zmq::Error::EAGAIN {
                        // If no message is available, continue polling.
                        continue;
                    } else {
                        eprintln!(
                            "[RelayRLAgent - loop_for_updated_model] Failed to receive model: {}",
                            e
                        );
                        continue;
                    }
                }
            }

            // Sleep briefly to avoid busy waiting.
            thread::sleep(Duration::from_millis(50));
        }
    }
}
