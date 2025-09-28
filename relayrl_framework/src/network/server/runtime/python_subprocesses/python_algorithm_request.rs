//! This module implements the Python Algorithm Request, which spawns and manages a Python subprocess
//! to handle commands from the Rust side (such as saving the model or receiving trajectories).
//! It uses asynchronous tasks to communicate with the Python process via its standard input/output.

use crate::types::trajectory::RL4SysTrajectory;
use serde::{Deserialize, Serialize};
use std::env;
use std::ops::Deref;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc::{self, Receiver as TokioReceiver, Sender as TokioSender};
use tokio::sync::{Notify, oneshot as tokio_oneshot};
use tokio::sync::{RwLock, RwLockReadGuard};
use tokio::task::spawn;

use crate::sys_utils::misc_utils::PythonResponse;

#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_trajectory::PyRL4SysTrajectory;

const PYTHON_ALGORITHM_REPLY_SCRIPT: &str = "src/native/python/python_algorithm_reply.py";

/// Commands that can be sent to the Python Algorithm Request.
///
/// - `SaveModel`: Instructs the Python subprocess to save the current model.
/// - `ReceiveTrajectory`: Sends a trajectory to the Python subprocess for processing.
#[derive(Debug)]
pub enum PythonAlgorithmCommand {
    SaveModel(tokio_oneshot::Sender<bool>),
    ReceiveTrajectory(tokio_oneshot::Sender<bool>, RL4SysTrajectory),
}

/// A request sent from Rust to the Python subprocess.
///
/// It contains a command string and an optional trajectory (wrapped in the Python representation).
#[derive(Serialize, Deserialize)]
struct PythonRequest {
    command: String,
    trajectory: Option<PyRL4SysTrajectory>,
}

/// The PythonAlgorithmRequest manages a Python subprocess that executes a command worker script.
///
/// It spawns the Python subprocess, sends commands to it via stdin, and reads responses from its stdout.
/// It also maintains status flags for the algorithm and tensorboard scripts.
#[derive(Debug)]
pub struct PythonAlgorithmRequest {
    algorithm_child: Arc<RwLock<Child>>,
    pub algorithm_pyscript_status: Arc<AtomicBool>,
    pub notify_algorithm_status: Arc<Notify>,
}

impl PythonAlgorithmRequest {
    /// Initializes the Python Algorithm Commander by spawning a Python subprocess.
    ///
    /// # Arguments
    /// * `receiver` - An mpsc receiver for receiving PythonCommand messages from Rust.
    /// * `tensorboard` - An atomic flag shared with the tensorboard subprocess status.
    /// * `args` - A vector of string arguments to pass to the Python script.
    ///
    /// # Returns
    /// A new instance of [PythonAlgorithmRequest] managing the spawned subprocess.
    pub async fn init_pyscript(
        receiver: TokioReceiver<PythonAlgorithmCommand>,
        args: Vec<String>,
    ) -> Self {
        println!("[PythonAlgorithmRequest - new] Initializing Python Algorithm Request...");

        // Construct the path to the Python command worker script.
        let python_script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(PYTHON_ALGORITHM_REPLY_SCRIPT);

        // Spawn the Python subprocess with unbuffered output, piping stdin and stdout.
        let algorithm_cmd = Command::new("python")
            .arg("-u")
            .arg(python_script_path)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Inherit stderr for debugging purposes.
            .spawn()
            .expect("[PythonAlgorithmRequest - init] Failed to spawn Python subprocess");

        let algorithm_child: Arc<RwLock<Child>> = Arc::new(RwLock::new(algorithm_cmd));
        let algorithm_child_clone: Arc<RwLock<Child>> = Arc::clone(&algorithm_child);

        // Initialize status flags and a notifier for algorithm pyscript status.
        let algorithm_pyscript_status: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let notify_algorithm_status: Arc<Notify> = Arc::new(Notify::new());

        let algorithm_status_clone: Arc<AtomicBool> = Arc::clone(&algorithm_pyscript_status);
        let notify_algorithm_status_clone: Arc<Notify> = Arc::clone(&notify_algorithm_status);

        // Spawn an asynchronous task to handle incoming commands from Rust and read from Python's stdout.
        tokio::spawn(async move {
            PythonAlgorithmRequest::handle_commands(
                algorithm_child_clone,
                receiver,
                notify_algorithm_status_clone,
                algorithm_status_clone,
            )
            .await;
        });

        PythonAlgorithmRequest {
            algorithm_child,
            algorithm_pyscript_status,
            notify_algorithm_status,
        }
    }

    /// Creates and returns a mpsc sender for sending PythonCommand messages.
    ///
    /// This function creates a mpsc channel with a capacity of 100000 messages and returns the sender.
    async fn create_sender() -> TokioSender<PythonAlgorithmCommand> {
        let (tx, _rx) = mpsc::channel(100000);
        tx
    }

    /// Handles incoming commands from Rust and reads responses from the Python subprocess.
    ///
    /// This function performs the following:
    /// - Takes control of the Python subprocess's stdin and stdout.
    /// - Spawns a task to continuously read lines from stdout, attempting to parse JSON responses.
    /// - In the main loop, listens for incoming PythonCommand messages, writes the corresponding JSON
    ///   request to the subprocess, and waits for a response.
    ///
    /// # Arguments
    /// * `child` - An Arc-wrapped Mutex protecting the Python subprocess.
    /// * `receiver` - The mpsc receiver for PythonCommand messages.
    /// * `notify_algorithm_status` - A Notify object used to signal status changes.
    /// * `algorithm_pyscript_status` - An atomic flag storing the algorithm pyscript's status.
    async fn handle_commands(
        child: Arc<RwLock<Child>>,
        mut receiver: TokioReceiver<PythonAlgorithmCommand>,
        notify_algorithm_status: Arc<Notify>,
        algorithm_pyscript_status: Arc<AtomicBool>,
    ) {
        // Take ownership of the subprocess's stdin.
        let mut stdin: ChildStdin = child
            .write()
            .await
            .stdin
            .take()
            .expect("Failed to open stdin");

        // Take ownership of the subprocess's stdout.
        let stdout: ChildStdout = child
            .write()
            .await
            .stdout
            .take()
            .expect("Failed to open stdout");

        // Create an mpsc channel to forward JSON responses received from the subprocess.
        let (response_tx, mut response_rx) = mpsc::channel::<PythonResponse>(100000);
        let mut reader: Lines<BufReader<ChildStdout>> = BufReader::new(stdout).lines();

        // Spawn a task to read lines from the Python subprocess's stdout in real-time.
        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                let trimmed: &str = line.trim();

                // Attempt to parse the line as a JSON response.
                match serde_json::from_str::<PythonResponse>(trimmed) {
                    Ok(response) => {
                        // Successfully parsed JSON; forward it via the mpsc channel.
                        let _ = response_tx.send(response).await;
                    }
                    Err(_) => {
                        // Not a valid JSON response; print the output to stdout.
                        if !trimmed.contains("[algorithm_pyscript_") {
                            println!("{}", trimmed);
                        }

                        // Update algorithm pyscript status based on specific markers in the output.
                        if trimmed.contains("[algorithm_pyscript_true]") {
                            algorithm_pyscript_status.store(true, Ordering::Relaxed);
                            notify_algorithm_status.notify_waiters();
                        } else if trimmed.contains("[algorithm_pyscript_false]") {
                            algorithm_pyscript_status.store(false, Ordering::Relaxed);
                            notify_algorithm_status.notify_waiters();
                        }
                    }
                }
            }
        });

        // Main loop to process commands from Rust and send them to Python.
        while let Some(command) = receiver.recv().await {
            match command {
                PythonAlgorithmCommand::SaveModel(oneshot_tx) => {
                    let request = PythonRequest {
                        command: "save_model".to_string(),
                        trajectory: None,
                    };
                    let request_json =
                        serde_json::to_string(&request).expect("Failed to serialize request");

                    // Send the JSON request to Python's stdin.
                    if let Err(e) = stdin.write_all(request_json.as_bytes()).await {
                        eprintln!("[SaveModel] Failed to write to stdin: {}", e);
                        let _ = oneshot_tx.send(false);
                        continue;
                    }
                    if let Err(e) = stdin.write_all(b"\n").await {
                        eprintln!("[SaveModel] Failed to write newline to stdin: {}", e);
                        let _ = oneshot_tx.send(false);
                        continue;
                    }

                    // Await a response from the Python subprocess.
                    if let Some(response) = response_rx.recv().await {
                        if response.status == "success" {
                            let _ = oneshot_tx.send(true);
                        } else {
                            eprintln!("[SaveModel] Python responded with error: {:?}", response);
                            let _ = oneshot_tx.send(false);
                        }
                    } else {
                        eprintln!("[SaveModel] No response from Python");
                        let _ = oneshot_tx.send(false);
                    }
                }

                PythonAlgorithmCommand::ReceiveTrajectory(oneshot_tx, trajectory) => {
                    let request = PythonRequest {
                        command: "receive_trajectory".to_string(),
                        trajectory: Some(trajectory.into_py()),
                    };
                    let request_json: String =
                        serde_json::to_string(&request).expect("Failed to serialize request");

                    // Write the trajectory request to Python's stdin.
                    if let Err(e) = stdin.write_all(request_json.as_bytes()).await {
                        eprintln!("[ReceiveTrajectory] Failed to write to stdin: {}", e);
                        let _ = oneshot_tx.send(false);
                        continue;
                    }
                    if let Err(e) = stdin.write_all(b"\n").await {
                        eprintln!("[ReceiveTrajectory] Failed to write newline: {}", e);
                        let _ = oneshot_tx.send(false);
                        continue;
                    }

                    // Await a response from the Python subprocess.
                    if let Some(response) = response_rx.recv().await {
                        if response.status == "success" {
                            let _ = oneshot_tx.send(true);
                        } else {
                            let _ = oneshot_tx.send(false);
                        }
                    } else {
                        eprintln!("[ReceiveTrajectory] No response from Python");
                        let _ = oneshot_tx.send(false);
                    }
                }
            }
        }
    }
}

/// Drop implementation for PythonAlgorithmRequest for graceful program termination
impl Drop for PythonAlgorithmRequest {
    fn drop(&mut self) {
        let algorithm_child_clone = Arc::clone(&self.algorithm_child);
        let handle = spawn(async move {
            let mut algorithm_child_write = algorithm_child_clone.write().await;
            let kill_result = algorithm_child_write.kill().await;
            match kill_result {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[PythonAlgorithmRequest - drop] Failed to kill algorithm subprocess: {}",
                        e
                    );
                }
            };
        });
        drop(handle);
    }
}
