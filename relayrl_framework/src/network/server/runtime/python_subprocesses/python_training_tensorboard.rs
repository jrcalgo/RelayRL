use crate::sys_utils::misc_utils::PythonResponse;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout, Command};
use tokio::sync::{RwLock as TokioRwLock, mpsc};
use tokio::task::spawn;

const PYTHON_TRAINING_TENSORBOARD_SCRIPT: &str = "src/native/python/training_tensorboard.py";

#[derive(Debug)]
pub struct PythonTrainingTensorboard {
    tensorboard_child: Arc<TokioRwLock<Child>>,
    pub tensorboard_pyscript_status: Arc<AtomicBool>,
}

impl PythonTrainingTensorboard {
    pub(crate) fn init_pyscript(args: Vec<String>) -> Self {
        let python_script_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(PYTHON_TRAINING_TENSORBOARD_SCRIPT);

        let tensorboard_cmd = Command::new("python")
            .arg("-u")
            .arg(python_script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let (tensorboard_child, tensorboard_pyscript_status) = match tensorboard_cmd {
            Ok(child) => {
                let child = Arc::new(TokioRwLock::new(child));
                let tensorboard_status = Arc::new(AtomicBool::new(true));
                (child, tensorboard_status)
            }
            Err(e) => {
                eprintln!("Failed to start TensorBoard process: {}", e);
                panic!("Failed to start TensorBoard process");
            }
        };

        let tensorboard_child_clone = Arc::clone(&tensorboard_child);

        tokio::spawn(async move {
            PythonTrainingTensorboard::pipe_output(tensorboard_child_clone).await;
        });

        PythonTrainingTensorboard {
            tensorboard_child,
            tensorboard_pyscript_status,
        }
    }

    async fn pipe_output(child: Arc<TokioRwLock<Child>>) {
        let mut child_write = child.write().await;
        let stdout = child_write.stdout.take().expect("Failed to open stdout");

        let mut reader: Lines<BufReader<ChildStdout>> = BufReader::new(stdout).lines();

        // Spawn tasks to read from stdout and stderr
        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                let trimmed: &str = line.trim();

                match serde_json::from_str::<PythonResponse>(trimmed) {
                    Ok(_) => {
                        // Handle the response
                        println!("{}", trimmed);
                    }
                    Err(e) => {
                        eprintln!(
                            "[Python Subprocess - pipe_output] Failed to parse TensorBoard response: {}",
                            e
                        );
                    }
                }
            }
        });
    }
}

/// Drop implementation for PythonTrainingTensorboard for graceful program termination
impl Drop for PythonTrainingTensorboard {
    fn drop(&mut self) {
        let tensorboard_child_clone = Arc::clone(&self.tensorboard_child);
        let handle = spawn(async move {
            let mut tensorboard_child_write = tensorboard_child_clone.write().await;
            let kill_result = tensorboard_child_write.kill().await;
            match kill_result {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[Python Subprocess - drop] Failed to kill TensorBoard subprocess: {}",
                        e
                    );
                }
            };
        });
        drop(handle);
    }
}
