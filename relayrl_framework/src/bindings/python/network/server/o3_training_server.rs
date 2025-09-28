//! This module defines a Python-exposed wrapper (`PyTrainingServer`) around the RelayRL `TrainingServer` struct.
//! It allows Python code to instantiate and manage the training server for RL tasks using either
//! ZMQ or gRPC. Additionally, it supports custom hyperparameters, environment directories, and
//! toggling TensorBoard functionality.

use crate::network::server::training_server_wrapper::{Hyperparams, TrainingServer};
use crate::sys_utils::tokio_utils::get_or_init_tokio_runtime;
use pyo3::prelude::*;
use pyo3::{PyResult, pyclass, pymethods};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime as TokioRuntime;
use tokio::sync::{RwLock as TokioRwLock, RwLockWriteGuard};

#[cfg(feature = "console-subscriber")]
use crate::sys_utils::tokio::utils::get_or_init_console_subscriber;

/// A Python-friendly wrapper for parsing `Hyperparams` from Python objects.
/// This struct supports either a dictionary of string-to-string parameters
/// or a list of string arguments.
impl<'source> FromPyObject<'source> for Hyperparams {
    /// Attempts to convert a Python object into `Hyperparams`.
    ///
    /// Accepts:
    /// - A dictionary (`HashMap<String, String>`) for key-value parameters.
    /// - A list of strings (`Vec<String>`) for argument-based hyperparameters.
    ///
    /// Returns a `TypeError` if neither conversion succeeds.
    fn extract_bound(obj: &Bound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = obj.extract::<HashMap<String, String>>() {
            Ok(Hyperparams::Map(dict))
        } else if let Ok(list) = obj.extract::<Vec<String>>() {
            Ok(Hyperparams::Args(list))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a dict or list",
            ))
        }
    }
}

/// A Python-exposed struct wrapping the `TrainingServer` in an async-friendly manner.
/// It uses a Tokio runtime internally to spawn tasks and interact with the server asynchronously.
#[pyclass(name = "TrainingServer")]
#[derive(Clone)]
pub struct PyTrainingServer {
    /// The underlying `TrainingServer`, protected by an `Arc<TokioRwLock<...>>` for concurrency.
    inner: Arc<TokioRwLock<TrainingServer>>,
    /// The dedicated Tokio runtime for blocking on async calls from Python.
    tokio_runtime: Arc<TokioRuntime>,
}

#[pymethods]
impl PyTrainingServer {
    /// Creates a new Python-facing `TrainingServer` instance.
    ///
    /// # Arguments
    /// * `algorithm_name` - The name of the RL algorithm (DQN, PPO, etc.).
    /// * `obs_dim` - The dimension of the observation space.
    /// * `act_dim` - The dimension of the action space.
    /// * `buf_size` - The buffer size for replay or memory usage.
    /// * `tensorboard` - Boolean indicating whether to enable TensorBoard logging.
    /// * `multiactor` - Boolean indicating support for multi-actor setups.
    /// * `env_dir` - The directory for environment files or assets.
    /// * `algorithm_dir` - Directory containing the algorithm scripts or references.
    /// * `config_path` - The JSON config file path for RelayRL.
    /// * `hyperparams` - Optional hyperparameters (dict or list) for advanced customization.
    /// * `server_type` - Either "zmq" or "grpc", specifying which server approach to use.
    /// * `training_prefix`, `training_host`, `training_port` - Additional commands
    ///  for manually configuring the training server address.
    ///
    /// Internally, this method:
    /// 1. Enables `TOKIO_DEBUG` for better async debugging.
    /// 2. Creates a new Tokio runtime.
    /// 3. Invokes `TrainingServer::new(...)` within that runtime to build a server instance.
    /// 4. Wraps the result in a `PyTrainingServer`.
    #[new]
    #[pyo3(signature = (
        algorithm_name,
        obs_dim,
        act_dim,
        buf_size,
        tensorboard = false,
        multiactor = false,
        env_dir = "./env",
        algorithm_dir = None,
        config_path = "./config.json",
        hyperparams = None,
        server_type = "zmq",
        training_prefix = None,
        training_host = None,
        training_port = None,
    ))]
    pub fn new(
        algorithm_name: String,
        obs_dim: i32,
        act_dim: i32,
        buf_size: i32,
        tensorboard: bool,
        multiactor: bool,
        env_dir: Option<&str>,
        algorithm_dir: Option<String>,
        config_path: Option<&str>,
        hyperparams: Option<Hyperparams>,
        server_type: Option<&str>,
        training_prefix: Option<String>,
        training_host: Option<String>,
        training_port: Option<String>,
    ) -> PyResult<Self> {
        // Enable extra logging for Tokio-based debugging
        unsafe {
            std::env::set_var("TOKIO_DEBUG", "1");
        }

        #[cfg(feature = "console-subscriber")]
        get_or_init_console_subscriber();

        // Create a new Tokio runtime for async tasks
        let tokio_runtime: Arc<TokioRuntime> = get_or_init_tokio_runtime();

        // Convert the optional &str to owned Strings
        let env_dir: Option<String> = Some(env_dir.expect("env_dir is None").to_string());

        let config_path: Option<PathBuf> = config_path.map(|s| PathBuf::from(s));

        // Block on the async creation of the training server
        let training_server_arc: Arc<TokioRwLock<TrainingServer>> =
            tokio_runtime.block_on(TrainingServer::new(
                algorithm_name,
                obs_dim,
                act_dim,
                buf_size,
                tensorboard,
                multiactor,
                Some(env_dir.expect("env_dir is None").to_string()),
                algorithm_dir,
                config_path,
                hyperparams,
                Some(server_type.expect("server_type is None").to_string()),
                training_prefix,
                training_host,
                training_port,
            ));

        // Wrap and return as a PyTrainingServer
        Ok(PyTrainingServer {
            inner: training_server_arc,
            tokio_runtime,
        })
    }

    #[pyo3(signature = (training_server_address = None))]
    fn restart_server(
        &self,
        py: Python,
        training_server_address: Option<String>,
    ) -> PyResult<Vec<String>> {
        let mut server = self.tokio_runtime.block_on(self.inner.write());

        let mut result: PyResult<Vec<String>> = Ok(vec![]);
        py.allow_threads(|| {
            if server.ts_zmq.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(server
                        .ts_zmq
                        .as_mut()
                        .expect("[PyO3] ZMQ server not initialized")
                        .write()
                        .await
                        .restart_server(training_server_address)
                        .await
                        .into_iter()
                        .map(|result| match result {
                            Ok(_) => "OK".to_string(),
                            Err(e) => format!("Error: {}", e),
                        })
                        .collect());
                })
            } else if server.ts_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(server
                        .ts_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC server not initialized")
                        .write()
                        .await
                        .restart_server(training_server_address)
                        .await
                        .into_iter()
                        .map(|result| match result {
                            Ok(_) => "OK".to_string(),
                            Err(e) => format!("Error: {}", e),
                        })
                        .collect());
                })
            }
        });
        result
    }

    #[pyo3(signature = ())]
    fn disable_server(&self, py: Python) -> PyResult<()> {
        let mut server = self.tokio_runtime.block_on(self.inner.write());

        let mut result: PyResult<()> = Ok(());
        py.allow_threads(|| {
            if server.ts_zmq.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(server
                        .ts_zmq
                        .as_mut()
                        .expect("[PyO3] ZMQ server not initialized")
                        .write()
                        .await
                        .disable_server()
                        .await
                        .expect("[PyO3] ZMQ server not initialized"));
                })
            } else if server.ts_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(server
                        .ts_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC server not initialized")
                        .write()
                        .await
                        .disable_server()
                        .await
                        .expect("[PyO3] gRPC server not initialized"));
                })
            }
        });
        result
    }

    #[pyo3(signature = (training_server_address = None))]
    fn enable_server(&self, py: Python, training_server_address: Option<String>) -> PyResult<()> {
        let mut server_read = self.tokio_runtime.block_on(self.inner.write());

        let mut result: PyResult<()> = Ok(());
        py.allow_threads(|| {
            if server_read.ts_zmq.is_some() {
                self.tokio_runtime.block_on(async {
                    let mut ts_zmq_write = server_read
                        .ts_zmq
                        .as_mut()
                        .expect("[PyO3] ZMQ server not initialized")
                        .write()
                        .await;
                    result = Ok(ts_zmq_write
                        .enable_server(training_server_address)
                        .await
                        .expect("[PyO3] ZMQ server not initialized"));
                })
            } else if server_read.ts_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    let mut ts_grpc_write = server_read
                        .ts_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC server not initialized")
                        .write()
                        .await;
                    result = Ok(ts_grpc_write
                        .enable_server(training_server_address)
                        .await
                        .expect("[PyO3] gRPC server not initialized"));
                })
            }
        });
        result
    }
}
