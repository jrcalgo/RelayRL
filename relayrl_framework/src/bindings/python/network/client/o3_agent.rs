//! This module exposes a Python class `RelayRLAgent` that wraps the underlying Rust `RelayRLAgent` struct,
//! allowing Python code to interact with either a gRPC-based or ZMQ-based RL agent. It uses PyO3
//! for interop, and a Tokio runtime for handling asynchronous operations where necessary.

use crate::bindings::python::o3_action::PyRelayRLAction;
use crate::types::action::{RelayRLAction, TensorData};
use crate::network::client::agent_grpc::RelayRLAgentGrpcTrait;
use crate::network::client::agent_wrapper::RelayRLAgent;
use crate::network::client::agent_zmq::RelayRLAgentZmqTrait;
use crate::sys_utils::tokio_utils::get_or_init_tokio_runtime;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, pyclass, pyfunction, pymethods};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::Arc;
use tch::{CModule, Tensor};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::sync::{RwLock as TokioRwLock, RwLock, RwLockWriteGuard};

#[cfg(feature = "console-subscriber")]
use crate::sys_utils::tokio::utils::get_or_init_console_subscriber;

/// A Python-facing wrapper around RelayRLAgent.
///
/// This struct holds an async Tokio runtime and a reference-counted, thread-safe lock (`RwLock`)
/// to the Rust `RelayRLAgent`. It unifies both gRPC and ZMQ agent protocols behind a single API.
#[pyclass(name = "RelayRLAgent")]
#[derive(Clone)]
pub struct PyRelayRLAgent {
    /// A thread-safe reference to the core RelayRLAgent implementation, wrapped in a
    /// tokio::sync::RwLock for concurrent async usage.
    inner: Arc<TokioRwLock<RelayRLAgent>>,
    /// A dedicated Tokio runtime to handle async tasks (like gRPC calls) from synchronous Python code.
    tokio_runtime: Arc<TokioRuntime>,
}

#[pymethods]
impl PyRelayRLAgent {
    /// Creates a new `PyRelayRLAgent` from optional arguments specifying model path, config,
    /// server type (zmq or grpc), and ZeroMQ details (prefix, host, port).
    ///
    /// # Arguments
    /// * `model_path` - Optional path to a TorchScript model file (.pt). If provided, it is loaded into a CModule.
    /// * `config_path` - Optional path to the RelayRL configuration JSON.
    /// * `server_type` - "zmq" (default) or "grpc".
    /// * `training_port` - Optional port for the training server.
    /// * 'training_prefix' - Optional prefix for the training server.
    /// * 'training_host' - Optional host for the training server.
    #[new]
    #[pyo3(signature = (
        model_path = None,
        config_path = "./config.json",
        server_type = "zmq",
        training_port = None,
        training_prefix = None,
        training_host = None,
    ))]
    fn new(
        py: Python,
        model_path: Option<String>,
        config_path: Option<&str>,
        server_type: Option<&str>,
        training_port: Option<String>,
        training_prefix: Option<String>,
        training_host: Option<String>,
    ) -> PyResult<Self> {
        #[cfg(feature = "console-subscriber")]
        get_or_init_console_subscriber();

        // Create a Tokio runtime for asynchronous tasks
        let tokio_runtime: Arc<TokioRuntime> = get_or_init_tokio_runtime();

        // Load the TorchScript model if a path is provided
        let model: Option<CModule> = if let Some(path) = model_path {
            Some(CModule::load(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "[PyO3] Failed to load model: {}",
                    e
                ))
            })?)
        } else {
            None
        };

        let config_path: Option<PathBuf> = config_path.map(|s| PathBuf::from(s));

        // Block on the creation of the RelayRLAgent in an async context
        let agent_arc: RelayRLAgent = py.allow_threads(|| {
            tokio_runtime.block_on(RelayRLAgent::new(
                model,
                config_path,
                Some(server_type.expect("server_type is required").to_string()),
                training_port,
                training_prefix,
                training_host,
            ))
        });

        Ok(PyRelayRLAgent {
            inner: Arc::new(TokioRwLock::new(agent_arc)),
            tokio_runtime,
        })
    }

    /// Requests an action from the agent by passing in observation, mask, and reward from Python.
    ///
    /// - Converts Python's NumPy arrays / PyTorch tensors into tch::Tensor (serialized as TensorData).
    /// - If the agent is gRPC-based, it awaits the async `request_for_action` call.
    /// - If the agent is ZMQ-based, it calls the synchronous `request_for_action`.
    /// - Returns a Python-wrapped `PyRelayRLAction`.
    ///
    /// # Arguments
    /// * `obs` - A Python object representing the observation (either PyTorch tensor or NumPy array).
    /// * `mask` - A Python object representing the mask tensor.
    /// * `reward` - A float representing the immediate reward.
    #[pyo3(signature = (obs, mask, reward))]
    fn request_for_action(
        &self,
        py: Python,
        obs: &Bound<'_, PyAny>,
        mask: &Bound<'_, PyAny>,
        reward: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyRelayRLAction>> {
        let obs_tensor_data: TensorData = PyRelayRLAction::ndarray_to_tensor_data(Some(obs))
            .expect("[PyO3] Failed to convert obs");
        let mask_tensor_data: TensorData = PyRelayRLAction::ndarray_to_tensor_data(Some(mask))
            .expect("[PyO3] Failed to convert mask");
        let obs_tensor: Option<Tensor> = Tensor::try_from(obs_tensor_data).ok();
        let mask_tensor: Option<Tensor> = Tensor::try_from(mask_tensor_data).ok();
        let reward_value: f32 = reward.extract::<f32>()?;

        let action_arc_option: Option<Arc<RelayRLAction>> = {
            let mut agent_guard: RwLockWriteGuard<RelayRLAgent> = self
                .tokio_runtime
                .block_on(async { self.inner.write().await });

            if let Some(grpc_agent) = &mut agent_guard.agent_grpc {
                // Synchronously wait for the result
                match self.tokio_runtime.block_on(grpc_agent.request_for_action(
                    obs_tensor.expect("[PyO3] Failed to convert obs"),
                    mask_tensor.expect("[PyO3] Failed to convert mask"),
                    reward_value,
                )) {
                    Ok(action_arc) => Some(action_arc),
                    Err(e) => {
                        eprintln!("[PyO3] gRPC agent request_for_action error: {:?}", e);
                        None
                    }
                }
            } else if let Some(zmq_agent) = &mut agent_guard.agent_zmq {
                // ZMQ agent is synchronous
                match zmq_agent.request_for_action(
                    &obs_tensor.expect("[PyO3] Failed to convert obs"),
                    &mask_tensor.expect("[PyO3] Failed to convert mask"),
                    reward_value,
                ) {
                    Ok(action_arc) => Some(action_arc),
                    Err(e) => {
                        eprintln!("[PyO3] ZMQ agent request_for_action error: {:?}", e);
                        None
                    }
                }
            } else {
                // No agent was initialized
                eprintln!("No agent initialized");
                None
            }
        };

        // 3. Convert the resulting Arc<RelayRLAction> into a PyRelayRLAction
        match action_arc_option {
            Some(action_arc) => Ok(Py::new(
                py,
                PyRelayRLAction {
                    inner: (*action_arc).clone(),
                },
            )?),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "[PyO3] No action returned",
            )),
        }
    }

    /// Flags the last action in the agent with a final reward, indicating episode termination.
    ///
    /// If using ZMQ, it's a synchronous call. If using gRPC, it awaits the async call. This
    /// method finalizes the trajectory, potentially sending it to the training server.
    ///
    /// # Arguments
    /// * `reward` - Final reward for the last action (defaults to 0.0 if not specified).
    #[pyo3(signature = (reward = 0.0))]
    fn flag_last_action(&self, py: Python, reward: f32) {
        // Acquire a write guard to ensure exclusive access to the agent
        let mut agent: RwLockWriteGuard<RelayRLAgent> = self
            .tokio_runtime
            .block_on(async { self.inner.write().await });

        // Since ZMQ or gRPC calls could block, allow Python to release the GIL
        py.allow_threads(|| {
            if agent.agent_zmq.is_some() {
                agent
                    .agent_zmq
                    .as_mut()
                    .expect("[PyO3] ZMQ agent not initialized")
                    .flag_last_action(reward);
            } else if agent.agent_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    agent
                        .agent_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC agent not initialized")
                        .flag_last_action(reward)
                        .await
                });
            }
        });
    }

    #[pyo3(signature = (training_server_address = None))]
    fn restart_agent(
        &mut self,
        py: Python<'_>,
        training_server_address: Option<String>,
    ) -> PyResult<bool> {
        let mut agent: RwLockWriteGuard<RelayRLAgent> = self
            .tokio_runtime
            .block_on(async { self.inner.write().await });

        let mut result: PyResult<bool> = Ok(false);
        py.allow_threads(|| {
            if agent.agent_zmq.is_some() {
                self.tokio_runtime.block_on(async {
                    let result_vec = agent
                        .agent_zmq
                        .as_mut()
                        .expect("[PyO3] ZMQ agent not initialized")
                        .restart_agent(training_server_address)
                        .await;
                    let restart_result = result_vec.iter().map(|r| !r.is_err()).all(|x| x);
                    result = Ok(restart_result);
                });
            } else if agent.agent_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    let result_vec = agent
                        .agent_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC agent not initialized")
                        .restart_agent(training_server_address)
                        .await;
                    let restart_result = result_vec.iter().map(|r| !r.is_err()).all(|x| x);
                    result = Ok(restart_result);
                });
            } else {
                eprintln!("[PyO3] No agent initialized");
            }
        });
        result
    }

    #[pyo3(signature = ())]
    fn disable_agent(&self, py: Python) -> PyResult<()> {
        let mut agent: RwLockWriteGuard<RelayRLAgent> = self
            .tokio_runtime
            .block_on(async { self.inner.write().await });

        let mut result: PyResult<()> = Ok(());

        py.allow_threads(|| {
            if agent.agent_zmq.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(agent
                        .agent_zmq
                        .as_mut()
                        .expect("[PyO3] ZMQ agent not initialized")
                        .disable_agent()
                        .await
                        .expect("[PyO3] Failed to disable ZMQ agent"));
                });
            } else if agent.agent_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(agent
                        .agent_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC agent not initialized")
                        .disable_agent()
                        .await
                        .expect("[PyO3] Failed to disable gRPC agent"));
                });
            } else {
                eprintln!("[PyO3] No agent initialized");
            }
        });
        result
    }

    #[pyo3(signature = (training_server_address = None))]
    fn enable_agent(&self, py: Python, training_server_address: Option<String>) -> PyResult<()> {
        let mut agent: RwLockWriteGuard<RelayRLAgent> = self
            .tokio_runtime
            .block_on(async { self.inner.write().await });

        let mut result: PyResult<()> = Ok(());
        py.allow_threads(|| {
            if agent.agent_zmq.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(agent
                        .agent_zmq
                        .as_mut()
                        .expect("[PyO3] ZMQ agent not initialized")
                        .enable_agent(training_server_address)
                        .await
                        .expect("[PyO3] Failed to enable ZMQ agent"));
                });
            } else if agent.agent_grpc.is_some() {
                self.tokio_runtime.block_on(async {
                    result = Ok(agent
                        .agent_grpc
                        .as_mut()
                        .expect("[PyO3] gRPC agent not initialized")
                        .enable_agent(training_server_address)
                        .await
                        .expect("[PyO3] Failed to enable gRPC agent"));
                });
            } else {
                eprintln!("[PyO3] No agent initialized");
            }
        });
        result
    }
}
