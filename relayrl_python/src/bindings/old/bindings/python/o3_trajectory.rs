//! This module defines a Python wrapper for the RelayRLTrajectory struct, allowing Python code
//! to create and manage trajectories of actions in the RelayRL framework. It provides methods
//! for adding actions, converting trajectories to JSON, and reconstructing them from JSON.

use crate::bindings::python::o3_action::PyRelayRLAction;
use crate::types::action::RelayRLAction;
use crate::types::trajectory;
use crate::types::trajectory::RelayRLTrajectoryTrait;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{Bound, PyAny, pyclass, pymethods};
use serde::{Deserialize, Serialize};

/// A Python-facing wrapper around the RelayRLTrajectory struct.
///
/// This wrapper allows Python code to manipulate a list of RelayRLAction objects (wrapped as
/// PyRelayRLAction) in a trajectory format. Users can add actions, retrieve them, convert the
/// entire trajectory to JSON, or rebuild it from JSON data.
#[pyclass(name = "RelayRLTrajectory")]
#[derive(Serialize, Deserialize, Debug)]
pub struct PyRelayRLTrajectory {
    /// The internal Rust RelayRLTrajectory object, storing a list of RelayRLAction instances
    /// as well as optional server configuration (trajectory_server) and maximum length.
    pub inner: trajectory::RelayRLTrajectory,
}

#[pymethods]
impl PyRelayRLTrajectory {
    /// Constructs a new PyRelayRLTrajectory with a given maximum length and an optional trajectory server address.
    ///
    /// # Arguments
    /// * `max_length` - The maximum number of actions this trajectory can hold (defaults to 1000).
    /// * `trajectory_server` - The address of the trajectory server as a string (defaults to tcp://127.0.0.1:5556).
    #[new]
    #[pyo3(signature = (max_length = 1000, trajectory_server = "tcp://127.0.0.1:5556"))]
    fn new(max_length: Option<u32>, trajectory_server: Option<&str>) -> Self {
        // Convert the optional server address into an owned String if provided
        let traj_server = trajectory_server.map(|server| server.to_string());
        PyRelayRLTrajectory {
            inner: trajectory::RelayRLTrajectory::new(max_length, traj_server),
        }
    }

    /// Retrieves the list of actions from the RelayRLTrajectory as a vector of PyRelayRLAction objects.
    ///
    /// # Returns
    /// A vector of PyRelayRLAction objects corresponding to each RelayRLAction in the trajectory.
    #[pyo3(signature = ())]
    fn get_actions(&self) -> Vec<PyRelayRLAction> {
        let mut py_actions: Vec<PyRelayRLAction> = Vec::new();
        let actions: &Vec<RelayRLAction> = &self.inner.actions;

        // Convert each RelayRLAction to its Python wrapper variant
        for action in actions {
            py_actions.push((*action).clone().into_py());
        }

        py_actions
    }

    /// Adds an action (PyRelayRLAction) to the trajectory, optionally sending it if the
    /// action is marked as done.
    ///
    /// # Arguments
    /// * `action` - The action to add (in Python-wrapped form).
    #[pyo3(signature = (action))]
    fn add_action(&mut self, action: &PyRelayRLAction) {
        self.inner.add_action(&action.inner, true);
    }

    /// Serializes the trajectory (including all actions) into a JSON string.
    ///
    /// # Returns
    /// A JSON representation of the PyRelayRLTrajectory object, including all fields and actions.
    #[pyo3(signature = ())]
    fn to_json(&self) -> String {
        serde_json::to_string(&self).expect("Failed to serialize trajectory to JSON")
    }

    /// Constructs a PyRelayRLTrajectory from a Python dictionary containing serialized fields.
    ///
    /// This function expects the Python dictionary to match the JSON structure of a
    /// serialized RelayRLTrajectory, which includes fields such as `trajectory_server`,
    /// `max_length`, and an `actions` list. Each action is similarly structured with serialized
    /// tensor data if applicable.
    ///
    /// # Arguments
    /// * `trajectory_dict` - A Python dictionary describing the trajectory fields.
    ///
    /// # Returns
    /// A newly constructed PyRelayRLTrajectory.
    ///
    /// Example of the required dictionary structure:
    /// ```
    /// {
    ///     "inner": {
    ///         "trajectory_server": "tcp://127.0.0.1:5556",
    ///         "max_length": 1000,
    ///         "actions": [
    ///             {
    ///                 "obs": { "shape": [3], "dtype": "Float", "data": [1.0, 2.0, 3.0] },
    ///                 "act": { "shape": [3], "dtype": "Float", "data": [1.0, 2.0, 3.0] },
    ///                 "mask": { "shape": [3], "dtype": "Float", "data": [1.0, 2.0, 3.0] },
    ///                 "rew": 0.0,
    ///                 "data": {},
    ///                 "done": false
    ///             },
    ///             ...
    ///         ]
    ///     }
    /// }
    /// ```
    #[staticmethod]
    #[pyo3(signature = (trajectory_dict))]
    fn traj_from_json(trajectory_dict: &Bound<'_, PyDict>) -> PyRelayRLTrajectory {
        // Extract the `inner` field from the dictionary, which holds the RelayRLTrajectory data
        let inner_any: Bound<PyAny> = trajectory_dict
            .get_item("inner")
            .expect("Missing 'inner' field")
            .expect("Missing 'inner' field");
        let inner_dict: &Bound<PyDict> = inner_any
            .downcast::<PyDict>()
            .expect("Expected 'inner' to be a dictionary");

        // Extract `max_length` or default to 1000
        let max_length: u32 = match inner_dict.get_item("max_length") {
            Ok(Some(val)) => val.extract::<u32>().expect("Failed to extract max_length"),
            _ => 1000,
        };

        // Extract `trajectory_server` or default to None
        let trajectory_binding: Bound<PyAny> = inner_dict
            .get_item("trajectory_server")
            .ok()
            .flatten()
            .expect("Missing 'trajectory_server' field");

        let trajectory_server_result: Option<&str> = trajectory_binding
            .downcast::<PyString>()
            .ok()
            .and_then(|py_str| py_str.to_str().ok());

        let trajectory_server: Option<&str> = trajectory_server_result
            .map(|_server| trajectory_server_result.expect("Failed to extract trajectory_server"));

        // Create a new PyRelayRLTrajectory
        let mut py_trajectory: PyRelayRLTrajectory =
            PyRelayRLTrajectory::new(Some(max_length), trajectory_server);

        // Extract the 'actions' field if present
        if let Ok(Some(actions_obj)) = inner_dict.get_item("actions") {
            let actions_list: &Bound<PyList> = actions_obj
                .downcast::<PyList>()
                .expect("'actions' must be a list");

            // Reconstruct each action from JSON and add it to the trajectory
            for action_item in actions_list.iter() {
                let action_dict: &Bound<PyDict> = action_item
                    .downcast::<PyDict>()
                    .expect("Action must be a dictionary");
                let action = PyRelayRLAction::action_from_json(action_dict);
                py_trajectory.inner.add_action(&action.inner, false);
            }
        }
        py_trajectory
    }
}
