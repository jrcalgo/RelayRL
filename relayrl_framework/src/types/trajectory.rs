//! This module provides utilities for serializing and sending trajectories as well as defining
//! the RL4SysTrajectory type and trait. It uses serde_pickle for serialization and ZMQ for sending
//! the serialized data to a trajectory server.

use crate::types::action::RL4SysAction;
use crate::types::action::RL4SysActionTrait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[cfg(feature = "zmq_network")]
use zmq;
#[cfg(feature = "zmq_network")]
use zmq::{Context, Socket};

#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_trajectory::PyRL4SysTrajectory;
use crate::sys_utils::configuration::{ClientConfigLoader, ServerConfigLoader};

/// Trait that defines the interface for trajectory handling.
///
/// Any trajectory struct must implement this trait, which requires a method to add an action
/// to the trajectory. The method may also send the trajectory if a terminal action is encountered.
pub trait RL4SysTrajectoryTrait {
    /// The associated action type that this trajectory holds.
    type Action: RL4SysActionTrait;
    /// Adds an action to the trajectory.
    ///
    /// # Arguments
    ///
    /// * `action` - A reference to the action to be added.
    /// * `send_if_done` - A boolean flag that, if true and the action is terminal, will trigger sending the trajectory.
    fn add_action(&mut self, action: &Self::Action);
}

pub enum NetworkParticipant {
    RL4SysAgent,
    RL4SysTrainingServer,
}

/// The RL4SysTrajectory struct represents a trajectory composed of a sequence of actions.
///
/// It stores an optional trajectory server address, a maximum trajectory length, and a vector of actions.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RL4SysTrajectory {
    /// The maximum number of actions allowed in the trajectory.
    max_length: u128,
    /// A vector storing the actions in the trajectory.
    pub actions: Vec<RL4SysAction>,
}

impl RL4SysTrajectory {
    /// Creates a new RL4SysTrajectory.
    ///
    /// # Arguments
    ///
    /// * `max_length` - An optional maximum trajectory length. Must be provided.
    /// * `trajectory_server` - An optional trajectory server address.
    ///
    /// # Returns
    ///
    /// A new instance of RL4SysTrajectory.
    pub fn new(max_length: Option<u128>, network_participant: NetworkParticipant, config_path: &PathBuf) -> Self {
        let max_length: u128 = max_length
            .unwrap_or_else(|| {
                match network_participant {
                    NetworkParticipant::RL4SysAgent => {
                        let loader = ClientConfigLoader::load_config(config_path);
                        loader.transport_config.max_traj_length
                    },
                    NetworkParticipant::RL4SysTrainingServer => {
                        let loader = ServerConfigLoader::load_config(config_path);
                        loader.transport_config.max_traj_length
                    },
                }
            });
        println!(
            "[RL4SysTrajectory] New {:?} length trajectory created",
            max_length
        );

        RL4SysTrajectory {
            max_length,
            actions: Vec::new(),
        }
    }

    /// Converts the RL4SysTrajectory into its Python wrapper representation.
    ///
    /// # Returns
    ///
    /// A PyRL4SysTrajectory that wraps the current trajectory.
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network",
        feature = "python_bindings"
    ))]
    pub fn into_py(self) -> PyRL4SysTrajectory {
        PyRL4SysTrajectory {
            inner: RL4SysTrajectory {
                max_length: self.max_length,
                actions: self.actions,
            },
        }
    }
}

/// Implementation of the RL4SysTrajectoryTrait for RL4SysTrajectory.
///
/// This implementation defines how an action is added to the trajectory. If the trajectory reaches
/// its maximum length and the send_if_done flag is set along with the action being terminal,
/// the trajectory is sent to the training server and then cleared.
impl RL4SysTrajectoryTrait for RL4SysTrajectory {
    type Action = RL4SysAction;

    /// Adds an action to the trajectory and conditionally sends it if the trajectory is full and the action is terminal.
    ///
    /// # Arguments
    ///
    /// * `action` - A reference to the RL4SysAction to be added.
    fn add_action(&mut self, action: &RL4SysAction) {
        let action_done: bool = action.done;

        self.actions.push(action.to_owned());

        if action_done
            && self.actions.len()
                >= <u128 as TryInto<usize>>::try_into(self.max_length)
                    .expect("Failed to convert max_length to usize")
        {
            self.actions.clear();
        }
    }
}
