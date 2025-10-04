//! This module provides utilities for serializing and sending trajectories as well as defining
//! the RelayRLTrajectory type and trait. It uses serde_pickle for serialization and ZMQ for sending
//! the serialized data to a trajectory server.

use crate::NetworkParticipant;
use crate::types::action::RelayRLAction;
use crate::types::action::RelayRLActionTrait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use relayrl_framework::utilities::configuration::{ClientConfigLoader, ServerConfigLoader};

/// Trait that defines the interface for trajectory handling.
///
/// Any trajectory struct must implement this trait, which requires a method to add an action
/// to the trajectory. The method may also send the trajectory if a terminal action is encountered.
pub trait RelayRLTrajectoryTrait {
    /// The associated action type that this trajectory holds.
    type Action: RelayRLActionTrait;
    /// Adds an action to the trajectory.
    ///
    /// # Arguments
    ///
    /// * `action` - A reference to the action to be added.
    /// * `send_if_done` - A boolean flag that, if true and the action is terminal, will trigger sending the trajectory.
    fn add_action(&mut self, action: &Self::Action);
}

/// The RelayRLTrajectory struct represents a trajectory composed of a sequence of actions.
///
/// It stores an optional trajectory server address, a maximum trajectory length, and a vector of actions.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RelayRLTrajectory {
    /// The maximum number of actions allowed in the trajectory.
    max_length: u128,
    /// A vector storing the actions in the trajectory.
    pub actions: Vec<RelayRLAction>,
}

impl RelayRLTrajectory {
    /// Creates a new RelayRLTrajectory.
    ///
    /// # Arguments
    ///
    /// * `max_length` - An optional maximum trajectory length. Must be provided.
    /// * `trajectory_server` - An optional trajectory server address.
    ///
    /// # Returns
    ///
    /// A new instance of RelayRLTrajectory.
    pub fn new(
        max_length: Option<u128>,
        network_participant: NetworkParticipant,
        config_path: &PathBuf,
    ) -> Self {
        let max_length: u128 = max_length.unwrap_or_else(|| match network_participant {
            NetworkParticipant::RelayRLAgent => {
                let loader = ClientConfigLoader::load_config(config_path);
                loader.transport_config.max_traj_length
            }
            NetworkParticipant::RelayRLTrainingServer => {
                let loader = ServerConfigLoader::load_config(config_path);
                loader.transport_config.max_traj_length
            }
        });
        println!(
            "[RelayRLTrajectory] New {:?} length trajectory created",
            max_length
        );

        RelayRLTrajectory {
            max_length,
            actions: Vec::new(),
        }
    }
}

/// Implementation of the RelayRLTrajectoryTrait for RelayRLTrajectory.
///
/// This implementation defines how an action is added to the trajectory. If the trajectory reaches
/// its maximum length and the send_if_done flag is set along with the action being terminal,
/// the trajectory is sent to the training server and then cleared.
impl RelayRLTrajectoryTrait for RelayRLTrajectory {
    type Action = RelayRLAction;

    /// Adds an action to the trajectory and conditionally sends it if the trajectory is full and the action is terminal.
    ///
    /// # Arguments
    ///
    /// * `action` - A reference to the RelayRLAction to be added.
    fn add_action(&mut self, action: &RelayRLAction) {
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
