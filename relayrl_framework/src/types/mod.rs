use std::collections::HashMap;

/// **Core RL4Sys Data Types**: Define the primary data structures used
/// throughout the framework. These include:
/// - `trajectory`: Defines trajectory management and serialization logic.
/// - `action`: Handles action structures and data.
#[cfg(feature = "data_types")]
pub mod action;
#[cfg(feature = "data_types")]
pub mod trajectory;

pub(crate) enum NetworkParticipant {
    RL4SysAgent,
    RL4SysTrainingServer,
}

/// Hyperparams enum represents hyperparameter inputs which can be provided either as a map
/// or as a list of argument strings.
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(HashMap<String, String>),
    Args(Vec<String>),
}
