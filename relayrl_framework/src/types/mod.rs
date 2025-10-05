use std::collections::HashMap;

/// **Core RelayRL Data Types**: Define the primary data structures used
/// throughout the framework. These include:
/// - `trajectory`: Defines trajectory management and serialization logic.
/// - `action`: Handles action structures and data.
pub mod action;
pub mod trajectory;

pub(crate) enum NetworkParticipant {
    RelayRLAgent,
    RelayRLTrainingServer,
}

/// Hyperparams enum represents hyperparameter inputs which can be provided either as a map
/// or as a list of argument strings.
#[derive(Clone, Debug)]
pub enum Hyperparams {
    Map(HashMap<String, String>),
    Args(Vec<String>),
}
