use std::collections::HashMap;

pub mod types {
    pub mod action;
    pub mod trajectory;
}

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
