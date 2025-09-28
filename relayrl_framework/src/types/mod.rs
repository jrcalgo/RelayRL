/// **Core RL4Sys Data Types**: Define the primary data structures used
/// throughout the framework. These include:
/// - `trajectory`: Defines trajectory management and serialization logic.
/// - `action`: Handles action structures and data.
#[cfg(feature = "data_types")]
pub mod action;
#[cfg(feature = "data_types")]
pub mod trajectory;
