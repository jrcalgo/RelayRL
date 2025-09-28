/// **Python Bindings for RL4Sys**: This module contains the Rust-to-Python bindings,
/// exposing RL4Sys components as Python classes. The `o3_*` modules implement PyO3-compatible
/// wrappers for core structures, enabling smooth Python interaction.
pub(crate) mod bindings {
    pub(crate) mod python {
        pub(crate) mod network {
            pub(crate) mod client {
                pub(crate) mod agent;
            }
        }
        pub(crate) mod types {
            pub(crate) mod action;
            pub(crate) mod trajectory;
        }
        pub(crate) mod utilities {
            pub(crate) mod configuration;
        }
    }
}

/// ### RL4Sys Python Module Definition
///
/// This function defines `rl4sys_framework`, the Python module for RL4Sys bindings.
///
/// It registers the following Python classes:
/// - `ConfigLoader`
/// - `TrainingServer`
/// - `RL4SysAgent`
/// - `RL4SysTrajectory`
/// - `RL4SysAction`
///
/// This allows Python users to easily import and use RL4Sys functionalities via:
///
/// ```python
/// from rl4sys_framework import RL4SysAgent, RL4SysTrajectory, RL4SysAction
/// ```
///
#[pymodule(name = "relayrl_framework")]
fn rl4sys_framework(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Python-accessible classes from the Rust implementation.
    m.add_class::<PyConfiguration>()?;
    m.add_class::<PyRelayRLTrainingServer>()?;
    m.add_class::<PyRelayRLAgent>()?;
    m.add_class::<PyRelayRLTrajectory>()?;
    m.add_class::<PyRelayRLAction>()?;

    // Define Python `__all__` to indicate available imports.
    m.add(
        "__all__",
        vec![
            "Configuration",
            "RelayRLTrainingServer",
            "RelayRLAgent",
            "RelayRLTrajectory",
            "RelayRLAction",
        ],
    )?;

    Ok(())
}