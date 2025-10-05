use serde::{Deserialize, Serialize};

/// **Python Bindings for RelayRL**: This module contains the Rust-to-Python bindings,
/// exposing RelayRL components as Python classes. The `o3_*` modules implement PyO3-compatible
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

/// A response received from the Python subprocess.
///
/// It contains a status string (e.g., "success") and an optional message.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct PythonResponse {
    pub(crate) status: String,
    message: Option<String>,
}


/// ### RelayRL Python Module Definition
///
/// This function defines `relayrl_framework`, the Python module for RelayRL bindings.
///
/// It registers the following Python classes:
/// - `ConfigLoader`
/// - `TrainingServer`
/// - `RelayRLAgent`
/// - `RelayRLTrajectory`
/// - `RelayRLAction`
///
/// This allows Python users to easily import and use RelayRL functionalities via:
///
/// ```python
/// from relayrl_framework import RelayRLAgent, RelayRLTrajectory, RelayRLAction
/// ```
///
#[pymodule(name = "relayrl_framework")]
fn relayrl_framework(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
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