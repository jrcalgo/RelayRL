//! RelayRL client runtime.
//!
//! This module is split into a small public API surface and a larger internal runtime:
//! - [`agent`]: public construction and control APIs for client applications
//! - `runtime::coordination`: coordinator, lifecycle, scaling, and state management
//! - `runtime::router`: message routing between actors and data sinks
//! - `runtime::data`: local file sinks plus experimental transport-backed sinks
//!
//! In `0.5.0-beta`, the supported path is the local/default runtime exposed through
//! [`agent`]. Transport-backed flows behind `zmq-transport` and `nats-transport` remain
//! experimental.
//!
//! The local/default runtime follows this flow:
//! `AgentBuilder` -> `RelayRLAgent` -> coordinator -> router/actors -> local file sink.
pub mod agent;
pub(crate) mod runtime {
    pub(crate) mod actor;
    pub(crate) mod coordination {
        pub(crate) mod coordinator;
        pub(crate) mod lifecycle_manager;
        pub(crate) mod scale_manager;
        pub(crate) mod state_manager;
    }
    pub(crate) mod router;

    pub(crate) mod data {
        pub(crate) mod file_sink;
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        pub(crate) mod transport_sink;
    }
}
