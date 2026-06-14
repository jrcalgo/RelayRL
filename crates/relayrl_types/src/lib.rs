//! # RelayRL Types
//!
//! Core data types and codec utilities shared across the RelayRL stack. This is the
//! lowest-level crate: it owns the on-the-wire and in-memory representations that every
//! other crate (`relayrl_algorithms`, `relayrl_framework`) builds on, and it deliberately
//! contains no runtime, networking, or training logic.
//!
//! ## Where this crate sits
//!
//! ```text
//! relayrl_framework  (client runtime, routing, sinks)
//!        |
//!        +-- relayrl_algorithms  (PPO/IPPO/MAPPO, networks)
//!        |
//!        +-- relayrl_types  (this crate: actions, tensors, trajectories, codecs)
//! ```
//!
//! ## System layout
//!
//! - [`data`]: the public data model.
//!   - [`data::action`]: [`RelayRLAction`](data::action::RelayRLAction), the unit of
//!     experience (observation, action, mask, reward, done, auxiliary data, agent id),
//!     plus the [`EncodedAction`](data::action::EncodedAction) codec wrapper.
//!   - [`data::tensor`]: the backend-agnostic [`TensorData`](data::tensor::TensorData)
//!     container, [`DType`](data::tensor::DType) / [`DeviceType`](data::tensor::DeviceType),
//!     and the [`BackendMatcher`](data::tensor::BackendMatcher) /
//!     [`SupportedTensorBackend`](data::tensor::SupportedTensorBackend) machinery that maps
//!     RelayRL types onto concrete Burn backends.
//!   - [`data::trajectory`]: [`RelayRLTrajectory`](data::trajectory::RelayRLTrajectory), an
//!     ordered buffer of actions with provenance metadata (agent/env ids, episode, policy
//!     version).
//!   - `data::records`: Arrow and CSV adapters for persisting trajectories to disk.
//!   - `data::utilities`: the codec pipeline (compression, encryption, integrity, metadata,
//!     quantization, chunking) used by transport-backed workflows.
//! - [`model`]: hot-reloadable inference model wrappers
//!   ([`ModelModule`](model::ModelModule), [`HotReloadableModel`](model::HotReloadableModel)),
//!   compiled only when an inference model feature and a tensor backend are both enabled.
//! - [`prelude`]: grouped re-exports (`action`, `tensor`, `trajectory`, `records`, `model`,
//!   `codec`) intended as the primary import surface for downstream crates.
//! - [`HyperparameterArgs`]: shared hyperparameter input shape (map or `key=value` list).
//!
//! ## Design notes
//!
//! - **Backend-agnostic by construction.** [`TensorData`](data::tensor::TensorData) stores
//!   raw bytes plus a [`DType`](data::tensor::DType) and a target backend, and is only
//!   materialized into a concrete Burn `Tensor` on demand via the `to_*_tensor` helpers.
//!   This keeps the data model serializable and independent of any one tensor library.
//! - **Exactly one numeric backend at a time.** `ndarray-backend` (CPU) and `tch-backend`
//!   (LibTorch/GPU) are mutually exclusive in practice; most public items are gated behind
//!   `any(feature = "ndarray-backend", feature = "tch-backend")`.
//! - **Codec is opt-in and layered.** Each codec stage is its own feature so a build only
//!   pays for what it uses; the `codec-basic`/`codec-secure`/`codec-full` bundles compose them.
//!
//! ## Feature flags
//!
//! - `ndarray-backend` (default): CPU tensors via `burn-ndarray`.
//! - `tch-backend`: LibTorch tensors via `burn-tch`.
//! - `onnx-model` (default) / `tch-model` / `inference-models`: inference model support.
//! - `compression`, `encryption`, `integrity`, `metadata`, `quantization`, `zerocopy`:
//!   individual codec stages.
//! - `codec-basic`, `codec-secure`, `codec-full`: convenience bundles of the above.
//!
//! ## Quick start
//!
//! Build actions and collect them into a trajectory (works with the default
//! `ndarray-backend` feature):
//!
//! ```no_run
//! use relayrl_types::prelude::action::RelayRLAction;
//! use relayrl_types::prelude::trajectory::RelayRLTrajectory;
//! use uuid::Uuid;
//!
//! // A trajectory is an ordered, capacity-bounded buffer of actions.
//! let mut trajectory = RelayRLTrajectory::with_agent_id(1_000, Uuid::new_v4());
//!
//! // `minimal` records only reward + done; `add_action` returns `true` when the
//! // trajectory should be flushed (capacity reached or `done == true`).
//! trajectory.add_action(RelayRLAction::minimal(1.0, false));
//! let should_flush = trajectory.add_action(RelayRLAction::minimal(2.0, true));
//!
//! assert!(should_flush);
//! assert_eq!(trajectory.len(), 2);
//! assert_eq!(trajectory.total_reward(), 3.0);
//! ```
//!
//! For tensor-backed actions, convert a Burn tensor into
//! [`TensorData`](data::tensor::TensorData) and pass it to
//! [`RelayRLAction::new`](data::action::RelayRLAction::new); see the crate README for a full
//! tensor round-trip example.

pub mod data;
#[cfg(all(
    any(feature = "tch-model", feature = "onnx-model"),
    any(feature = "ndarray-backend", feature = "tch-backend")
))]
pub mod model;

pub mod prelude {
    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub mod action {
        pub use crate::data::action::{
            ActionError, CodecConfig, EncodedAction, RelayRLAction, RelayRLData,
        };
    }

    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub mod tensor {
        pub mod relayrl {
            pub use crate::data::tensor::{
                AnyBurnTensor, BackendMatcher, BoolBurnTensor, DType, DeviceType, FloatBurnTensor,
                IntBurnTensor, SupportedTensorBackend, TensorData, TensorError,
            };
        }

        pub mod burn {
            pub use burn_tensor::*;
        }
    }

    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub mod trajectory {
        pub use crate::data::trajectory::{
            EncodedTrajectory, RelayRLTrajectory, RelayRLTrajectoryTrait, TrajectoryError,
        };
    }

    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    pub mod records {
        pub use crate::data::records::arrow::{ArrowTrajectory, ArrowTrajectoryError};
        pub use crate::data::records::csv::{CsvTrajectory, CsvTrajectoryError};
    }

    #[cfg(all(
        any(feature = "tch-model", feature = "onnx-model"),
        any(feature = "ndarray-backend", feature = "tch-backend")
    ))]
    pub mod model {
        pub use crate::model::{HotReloadableModel, ModelError, ModelModule};
    }

    pub mod codec {
        #[cfg(feature = "compression")]
        pub use crate::data::utilities::compress::{CompressedData, CompressionScheme};

        #[cfg(feature = "integrity")]
        pub use crate::data::utilities::integrity::{VerifiedData, compute_checksum};

        #[cfg(feature = "encryption")]
        pub use crate::data::utilities::encrypt::{EncryptedData, EncryptionKey};

        #[cfg(feature = "metadata")]
        pub use crate::data::utilities::metadata::TensorMetadata;

        #[cfg(feature = "quantization")]
        pub use crate::data::utilities::quantize::{QuantizationScheme, QuantizedData};

        #[cfg(feature = "integrity")]
        pub use crate::data::utilities::chunking::{ChunkedTensor, TensorChunk};
    }
}

use std::collections::HashMap;

/// Hyperparameter inputs for algorithms — either a key-value map or a flat `key=value` argument list.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HyperparameterArgs {
    /// Pre-parsed key-value map.
    Map(HashMap<String, String>),
    /// Flat `"key=value"` strings, parsed at handshake time.
    List(Vec<String>),
}
