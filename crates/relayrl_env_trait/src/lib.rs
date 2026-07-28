//! # RelayRL Environment Traits
//!
//! The environment abstraction layer for RelayRL. This crate is intentionally tiny and
//! dependency-light (`thiserror` + `uuid` only): it defines *how* the runtime talks to an
//! environment, but contains no simulator, no tensor backend, and no runtime logic. The
//! `relayrl_framework` client drives any type that implements these traits.
//!
//! ## System layout
//!
//! All public items live in the [`traits`] module and are re-exported at the crate root:
//!
//! - [`Environment`]: the base contract every environment shares (observation/mask building,
//!   dtypes and dimensions, flat-bytes accessors, discreteness, and conversion into an
//!   [`EnvironmentHandle`]).
//! - [`ScalarEnvironment`]: one object per logical environment, stepped with a single action.
//! - [`VectorEnvironment`]: one object that owns a *batch* of logical environments, stepped
//!   with a batch of actions in a single call.
//! - [`EnvironmentHandle`]: a runtime-facing enum unifying boxed scalar and vector
//!   environments, with [`DynScalarEnvironment`] providing object-safe, clonable scalar envs.
//! - Supporting types: [`EnvironmentUuid`] (stable per-env identity), [`EnvDType`] /
//!   [`EnvironmentKind`], [`ScalarEnvReset`] / [`VectorEnvReset`], the byte aliases
//!   [`Observation`] / [`Mask`] / [`Reward`] / [`Done`] / [`Truncated`], and
//!   [`TrainingPerformanceReturnFn`] for custom training signals.
//!
//! ## Scalar vs. vector execution
//!
//! The framework may run **many logical environments in parallel** (one
//! [`ScalarEnvironment`] per worker) or a single **batched** simulator that implements
//! [`VectorEnvironment`]:
//!
//! - Use [`ScalarEnvironment`] when each sub-environment is its own object with a scalar
//!   step. A parallel runner holds many handles, assigns one stable [`EnvironmentUuid`] per
//!   sub-env, and steps each worker independently.
//! - Use [`VectorEnvironment`] when one implementation can apply a batch of actions keyed by
//!   [`EnvironmentUuid`] in a single call (GPU batching, vectorized physics, a remote batched
//!   service, etc.).
//!
//! ## Design notes and implementor contracts
//!
//! - **`Send + Sync` everywhere.** All traits require `Send + Sync`, so mutable simulation
//!   state should live behind interior mutability (e.g. `Mutex`, atomics) rather than `&mut
//!   self` — every method takes `&self`.
//! - **Opaque identity.** Treat [`EnvironmentUuid`] as opaque; the same uuid must refer to
//!   one logical env across `reset`/`step` and any runtime routing.
//! - **Ordering.** Unless your concrete type documents otherwise, callers should not assume
//!   [`VectorEnvironment::step_bytes`] output order matches input order; key results by
//!   [`EnvironmentUuid`].
//! - **Errors are whole-operation.** [`EnvironmentError`] describes the entire call; partial
//!   success is not expressed in the type system. Surface per-env failures inside your info
//!   payloads if you need them.
//! - **Type-erased observations.** [`Environment::build_observation`] returns
//!   [`std::any::Any`] for framework integration; pair it with a documented downcasting
//!   convention. The `flat_*_bytes` accessors provide the byte-oriented path the runtime uses.
//!
//! ## Quick start
//!
//! A minimal scalar environment skeleton. Note that every method takes `&self`, so any
//! mutable state must use interior mutability:
//!
//! ```no_run
//! use relayrl_env_trait::*;
//! use std::any::Any;
//!
//! #[derive(Clone)]
//! struct MyEnv;
//!
//! impl Environment for MyEnv {
//!     fn run_environment(&self) -> Result<(), EnvironmentError> { Ok(()) }
//!     fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
//!         Ok(Box::new(vec![0u8; self.observation_dim()]))
//!     }
//!     fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> { Ok(Box::new(())) }
//!     fn observation_dtype(&self) -> EnvDType { EnvDType::NdArray(EnvNdArrayDType::F32) }
//!     fn action_dtype(&self) -> EnvDType { EnvDType::NdArray(EnvNdArrayDType::I64) }
//!     fn observation_dim(&self) -> usize { 8 }
//!     fn action_dim(&self) -> usize { 4 }
//!     fn flat_observation_bytes(&self) -> Observation { vec![0u8; self.observation_dim()] }
//!     fn flat_mask_bytes(&self) -> Mask { None }
//!     fn action_is_discrete(&self) -> bool { true }
//!     fn kind(&self) -> EnvironmentKind { EnvironmentKind::Scalar }
//!     fn into_handle(self: Box<Self>) -> EnvironmentHandle {
//!         EnvironmentHandle::Scalar(Box::new(*self))
//!     }
//! }
//!
//! impl ScalarEnvironment for MyEnv {
//!     fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
//!         Ok(ScalarEnvReset { observation: self.flat_observation_bytes(), info: None })
//!     }
//!     fn step_bytes(
//!         &self,
//!         _action: &[u8],
//!     ) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
//!         Some((self.flat_observation_bytes(), None, 0.0, false, false))
//!     }
//! }
//! ```

pub mod traits {
    //! Environment abstraction contracts and supporting types for the RelayRL runtime.
    //!
    //! All items here are re-exported at the crate root via `pub use traits::*`, so they are
    //! typically imported as `use relayrl_env_trait::*`. See the [crate-level documentation](super)
    //! for implementor guidance, design notes, and a full quick-start example.

    use std::any::Any;
    pub use thiserror::Error;
    pub use uuid::Uuid;

    /// Errors returned by environment operations.
    #[derive(Debug, Error, Clone)]
    pub enum EnvironmentError {
        #[error("Environment error: {0}")]
        EnvironmentError(String),
        #[error("Observation building error: {0}")]
        ObservationBuildingError(String),
        #[error("Training performance return error: {0}")]
        TrainingPerformanceReturnError(String),
    }

    /// Flat byte observation buffer; its element encoding is given by the env's [`EnvDType`].
    pub type Observation = Vec<u8>;
    /// Optional action mask as raw bytes; `None` means all actions are valid.
    pub type Mask = Option<Vec<u8>>;
    /// `true` when the episode has terminated naturally.
    pub type Done = bool;
    /// `true` when the episode was cut short (e.g. a step limit) rather than terminating naturally.
    pub type Truncated = bool;
    /// Scalar reward for the most recent step.
    pub type Reward = f32;

    /// Selects the scalar or vector execution path the runtime uses for an environment.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum EnvironmentKind {
        /// One object per logical environment, stepped with a single action.
        Scalar,
        /// One object owning a batch of logical environments, stepped with a batch of actions.
        Vector,
    }

    /// Observation or action element dtype, keyed to the active tensor backend.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum EnvDType {
        /// Element dtype for the NdArray backend.
        NdArray(EnvNdArrayDType),
        /// Element dtype for the LibTorch (`tch`) backend.
        Tch(EnvTchDType),
    }

    impl std::fmt::Display for EnvDType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                EnvDType::NdArray(ndarray) => write!(f, "NdArray({})", ndarray),
                EnvDType::Tch(tch) => write!(f, "Tch({})", tch),
            }
        }
    }

    /// Element dtype variants supported by the LibTorch (`tch`) backend.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum EnvTchDType {
        F16,
        Bf16,
        F32,
        F64,
        I8,
        I16,
        I32,
        I64,
        U8,
        Bool,
    }

    impl std::fmt::Display for EnvTchDType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                EnvTchDType::F16 => write!(f, "F16"),
                EnvTchDType::Bf16 => write!(f, "Bf16"),
                EnvTchDType::F32 => write!(f, "F32"),
                EnvTchDType::F64 => write!(f, "F64"),
                EnvTchDType::I8 => write!(f, "I8"),
                EnvTchDType::I16 => write!(f, "I16"),
                EnvTchDType::I32 => write!(f, "I32"),
                EnvTchDType::I64 => write!(f, "I64"),
                EnvTchDType::U8 => write!(f, "U8"),
                EnvTchDType::Bool => write!(f, "Bool"),
            }
        }
    }

    /// Element dtype variants supported by the NdArray backend.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum EnvNdArrayDType {
        F16,
        F32,
        F64,
        I8,
        I16,
        I32,
        I64,
        Bool,
    }

    impl std::fmt::Display for EnvNdArrayDType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                EnvNdArrayDType::F16 => write!(f, "F16"),
                EnvNdArrayDType::F32 => write!(f, "F32"),
                EnvNdArrayDType::F64 => write!(f, "F64"),
                EnvNdArrayDType::I8 => write!(f, "I8"),
                EnvNdArrayDType::I16 => write!(f, "I16"),
                EnvNdArrayDType::I32 => write!(f, "I32"),
                EnvNdArrayDType::I64 => write!(f, "I64"),
                EnvNdArrayDType::Bool => write!(f, "Bool"),
            }
        }
    }

    /// Stable identity for a logical sub-environment in batched or parallel execution (for example
    /// one slot in a [`VectorEnvironment`] step, or the id assigned by a parallel vec-env runner).
    pub type EnvironmentUuid = Uuid;

    /// Arbitrary string key-value metadata returned alongside a reset.
    pub type EnvInfo = Vec<(String, String)>;

    /// Result of a scalar [`ScalarEnvironment::reset`]: the initial observation plus optional info.
    #[derive(Debug, Clone)]
    pub struct ScalarEnvReset {
        pub observation: Vec<u8>,
        pub info: Option<EnvInfo>,
    }

    /// Result of resetting one environment slot within a batched [`VectorEnvironment::reset`].
    #[derive(Debug, Clone)]
    pub struct VectorEnvReset {
        pub env_id: EnvironmentUuid,
        pub observation: Vec<u8>,
        pub info: Option<EnvInfo>,
    }

    /// Object-safe type alias for a boxed [`VectorEnvironment`] trait object.
    pub type DynVectorEnv = dyn VectorEnvironment;
    /// Object-safe, clonable extension of [`ScalarEnvironment`], blanket-implemented for every
    /// `Clone + Send + Sync + 'static` scalar environment so it can be stored as a trait object.
    pub trait DynScalarEnvironment: ScalarEnvironment + Send + Sync {
        /// Clones this environment into a fresh boxed trait object.
        fn clone_box(&self) -> Box<dyn DynScalarEnvironment>;

        /// Object-safe wrapper around [`Environment::flat_observation_bytes`].
        fn dyn_flat_obs(&self) -> Observation {
            self.flat_observation_bytes()
        }
        /// Object-safe wrapper around [`Environment::flat_mask_bytes`].
        fn dyn_flat_mask(&self) -> Mask {
            self.flat_mask_bytes()
        }
        /// Object-safe wrapper around [`ScalarEnvironment::step_bytes`].
        fn dyn_step(&self, action: &[u8]) -> Option<(Observation, Mask, Reward, Done, Truncated)> {
            self.step_bytes(action)
        }
        /// Object-safe wrapper around [`Environment::action_dim`].
        fn dyn_act_dim(&self) -> usize {
            self.action_dim()
        }
    }
    impl<T> DynScalarEnvironment for T
    where
        T: ScalarEnvironment + Clone + Send + Sync + 'static,
    {
        fn clone_box(&self) -> Box<dyn DynScalarEnvironment> {
            Box::new(self.clone())
        }
    }
    impl Clone for Box<dyn DynScalarEnvironment> {
        fn clone(&self) -> Self {
            self.clone_box()
        }
    }
    /// Runtime-facing enum unifying a boxed scalar or vector environment behind one type.
    pub enum EnvironmentHandle {
        /// A single-action scalar environment.
        Scalar(Box<dyn DynScalarEnvironment>),
        /// A batched vector environment.
        Vector(Box<DynVectorEnv>),
    }

    /// An environment stepped one logical instance at a time with a single action.
    pub trait ScalarEnvironment: Environment + Send + Sync {
        /// Resets the environment and returns the initial observation plus optional info.
        fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError>;
        /// Applies one action (raw bytes) and returns the next observation, mask, reward, done, and truncated flags.
        fn step_bytes(&self, action: &[u8])
        -> Option<(Observation, Mask, Reward, Done, Truncated)>;
    }

    /// An environment that owns a batch of logical instances stepped together in one call.
    pub trait VectorEnvironment: Environment + Send + Sync {
        /// Allocates `num_envs` logical environments and returns their stable [`EnvironmentUuid`]s.
        fn init_num_envs(&self, num_envs: usize) -> Result<Vec<EnvironmentUuid>, EnvironmentError>;
        /// Resets the given environment slots, returning one [`VectorEnvReset`] per id.
        fn reset(
            &self,
            env_ids: &[EnvironmentUuid],
        ) -> Result<Vec<VectorEnvReset>, EnvironmentError>;
        /// Returns the number of logical environments currently held.
        fn n_envs(&self) -> usize;
        /// Applies a batch of actions (raw bytes) and returns batched observations, mask, rewards, done, and truncated flags.
        #[allow(clippy::type_complexity)]
        fn step_bytes(
            &self,
            actions: &[u8],
        ) -> Option<(Observation, Mask, Vec<Reward>, Vec<Done>, Vec<Truncated>)>;
    }

    /// Interface for environments where a model can be trained or evaluated.
    ///
    /// Methods are intentionally parameterless: configuration and mutable state live on the
    /// implementing type (often with interior mutability when shared across threads).
    pub trait Environment: Send + Sync {
        /// Runs any internal stepping/simulation the environment needs to advance its own state.
        fn run_environment(&self) -> Result<(), EnvironmentError>;
        /// Builds the current observation as a type-erased value for framework downcasting.
        fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError>;
        /// Builds the current action mask as a type-erased value for framework downcasting.
        fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError>;
        /// Returns the element dtype of observations.
        fn observation_dtype(&self) -> EnvDType;
        /// Returns the element dtype of actions.
        fn action_dtype(&self) -> EnvDType;
        /// Returns the flattened observation dimension (element count).
        fn observation_dim(&self) -> usize;
        /// Returns the flattened action dimension (element count).
        fn action_dim(&self) -> usize;
        /// Returns the current observation as a flat byte buffer (the runtime's primary path).
        fn flat_observation_bytes(&self) -> Observation;
        /// Returns the current action mask as flat bytes, or `None` when unmasked.
        fn flat_mask_bytes(&self) -> Mask;
        /// Returns `true` for discrete action spaces, `false` for continuous ones.
        fn action_is_discrete(&self) -> bool;
        /// Returns whether this environment is scalar or vector.
        fn kind(&self) -> EnvironmentKind;
        /// Consumes the boxed environment into an [`EnvironmentHandle`] for runtime storage.
        fn into_handle(self: Box<Self>) -> EnvironmentHandle;
    }

    /// Computes a performance signal (for example return-to-go) for training feedback.
    pub trait TrainingPerformanceReturnFn {
        fn calculate_performance_return(&self) -> Result<Box<dyn Any>, EnvironmentError>;
    }
}

pub use traits::*;
