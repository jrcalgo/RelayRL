//! Traits for training and testing environments in RelayRL.
//!
//! # VecEnv and parallel stepping
//!
//! The framework may run **many logical environments in parallel** (array-of-structs / one
//! [`ScalarEnvironment`] per worker) or a single **batched** simulator that implements
//! [`VectorEnvironment`]. Both paths share the base [`Environment`] contract for observation
//! building and optional high-level loops.
//!
//! ## [`ScalarEnvironment`]
//!
//! Use when each sub-environment is its own object with a scalar `(observation, step_info)` step.
//! A parallel runner typically holds `Vec<Arc<dyn ScalarEnvironment<...>>>` (or concrete handles),
//! assigns one **stable** [`EnvironmentUuid`] per sub-env, and calls `step` on each worker. Types
//! are `Send + Sync` so implementations often use interior mutability (for example `Mutex` or
//! atomics) if physical simulation state mutates across calls.
//!
//! ## [`VectorEnvironment`]
//!
//! Use when one implementation can apply a **batch** of actions keyed by [`EnvironmentUuid`] in a
//! single call (GPU batching, vectorized physics, remote batched service, etc.). Callers should
//! treat identities as opaque: the same uuid must be used for one logical env across `step` and
//! any routing in the runtime.
//!
//! ## Contracts (implementors)
//!
//! - **Ordering**: Unless documented otherwise by your concrete type, callers should not assume
//!   that output order matches input order for [`VectorEnvironment::step`]; they should key by
//!   [`EnvironmentUuid`].
//! - **Errors**: [`EnvironmentError`] is returned for the whole operation; partial success is not
//!   expressed in the type system. If you need per-env errors, document that on your concrete type
//!   or surface it inside [`StepInfo`] / [`ResetInfo`].
//! - **Observations**: [`Environment::build_observation`] is intentionally type-erased
//!   ([`std::any::Any`]) for framework integration; pair it with a documented convention for
//!   downcasting where the runtime requires a concrete layout.

pub mod traits {
    pub use burn_tensor::{Tensor, TensorKind, backend::Backend};
    use std::any::Any;
    pub use thiserror::Error;
    pub use uuid::Uuid;

    #[derive(Debug, Error, Clone)]
    pub enum EnvironmentError {
        #[error("Environment error: {0}")]
        EnvironmentError(String),
        #[error("Observation building error: {0}")]
        ObservationBuildingError(String),
        #[error("Training performance return error: {0}")]
        TrainingPerformanceReturnError(String),
    }

    /// Stable identity for a logical sub-environment in batched or parallel execution (for example
    /// one slot in a [`VectorEnvironment`] step, or the id assigned by a parallel vec-env runner).
    pub type EnvironmentUuid = Uuid;

    pub trait ScalarEnvironment<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KInput: TensorKind<B>,
        KOutput: TensorKind<B>,
    >: Environment + Send + Sync + Sized
    where
        Self: Sized,
    {
        type ResetInfo: IntoIterator<Item = (String, String)>;
        type StepInfo: IntoIterator<Item = (String, String)>;

        fn step(
            &self,
            action: Tensor<B, D_OUT, KOutput>,
        ) -> Result<(Tensor<B, D_IN, KInput>, Self::StepInfo), EnvironmentError>;
        fn reset(&self) -> Result<Option<Self::ResetInfo>, EnvironmentError>;
    }

    pub trait VectorEnvironment<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KInput: TensorKind<B>,
        KOutput: TensorKind<B>,
    >: Environment + Send + Sync + Sized
    where
        Self: Sized,
    {
        type ResetInfo: IntoIterator<Item = (String, String)>;
        type StepInfo: IntoIterator<Item = (String, String)>;

        fn init_num_envs(&self, num_envs: usize) -> Result<Vec<EnvironmentUuid>, EnvironmentError>;
        fn step(
            &self,
            actions: &[Tensor<B, D_OUT, KOutput>],
        ) -> Result<Vec<Tensor<B, D_IN, KInput>>, EnvironmentError>;
        fn reset(&self) -> Result<Option<Self::ResetInfo>, EnvironmentError>;
    }

    /// Interface for environments where a model can be trained or evaluated.
    ///
    /// Methods are intentionally parameterless: configuration and mutable state live on the
    /// implementing type (often with interior mutability when shared across threads).
    pub trait Environment: Clone
    where
        Self: Sized,
    {
        fn run_environment(&self) -> Result<(), EnvironmentError>;
        fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError>;
    }

    /// Computes a performance signal (for example return-to-go) for training feedback.
    pub trait TrainingPerformanceReturnFn {
        fn calculate_performance_return(&self) -> Result<Box<dyn Any>, EnvironmentError>;
    }
}

pub use traits::*;
