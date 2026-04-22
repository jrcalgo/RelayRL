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

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum EnvironmentKind {
        Scalar,
        Vector,
        Other(String),
        Unknown,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum EnvDType {
        NdArray(EnvNdArrayDType),
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

    pub type EnvInfo = Vec<(String, String)>;

    #[derive(Debug, Clone)]
    pub struct ScalarEnvReset<B: Backend, const D_IN: usize, KInput: TensorKind<B>> {
        pub observation: Tensor<B, D_IN, KInput>,
        pub info: Option<EnvInfo>,
    }

    #[derive(Debug, Clone)]
    pub struct ScalarEnvStep<B: Backend, const D_IN: usize, KInput: TensorKind<B>> {
        pub observation: Tensor<B, D_IN, KInput>,
        pub reward: f32,
        pub terminated: bool,
        pub truncated: bool,
        pub info: Option<EnvInfo>,
    }

    #[derive(Debug, Clone)]
    pub struct VectorEnvReset<B: Backend, const D_IN: usize, KInput: TensorKind<B>> {
        pub env_id: EnvironmentUuid,
        pub observation: Tensor<B, D_IN, KInput>,
        pub info: Option<EnvInfo>,
    }

    #[derive(Debug, Clone)]
    pub struct VectorEnvStep<B: Backend, const D_IN: usize, KInput: TensorKind<B>> {
        pub env_id: EnvironmentUuid,
        pub observation: Tensor<B, D_IN, KInput>,
        pub reward: f32,
        pub terminated: bool,
        pub truncated: bool,
        pub info: Option<EnvInfo>,
    }

    pub type DynVectorEnv<B, const D_IN: usize, const D_OUT: usize, KInput, KOutput> =
        dyn VectorEnvironment<
                B,
                D_IN,
                D_OUT,
                KInput,
                KOutput,
            >;
    pub trait DynScalarEnvironment<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KInput: TensorKind<B>,
        KOutput: TensorKind<B>,
    >:
        ScalarEnvironment<
            B,
            D_IN,
            D_OUT,
            KInput,
            KOutput,
        > + Send
        + Sync
    {
        fn clone_box(&self) -> Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>>;
    }
    impl<B, const D_IN: usize, const D_OUT: usize, KInput, KOutput, T>
        DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput> for T
    where
        B: Backend,
        KInput: TensorKind<B>,
        KOutput: TensorKind<B>,
        T: ScalarEnvironment<
                B,
                D_IN,
                D_OUT,
                KInput,
                KOutput,
            > + Clone
            + Send
            + Sync
            + 'static,
    {
        fn clone_box(&self) -> Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>> {
            Box::new(self.clone())
        }
    }
    impl<B, const D_IN: usize, const D_OUT: usize, KInput, KOutput> Clone
        for Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>>
    where
        B: Backend,
        KInput: TensorKind<B>,
        KOutput: TensorKind<B>,
    {
        fn clone(&self) -> Self {
            self.clone_box()
        }
    }
    pub enum EnvironmentHandle<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KInput: TensorKind<B>,
        KOutput: TensorKind<B>,
    > {
        Scalar(Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>>),
        Vector(Box<DynVectorEnv<B, D_IN, D_OUT, KInput, KOutput>>),
    }

    pub trait ScalarEnvironment<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B>,
        KindOut: TensorKind<B>,
    >: Environment<B, D_IN, D_OUT, KindIn, KindOut> + Send + Sync
    {
        fn step(
            &self,
            action: Tensor<B, D_OUT, KindOut>,
        ) -> Result<ScalarEnvStep<B, D_IN, KindIn>, EnvironmentError>;
        fn reset(&self) -> Result<ScalarEnvReset<B, D_IN, KindIn>, EnvironmentError>;
    }

    pub trait VectorEnvironment<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B>,
        KindOut: TensorKind<B>,
    >: Environment<B, D_IN, D_OUT, KindIn, KindOut> + Send + Sync
    {
        fn init_num_envs(&self, num_envs: usize) -> Result<Vec<EnvironmentUuid>, EnvironmentError>;
        fn step(
            &self,
            actions: &[(EnvironmentUuid, Tensor<B, D_OUT, KindOut>)],
        ) -> Result<Vec<VectorEnvStep<B, D_IN, KindIn>>, EnvironmentError>;
        fn reset(
            &self,
            env_ids: &[EnvironmentUuid],
        ) -> Result<Vec<VectorEnvReset<B, D_IN, KindIn>>, EnvironmentError>;
    }

    /// Interface for environments where a model can be trained or evaluated.
    ///
    /// Methods are intentionally parameterless: configuration and mutable state live on the
    /// implementing type (often with interior mutability when shared across threads).
    pub trait Environment<
        B: Backend,
        const D_IN: usize,
        const D_OUT: usize,
        KindIn: TensorKind<B>,
        KindOut: TensorKind<B>,
    >: Send + Sync
    {
        fn run_environment(&self) -> Result<(), EnvironmentError>;
        fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError>;
        fn observation_dtype(&self) -> EnvDType;
        fn action_dtype(&self) -> EnvDType;
        fn kind(&self) -> EnvironmentKind;
        fn into_handle(self: Box<Self>) -> EnvironmentHandle<B, D_IN, D_OUT, KindIn, KindOut>;
    }

    /// Computes a performance signal (for example return-to-go) for training feedback.
    pub trait TrainingPerformanceReturnFn {
        fn calculate_performance_return(&self) -> Result<Box<dyn Any>, EnvironmentError>;
    }
}

pub use traits::*;
