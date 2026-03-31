//! RelayRL learning algorithms and a small **trainer façade** for constructing them.
//!
//! ## Layout
//!
//! - **[`PpoTrainer`]** — independent PPO (and the `IPPO` naming alias for the same type).
//! - **[`ReinforceTrainer`]** — independent REINFORCE (and `IREINFORCE` alias).
//! - **[`MultiagentTrainer`]** — MAPPO / MAREINFORCE; **no** external step kernel type parameter.
//!
//! Use **[`RelayRLTrainer`]** as a convenience namespace with the same constructors as those types.
//!
//! After construction, drive training through **[`AlgorithmTrait`]**: pass trajectories whose type
//! implements [`TrajectoryData`] (for example RelayRL, CSV, or Arrow trajectory wrappers from
//! `relayrl_types`), then call `receive_trajectory`, `train_model`, `log_epoch`, and `save` as your
//! integration loop requires.
//!
//! ## Re-exports
//!
//! This module re-exports algorithm structs and hyperparameter types (`PPOParams`, `MAPPOParams`,
//! …) plus kernel traits ([`PPOKernelTrait`], [`StepKernelTrait`], [`TrainableKernelTrait`]) so
//! callers can supply custom kernels without digging through submodule paths.
//!
//! ## Generics
//!
//! `B`, `InK`, and `OutK` are the Burn backend and tensor kinds your environment uses. Independent
//! trainers also take a kernel type `K`. If the compiler cannot infer them, use a turbofish on the
//! constructor, e.g. `RelayRLTrainer::mappo::<B, InK, OutK>(args, None)?`, or give the result a
//! concrete type in a `let` binding.
//!
//! ## Examples in this file
//!
//! Fenced examples are **illustrative** (`ignore`): substitute your real `B`, `InK`, `OutK`, kernel
//! `K`, and async runtime. They are not run as doctests by default so environments without optional
//! backends (for example libtorch) still build docs cleanly.

pub mod algorithms;
pub mod logging;
pub mod templates;

use burn_tensor::backend::Backend;
use burn_tensor::TensorKind;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use std::path::PathBuf;

pub use algorithms::PPO::{
    IPPOAlgorithm, IPPOParams, MAPPOAlgorithm, MAPPOParams, PPOAlgorithm, PPOKernelTrait,
    PPOParams,
};
pub use algorithms::REINFORCE::{
    IREINFORCEAlgorithm, IREINFORCEParams, MAREINFORCEAlgorithm, MAREINFORCEParams,
    REINFORCEParams, ReinforceAlgorithm,
};
pub use templates::base_algorithm::{
    AlgorithmError, AlgorithmTrait, StepKernelTrait, TrainableKernelTrait, TrajectoryData,
};

/// Shared filesystem and shape arguments for every trainer constructor in this module.
///
/// These values are forwarded into the underlying algorithm implementations (replay sizing,
/// logging paths, and similar runtime configuration).
///
/// # Fields
///
/// - **`env_dir`**: working directory for environment-related assets the algorithm may expect.
/// - **`save_model_path`**: base path or directory used when persisting checkpoints (see
///   [`AlgorithmTrait::save`] on the concrete algorithm).
/// - **`obs_dim`**, **`act_dim`**: observation and action space sizes used when wiring kernels and
///   buffers.
/// - **`buffer_size`**: replay / trajectory buffer capacity for independent trainers; multi-agent
///   trainers use the same field for their buffers.
///
/// # Examples
///
/// ```ignore
/// use std::path::PathBuf;
/// use relayrl_algorithms::TrainerArgs;
///
/// let args = TrainerArgs {
///     env_dir: PathBuf::from("./env"),
///     save_model_path: PathBuf::from("./checkpoints"),
///     obs_dim: 64,
///     act_dim: 8,
///     buffer_size: 1_000_000,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct TrainerArgs {
    /// Directory the algorithm treats as the environment root.
    pub env_dir: PathBuf,
    /// Where to persist models or session output (algorithm-specific).
    pub save_model_path: PathBuf,
    /// Observation dimensionality expected by the policy / value stack.
    pub obs_dim: usize,
    /// Action dimensionality (or discrete action count, depending on your kernel).
    pub act_dim: usize,
    /// Experience buffer capacity in transitions or slots, depending on the algorithm.
    pub buffer_size: usize,
}

/// Describes **which** independent PPO trainer to build, before you supply a kernel `K`.
///
/// The `PPO` and `IPPO` variants differ only in naming and which hyperparameter type you pass;
/// they construct the same underlying algorithm type (`PPOAlgorithm` / `IPPOAlgorithm` are aliases).
///
/// Pair this with [`PpoTrainer::new`] and a kernel `K` that implements [`PPOKernelTrait`] and [`Default`].
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{PpoTrainerSpec, TrainerArgs};
///
/// let args = TrainerArgs { /* ... */ };
/// let spec = PpoTrainerSpec::ppo(args, None);
/// // Then: PpoTrainer::<B, InK, OutK, K>::new(spec, kernel)?
/// ```
pub enum PpoTrainerSpec {
    /// Independent PPO with [`PPOParams`].
    PPO {
        args: TrainerArgs,
        hyperparams: Option<PPOParams>,
    },
    /// Same algorithm as the `PPO` variant, using [`IPPOParams`] for naming consistency.
    IPPO {
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
    },
}

impl PpoTrainerSpec {
    /// Builds a [`PpoTrainerSpec::PPO`] variant.
    pub fn ppo(args: TrainerArgs, hyperparams: Option<PPOParams>) -> Self {
        Self::PPO { args, hyperparams }
    }

    /// Builds a [`PpoTrainerSpec::IPPO`] variant.
    pub fn ippo(args: TrainerArgs, hyperparams: Option<IPPOParams>) -> Self {
        Self::IPPO { args, hyperparams }
    }
}

/// Describes **which** independent REINFORCE trainer to build, before you supply a kernel `K`.
///
/// [`REINFORCE`] and [`IREINFORCE`](Self::IREINFORCE) mirror the PPO case: same underlying type,
/// different hyperparameter names.
///
/// Use with [`ReinforceTrainer::new`] and `K: StepKernelTrait + TrainableKernelTrait + Default`.
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{ReinforceTrainerSpec, TrainerArgs};
///
/// let args = TrainerArgs { /* ... */ };
/// let spec = ReinforceTrainerSpec::reinforce(args, None);
/// ```
pub enum ReinforceTrainerSpec {
    /// Independent REINFORCE with [`REINFORCEParams`].
    REINFORCE {
        args: TrainerArgs,
        hyperparams: Option<REINFORCEParams>,
    },
    /// Same algorithm with [`IREINFORCEParams`].
    IREINFORCE {
        args: TrainerArgs,
        hyperparams: Option<IREINFORCEParams>,
    },
}

impl ReinforceTrainerSpec {
    /// Builds a [`ReinforceTrainerSpec::REINFORCE`] variant.
    pub fn reinforce(args: TrainerArgs, hyperparams: Option<REINFORCEParams>) -> Self {
        Self::REINFORCE { args, hyperparams }
    }

    /// Builds a [`ReinforceTrainerSpec::IREINFORCE`] variant.
    pub fn ireinforce(args: TrainerArgs, hyperparams: Option<IREINFORCEParams>) -> Self {
        Self::IREINFORCE { args, hyperparams }
    }
}

/// Describes **which** multi-agent trainer to build (MAPPO or MAREINFORCE).
///
/// Unlike independent trainers, this spec carries **no** kernel: the multi-agent implementations
/// construct their internal kernels from hyperparameters.
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{MultiagentTrainerSpec, TrainerArgs};
///
/// let args = TrainerArgs { /* ... */ };
/// let spec = MultiagentTrainerSpec::mappo(args, None);
/// // Then: MultiagentTrainer::<B, InK, OutK>::new(spec)?
/// ```
pub enum MultiagentTrainerSpec {
    /// Multi-agent PPO with [`MAPPOParams`].
    MAPPO {
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
    },
    /// Multi-agent REINFORCE with [`MAREINFORCEParams`].
    MAREINFORCE {
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
    },
}

impl MultiagentTrainerSpec {
    /// Builds a [`MultiagentTrainerSpec::MAPPO`] variant.
    pub fn mappo(args: TrainerArgs, hyperparams: Option<MAPPOParams>) -> Self {
        Self::MAPPO { args, hyperparams }
    }

    /// Builds a [`MultiagentTrainerSpec::MAREINFORCE`] variant.
    pub fn mareinforce(args: TrainerArgs, hyperparams: Option<MAREINFORCEParams>) -> Self {
        Self::MAREINFORCE { args, hyperparams }
    }
}

/// Runtime wrapper for **independent** PPO-family algorithms, parameterized by your step kernel `K`.
///
/// - `B`: Burn backend (`Backend` + `BackendMatcher` from `burn_tensor` / `relayrl_types`).
/// - `InK`, `OutK`: input and mask tensor kinds for [`StepKernelTrait`].
/// - `K`: your policy/value kernel; must implement [`PPOKernelTrait`] (includes stepping) and
///   [`Default`] for agent slot initialization inside the algorithm.
///
/// Prefer [`PpoTrainer::ppo`] / [`PpoTrainer::ippo`] when you do not need a separate [`PpoTrainerSpec`]
/// value. Use [`PpoTrainer::new`] when the spec is built dynamically (for example from config).
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{PpoTrainer, RelayRLTrainer, TrainerArgs};
///
/// async fn run<B, InK, OutK, K>(args: TrainerArgs, kernel: K)
/// where
///     // ... your bounds ...
/// {
///     let mut trainer = RelayRLTrainer::ppo(args, None, kernel)?;
///     // let traj: T = ...;
///     // trainer.receive_trajectory(traj).await?;
///     Ok(())
/// }
/// ```
///
/// ## [`AlgorithmTrait`] behavior
///
/// [`AlgorithmTrait`] is implemented for this type: `save`, `receive_trajectory`, `train_model`, and
/// `log_epoch` forward to whichever concrete `PPOAlgorithm` / `IPPOAlgorithm` variant is held. The
/// trajectory type `T` must implement [`TrajectoryData`].
#[allow(clippy::large_enum_variant)]
pub enum PpoTrainer<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: StepKernelTrait<B, InK, OutK>,
> {
    /// Holds a [`PPOAlgorithm`] instance.
    PPO(PPOAlgorithm<B, InK, OutK, K>),
    /// Holds an [`IPPOAlgorithm`] instance (type alias of the same struct as the `PPO` variant).
    IPPO(IPPOAlgorithm<B, InK, OutK, K>),
}

impl<B, InK, OutK, K> PpoTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: PPOKernelTrait<B, InK, OutK> + Default,
{
    /// Constructs a trainer from a [`PpoTrainerSpec`] and a kernel instance.
    pub fn new(spec: PpoTrainerSpec, kernel: K) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            PpoTrainerSpec::PPO { args, hyperparams } => {
                Self::PPO(PPOAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?)
            }
            PpoTrainerSpec::IPPO { args, hyperparams } => {
                Self::IPPO(IPPOAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?)
            }
        };

        Ok(trainer)
    }

    /// Shorthand for `PpoTrainer::new(PpoTrainerSpec::ppo(args, hyperparams), kernel)`.
    pub fn ppo(
        args: TrainerArgs,
        hyperparams: Option<PPOParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(PpoTrainerSpec::ppo(args, hyperparams), kernel)
    }

    /// Shorthand for `PpoTrainer::new(PpoTrainerSpec::ippo(args, hyperparams), kernel)`.
    pub fn ippo(
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(PpoTrainerSpec::ippo(args, hyperparams), kernel)
    }
}

/// Runtime wrapper for **independent** REINFORCE-family algorithms with kernel `K`.
///
/// `K` must support stepping ([`StepKernelTrait`]) and the scalar training hooks ([`TrainableKernelTrait`]),
/// plus [`Default`] for per-agent kernel cloning inside the algorithm.
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{ReinforceTrainer, TrainerArgs};
///
/// fn make(args: TrainerArgs, kernel: K) -> Result<ReinforceTrainer<B, InK, OutK, K>, _> {
///     ReinforceTrainer::reinforce(args, None, kernel)
/// }
/// ```
///
/// ## [`AlgorithmTrait`] behavior
///
/// Implements [`AlgorithmTrait`] by delegating to the inner [`ReinforceAlgorithm`] or
/// [`IREINFORCEAlgorithm`], with `T: TrajectoryData` for incoming trajectories.
#[allow(clippy::large_enum_variant)]
pub enum ReinforceTrainer<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: StepKernelTrait<B, InK, OutK>,
> {
    /// Holds a [`ReinforceAlgorithm`].
    REINFORCE(ReinforceAlgorithm<B, InK, OutK, K>),
    /// Holds an [`IREINFORCEAlgorithm`] (alias of the same struct as the `REINFORCE` variant).
    IREINFORCE(IREINFORCEAlgorithm<B, InK, OutK, K>),
}

impl<B, InK, OutK, K> ReinforceTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: StepKernelTrait<B, InK, OutK> + TrainableKernelTrait + Default,
{
    /// Constructs a trainer from a [`ReinforceTrainerSpec`] and a kernel instance.
    pub fn new(spec: ReinforceTrainerSpec, kernel: K) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            ReinforceTrainerSpec::REINFORCE { args, hyperparams } => Self::REINFORCE(
                ReinforceAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?,
            ),
            ReinforceTrainerSpec::IREINFORCE { args, hyperparams } => Self::IREINFORCE(
                IREINFORCEAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?,
            ),
        };

        Ok(trainer)
    }

    /// Shorthand for `ReinforceTrainer::new(ReinforceTrainerSpec::reinforce(args, hyperparams), kernel)`.
    pub fn reinforce(
        args: TrainerArgs,
        hyperparams: Option<REINFORCEParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(ReinforceTrainerSpec::reinforce(args, hyperparams), kernel)
    }

    /// Shorthand for `ReinforceTrainer::new(ReinforceTrainerSpec::ireinforce(args, hyperparams), kernel)`.
    pub fn ireinforce(
        args: TrainerArgs,
        hyperparams: Option<IREINFORCEParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(ReinforceTrainerSpec::ireinforce(args, hyperparams), kernel)
    }
}

/// Runtime wrapper for **multi-agent** MAPPO and MAREINFORCE.
///
/// No kernel type parameter: hyperparameters drive internal multi-agent kernels. Use
/// [`MultiagentTrainer::mappo`] / [`MultiagentTrainer::mareinforce`] or [`MultiagentTrainer::new`]
/// with a [`MultiagentTrainerSpec`].
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{MultiagentTrainer, TrainerArgs};
///
/// fn open<B, InK, OutK>(args: TrainerArgs) -> Result<MultiagentTrainer<B, InK, OutK>, _> {
///     MultiagentTrainer::mappo(args, None)
/// }
/// ```
///
/// ## [`AlgorithmTrait`] behavior
///
/// Implements [`AlgorithmTrait`] by delegating to the inner [`MAPPOAlgorithm`] or
/// [`MAREINFORCEAlgorithm`].
#[allow(clippy::large_enum_variant)]
pub enum MultiagentTrainer<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
> {
    /// Multi-agent PPO; see field `trainer`.
    MAPPO {
        /// The constructed [`MAPPOAlgorithm`].
        trainer: MAPPOAlgorithm<B, InK, OutK>,
    },
    /// Multi-agent REINFORCE; see field `trainer`.
    MAREINFORCE {
        /// The constructed [`MAREINFORCEAlgorithm`].
        trainer: MAREINFORCEAlgorithm<B, InK, OutK>,
    },
}

impl<B, InK, OutK> MultiagentTrainer<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
{
    /// Builds from a [`MultiagentTrainerSpec`] (MAPPO or MAREINFORCE).
    pub fn new(spec: MultiagentTrainerSpec) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            MultiagentTrainerSpec::MAPPO { args, hyperparams } => Self::MAPPO {
                trainer: MAPPOAlgorithm::<B, InK, OutK>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                )?,
            },
            MultiagentTrainerSpec::MAREINFORCE { args, hyperparams } => Self::MAREINFORCE {
                trainer: MAREINFORCEAlgorithm::<B, InK, OutK>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                )?,
            },
        };

        Ok(trainer)
    }

    /// Shorthand for `MultiagentTrainer::new(MultiagentTrainerSpec::mappo(args, hyperparams))`.
    pub fn mappo(args: TrainerArgs, hyperparams: Option<MAPPOParams>) -> Result<Self, AlgorithmError> {
        Self::new(MultiagentTrainerSpec::mappo(args, hyperparams))
    }

    /// Shorthand for `MultiagentTrainer::new(MultiagentTrainerSpec::mareinforce(args, hyperparams))`.
    pub fn mareinforce(
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
    ) -> Result<Self, AlgorithmError> {
        Self::new(MultiagentTrainerSpec::mareinforce(args, hyperparams))
    }
}

/// Namespace type with static constructors for each trainer family.
///
/// Each method delegates to the corresponding inherent constructor on [`PpoTrainer`],
/// [`ReinforceTrainer`], or [`MultiagentTrainer`]. Use it when you prefer
/// `RelayRLTrainer::mappo(...)` over `MultiagentTrainer::mappo(...)`.
///
/// # Type inference
///
/// If the compiler cannot infer `B`, `InK`, `OutK`, or `K`, spell them explicitly with turbofish, for
/// example `RelayRLTrainer::mappo::<B, InK, OutK>(args, None)?`, or assign the result to a
/// variable with a concrete type.
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{RelayRLTrainer, TrainerArgs};
///
/// fn ppo<B, InK, OutK, K>(args: TrainerArgs, kernel: K) -> _ {
///     RelayRLTrainer::ppo(args, None, kernel)
/// }
///
/// fn mappo<B, InK, OutK>(args: TrainerArgs) -> _ {
///     RelayRLTrainer::mappo::<B, InK, OutK>(args, None)
/// }
/// ```
pub struct RelayRLTrainer;

impl RelayRLTrainer {
    /// See [`PpoTrainer::ppo`].
    pub fn ppo<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<PPOParams>,
        kernel: K,
    ) -> Result<PpoTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: PPOKernelTrait<B, InK, OutK> + Default,
    {
        PpoTrainer::<B, InK, OutK, K>::ppo(args, hyperparams, kernel)
    }

    /// See [`PpoTrainer::ippo`].
    pub fn ippo<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<IPPOParams>,
        kernel: K,
    ) -> Result<PpoTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: PPOKernelTrait<B, InK, OutK> + Default,
    {
        PpoTrainer::<B, InK, OutK, K>::ippo(args, hyperparams, kernel)
    }

    /// See [`ReinforceTrainer::reinforce`].
    pub fn reinforce<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<REINFORCEParams>,
        kernel: K,
    ) -> Result<ReinforceTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: StepKernelTrait<B, InK, OutK> + TrainableKernelTrait + Default,
    {
        ReinforceTrainer::<B, InK, OutK, K>::reinforce(args, hyperparams, kernel)
    }

    /// See [`ReinforceTrainer::ireinforce`].
    pub fn ireinforce<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<IREINFORCEParams>,
        kernel: K,
    ) -> Result<ReinforceTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: StepKernelTrait<B, InK, OutK> + TrainableKernelTrait + Default,
    {
        ReinforceTrainer::<B, InK, OutK, K>::ireinforce(args, hyperparams, kernel)
    }

    /// See [`MultiagentTrainer::mappo`].
    pub fn mappo<B, InK, OutK>(
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
    ) -> Result<MultiagentTrainer<B, InK, OutK>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
    {
        MultiagentTrainer::<B, InK, OutK>::mappo(args, hyperparams)
    }

    /// See [`MultiagentTrainer::mareinforce`].
    pub fn mareinforce<B, InK, OutK>(
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
    ) -> Result<MultiagentTrainer<B, InK, OutK>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
    {
        MultiagentTrainer::<B, InK, OutK>::mareinforce(args, hyperparams)
    }
}

impl<B, InK, OutK, K, T> AlgorithmTrait<T> for PpoTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: PPOKernelTrait<B, InK, OutK> + Default,
    T: TrajectoryData,
{
    fn save(&self, filename: &str) {
        match self {
            Self::PPO(algorithm) => AlgorithmTrait::<T>::save(algorithm, filename),
            Self::IPPO(algorithm) => AlgorithmTrait::<T>::save(algorithm, filename),
        }
    }

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        match self {
            Self::PPO(algorithm) => {
                AlgorithmTrait::<T>::receive_trajectory(algorithm, trajectory).await
            }
            Self::IPPO(algorithm) => {
                AlgorithmTrait::<T>::receive_trajectory(algorithm, trajectory).await
            }
        }
    }

    fn train_model(&mut self) {
        match self {
            Self::PPO(algorithm) => AlgorithmTrait::<T>::train_model(algorithm),
            Self::IPPO(algorithm) => AlgorithmTrait::<T>::train_model(algorithm),
        }
    }

    fn log_epoch(&mut self) {
        match self {
            Self::PPO(algorithm) => AlgorithmTrait::<T>::log_epoch(algorithm),
            Self::IPPO(algorithm) => AlgorithmTrait::<T>::log_epoch(algorithm),
        }
    }
}

impl<B, InK, OutK, K, T> AlgorithmTrait<T> for ReinforceTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: StepKernelTrait<B, InK, OutK> + TrainableKernelTrait + Default,
    T: TrajectoryData,
{
    fn save(&self, filename: &str) {
        match self {
            Self::REINFORCE(algorithm) => AlgorithmTrait::<T>::save(algorithm, filename),
            Self::IREINFORCE(algorithm) => AlgorithmTrait::<T>::save(algorithm, filename),
        }
    }

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        match self {
            Self::REINFORCE(algorithm) => {
                AlgorithmTrait::<T>::receive_trajectory(algorithm, trajectory).await
            }
            Self::IREINFORCE(algorithm) => {
                AlgorithmTrait::<T>::receive_trajectory(algorithm, trajectory).await
            }
        }
    }

    fn train_model(&mut self) {
        match self {
            Self::REINFORCE(algorithm) => AlgorithmTrait::<T>::train_model(algorithm),
            Self::IREINFORCE(algorithm) => AlgorithmTrait::<T>::train_model(algorithm),
        }
    }

    fn log_epoch(&mut self) {
        match self {
            Self::REINFORCE(algorithm) => AlgorithmTrait::<T>::log_epoch(algorithm),
            Self::IREINFORCE(algorithm) => AlgorithmTrait::<T>::log_epoch(algorithm),
        }
    }
}

impl<B, InK, OutK, T> AlgorithmTrait<T> for MultiagentTrainer<B, InK, OutK>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    T: TrajectoryData,
{
    fn save(&self, filename: &str) {
        match self {
            Self::MAPPO { trainer } => AlgorithmTrait::<T>::save(trainer, filename),
            Self::MAREINFORCE { trainer } => AlgorithmTrait::<T>::save(trainer, filename),
        }
    }

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        match self {
            Self::MAPPO { trainer } => {
                AlgorithmTrait::<T>::receive_trajectory(trainer, trajectory).await
            }
            Self::MAREINFORCE { trainer } => {
                AlgorithmTrait::<T>::receive_trajectory(trainer, trajectory).await
            }
        }
    }

    fn train_model(&mut self) {
        match self {
            Self::MAPPO { trainer } => AlgorithmTrait::<T>::train_model(trainer),
            Self::MAREINFORCE { trainer } => AlgorithmTrait::<T>::train_model(trainer),
        }
    }

    fn log_epoch(&mut self) {
        match self {
            Self::MAPPO { trainer } => AlgorithmTrait::<T>::log_epoch(trainer),
            Self::MAREINFORCE { trainer } => AlgorithmTrait::<T>::log_epoch(trainer),
        }
    }
}
