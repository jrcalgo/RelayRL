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
//!
//! ### End-to-end training flow
//!
//! ```ignore
//! use std::path::PathBuf;
//!
//! use relayrl_algorithms::{AlgorithmError, AlgorithmTrait, RelayRLTrainer, TrainerArgs};
//! use relayrl_types::prelude::trajectory::RelayRLTrajectory;
//!
//! async fn run_training_loop<B, InK, OutK>() -> Result<(), AlgorithmError> {
//!     let args = TrainerArgs {
//!         env_dir: PathBuf::from("./env"),
//!         save_model_path: PathBuf::from("./checkpoints"),
//!         obs_dim: 64,
//!         act_dim: 8,
//!         buffer_size: 1_000_000,
//!     };
//!
//!     let mut trainer = RelayRLTrainer::mappo::<B, InK, OutK>(args, None)?;
//!
//!     let mut trajectory = RelayRLTrajectory::new(1024);
//!     // Populate `trajectory` from your environment loop before handing it to the trainer.
//!
//!     trainer.receive_trajectory(trajectory).await?;
//!     trainer.train_model();
//!     trainer.log_epoch();
//!     trainer.save("epoch-0001");
//!
//!     Ok(())
//! }
//! ```

pub mod algorithms;
pub mod logging;
pub mod templates;

use burn_tensor::TensorKind;
use burn_tensor::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use std::path::PathBuf;

pub use algorithms::DDPG::{
    DDPGAlgorithm, DDPGKernelTrait, DDPGParams, IDDPGAlgorithm, IDDPGParams, MADDPGAlgorithm,
    MADDPGParams, MultiagentDDPGKernelTrait,
};
pub use algorithms::PPO::{
    IPPOAlgorithm, IPPOParams, MAPPOAlgorithm, MAPPOParams, MultiagentPPOKernelTrait, PPOAlgorithm,
    PPOKernelTrait, PPOParams,
};
pub use algorithms::REINFORCE::{
    IREINFORCEAlgorithm, IREINFORCEParams, MAREINFORCEAlgorithm, MAREINFORCEParams,
    MultiagentReinforceKernelTrait, REINFORCEParams, ReinforceAlgorithm,
};
pub use algorithms::TD3::{
    ITD3Algorithm, ITD3Params, MATD3Algorithm, MATD3Params, MultiagentTD3KernelTrait, TD3Algorithm,
    TD3KernelTrait, TD3Params,
};
pub use templates::base_algorithm::{
    AlgorithmError, AlgorithmTrait, MultiagentKernelTrait, StepKernelTrait, TrainableKernelTrait,
    TrajectoryData, WeightProvider,
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
///
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

/// Describes **which** multi-agent trainer to build and carries a user-supplied kernel `K`.
///
/// `K` must implement the algorithm-specific sub-trait for whichever variant is chosen.  
/// When passed to [`MultiagentTrainer::new`], `K` must satisfy all four sub-traits so the
/// compiler can validate it at the call site regardless of which variant is selected.
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{MultiagentTrainerSpec, TrainerArgs};
///
/// let args = TrainerArgs { /* ... */ };
/// let spec = MultiagentTrainerSpec::mappo(args, None, kernel);
/// // Then: MultiagentTrainer::<B, InK, OutK, _>::new(spec)?
/// ```
pub enum MultiagentTrainerSpec<K> {
    /// Multi-agent DDPG with [`MADDPGParams`].
    MADDPG {
        args: TrainerArgs,
        hyperparams: Option<MADDPGParams>,
        kernel: K,
    },
    /// Multi-agent PPO with [`MAPPOParams`].
    MAPPO {
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
        kernel: K,
    },
    /// Multi-agent REINFORCE with [`MAREINFORCEParams`].
    MAREINFORCE {
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
        kernel: K,
    },
    /// Multi-agent TD3 with [`MATD3Params`].
    MATD3 {
        args: TrainerArgs,
        hyperparams: Option<MATD3Params>,
        kernel: K,
    },
}

impl<K> MultiagentTrainerSpec<K> {
    /// Builds a [`MultiagentTrainerSpec::MADDPG`] variant.
    pub fn maddpg(args: TrainerArgs, hyperparams: Option<MADDPGParams>, kernel: K) -> Self {
        Self::MADDPG {
            args,
            hyperparams,
            kernel,
        }
    }

    /// Builds a [`MultiagentTrainerSpec::MAPPO`] variant.
    pub fn mappo(args: TrainerArgs, hyperparams: Option<MAPPOParams>, kernel: K) -> Self {
        Self::MAPPO {
            args,
            hyperparams,
            kernel,
        }
    }

    /// Builds a [`MultiagentTrainerSpec::MAREINFORCE`] variant.
    pub fn mareinforce(
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
        kernel: K,
    ) -> Self {
        Self::MAREINFORCE {
            args,
            hyperparams,
            kernel,
        }
    }

    /// Builds a [`MultiagentTrainerSpec::MATD3`] variant.
    pub fn matd3(args: TrainerArgs, hyperparams: Option<MATD3Params>, kernel: K) -> Self {
        Self::MATD3 {
            args,
            hyperparams,
            kernel,
        }
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

    /// Reset per-actor trajectory counts.
    ///
    /// Prevents `receive_trajectory` from auto-triggering `train_model` when called
    /// from an async context at the start of a new epoch.
    pub fn reset_epoch(&mut self) {
        match self {
            Self::PPO(algorithm) => algorithm.reset_epoch(),
            Self::IPPO(algorithm) => algorithm.reset_epoch(),
        }
    }
}

#[cfg(feature = "ndarray-backend")]
impl<B, InK, OutK, K> PpoTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher<Backend = B>,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: PPOKernelTrait<B, InK, OutK> + WeightProvider + Default,
{
    /// Export the trained policy as an in-memory ONNX model.
    ///
    /// Returns `None` before the first training epoch or when no actors have been
    /// registered.
    pub fn acquire_model_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        match self {
            Self::PPO(algorithm) => algorithm.acquire_model_module(),
            Self::IPPO(algorithm) => algorithm.acquire_model_module(),
        }
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
            ReinforceTrainerSpec::REINFORCE { args, hyperparams } => {
                Self::REINFORCE(ReinforceAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?)
            }
            ReinforceTrainerSpec::IREINFORCE { args, hyperparams } => {
                Self::IREINFORCE(IREINFORCEAlgorithm::<B, InK, OutK, K>::new(
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

    /// No-op for REINFORCE trainers; trajectory counts are managed internally.
    pub fn reset_epoch(&mut self) {}

    /// REINFORCE does not support in-memory ONNX export; always returns `None`.
    pub fn acquire_model_module(&self) -> Option<relayrl_types::model::ModelModule<B>>
    where
        B: BackendMatcher<Backend = B>,
    {
        None
    }
}

/// Runtime wrapper for **multi-agent** algorithms, parameterized by your kernel `K`.
///
/// `K` must implement all four multi-agent sub-traits so that a single `MultiagentTrainer<K>`
/// value can hold any of the four algorithm variants. Use [`MultiagentTrainer::new`] with a
/// [`MultiagentTrainerSpec`] or the convenience constructors (`mappo`, `mareinforce`, `maddpg`,
/// `matd3`).
///
/// # Examples
///
/// ```ignore
/// use relayrl_algorithms::{MultiagentTrainer, TrainerArgs};
///
/// fn open<B, InK, OutK, K>(args: TrainerArgs, kernel: K)
///     -> Result<MultiagentTrainer<B, InK, OutK, K>, _>
/// {
///     MultiagentTrainer::mappo(args, None, kernel)
/// }
/// ```
///
/// ## [`AlgorithmTrait`] behavior
///
/// Implements [`AlgorithmTrait`] by delegating to the inner algorithm variant.
#[allow(clippy::large_enum_variant)]
pub enum MultiagentTrainer<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: MultiagentPPOKernelTrait<B, InK, OutK>
        + MultiagentReinforceKernelTrait<B, InK, OutK>
        + MultiagentDDPGKernelTrait<B, InK, OutK>
        + MultiagentTD3KernelTrait<B, InK, OutK>,
> {
    /// Multi-agent DDPG; see field `trainer`.
    MADDPG {
        /// The constructed [`MADDPGAlgorithm`].
        trainer: MADDPGAlgorithm<B, InK, OutK, K>,
    },
    /// Multi-agent PPO; see field `trainer`.
    MAPPO {
        /// The constructed [`MAPPOAlgorithm`].
        trainer: MAPPOAlgorithm<B, InK, OutK, K>,
    },
    /// Multi-agent REINFORCE; see field `trainer`.
    MAREINFORCE {
        /// The constructed [`MAREINFORCEAlgorithm`].
        trainer: MAREINFORCEAlgorithm<B, InK, OutK, K>,
    },
    /// Multi-agent TD3; see field `trainer`.
    MATD3 {
        /// The constructed [`MATD3Algorithm`].
        trainer: MATD3Algorithm<B, InK, OutK, K>,
    },
}

impl<B, InK, OutK, K> MultiagentTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: MultiagentPPOKernelTrait<B, InK, OutK>
        + MultiagentReinforceKernelTrait<B, InK, OutK>
        + MultiagentDDPGKernelTrait<B, InK, OutK>
        + MultiagentTD3KernelTrait<B, InK, OutK>
        + Default,
{
    /// Builds from a [`MultiagentTrainerSpec`].
    pub fn new(spec: MultiagentTrainerSpec<K>) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            MultiagentTrainerSpec::MADDPG {
                args,
                hyperparams,
                kernel,
            } => Self::MADDPG {
                trainer: MADDPGAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?,
            },
            MultiagentTrainerSpec::MAPPO {
                args,
                hyperparams,
                kernel,
            } => Self::MAPPO {
                trainer: MAPPOAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?,
            },
            MultiagentTrainerSpec::MAREINFORCE {
                args,
                hyperparams,
                kernel,
            } => Self::MAREINFORCE {
                trainer: MAREINFORCEAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?,
            },
            MultiagentTrainerSpec::MATD3 {
                args,
                hyperparams,
                kernel,
            } => Self::MATD3 {
                trainer: MATD3Algorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?,
            },
        };

        Ok(trainer)
    }

    /// Shorthand for building a MAPPO trainer.
    pub fn mappo(
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(MultiagentTrainerSpec::mappo(args, hyperparams, kernel))
    }

    /// Shorthand for building a MAREINFORCE trainer.
    pub fn mareinforce(
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(MultiagentTrainerSpec::mareinforce(
            args,
            hyperparams,
            kernel,
        ))
    }

    /// Shorthand for building a MADDPG trainer.
    pub fn maddpg(
        args: TrainerArgs,
        hyperparams: Option<MADDPGParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(MultiagentTrainerSpec::maddpg(args, hyperparams, kernel))
    }

    /// Shorthand for building a MATD3 trainer.
    pub fn matd3(
        args: TrainerArgs,
        hyperparams: Option<MATD3Params>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(MultiagentTrainerSpec::matd3(args, hyperparams, kernel))
    }

    /// No-op for multi-agent trainers; trajectory counts are managed internally.
    pub fn reset_epoch(&mut self) {}
}

#[cfg(feature = "ndarray-backend")]
impl<B, InK, OutK, K> MultiagentTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher<Backend = B>,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: MultiagentPPOKernelTrait<B, InK, OutK>
        + MultiagentReinforceKernelTrait<B, InK, OutK>
        + MultiagentDDPGKernelTrait<B, InK, OutK>
        + MultiagentTD3KernelTrait<B, InK, OutK>
        + WeightProvider
        + Default,
{
    /// Export the trained policy as an in-memory ONNX model.
    pub fn acquire_model_module(&self) -> Option<relayrl_types::model::ModelModule<B>> {
        match self {
            Self::MADDPG { trainer } => trainer.acquire_model_module(),
            Self::MAPPO { trainer } => trainer.acquire_model_module(),
            Self::MAREINFORCE { trainer } => trainer.acquire_model_module(),
            Self::MATD3 { trainer } => trainer.acquire_model_module(),
        }
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
    pub fn mappo<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<MAPPOParams>,
        kernel: K,
    ) -> Result<MultiagentTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: MultiagentPPOKernelTrait<B, InK, OutK>
            + MultiagentReinforceKernelTrait<B, InK, OutK>
            + MultiagentDDPGKernelTrait<B, InK, OutK>
            + MultiagentTD3KernelTrait<B, InK, OutK>
            + Default,
    {
        MultiagentTrainer::<B, InK, OutK, K>::mappo(args, hyperparams, kernel)
    }

    /// See [`MultiagentTrainer::mareinforce`].
    pub fn mareinforce<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<MAREINFORCEParams>,
        kernel: K,
    ) -> Result<MultiagentTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: MultiagentPPOKernelTrait<B, InK, OutK>
            + MultiagentReinforceKernelTrait<B, InK, OutK>
            + MultiagentDDPGKernelTrait<B, InK, OutK>
            + MultiagentTD3KernelTrait<B, InK, OutK>
            + Default,
    {
        MultiagentTrainer::<B, InK, OutK, K>::mareinforce(args, hyperparams, kernel)
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

impl<B, InK, OutK, K, T> AlgorithmTrait<T> for MultiagentTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: MultiagentPPOKernelTrait<B, InK, OutK>
        + MultiagentReinforceKernelTrait<B, InK, OutK>
        + MultiagentDDPGKernelTrait<B, InK, OutK>
        + MultiagentTD3KernelTrait<B, InK, OutK>
        + Default,
    T: TrajectoryData,
{
    fn save(&self, filename: &str) {
        match self {
            Self::MADDPG { trainer } => AlgorithmTrait::<T>::save(trainer, filename),
            Self::MAPPO { trainer } => AlgorithmTrait::<T>::save(trainer, filename),
            Self::MAREINFORCE { trainer } => AlgorithmTrait::<T>::save(trainer, filename),
            Self::MATD3 { trainer } => AlgorithmTrait::<T>::save(trainer, filename),
        }
    }

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        match self {
            Self::MADDPG { trainer } => {
                AlgorithmTrait::<T>::receive_trajectory(trainer, trajectory).await
            }
            Self::MAPPO { trainer } => {
                AlgorithmTrait::<T>::receive_trajectory(trainer, trajectory).await
            }
            Self::MAREINFORCE { trainer } => {
                AlgorithmTrait::<T>::receive_trajectory(trainer, trajectory).await
            }
            Self::MATD3 { trainer } => {
                AlgorithmTrait::<T>::receive_trajectory(trainer, trajectory).await
            }
        }
    }

    fn train_model(&mut self) {
        match self {
            Self::MADDPG { trainer } => AlgorithmTrait::<T>::train_model(trainer),
            Self::MAPPO { trainer } => AlgorithmTrait::<T>::train_model(trainer),
            Self::MAREINFORCE { trainer } => AlgorithmTrait::<T>::train_model(trainer),
            Self::MATD3 { trainer } => AlgorithmTrait::<T>::train_model(trainer),
        }
    }

    fn log_epoch(&mut self) {
        match self {
            Self::MADDPG { trainer } => AlgorithmTrait::<T>::log_epoch(trainer),
            Self::MAPPO { trainer } => AlgorithmTrait::<T>::log_epoch(trainer),
            Self::MAREINFORCE { trainer } => AlgorithmTrait::<T>::log_epoch(trainer),
            Self::MATD3 { trainer } => AlgorithmTrait::<T>::log_epoch(trainer),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DdpgTrainer
// ─────────────────────────────────────────────────────────────────────────────

/// Which independent DDPG trainer to build.
pub enum DdpgTrainerSpec {
    /// Independent DDPG with [`DDPGParams`].
    DDPG {
        args: TrainerArgs,
        hyperparams: Option<DDPGParams>,
    },
    /// Same algorithm named with the `I`-prefix convention.
    IDDPG {
        args: TrainerArgs,
        hyperparams: Option<IDDPGParams>,
    },
}

impl DdpgTrainerSpec {
    pub fn ddpg(args: TrainerArgs, hyperparams: Option<DDPGParams>) -> Self {
        Self::DDPG { args, hyperparams }
    }

    pub fn iddpg(args: TrainerArgs, hyperparams: Option<IDDPGParams>) -> Self {
        Self::IDDPG { args, hyperparams }
    }
}

/// Runtime wrapper for independent DDPG-family algorithms with kernel `K`.
#[allow(clippy::large_enum_variant)]
pub enum DdpgTrainer<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: StepKernelTrait<B, InK, OutK>,
> {
    DDPG(DDPGAlgorithm<B, InK, OutK, K>),
    IDDPG(IDDPGAlgorithm<B, InK, OutK, K>),
}

impl<B, InK, OutK, K> DdpgTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: DDPGKernelTrait<B, InK, OutK> + Default,
{
    pub fn new(spec: DdpgTrainerSpec, kernel: K) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            DdpgTrainerSpec::DDPG { args, hyperparams } => {
                Self::DDPG(DDPGAlgorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?)
            }
            DdpgTrainerSpec::IDDPG { args, hyperparams } => {
                Self::IDDPG(IDDPGAlgorithm::<B, InK, OutK, K>::new(
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

    pub fn ddpg(
        args: TrainerArgs,
        hyperparams: Option<DDPGParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(DdpgTrainerSpec::ddpg(args, hyperparams), kernel)
    }

    pub fn iddpg(
        args: TrainerArgs,
        hyperparams: Option<IDDPGParams>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(DdpgTrainerSpec::iddpg(args, hyperparams), kernel)
    }

    pub fn reset_epoch(&mut self) {
        match self {
            Self::DDPG(a) => a.reset_epoch(),
            Self::IDDPG(a) => a.reset_epoch(),
        }
    }
}

impl<B, InK, OutK, K, T> AlgorithmTrait<T> for DdpgTrainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: DDPGKernelTrait<B, InK, OutK> + Default,
    T: TrajectoryData,
{
    fn save(&self, filename: &str) {
        match self {
            Self::DDPG(a) => AlgorithmTrait::<T>::save(a, filename),
            Self::IDDPG(a) => AlgorithmTrait::<T>::save(a, filename),
        }
    }

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        match self {
            Self::DDPG(a) => AlgorithmTrait::<T>::receive_trajectory(a, trajectory).await,
            Self::IDDPG(a) => AlgorithmTrait::<T>::receive_trajectory(a, trajectory).await,
        }
    }

    fn train_model(&mut self) {
        match self {
            Self::DDPG(a) => AlgorithmTrait::<T>::train_model(a),
            Self::IDDPG(a) => AlgorithmTrait::<T>::train_model(a),
        }
    }

    fn log_epoch(&mut self) {
        match self {
            Self::DDPG(a) => AlgorithmTrait::<T>::log_epoch(a),
            Self::IDDPG(a) => AlgorithmTrait::<T>::log_epoch(a),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Td3Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Which independent TD3 trainer to build.
pub enum Td3TrainerSpec {
    TD3 {
        args: TrainerArgs,
        hyperparams: Option<TD3Params>,
    },
    ITD3 {
        args: TrainerArgs,
        hyperparams: Option<ITD3Params>,
    },
}

impl Td3TrainerSpec {
    pub fn td3(args: TrainerArgs, hyperparams: Option<TD3Params>) -> Self {
        Self::TD3 { args, hyperparams }
    }

    pub fn itd3(args: TrainerArgs, hyperparams: Option<ITD3Params>) -> Self {
        Self::ITD3 { args, hyperparams }
    }
}

/// Runtime wrapper for independent TD3-family algorithms with kernel `K`.
#[allow(clippy::large_enum_variant)]
pub enum Td3Trainer<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: StepKernelTrait<B, InK, OutK>,
> {
    TD3(TD3Algorithm<B, InK, OutK, K>),
    ITD3(ITD3Algorithm<B, InK, OutK, K>),
}

impl<B, InK, OutK, K> Td3Trainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: TD3KernelTrait<B, InK, OutK> + Default,
{
    pub fn new(spec: Td3TrainerSpec, kernel: K) -> Result<Self, AlgorithmError> {
        let trainer = match spec {
            Td3TrainerSpec::TD3 { args, hyperparams } => {
                Self::TD3(TD3Algorithm::<B, InK, OutK, K>::new(
                    hyperparams,
                    &args.env_dir,
                    &args.save_model_path,
                    args.obs_dim,
                    args.act_dim,
                    args.buffer_size,
                    kernel,
                )?)
            }
            Td3TrainerSpec::ITD3 { args, hyperparams } => {
                Self::ITD3(ITD3Algorithm::<B, InK, OutK, K>::new(
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

    pub fn td3(
        args: TrainerArgs,
        hyperparams: Option<TD3Params>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(Td3TrainerSpec::td3(args, hyperparams), kernel)
    }

    pub fn itd3(
        args: TrainerArgs,
        hyperparams: Option<ITD3Params>,
        kernel: K,
    ) -> Result<Self, AlgorithmError> {
        Self::new(Td3TrainerSpec::itd3(args, hyperparams), kernel)
    }

    pub fn reset_epoch(&mut self) {
        match self {
            Self::TD3(a) => a.reset_epoch(),
            Self::ITD3(a) => a.reset_epoch(),
        }
    }
}

impl<B, InK, OutK, K, T> AlgorithmTrait<T> for Td3Trainer<B, InK, OutK, K>
where
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
    K: TD3KernelTrait<B, InK, OutK> + Default,
    T: TrajectoryData,
{
    fn save(&self, filename: &str) {
        match self {
            Self::TD3(a) => AlgorithmTrait::<T>::save(a, filename),
            Self::ITD3(a) => AlgorithmTrait::<T>::save(a, filename),
        }
    }

    async fn receive_trajectory(&mut self, trajectory: T) -> Result<bool, AlgorithmError> {
        match self {
            Self::TD3(a) => AlgorithmTrait::<T>::receive_trajectory(a, trajectory).await,
            Self::ITD3(a) => AlgorithmTrait::<T>::receive_trajectory(a, trajectory).await,
        }
    }

    fn train_model(&mut self) {
        match self {
            Self::TD3(a) => AlgorithmTrait::<T>::train_model(a),
            Self::ITD3(a) => AlgorithmTrait::<T>::train_model(a),
        }
    }

    fn log_epoch(&mut self) {
        match self {
            Self::TD3(a) => AlgorithmTrait::<T>::log_epoch(a),
            Self::ITD3(a) => AlgorithmTrait::<T>::log_epoch(a),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RelayRLTrainer extensions for DDPG and TD3
// ─────────────────────────────────────────────────────────────────────────────

impl RelayRLTrainer {
    /// See [`DdpgTrainer::ddpg`].
    pub fn ddpg<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<DDPGParams>,
        kernel: K,
    ) -> Result<DdpgTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: DDPGKernelTrait<B, InK, OutK> + Default,
    {
        DdpgTrainer::<B, InK, OutK, K>::ddpg(args, hyperparams, kernel)
    }

    /// See [`DdpgTrainer::iddpg`].
    pub fn iddpg<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<IDDPGParams>,
        kernel: K,
    ) -> Result<DdpgTrainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: DDPGKernelTrait<B, InK, OutK> + Default,
    {
        DdpgTrainer::<B, InK, OutK, K>::iddpg(args, hyperparams, kernel)
    }

    /// See [`Td3Trainer::td3`].
    pub fn td3<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<TD3Params>,
        kernel: K,
    ) -> Result<Td3Trainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: TD3KernelTrait<B, InK, OutK> + Default,
    {
        Td3Trainer::<B, InK, OutK, K>::td3(args, hyperparams, kernel)
    }

    /// See [`Td3Trainer::itd3`].
    pub fn itd3<B, InK, OutK, K>(
        args: TrainerArgs,
        hyperparams: Option<ITD3Params>,
        kernel: K,
    ) -> Result<Td3Trainer<B, InK, OutK, K>, AlgorithmError>
    where
        B: Backend + BackendMatcher,
        InK: TensorKind<B>,
        OutK: TensorKind<B>,
        K: TD3KernelTrait<B, InK, OutK> + Default,
    {
        Td3Trainer::<B, InK, OutK, K>::itd3(args, hyperparams, kernel)
    }
}
