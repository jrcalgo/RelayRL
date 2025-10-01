use crate::types::Hyperparams;
use crate::types::action::{RL4SysAction, RL4SysData, TensorData};
use rand::Rng;
use std::collections::HashMap;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    },
};
use tch::{CModule, Device, IValue, Kind, Tensor, no_grad};
use tokio::sync::{RwLock, RwLockReadGuard};
use uuid::Uuid;

/// **Client Modules**: Handles client-side runtime coordination and actor management.
///
/// The client module provides a comprehensive runtime system for managing RL agents:
/// - `agent`: Public interface for agent implementations and wrappers
/// - `runtime`: Internal runtime system including:
///   - `actor`: Individual agent actor implementations
///   - `coordination`: Manages lifecycle, scaling, metrics, and state across actors
///   - `router`: Message routing between actors and transport layers
///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
pub mod client {
    pub mod agent;
    pub(crate) mod runtime {
        pub(crate) mod actor;
        pub(crate) mod coordination {
            pub(crate) mod coordinator;
            pub(crate) mod lifecycle_manager;
            pub(crate) mod metrics_manager;
            pub(crate) mod scale_manager;
            pub(crate) mod state_manager;
        }
        pub(crate) mod router;
        #[cfg(feature = "networks")]
        pub(crate) mod transport;
    }
}

/// **Server Modules**: Implements RelayRL training servers and communication channels.
///
/// The server module provides a comprehensive runtime system for managing RL training:
/// - `training_server`: Public interface for training server implementations
/// - `runtime`: Internal runtime system including:
///   - `coordination`: Manages lifecycle, scaling, metrics, and state for training
///   - `python_subprocesses`: Python subprocess management for server interactions
///     - `python_algorithm_request`: Manages Python-command-based algorithm interactions
///     - `python_training_tensorboard`: Manages TensorBoard integration for training visualization
///   - `transport`: Network transport implementations (gRPC, ZeroMQ)
///   - `router`: Message routing between workers and transport layers
///   - `worker`: Individual training worker implementations
pub mod server {
    pub mod training_server;
    pub(crate) mod runtime {
        pub(crate) mod coordination {
            pub(crate) mod coordinator;
            pub(crate) mod lifecycle_manager;
            pub(crate) mod metrics_manager;
            pub(crate) mod scale_manager;
            pub(crate) mod state_manager;
        }
        #[cfg(feature = "networks")]
        pub(crate) mod transport;
        pub(crate) mod router;
        pub(crate) mod worker;
    }
}

pub fn random_uuid(base: u32) -> Uuid {
    let random_num = base * rand::thread_rng().gen_range(11..100)
        + base
        + 1 * rand::thread_rng().gen_range(11..100)
        - rand::thread_rng().gen_range(1..10);
    Uuid::new_v8([random_num as u8; 16])
}

/// Converts a generic dictionary (represented as a Vec of (IValue, IValue)) into a HashMap
/// with String keys and RL4SysData values.
///
/// The function iterates over each key-value pair in the generic dictionary. If the key is a
/// string and the value is one of the supported types (Tensor, Int, Double), it converts the value
/// into the corresponding RL4SysData variant. For tensors, the value is first converted to a Float
/// tensor before being transformed into TensorData.
///
/// # Arguments
///
/// * `dict` - A reference to a vector of (IValue, IValue) tuples representing a generic dictionary.
///
/// # Returns
///
/// An Option containing a HashMap with String keys and RL4SysData values if conversion is successful;
/// otherwise, None.
pub fn convert_generic_dict(dict: &Vec<(IValue, IValue)>) -> Option<HashMap<String, RL4SysData>> {
    let mut map: HashMap<String, RL4SysData> = HashMap::new();

    for (k, v) in dict {
        if let IValue::String(s) = k {
            if let IValue::Tensor(tensor) = v {
                map.insert(
                    s.clone(),
                    RL4SysData::Tensor(
                        TensorData::try_from(&tensor.to_kind(Kind::Float))
                            .expect("Failed to convert tensor to TensorData"),
                    ),
                );
            } else if let IValue::Int(i) = v {
                map.insert(
                    s.clone(),
                    RL4SysData::Int((*i).try_into().expect("Failed to convert int to i32")),
                );
            } else if let IValue::Double(f) = v {
                map.insert(s.clone(), RL4SysData::Double(*f));
            }
        }
    }

    Some(map)
}

/// Validates a TorchScript model (CModule) by running a forward pass with a dummy tensor.
///
/// This function checks that:
/// 1. The model's forward pass returns a tuple (IValue::Tuple) with exactly two elements.
/// 2. The first element is a Tensor.
/// 3. The second element is a dictionary (IValue::GenericDict) that could be empty.
///
/// # Arguments
///
/// * `model` - A reference to the TorchScript model (CModule) to be validated.
/// * `input_dim` - The dimensionality of the input vector.
pub fn validate_model(model: &CModule) {
    // check if input_dim is a model attribute.
    let input_dim: IValue = model
        .method_is::<IValue>("get_input_dim", &[])
        .expect("Failed to get input dimension");
    let input_dim_usize: usize = if let IValue::Int(dim) = input_dim {
        if dim < 0 {
            panic!("Input dimension must be non-negative");
        }
        usize::try_from(dim).expect("Input dimension too large")
    } else {
        panic!("Input dimension must be an integer");
    };

    // check if output_dim is a model attribute.
    let output_dim: IValue = model
        .method_is::<IValue>("get_output_dim", &[])
        .expect("Failed to get output dimension");
    let output_dim_usize: usize = if let IValue::Int(dim) = output_dim {
        if dim < 0 {
            panic!("Output dimension must be non-negative");
        }
        usize::try_from(dim).expect("Output dimension too large")
    } else {
        panic!("Output dimension must be an integer");
    };

    // Create a dummy input vector filled with zeros.
    let input_test_vec: Vec<f64> = vec![0.0; input_dim_usize];
    // Convert the vector into a tensor and reshape it to have a batch dimension.
    let input_test_tensor: Tensor = Tensor::f_from_slice(&input_test_vec)
        .expect("Failed to convert slice to tensor")
        .reshape([1, input_dim_usize as i64]);
    // Create a dummy output vector filled with zeros.
    let output_test_vec: Vec<f64> = vec![0.0; output_dim_usize];
    // Convert the vector into a tensor and reshape it to have a batch dimension.
    let output_test_tensor: Tensor = Tensor::f_from_slice(&output_test_vec)
        .expect("Failed to convert slice to tensor")
        .reshape([1, output_dim_usize as i64]);
    // Convert the tensor to a device tensor.
    let obs_tensor: Tensor = input_test_tensor.to_device(Device::Cpu).contiguous();
    let mask_tensor: Tensor = output_test_tensor.to_device(Device::Cpu).contiguous();
    // Construct IValue tensor input vec
    let obs_ivalue = IValue::Tensor(obs_tensor.to_kind(Kind::Float));
    let mask_ivalue = IValue::Tensor(mask_tensor.to_kind(Kind::Float));
    let test_input: Vec<IValue> = vec![obs_ivalue, mask_ivalue];

    // Run the forward pass (step).
    let output: IValue = no_grad(|| model.method_is::<IValue>("step", &test_input))
        .expect("Failed to run forward 'step' pass");

    // Validate that the output is a tuple.
    match output {
        IValue::Tuple(ref values) => {
            // Assert that the tuple has exactly two elements.
            assert_eq!(
                values.len(),
                2,
                "Model forward must return a tuple of length 2"
            );

            // Check that the first element is a Tensor.
            if let IValue::Tensor(ref _tensor) = values[0] {
                // Optionally: Add additional checks for tensor shape or content here.
            } else {
                panic!("First element of tuple must be a Tensor");
            }

            // Check that the second element is a dictionary (GenericDict).
            if let IValue::GenericDict(ref dict) = values[1] {
                assert!(
                    !dict.is_empty(),
                    "Second element of tuple must be a non-empty dictionary"
                );
            } else {
                panic!("Second element of tuple must be a Dictionary");
            }
        }
        _ => panic!("Model forward must return a tuple"),
    }
}

/// Extend for future utility with other transport protocols (extend transport.rs accordingly)
#[derive(Clone, Copy, Debug)]
pub enum TransportType {
    GRPC,
    ZMQ,
}

type Model = Arc<CModule>;

/// A `CModule` plus a version counter, all wrapped in async locks so you can swap it at runtime.
pub struct HotReloadableModel {
    inner: RwLock<Model>,
    version: Arc<AtomicI64>,
    device: Device,
}

impl HotReloadableModel {
    /// Load the initial model from disk.
    pub async fn new_from_path<P: AsRef<Path>>(
        path: P,
        device: Device,
    ) -> Result<Self, tch::TchError> {
        let module: Model = Arc::new(CModule::load(path)?);
        Ok(Self {
            inner: RwLock::new(module),
            version: Arc::from(AtomicI64::new(0)),
            device,
        })
    }

    pub async fn new_from_model(model: CModule, device: Device) -> Result<Self, tch::TchError> {
        let module: Model = Arc::new(model);
        Ok(Self {
            inner: RwLock::new(module),
            version: Arc::from(AtomicI64::new(0)),
            device,
        })
    }

    /// Atomically swap in a new module from the given path.
    /// Returns the new version number.
    pub async fn reload<P: AsRef<Path>>(
        &self,
        path: P,
        version: i64,
    ) -> Result<i64, tch::TchError> {
        // 1) load outside the lock so you don’t block readers on I/O
        let new_model = Arc::new(CModule::load(path)?);
        {
            let mut guard = self.inner.write().await;
            *guard = new_model;
        }
        // 3) bump version
        self.version.store(version, Ordering::SeqCst);
        let v = self.version.load(Ordering::SeqCst);
        Ok(v)
    }

    /// Run inference with the currently loaded module.
    pub async fn forward(
        &self,
        observation: Tensor,
        mask: Tensor,
        reward: f32,
    ) -> Result<RL4SysAction, String> {
        let action_result = {
            // Lock the model
            let model_guard: RwLockReadGuard<Model> = self.inner.read().await;
            let model: &Arc<CModule> = &*model_guard;

            // Move Tensors to CPU contiguously
            let obs: Tensor = observation.to_device(Device::Cpu).contiguous();
            let mask: Tensor = mask.to_device(Device::Cpu).contiguous();

            // Convert Tensors -> IValue
            let obs_ivalue = IValue::Tensor(obs.to_kind(Kind::Float));
            let mask_ivalue = IValue::Tensor(mask.to_kind(Kind::Float));
            let inputs: Vec<IValue> = vec![obs_ivalue, mask_ivalue];

            // Execute step(...) in a blocking context
            no_grad(|| {
                let output_ivalue: IValue = model
                    .method_is("step", &inputs)
                    .map_err(|e| format!("Failed to call model.step: {}", e))?;

                // Expect output to be a 2-tuple: (action_tensor, data_dict)
                let outputs: &Vec<IValue> = match output_ivalue {
                    IValue::Tuple(ref tup) if tup.len() == 2 => tup,
                    _ => return Err("step() did not return (action, data_dict) tuple".to_string()),
                };

                // Extract action
                let action_tensor: Tensor = if let IValue::Tensor(t) = &outputs[0] {
                    t.to_kind(Kind::Float)
                } else {
                    Tensor::zeros([], (Kind::Float, Device::Cpu))
                };

                // Convert data
                let data_dict: Option<HashMap<String, RL4SysData>> = match &outputs[1] {
                    IValue::GenericDict(dict) => {
                        Some(convert_generic_dict(dict).expect("Failed to convert data dict"))
                    }
                    _ => Some(HashMap::new()),
                };

                // Build RL4SysAction with Tensors turned into `TensorData`
                let obs_td: TensorData =
                    TensorData::try_from(&obs).expect("Failed to convert obs to TensorData");
                let act_td: TensorData = TensorData::try_from(&action_tensor)
                    .expect("Failed to convert act to TensorData");
                let mask_td: TensorData =
                    TensorData::try_from(&mask).expect("Failed to convert mask to TensorData");

                let r4sa: RL4SysAction = RL4SysAction::new(
                    Some(obs_td),
                    Some(act_td),
                    Some(mask_td),
                    reward,
                    data_dict,
                    false,
                );

                Ok(r4sa)
            })
        };
        action_result
    }

    /// Inspect the current model version.
    pub fn version(&self) -> i64 {
        self.version.load(Ordering::SeqCst)
    }
}

/// Parses hyperparameter arguments into a HashMap.
///
/// The function accepts an optional `Hyperparams` enum value, which may be provided as either
/// a map or a vector of argument strings. It returns a HashMap mapping hyperparameter keys to
/// their corresponding string values.
///
/// # Arguments
///
/// * `hyperparams` - An optional [Hyperparams] enum that contains either a map or vector of strings.
///
/// # Returns
///
/// A [HashMap] where the keys and values are both strings.
pub fn parse_args(hyperparams: &Option<Hyperparams>) -> HashMap<String, String> {
    let mut hyperparams_map: HashMap<String, String> = HashMap::new();

    match hyperparams {
        Some(Hyperparams::Map(map)) => {
            for (key, value) in map {
                hyperparams_map.insert(key.to_string(), value.to_string());
            }
        }
        Some(Hyperparams::Args(args)) => {
            for arg in args {
                // Split the argument string on '=' or ' ' if possible.
                let split: Vec<&str> = if arg.contains("=") {
                    arg.split('=').collect()
                } else if arg.contains(' ') {
                    arg.split(' ').collect()
                } else {
                    panic!(
                        "[TrainingServer - new] Invalid hyperparameter argument: {}",
                        arg
                    );
                };
                // Ensure exactly two parts are obtained: key and value.
                if split.len() != 2 {
                    panic!(
                        "[TrainingServer - new] Invalid hyperparameter argument: {}",
                        arg
                    );
                }
                hyperparams_map.insert(split[0].to_string(), split[1].to_string());
            }
        }
        None => {}
    }

    hyperparams_map
}
