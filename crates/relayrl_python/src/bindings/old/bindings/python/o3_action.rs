//! This module defines a Python-exposed wrapper class (`PyRelayRLAction`) around the Rust
//! `RelayRLAction` struct. It uses PyO3 for interop, allowing Python code to create, inspect,
//! and manipulate RelayRLAction objects (which may contain serialized tensor data).

use pyo3::{Bound, IntoPyObject, Py, PyAny, Python, pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tch::Tensor;

use crate::types::action::{RelayRLAction, RelayRLData, TensorData};

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A Python-facing wrapper around the core `RelayRLAction` struct.
///
/// This struct holds an internal `RelayRLAction` and provides various methods to construct
/// and extract data from an RelayRLAction in a Python-compatible manner. It also includes
/// serialization helpers for translating between Python objects (numpy arrays, etc.) and
/// the Rust tensor representation.
#[pyclass(name = "RelayRLAction")]
#[derive(Serialize, Deserialize, Debug)]
pub struct PyRelayRLAction {
    /// The internal Rust RelayRLAction object containing observation, action, mask,
    /// reward, auxiliary data, and done/reward-updated flags.
    pub inner: RelayRLAction,
}

#[pymethods]
impl PyRelayRLAction {
    /// Creates a new `PyRelayRLAction` from Python-provided observation, action, mask,
    /// reward, data, done flag, and reward_updated flag.
    ///
    /// This method expects the `obs`, `act`, and `mask` arguments to be either
    /// a PyTorch tensor or a NumPy array in Python. It converts them into serialized
    /// `TensorData` objects for the underlying RelayRLAction. The `data` argument,
    /// if provided, should be a Python dictionary containing auxiliary data.
    ///
    /// # Arguments
    ///
    /// * `obs` - Optional Python object representing the observation (PyTorch or NumPy).
    /// * `act` - Optional Python object representing the action (PyTorch or NumPy).
    /// * `mask` - Optional Python object representing the mask (PyTorch or NumPy).
    /// * `rew` - A float value for the reward.
    /// * `data` - An optional Python dictionary of auxiliary data.
    /// * `done` - A boolean indicating whether this action terminates an episode.
    /// * `reward_updated` - A boolean indicating if the reward was externally updated.
    #[new]
    #[pyo3(signature = (
        obs = None,
        act = None,
        mask = None,
        rew = 0.0,
        data = None,
        done = false,
        reward_updated = false
    ))]
    fn new(
        obs: Option<&Bound<'_, PyAny>>,
        act: Option<&Bound<'_, PyAny>>,
        mask: Option<&Bound<'_, PyAny>>,
        rew: f32,
        data: Option<&Bound<'_, PyDict>>,
        done: bool,
        reward_updated: bool,
    ) -> Self {
        // Convert Python objects (PyTorch / NumPy) to TensorData
        let obs_tensor_data: Option<TensorData> =
            Some(Self::ndarray_to_tensor_data(obs)).expect("Failed to convert obs");
        let act_tensor_data: Option<TensorData> =
            Some(Self::ndarray_to_tensor_data(act)).expect("Failed to convert act");
        let mask_tensor_data: Option<TensorData> =
            Some(Self::ndarray_to_tensor_data(mask)).expect("Failed to convert mask");

        // Convert the Python dictionary of data to RelayRLData.
        let data: Option<HashMap<String, RelayRLData>> =
            crate::PyRelayRLAction::pytensor_data_dict_to_relayrldata(data);

        Self {
            inner: RelayRLAction::new(
                obs_tensor_data,
                act_tensor_data,
                mask_tensor_data,
                rew,
                data,
                done,
                reward_updated,
            ),
        }
    }

    /// Returns the observation from the internal RelayRLAction as a Python object (NumPy array).
    ///
    /// Converts the underlying serialized TensorData to a tch::Tensor, then to a NumPy array.
    #[pyo3(signature = ())]
    fn get_obs(&self, py: Python) -> Py<PyAny> {
        let obs_tensor: Tensor =
            Tensor::try_from(self.inner.get_obs().expect("Could not get obs").clone())
                .expect("Failed to convert obs");
        Self::tch_tensor_to_ndarray(py, obs_tensor)
    }

    /// Returns the action from the internal RelayRLAction as a Python object (NumPy array).
    #[pyo3(signature = ())]
    fn get_act(&self, py: Python) -> Py<PyAny> {
        let act_tensor: Tensor =
            Tensor::try_from(self.inner.get_act().expect("Could not get act").clone())
                .expect("Failed to convert act");
        Self::tch_tensor_to_ndarray(py, act_tensor)
    }

    /// Returns the mask from the internal RelayRLAction as a Python object (NumPy array).
    #[pyo3(signature = ())]
    fn get_mask(&self, py: Python) -> Py<PyAny> {
        let mask_tensor: Tensor =
            Tensor::try_from(self.inner.get_mask().expect("Could not get mask").clone())
                .expect("Failed to convert mask");
        Self::tch_tensor_to_ndarray(py, mask_tensor)
    }

    /// Returns the stored reward from the RelayRLAction.
    #[pyo3(signature = ())]
    fn get_rew(&self) -> f32 {
        self.inner.get_rew()
    }

    /// Returns the auxiliary data dictionary as a Python `dict`.
    ///
    /// Any `Tensor` objects in the RelayRLData are converted to NumPy arrays.
    #[pyo3(signature = ())]
    fn get_data(&self, py: Python) -> Py<PyDict> {
        let data: Bound<PyDict> =
            Self::relayrldata_to_data_dict(py, self.inner.get_data().cloned());
        data.into()
    }

    /// Returns the done flag indicating if this action terminates an episode.
    #[pyo3(signature = ())]
    fn get_done(&self) -> bool {
        self.inner.get_done()
    }

    /// Updates the reward in the underlying RelayRLAction with a new float value.
    ///
    /// # Arguments
    /// * `reward` - The new reward value.
    #[pyo3(signature = (reward))]
    fn update_reward(&mut self, reward: &Bound<'_, PyAny>) {
        self.inner.update_reward(
            reward
                .extract::<f32>()
                .expect("Failed to convert reward to f32"),
        );
    }

    /// Serializes the entire RelayRLAction (including its TensorData fields) to a JSON string.
    ///
    /// This is useful for quick debugging or saving the action state outside of typical RL usage.
    #[pyo3(signature = ())]
    fn to_json(&self) -> String {
        serde_json::to_string(&self).expect("Failed to serialize action to JSON")
    }

    /// Constructs a `PyRelayRLAction` from a Python dictionary containing serialized fields.
    ///
    /// The dictionary must match the JSON format of a serialized RelayRLAction, including
    /// optional fields for "obs", "act", "mask", the "rew", "data", "done", and "reward_updated".
    /// Tensors are represented as sub-dictionaries containing `shape`, `dtype`, and `data`.
    ///
    /// # Arguments
    /// * `action_dict` - A Python dictionary describing the action fields.
    ///
    /// # Returns
    /// A newly constructed `PyRelayRLAction`.
    #[staticmethod]
    #[pyo3(signature = (action_dict))]
    pub fn action_from_json(action_dict: &Bound<'_, PyDict>) -> PyRelayRLAction {
        // Helper to safely get a sub-dict from the main dictionary
        let get_dict_item = |key: &str| -> Option<Bound<'_, PyDict>> {
            let item_any: Bound<PyAny> = action_dict
                .get_item(key)
                .ok()
                .flatten()
                .expect("Failed to get item");
            item_any.downcast::<PyDict>().ok().cloned()
        };

        // Extract main fields from the Python dictionary
        let obs = get_dict_item("obs");
        let act = get_dict_item("act");
        let mask = get_dict_item("mask");
        let rew: f32 = action_dict
            .get_item("rew")
            .expect("Failed to get rew")
            .expect("Rew is None")
            .extract::<f32>()
            .expect("Failed to convert rew to f32");
        let data = get_dict_item("data");
        let done: bool = action_dict
            .get_item("done")
            .expect("Failed to get done")
            .expect("Done is None")
            .extract::<bool>()
            .expect("Failed to convert done to bool");
        let reward_updated: bool = action_dict
            .get_item("reward_updated")
            .expect("Failed to get reward update")
            .expect("reward_update is None")
            .extract::<bool>()
            .expect("Failed to convert reward_updated to bool");

        // Convert the sub-dicts to TensorData
        let obs_tensor_data: Option<TensorData> =
            obs.and_then(|val| TensorData::try_from(val).ok());
        let act_tensor_data: Option<TensorData> =
            act.and_then(|val| TensorData::try_from(val).ok());
        let mask_tensor_data: Option<TensorData> =
            mask.and_then(|val| TensorData::try_from(val).ok());

        // Convert the data dictionary to RelayRLData
        let data_map: Option<HashMap<String, RelayRLData>> =
            Self::json_data_dict_to_relayrldata(data.as_ref());

        PyRelayRLAction {
            inner: RelayRLAction::new(
                obs_tensor_data,
                act_tensor_data,
                mask_tensor_data,
                rew,
                data_map,
                done,
                reward_updated,
            ),
        }
    }
}

/// Implementation of helper methods for `PyRelayRLAction` that deal with data conversions.
impl PyRelayRLAction {
    /// Converts either a PyTorch tensor or a NumPy array in Python into a `TensorData` object.
    ///
    /// This function will call `.numpy()` on a PyTorch tensor, or `.tolist()` on a NumPy array,
    /// then reconstruct a tch::Tensor on the Rust side, and finally convert to `TensorData`.
    /// If the Python object type is unrecognized, it returns `None`.
    ///
    /// # Arguments
    /// * `py_tensor` - An optional reference to a Python object (either tensor or ndarray).
    ///
    /// # Returns
    /// `Some(TensorData)` if conversion succeeds, or `None` if the Python object was None
    /// or of an unsupported type.
    pub(crate) fn ndarray_to_tensor_data(
        py_tensor: Option<&Bound<'_, PyAny>>,
    ) -> Option<TensorData> {
        // A helper to do the final conversion of the extracted data -> tch::Tensor -> TensorData
        fn ndarray_to_data(np_array_obj: &Bound<'_, PyAny>) -> TensorData {
            let ndarray: Bound<PyAny> = np_array_obj
                .call_method0("tolist")
                .expect("Failed to convert to list");
            let ndarray: Vec<f64> = ndarray
                .extract::<Vec<f64>>()
                .expect("Failed to convert to Vec<f64>");
            let ndarray: Tensor = Tensor::from_slice(&ndarray);
            TensorData::try_from(&ndarray).expect("Failed to convert to TensorData")
        }

        match py_tensor {
            Some(tensor) => {
                // Identify the object type
                match tensor
                    .get_type()
                    .name()
                    .expect("Failed to get tensor name")
                    .to_str()
                    .expect("Tensor name is not a string")
                {
                    "torch.Tensor" | "Tensor" | "tensor" => {
                        // Convert a PyTorch tensor to numpy array
                        let ndarray: Bound<PyAny> = tensor.call_method1("numpy", ()).ok()?;
                        Some(ndarray_to_data(&ndarray))
                    }
                    "numpy.ndarray" | "ndarray" => Some(ndarray_to_data(tensor)),
                    _ => None,
                }
            }
            None => None,
        }
    }

    /// Converts a tch::Tensor into a NumPy array (Python object) via PyO3 calls.
    ///
    /// This function reads the shape and data from the tch::Tensor, builds a NumPy array
    /// in Python with the same shape and data, and returns it as a Py<PyAny>.
    ///
    /// # Arguments
    /// * `py` - The active Python GIL token.
    /// * `tensor` - The tch::Tensor to convert.
    ///
    /// # Returns
    /// A Py<PyAny> referencing the constructed NumPy array in Python.
    pub(crate) fn tch_tensor_to_ndarray(py: Python, tensor: Tensor) -> Py<PyAny> {
        /// Builds a NumPy array from a Rust vector `data` and a `shape_usize` describing its shape.
        fn build_ndarray<'a, T: IntoPyObject<'a>>(
            python: Python<'a>,
            data: Vec<T>,
            shape_usize: Vec<usize>,
        ) -> Bound<'_, PyAny> {
            let numpy = PyModule::import(python, "numpy").expect("numpy unavailable in Python");
            let array = numpy
                .getattr("array")
                .expect("Failed to get numpy array()")
                .call1((data,))
                .expect("Failed to convert Rust data to numpy array");
            let ndarray = array
                .getattr("reshape")
                .expect("Failed to get reshape()")
                .call1((shape_usize,))
                .expect("Failed to reshape numpy array");
            ndarray
        }

        // Acquire shape and cast from tch::Tensor
        let shape: Vec<i64> = tensor.size();
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

        match tensor.kind() {
            tch::Kind::Float => {
                let data_f32: Vec<f32> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<f32>");
                build_ndarray::<f32>(py, data_f32, shape_usize).into()
            }
            tch::Kind::Double => {
                let data_f64: Vec<f64> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<f64>");
                build_ndarray::<f64>(py, data_f64, shape_usize).into()
            }
            tch::Kind::Uint8 => {
                let data_u8: Vec<u8> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<u8>");
                build_ndarray::<u8>(py, data_u8, shape_usize).into()
            }
            tch::Kind::Int8 => {
                let data_i8: Vec<i8> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<i8>");
                build_ndarray::<i8>(py, data_i8, shape_usize).into()
            }
            tch::Kind::Int16 => {
                let data_i16: Vec<i16> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<i16>");
                build_ndarray::<i16>(py, data_i16, shape_usize).into()
            }
            tch::Kind::Int => {
                let data_i32: Vec<i32> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<i32>");
                build_ndarray::<i32>(py, data_i32, shape_usize).into()
            }
            tch::Kind::Int64 => {
                let data_i64: Vec<i64> = tensor
                    .try_into()
                    .expect("Failed to convert tch::Tensor to Vec<i64>");
                build_ndarray::<i64>(py, data_i64, shape_usize).into()
            }
            _ => panic!("Unsupported tensor kind for conversion"),
        }
    }

    /// Converts a Python dictionary of RelayRLData-serialized fields back into a HashMap.
    ///
    /// The input dictionary might look like:
    /// {
    ///   "root_key": {
    ///       "root_value": {
    ///         "shape": [3],
    ///         "dtype": "Int64",
    ///         "data": [1.0, 2.0, 3.0]
    ///       }
    ///   },
    ///   ...
    /// }
    ///
    /// # Arguments
    /// * `dict_data` - An optional reference to a Python dictionary containing sub-entries.
    ///
    /// # Returns
    /// A HashMap mapping string keys to RelayRLData. If the dictionary is None or invalid, returns None.
    pub(crate) fn json_data_dict_to_relayrldata(
        dict_data: Option<&Bound<'_, PyDict>>,
    ) -> Option<HashMap<String, RelayRLData>> {
        let relayrldata: Option<HashMap<String, RelayRLData>> = match dict_data {
            Some(dict) => {
                let mut data_map: HashMap<String, RelayRLData> = HashMap::new();
                for (root_key, root_value) in dict.iter() {
                    // Convert the root_key to a String
                    let root_key: &String =
                        &root_key.extract::<String>().expect("Failed to extract key");
                    let sub_dict: &Bound<PyDict> = match root_value.downcast::<PyDict>() {
                        Ok(sub_dict) => sub_dict,
                        Err(_) => {
                            return None;
                        }
                    };

                    // Iterate over the sub-dictionary
                    for (sub_key, sub_value) in sub_dict.iter() {
                        let sub_value: RelayRLData = match sub_key.to_string().as_str() {
                            "torch.Tensor" | "Tensor" | "tensor" => {
                                let sub_value_dict: Bound<PyDict> = sub_value
                                    .downcast::<PyDict>()
                                    .expect("Failed to downcast")
                                    .clone();
                                RelayRLData::Tensor(
                                    TensorData::try_from(sub_value_dict)
                                        .expect("Failed to convert to TensorData"),
                                )
                            }
                            "numpy.ndarray" | "ndarray" => {
                                let sub_value_dict: Bound<PyDict> = sub_value
                                    .downcast::<PyDict>()
                                    .expect("Failed to downcast")
                                    .clone();
                                RelayRLData::Tensor(
                                    TensorData::try_from(sub_value_dict)
                                        .expect("Failed to convert to TensorData"),
                                )
                            }
                            "Byte" | "byte" => RelayRLData::Byte(
                                sub_value.extract::<u8>().expect("Failed to extract byte"),
                            ),
                            "Int" | "int" => RelayRLData::Int(
                                sub_value.extract::<i32>().expect("Failed to extract int"),
                            ),
                            "Long" | "long" => RelayRLData::Long(
                                sub_value.extract::<i64>().expect("Failed to extract long"),
                            ),
                            "Float" | "float" => RelayRLData::Float(
                                sub_value.extract::<f32>().expect("Failed to extract float"),
                            ),
                            "Double" | "double" => RelayRLData::Double(
                                sub_value
                                    .extract::<f64>()
                                    .expect("Failed to extract double"),
                            ),
                            "String" | "str" => RelayRLData::String(
                                sub_value
                                    .extract::<String>()
                                    .expect("Failed to extract string"),
                            ),
                            "bool" => RelayRLData::Bool(
                                sub_value.extract::<bool>().expect("Failed to extract bool"),
                            ),
                            _ => {
                                return None;
                            }
                        };
                        data_map.insert(root_key.to_string(), sub_value);
                    }
                }
                Some(data_map)
            }
            None => None,
        };
        relayrldata
    }

    /// Converts a Python dictionary of possible Python objects into a HashMap of RelayRLData.
    ///
    /// # Arguments
    /// * `dict_data` - An optional Python dictionary where each key maps to a potential
    ///   PyTorch tensor, NumPy array, or primitive type.
    ///
    /// # Returns
    /// An optional HashMap with `String -> RelayRLData`.
    pub(crate) fn pytensor_data_dict_to_relayrldata(
        dict_data: Option<&Bound<'_, PyDict>>,
    ) -> Option<HashMap<String, RelayRLData>> {
        let relayrldata: Option<HashMap<String, RelayRLData>> = match dict_data {
            Some(dict) => {
                let mut data: HashMap<String, RelayRLData> = HashMap::new();
                for (key, value) in dict.iter() {
                    let key_str: String = key.extract::<String>().expect("Failed to extract key");
                    let val_type = value
                        .get_type()
                        .name()
                        .expect("Failed to get type")
                        .to_string();
                    let sub_value: RelayRLData = match val_type.as_str() {
                        "torch.Tensor" | "Tensor" | "tensor" => {
                            // Convert the PyTorch tensor to a NumPy array
                            let ndarray = value.call_method1("to_numpy", ()).ok()?;
                            RelayRLData::Tensor(Self::ndarray_to_tensor_data(Some(&ndarray))?)
                        }
                        "numpy.ndarray" | "ndarray" => {
                            RelayRLData::Tensor(Self::ndarray_to_tensor_data(Some(&value))?)
                        }
                        "byte" => RelayRLData::Byte(
                            value.extract::<u8>().expect("Failed to extract byte"),
                        ),
                        "int" => RelayRLData::Int(
                            value.extract::<i32>().expect("Failed to extract integer"),
                        ),
                        "long" => RelayRLData::Long(
                            value.extract::<i64>().expect("Failed to extract long"),
                        ),
                        "float" => RelayRLData::Float(
                            value.extract::<f32>().expect("Failed to extract float"),
                        ),
                        "double" => RelayRLData::Double(
                            value.extract::<f64>().expect("Failed to extract double"),
                        ),
                        "str" => RelayRLData::String(
                            value.extract::<String>().expect("Failed to extract string"),
                        ),
                        "bool" => RelayRLData::Bool(
                            value.extract::<bool>().expect("Failed to extract bool"),
                        ),
                        _ => {
                            return None;
                        }
                    };
                    data.insert(key_str, sub_value);
                }
                Some(data)
            }
            None => None,
        };
        relayrldata
    }

    /// Converts an optional HashMap of RelayRLData into a Python dictionary,
    /// turning any TensorData into NumPy arrays.
    ///
    /// # Arguments
    /// * `py` - The active Python interpreter token.
    /// * `relayrldata` - The optional HashMap of string -> RelayRLData.
    ///
    /// # Returns
    /// A `PyDict` representing the data, with Tensors replaced by NumPy arrays.
    pub(crate) fn relayrldata_to_data_dict(
        py: Python,
        relayrldata: Option<HashMap<String, RelayRLData>>,
    ) -> Bound<'_, PyDict> {
        let dict: pyo3::Bound<PyDict> = PyDict::new(py);
        if relayrldata.is_none() {
            return dict;
        }

        for (key, value) in relayrldata.expect("RelayRLData is None").iter() {
            match value {
                RelayRLData::Tensor(tensor_data) => {
                    // Convert TensorData -> tch::Tensor -> NumPy
                    let tensor = Tensor::try_from(tensor_data.clone())
                        .expect("Failed to convert tensordata to Tensor");
                    let ndarray = Self::tch_tensor_to_ndarray(py, tensor);
                    dict.set_item(key, ndarray)
                        .expect("Failed to set tensor data");
                }
                RelayRLData::Byte(byte_data) => {
                    dict.set_item(key, byte_data)
                        .expect("Failed to set byte data");
                }
                RelayRLData::Short(short_data) => {
                    dict.set_item(key, short_data)
                        .expect("Failed to set short data");
                }
                RelayRLData::Int(int_data) => {
                    dict.set_item(key, int_data)
                        .expect("Failed to set int data");
                }
                RelayRLData::Long(long_data) => {
                    dict.set_item(key, long_data)
                        .expect("Failed to set long data");
                }
                RelayRLData::Float(float_data) => {
                    dict.set_item(key, float_data)
                        .expect("Failed to set float data");
                }
                RelayRLData::Double(double_data) => {
                    dict.set_item(key, double_data)
                        .expect("Failed to set double data");
                }
                RelayRLData::String(string_data) => {
                    dict.set_item(key, string_data)
                        .expect("Failed to set string data");
                }
                RelayRLData::Bool(bool_data) => {
                    dict.set_item(key, bool_data)
                        .expect("Failed to set bool data");
                }
            }
        }
        dict
    }
}
