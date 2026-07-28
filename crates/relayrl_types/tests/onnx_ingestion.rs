//! Integration tests proving `relayrl_types` ingests arbitrary single-input/single-output
//! ONNX graphs rather than a hard-coded `"input"`/first-output schema: names and shapes are
//! discovered from the graph itself, and RelayRL-supplied `ModelMetadata` is validated
//! against that discovered signature instead of being trusted blindly.
#![cfg(all(feature = "onnx-model", feature = "ndarray-backend"))]

use burn_ndarray::NdArray;
use relayrl_types::data::tensor::{
    DType, DeviceType, NdArrayDType, SupportedTensorBackend, TensorData,
};
use relayrl_types::model::{ModelError, ModelMetadata, ModelModule};

type TestBackend = NdArray<f32>;

// ── Minimal ONNX protobuf wire-encoding helpers (test-only) ─────────────────
//
// These mirror the encoding used by `relayrl_algorithms::algorithms::onnx_builder`, kept
// self-contained here since `relayrl_types` cannot depend on `relayrl_algorithms` (the
// dependency runs the other way).

fn varint(mut val: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        if val < 0x80 {
            out.push(val as u8);
            break;
        }
        out.push((val as u8 & 0x7F) | 0x80);
        val >>= 7;
    }
    out
}

fn field_varint(field: u32, val: u64) -> Vec<u8> {
    let mut out = varint((field as u64) << 3);
    out.extend(varint(val));
    out
}

fn field_bytes(field: u32, data: &[u8]) -> Vec<u8> {
    let mut out = varint(((field as u64) << 3) | 2);
    out.extend(varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

fn field_str(field: u32, s: &str) -> Vec<u8> {
    field_bytes(field, s.as_bytes())
}

fn field_msg(field: u32, msg: &[u8]) -> Vec<u8> {
    field_bytes(field, msg)
}

/// A single ONNX `TensorShapeProto.Dimension`: either a fixed size or a dynamic
/// (optionally named) dimension.
enum Dim {
    Fixed(i64),
    Dynamic(Option<&'static str>),
}

fn build_dim(dim: &Dim) -> Vec<u8> {
    let mut msg = Vec::new();
    match dim {
        Dim::Fixed(v) => msg.extend(field_varint(1, *v as u64)),
        Dim::Dynamic(Some(p)) => msg.extend(field_str(2, p)),
        Dim::Dynamic(None) => {}
    }
    msg
}

/// ONNX `TensorProto.DataType` values relevant to these tests.
const ONNX_FLOAT: i64 = 1;
const ONNX_BOOL: i64 = 9;

fn build_type_proto_tensor(elem_type: i64, dims: &[Dim]) -> Vec<u8> {
    let mut shape_msg = Vec::new();
    for d in dims {
        shape_msg.extend(field_msg(1, &build_dim(d)));
    }
    let mut tensor_msg = Vec::new();
    tensor_msg.extend(field_varint(1, elem_type as u64));
    tensor_msg.extend(field_msg(2, &shape_msg));
    let mut type_msg = Vec::new();
    type_msg.extend(field_msg(1, &tensor_msg));
    type_msg
}

fn build_value_info(name: &str, elem_type: i64, dims: &[Dim]) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, name));
    msg.extend(field_msg(2, &build_type_proto_tensor(elem_type, dims)));
    msg
}

fn build_identity_node(name: &str, input: &str, output: &str) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, input));
    msg.extend(field_str(2, output));
    msg.extend(field_str(3, name));
    msg.extend(field_str(4, "Identity"));
    msg
}

fn build_add_node(name: &str, a: &str, b: &str, output: &str) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, a));
    msg.extend(field_str(1, b));
    msg.extend(field_str(2, output));
    msg.extend(field_str(3, name));
    msg.extend(field_str(4, "Add"));
    msg
}

fn build_graph_proto(
    name: &str,
    nodes: &[Vec<u8>],
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Vec<u8> {
    let mut msg = Vec::new();
    for node in nodes {
        msg.extend(field_msg(1, node));
    }
    msg.extend(field_str(2, name));
    for input in inputs {
        msg.extend(field_msg(11, input));
    }
    for output in outputs {
        msg.extend(field_msg(12, output));
    }
    msg
}

fn build_opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, domain));
    msg.extend(field_varint(2, version as u64));
    msg
}

fn build_model_proto(graph: Vec<u8>) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_varint(1, 7));
    msg.extend(field_msg(8, &build_opset_import("", 17)));
    msg.extend(field_msg(7, &graph));
    msg
}

/// Builds a single-input/single-output `Identity` ONNX graph with the given I/O names,
/// element type, and per-axis dimensions (fixed or dynamic).
fn build_identity_onnx(
    input_name: &str,
    output_name: &str,
    elem_type: i64,
    dims: &[Dim],
) -> Vec<u8> {
    let node = build_identity_node("Identity_0", input_name, output_name);
    let input_info = build_value_info(input_name, elem_type, dims);
    let output_info = build_value_info(output_name, elem_type, dims);
    let graph = build_graph_proto("identity_graph", &[node], &[input_info], &[output_info]);
    build_model_proto(graph)
}

/// Builds a two-input `Add` ONNX graph, used to prove multi-input graphs are rejected.
fn build_two_input_onnx(elem_type: i64, dims: &[Dim]) -> Vec<u8> {
    let node = build_add_node("Add_0", "a", "b", "sum");
    let a_info = build_value_info("a", elem_type, dims);
    let b_info = build_value_info("b", elem_type, dims);
    let out_info = build_value_info("sum", elem_type, dims);
    let graph = build_graph_proto("two_input_graph", &[node], &[a_info, b_info], &[out_info]);
    build_model_proto(graph)
}

fn f32_metadata(model_file: &str, shape: Vec<usize>) -> ModelMetadata {
    ModelMetadata {
        model_file: model_file.to_string(),
        model_type: relayrl_types::model::ModelFileType::Onnx,
        input_dtype: DType::NdArray(NdArrayDType::F32),
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: shape.clone(),
        output_shape: shape,
        default_device: Some(DeviceType::Cpu),
    }
}

fn float_tensor_data(shape: Vec<usize>, values: &[f32]) -> TensorData {
    TensorData::new(
        shape,
        DType::NdArray(NdArrayDType::F32),
        bytemuck::cast_slice(values).to_vec(),
        SupportedTensorBackend::NdArray,
    )
}

/// The graph declares non-standard I/O names (`"obs"` / `"action"`); this must succeed and
/// bind correctly purely from discovery, without any hard-coded `"input"`/`"output"` guess.
#[test]
fn discovers_nonstandard_input_and_output_names() {
    let bytes = build_identity_onnx("obs", "action", ONNX_FLOAT, &[Dim::Fixed(2)]);
    let metadata = f32_metadata("policy.onnx", vec![2]);

    let module = ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata)
        .expect("graph with non-standard I/O names should load and validate");

    let input = float_tensor_data(vec![2], &[1.0, 2.0]);
    let output = module
        .flat_batch_inference(input)
        .expect("identity graph should run");

    assert_eq!(output.shape, vec![2]);
    let values: &[f32] = bytemuck::cast_slice(&output.data);
    assert_eq!(values, &[1.0, 2.0]);
}

/// The legacy `"input"`/`"output"` naming convention (used by `relayrl_algorithms`'s ONNX
/// builder and older exported models) must keep working unchanged.
#[test]
fn legacy_input_output_names_still_work() {
    let bytes = build_identity_onnx("input", "output", ONNX_FLOAT, &[Dim::Fixed(3)]);
    let metadata = f32_metadata("policy.onnx", vec![3]);

    let module = ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata)
        .expect("legacy-named graph should load and validate");

    let input = float_tensor_data(vec![3], &[3.0, 4.0, 5.0]);
    let output = module
        .flat_batch_inference(input)
        .expect("identity graph should run");

    let values: &[f32] = bytemuck::cast_slice(&output.data);
    assert_eq!(values, &[3.0, 4.0, 5.0]);
}

/// A graph with a dynamic (symbolic) batch dimension must return the *actual* ORT runtime
/// output shape, not a shape reconstructed from `metadata.output_shape`.
#[test]
fn dynamic_batch_dimension_returns_actual_runtime_shape() {
    let dims = [Dim::Dynamic(Some("batch")), Dim::Fixed(3)];
    let bytes = build_identity_onnx("input", "output", ONNX_FLOAT, &dims);
    // Metadata declares a batch of 1 purely for validation purposes; the graph's batch
    // dimension is dynamic, so any concrete batch size must be accepted at runtime.
    let metadata = f32_metadata("policy.onnx", vec![1, 3]);

    let module = ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata)
        .expect("graph with a dynamic batch dimension should load and validate");

    let input = float_tensor_data(vec![4, 3], &[0.0; 12]);
    let output = module
        .flat_batch_inference(input)
        .expect("identity graph should run with a batch larger than metadata implies");

    assert_eq!(output.shape, vec![4, 3]);
}

/// A `Bool` tensor graph must round-trip through the native ORT `Bool` element type, not a
/// `Uint8` reinterpretation.
#[test]
fn bool_tensor_round_trips_through_native_onnx_bool_type() {
    let bytes = build_identity_onnx("input", "output", ONNX_BOOL, &[Dim::Fixed(3)]);
    let metadata = ModelMetadata {
        model_file: "bool_policy.onnx".to_string(),
        model_type: relayrl_types::model::ModelFileType::Onnx,
        input_dtype: DType::NdArray(NdArrayDType::Bool),
        output_dtype: DType::NdArray(NdArrayDType::Bool),
        input_shape: vec![3],
        output_shape: vec![3],
        default_device: Some(DeviceType::Cpu),
    };

    let module = ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata)
        .expect("bool graph should load and validate");

    let input = TensorData::new(
        vec![3],
        DType::NdArray(NdArrayDType::Bool),
        vec![1u8, 0u8, 1u8],
        SupportedTensorBackend::NdArray,
    );
    let output = module
        .flat_batch_inference(input)
        .expect("identity graph should run for bool tensors");

    assert_eq!(output.data, vec![1u8, 0u8, 1u8]);
}

/// Metadata declaring a dtype that doesn't match the graph's actual element type must be
/// rejected at construction time, not fail later inside `Session::run`.
#[test]
fn dtype_mismatch_is_rejected_at_construction() {
    let bytes = build_identity_onnx("input", "output", ONNX_FLOAT, &[Dim::Fixed(2)]);
    let metadata = ModelMetadata {
        model_file: "mismatch.onnx".to_string(),
        model_type: relayrl_types::model::ModelFileType::Onnx,
        input_dtype: DType::NdArray(NdArrayDType::F64), // graph is Float32
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: vec![2],
        output_shape: vec![2],
        default_device: Some(DeviceType::Cpu),
    };

    let err = match ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata) {
        Ok(_) => panic!("dtype mismatch between metadata and graph should be rejected"),
        Err(err) => err,
    };

    assert!(matches!(err, ModelError::DTypeError(_)), "got: {err:?}");
}

/// A fixed graph dimension that disagrees with metadata's declared size must be rejected.
#[test]
fn fixed_dimension_mismatch_is_rejected_at_construction() {
    let bytes = build_identity_onnx("input", "output", ONNX_FLOAT, &[Dim::Fixed(4)]);
    let metadata = f32_metadata("mismatch.onnx", vec![5]);

    let err = match ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata) {
        Ok(_) => panic!("fixed dimension mismatch should be rejected"),
        Err(err) => err,
    };

    assert!(
        matches!(err, ModelError::InvalidMetadata(_)),
        "got: {err:?}"
    );
}

/// A rank mismatch between metadata and the graph must be rejected.
#[test]
fn rank_mismatch_is_rejected_at_construction() {
    let bytes = build_identity_onnx(
        "input",
        "output",
        ONNX_FLOAT,
        &[Dim::Fixed(2), Dim::Fixed(3)],
    );
    let metadata = f32_metadata("mismatch.onnx", vec![2]);

    let err = match ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata) {
        Ok(_) => panic!("rank mismatch should be rejected"),
        Err(err) => err,
    };

    assert!(
        matches!(err, ModelError::UnsupportedRank(_)),
        "got: {err:?}"
    );
}

/// Graphs with more than one input are outside RelayRL's supported single-tensor contract
/// and must be rejected with a clear error instead of silently binding the wrong tensor.
#[test]
fn multi_input_graph_is_rejected_at_construction() {
    let bytes = build_two_input_onnx(ONNX_FLOAT, &[Dim::Fixed(2)]);
    let metadata = f32_metadata("two_input.onnx", vec![2]);

    let err = match ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata) {
        Ok(_) => panic!("multi-input graphs should be rejected"),
        Err(err) => err,
    };

    assert!(
        matches!(err, ModelError::UnsupportedModelType(_)),
        "got: {err:?}"
    );
    assert!(err.to_string().contains("exactly one input"));
}

/// Saving and reloading a validated module from disk must preserve the exact `metadata.json`
/// wire format and produce identical inference behavior (no hidden schema fields leak in).
#[test]
fn save_and_reload_round_trip_preserves_behavior() {
    let dir = std::env::temp_dir().join(format!("relayrl-onnx-ingestion-{}", uuid::Uuid::new_v4()));

    let bytes = build_identity_onnx("obs", "action", ONNX_FLOAT, &[Dim::Fixed(2)]);
    let metadata = f32_metadata("policy.onnx", vec![2]);
    let module = ModelModule::<TestBackend>::from_onnx_bytes(bytes, metadata)
        .expect("graph should load and validate");

    module.save(&dir).expect("module should save to disk");

    let meta_json =
        std::fs::read_to_string(dir.join("metadata.json")).expect("metadata.json should exist");
    let meta_value: serde_json::Value =
        serde_json::from_str(&meta_json).expect("metadata.json should be valid JSON");
    let mut keys: Vec<&str> = meta_value
        .as_object()
        .expect("metadata.json should be a JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "default_device",
            "input_dtype",
            "input_shape",
            "model_file",
            "model_type",
            "output_dtype",
            "output_shape",
        ]
    );

    let reloaded = ModelModule::<TestBackend>::load_from_path(&dir)
        .expect("reloaded module should load and validate identically");

    let input = float_tensor_data(vec![2], &[7.0, 8.0]);
    let output = reloaded
        .flat_batch_inference(input)
        .expect("reloaded identity graph should run");
    let values: &[f32] = bytemuck::cast_slice(&output.data);
    assert_eq!(values, &[7.0, 8.0]);

    let _ = std::fs::remove_dir_all(&dir);
}
