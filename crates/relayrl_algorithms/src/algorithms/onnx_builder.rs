/// Hand-rolled ONNX protobuf encoder for fully-connected MLPs.
///
/// No external protobuf library is required. The output is a valid ONNX `ModelProto`
/// (opset 17) that can be loaded directly by ORT's `commit_from_memory()`.
///
/// # Layout
///
/// `build_onnx_mlp_bytes` is the public entry point. It accepts a slice of layer
/// specs (produced by `WeightProvider::get_pi_layer_specs`) and returns the raw bytes
/// of the serialized model.
///
/// ## Burn weight layout
///
/// Burn's `Linear` stores weights in row-major `[in_features, out_features]` order.
/// ONNX `Gemm` computes `Y = alpha * A * B^transB + beta * C`.  With `transB=0` and
/// B shaped `[in, out]`, the result is `[batch, out]` — exactly the right shape when A
/// is `[batch, in]`.  The old code had `transB=1` (PyTorch convention) which is wrong
/// for Burn.

/// Build a serialized ONNX `ModelProto` for a fully-connected MLP.
///
/// `layer_specs`: `(in_dim, out_dim, flat_weights, flat_biases)` per layer, ordered
/// input→output.  ReLU is inserted between every consecutive layer pair; the last
/// layer has no activation.
pub fn build_onnx_mlp_bytes(layer_specs: &[(usize, usize, Vec<f32>, Vec<f32>)]) -> Vec<u8> {
    if layer_specs.is_empty() {
        return Vec::new();
    }

    let n = layer_specs.len();
    let in_dim = layer_specs[0].0;
    let out_dim = layer_specs[n - 1].1;

    let mut initializers: Vec<Vec<u8>> = Vec::new();
    let mut nodes: Vec<Vec<u8>> = Vec::new();

    for (idx, (layer_in, layer_out, weights, biases)) in layer_specs.iter().enumerate() {
        let w_name = format!("W{idx}");
        let b_name = format!("b{idx}");

        initializers.push(build_tensor_proto(
            &w_name,
            &[*layer_in as i64, *layer_out as i64],
            weights,
        ));
        initializers.push(build_tensor_proto(&b_name, &[*layer_out as i64], biases));

        let gemm_input = if idx == 0 {
            "obs".to_string()
        } else {
            format!("relu{}", idx - 1)
        };
        let gemm_output = format!("gemm{idx}");

        nodes.push(build_gemm_node(
            &format!("Gemm_{idx}"),
            &gemm_input,
            &w_name,
            &b_name,
            &gemm_output,
        ));

        if idx < n - 1 {
            nodes.push(build_relu_node(
                &format!("Relu_{idx}"),
                &gemm_output,
                &format!("relu{idx}"),
            ));
        }
    }

    let final_output = format!("gemm{}", n - 1);
    let input_info = build_value_info("obs", in_dim);
    let output_info = build_value_info(&final_output, out_dim);

    let graph = build_graph_proto("mlp", &nodes, &initializers, &[input_info], &[output_info]);
    build_model_proto(graph)
}

// ── Protobuf wire-encoding helpers ───────────────────────────────────────────

/// Encode a non-negative integer as a protobuf varint.
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

/// Encode a field with wire-type 0 (varint).
fn field_varint(field: u32, val: u64) -> Vec<u8> {
    let mut out = varint(((field as u64) << 3) | 0);
    out.extend(varint(val));
    out
}

/// Encode a field with wire-type 2 (length-delimited bytes / string / message).
fn field_bytes(field: u32, data: &[u8]) -> Vec<u8> {
    let mut out = varint(((field as u64) << 3) | 2);
    out.extend(varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

/// Encode a UTF-8 string field (wire-type 2).
fn field_str(field: u32, s: &str) -> Vec<u8> {
    field_bytes(field, s.as_bytes())
}

/// Encode a field with wire-type 5 (32-bit fixed — used for `float`).
fn field_fixed32(field: u32, val: f32) -> Vec<u8> {
    let mut out = varint(((field as u64) << 3) | 5);
    out.extend_from_slice(&val.to_le_bytes());
    out
}

/// Encode an embedded protobuf message field (wire-type 2).
fn field_msg(field: u32, msg: &[u8]) -> Vec<u8> {
    field_bytes(field, msg)
}

// ── ONNX structure builders ───────────────────────────────────────────────────

/// Build a `TensorProto` (initializer) from a flat f32 slice.
///
/// `dims` is the tensor shape; `data` is stored as `raw_data` (little-endian f32).
fn build_tensor_proto(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut msg = Vec::new();
    for &d in dims {
        msg.extend(field_varint(1, d as u64)); // dims (field 1, repeated int64)
    }
    msg.extend(field_varint(2, 1)); // data_type = FLOAT (field 2)
    msg.extend(field_str(8, name)); // name (field 8)
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    msg.extend(field_bytes(9, &raw)); // raw_data (field 9)
    msg
}

/// Build a float `AttributeProto`.
///
/// `type` field = 20, value FLOAT = 1.  `f` and `i` share field number 4 in the
/// ONNX proto; they are distinguished by wire-type (5 = 32-bit for float, 0 = varint
/// for int64).
fn build_attribute_float(name: &str, val: f32) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, name)); // name (field 1)
    msg.extend(field_varint(20, 1)); // type = FLOAT=1 (field 20)
    msg.extend(field_fixed32(4, val)); // f (field 4, wire-type 5)
    msg
}

/// Build an integer `AttributeProto`.
///
/// `type` field = 20, value INT = 2.  `i` is encoded at field 4 with wire-type 0
/// (varint), which coexists with `f` at the same field number (different wire-type).
fn build_attribute_int(name: &str, val: i64) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, name)); // name (field 1)
    msg.extend(field_varint(20, 2)); // type = INT=2 (field 20)
    msg.extend(field_varint(4, val as u64)); // i (field 4, wire-type 0)
    msg
}

/// Build a `NodeProto` for a `Gemm` operation.
///
/// `transB=0` because Burn Linear stores weights as `[in, out]` (no transposition
/// needed).  `alpha = beta = 1.0` are the standard scale factors.
fn build_gemm_node(
    name: &str,
    input: &str,
    weight: &str,
    bias: &str,
    output: &str,
) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, input)); // input[0]
    msg.extend(field_str(1, weight)); // input[1]
    msg.extend(field_str(1, bias)); // input[2]
    msg.extend(field_str(2, output)); // output[0]
    msg.extend(field_str(3, name)); // name
    msg.extend(field_str(4, "Gemm")); // op_type
    msg.extend(field_msg(6, &build_attribute_float("alpha", 1.0)));
    msg.extend(field_msg(6, &build_attribute_float("beta", 1.0)));
    msg.extend(field_msg(6, &build_attribute_int("transB", 0)));
    msg
}

/// Build a `NodeProto` for a `Relu` operation.
fn build_relu_node(name: &str, input: &str, output: &str) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, input));
    msg.extend(field_str(2, output));
    msg.extend(field_str(3, name));
    msg.extend(field_str(4, "Relu"));
    msg
}

/// Build a `TensorShapeProto.Dimension`.
fn build_dim(dim_value: Option<i64>, dim_param: Option<&str>) -> Vec<u8> {
    let mut msg = Vec::new();
    if let Some(v) = dim_value {
        msg.extend(field_varint(1, v as u64)); // dim_value (field 1)
    }
    if let Some(p) = dim_param {
        msg.extend(field_str(2, p)); // dim_param (field 2)
    }
    msg
}

/// Build a `TypeProto` wrapping a float tensor with shape `[batch_size, feature_dim]`.
///
/// The batch dimension is represented as a symbolic string `"batch_size"` to allow
/// dynamic batch sizes at inference time.
fn build_type_proto_float_tensor(feature_dim: usize) -> Vec<u8> {
    // TensorShapeProto: two dims — dynamic batch, static feature
    let mut shape_msg = Vec::new();
    shape_msg.extend(field_msg(1, &build_dim(None, Some("batch_size"))));
    shape_msg.extend(field_msg(1, &build_dim(Some(feature_dim as i64), None)));

    // TypeProto.Tensor: elem_type=FLOAT(1), shape
    let mut tensor_msg = Vec::new();
    tensor_msg.extend(field_varint(1, 1)); // elem_type = FLOAT
    tensor_msg.extend(field_msg(2, &shape_msg)); // shape

    // TypeProto: field 1 = tensor_type (the Tensor sub-message)
    let mut type_msg = Vec::new();
    type_msg.extend(field_msg(1, &tensor_msg));
    type_msg
}

/// Build a `ValueInfoProto` for a named float tensor input or output.
fn build_value_info(name: &str, feature_dim: usize) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, name)); // name (field 1)
    msg.extend(field_msg(2, &build_type_proto_float_tensor(feature_dim))); // type (field 2)
    msg
}

/// Build the `GraphProto`.
///
/// `initializer` is at field **5** (not 3 — a common source of bugs with older ONNX
/// proto references).
fn build_graph_proto(
    name: &str,
    nodes: &[Vec<u8>],
    initializers: &[Vec<u8>],
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Vec<u8> {
    let mut msg = Vec::new();
    for node in nodes {
        msg.extend(field_msg(1, node)); // node (field 1)
    }
    msg.extend(field_str(2, name)); // name (field 2)
    for init in initializers {
        msg.extend(field_msg(5, init)); // initializer (field 5)
    }
    for input in inputs {
        msg.extend(field_msg(11, input)); // input (field 11)
    }
    for output in outputs {
        msg.extend(field_msg(12, output)); // output (field 12)
    }
    msg
}

/// Build an `OperatorSetIdProto`.
fn build_opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_str(1, domain)); // domain (field 1)
    msg.extend(field_varint(2, version as u64)); // version (field 2)
    msg
}

/// Build the top-level `ModelProto`.
///
/// Uses IR version 7 and opset 17 (the default ONNX domain).
fn build_model_proto(graph: Vec<u8>) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend(field_varint(1, 7)); // ir_version = 7 (field 1)
    msg.extend(field_msg(8, &build_opset_import("", 17))); // opset_import (field 8)
    msg.extend(field_msg(7, &graph)); // graph (field 7)
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_single_byte() {
        assert_eq!(varint(0), vec![0]);
        assert_eq!(varint(127), vec![127]);
    }

    #[test]
    fn varint_multibyte() {
        // 128 = 0x80 → [0x80, 0x01]
        assert_eq!(varint(128), vec![0x80, 0x01]);
    }

    #[test]
    fn build_onnx_mlp_bytes_nonempty_for_single_layer() {
        let weights = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let biases = vec![0.0f32, 0.0];
        let specs = vec![(2usize, 2usize, weights, biases)];
        let bytes = build_onnx_mlp_bytes(&specs);
        assert!(!bytes.is_empty(), "ONNX bytes should not be empty");
    }

    #[test]
    fn build_onnx_mlp_bytes_empty_for_no_layers() {
        let bytes = build_onnx_mlp_bytes(&[]);
        assert!(bytes.is_empty());
    }

    #[test]
    fn build_onnx_mlp_bytes_two_layer_mlp() {
        let w1 = vec![0.1f32; 4 * 8]; // 4→8
        let b1 = vec![0.0f32; 8];
        let w2 = vec![0.2f32; 8 * 2]; // 8→2
        let b2 = vec![0.0f32; 2];
        let specs = vec![(4, 8, w1, b1), (8, 2, w2, b2)];
        let bytes = build_onnx_mlp_bytes(&specs);
        // Minimal sanity: should be a non-trivial byte sequence
        assert!(bytes.len() > 100, "expected a non-trivial ONNX blob");
    }
}
