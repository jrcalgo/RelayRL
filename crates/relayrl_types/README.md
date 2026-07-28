relayrl_types
==============

Core data types and encoding/decoding utilities for the RelayRL framework.

[![Changelog](https://img.shields.io/badge/Changelog-0.9.1-blue.svg)](CHANGELOG.md)

## Features

- **`RelayRLAction`**: Serializable action container (obs, act, mask, reward, data, done) with UUID agent tracking
- **`RelayRLTrajectory`**: In-memory trajectory buffer with metadata and provenance tracking
- **`Records`**: Records for converting trajectories to and from CSV and Arrow files
- **Burn backend support**: Compatible with both `burn-ndarray` (CPU) and `burn-tch` (GPU) backends
- **Codec pipeline**: Compression, encryption, integrity verification, and chunking
- **Utilities**: Metadata tracking, quantization, and network transport optimizations

## Feature Flags

The crate's actual `default` set is `["ndarray-backend", "onnx-model"]` — the CPU Burn
backend plus ONNX inference. Codec features (compression, encryption, integrity,
metadata, quantization, zerocopy) are **not** enabled by default; the codec examples
below require opting into them explicitly (individually, or via a bundle like
`codec-full`).

```toml
[features]
default = ["ndarray-backend", "onnx-model"]

# Backend selection
tch-backend = ["burn-tch", "half"]          # GPU (LibTorch) backend
ndarray-backend = ["burn-ndarray", "half"]  # CPU backend

# Inference model formats
inference-models = ["tch-model", "onnx-model"]  # All inference models
tch-model = ["tch", "tokio", "tempfile"]        # LibTorch inference
onnx-model = ["ort", "tokio", "tempfile", "ndarray"]  # ONNX inference

# Codec pipeline (opt-in)
compression = ["lz4_flex", "zstd", "bincode"]  # LZ4/Zstd compression
encryption = ["chacha20poly1305", "bincode"]   # ChaCha20-Poly1305 AEAD
integrity = ["blake3"]                          # BLAKE3 checksums
metadata = ["bincode"]                          # `encode`/`decode`/`to_bytes`/`from_bytes`
quantization = ["half"]                         # FP16/BF16 quantization
zerocopy = ["bytes"]                            # Zerocopy data conversions

# Convenience bundles
codec-basic = ["compression", "integrity", "zerocopy"]
codec-secure = ["codec-basic", "encryption"]
codec-full = ["codec-secure", "metadata", "quantization"]
```

To run the codec examples below, add `relayrl_types` with `features = ["codec-full"]`
(or the specific features each example needs) in addition to the default backend.

## Quick Start

### Basic Usage

```rust
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::tensor::relayrl::{DType, NdArrayDType, DeviceType, TensorData};
use relayrl_types::data::tensor::ConversionBurnTensor;
use uuid::Uuid;
use std::sync::Arc;
use burn_tensor::Tensor;
use burn_ndarray::NdArray; // enable feature: ndarray-backend

// Create a Burn tensor (NdArray backend) and store as RelayRL TensorData
let device = DeviceType::Cpu;

// 1) Burn → RelayRL: Convert any Burn tensor into TensorData with a target dtype/backend.
//    `ConversionBurnTensor` wraps the tensor in an `Arc` (it may be shared across conversions).
let obs_burn = Tensor::<NdArray, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &burn_tensor::Device::Cpu);
let obs_td: TensorData = ConversionBurnTensor {
    inner: Arc::new(obs_burn),
    conversion_dtype: DType::NdArray(NdArrayDType::F32),
}.try_into()?;

let act_burn = Tensor::<NdArray, 1>::from_floats([0.5, -0.3], &burn_tensor::Device::Cpu);
let act_td: TensorData = ConversionBurnTensor {
    inner: Arc::new(act_burn),
    conversion_dtype: DType::NdArray(NdArrayDType::F32),
}.try_into()?;

// 2) RelayRL → Burn: Build Burn tensors from stored TensorData with a chosen backend/device
//    Specify the backend type parameter; device is provided via DeviceType
let obs_tensor_any = RelayRLAction::to_tensor::<NdArray>(&obs_td, &device)?; // Box<dyn Any>
let act_tensor_any = RelayRLAction::to_tensor::<NdArray>(&act_td, &device)?;

// 3) Create an action with tensors
let action = RelayRLAction::new(
    Some(obs_td),                 // observation TensorData
    Some(act_td),                 // action TensorData
    None,                         // mask
    1.5,                          // reward
    false,                        // done
    None,                         // auxiliary data
    Some(Uuid::new_v4()),        // agent_id
);

// 4) Work with a trajectory
let mut trajectory = RelayRLTrajectory::with_agent_id(1000, Uuid::new_v4());
trajectory.add_action(action);

println!("Total reward: {}", trajectory.total_reward());
println!("Length: {}", trajectory.len());

// Minimal action without tensors
trajectory.add_action(RelayRLAction::minimal(1.0, false));
```

## Codec Functionality

### 1. Simple Serialization

```rust
use relayrl_types::prelude::action::RelayRLAction;

let action = RelayRLAction::minimal(1.0, false);

// Simple serialization (requires the "metadata" feature).
// `from_bytes` returns `(Self, usize)`: the decoded value and the number of bytes consumed.
let bytes = action.to_bytes()?;
let (decoded, _consumed) = RelayRLAction::from_bytes(&bytes)?;

assert_eq!(decoded.get_rew(), 1.0);
```

### 2. Compression

```rust
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::action::CodecConfig;
use relayrl_types::prelude::codec::CompressionScheme;

let trajectory = RelayRLTrajectory::new(100);
// ... add actions ...

// Configure codec with LZ4 compression (fast). Requires "compression", "integrity",
// and "metadata" features (e.g. via the "codec-full" bundle).
let config = CodecConfig {
    compression: Some(CompressionScheme::Lz4),
    encryption_key: None,
    verify_integrity: true,
    include_metadata: true,
};

// Encode with compression
let encoded = trajectory.encode(&config)?;
println!("Compressed from {} to {} bytes", 
    encoded.original_size, 
    encoded.data.len()
);

// Decode. `RelayRLTrajectory::decode` returns `(Self, usize)`.
let (decoded, _consumed) = RelayRLTrajectory::decode(&encoded, &config)?;
```

### 3. Compression + Encryption

```rust
use relayrl_types::prelude::action::{RelayRLAction, CodecConfig};
use relayrl_types::prelude::codec::CompressionScheme;
use relayrl_types::data::utilities::encrypt::generate_key;

let action = RelayRLAction::minimal(2.5, true);

// Generate encryption key (requires the "encryption" feature)
let key = generate_key();

// Configure codec with compression AND encryption.
// Requires "compression", "encryption", "integrity", and "metadata" features.
let config = CodecConfig {
    compression: Some(CompressionScheme::Zstd(3)),  // Zstd level 3
    encryption_key: Some(key),
    verify_integrity: true,
    include_metadata: true,
};

// Encode (compressed + encrypted)
let encoded = action.encode(&config)?;

// Decode (must use same key!). `RelayRLAction::decode` returns `Self` directly.
let decoded = RelayRLAction::decode(&encoded, &config)?;
assert_eq!(decoded.get_rew(), 2.5);
```

### 4. Full Pipeline with Integrity Verification

```rust
use relayrl_types::prelude::action::{RelayRLAction, CodecConfig};
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use relayrl_types::prelude::codec::CompressionScheme;
use relayrl_types::data::utilities::encrypt::generate_key;

let mut trajectory = RelayRLTrajectory::new(100);
for i in 0..50 {
    trajectory.add_action(RelayRLAction::minimal(i as f32, false));
}

// Full codec configuration. Requires "codec-full" (compression + encryption +
// integrity + metadata + quantization).
let key = generate_key();
let config = CodecConfig {
    compression: Some(CompressionScheme::Lz4),
    encryption_key: Some(key),
    verify_integrity: true,      // Enable BLAKE3 checksums
    include_metadata: true,
};

// Encode: Serialize → Compress → Encrypt → Checksum
let encoded = trajectory.encode(&config)?;

// Integrity is automatically verified during decode.
// `RelayRLTrajectory::decode` returns `(Self, usize)`.
let (decoded, _consumed) = RelayRLTrajectory::decode(&encoded, &config)?;

println!("Encoded {} actions", decoded.len());
println!("Total reward: {}", decoded.total_reward());
```

### 5. Chunking for Large Data

```rust
use relayrl_types::prelude::action::CodecConfig;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;

let mut trajectory = RelayRLTrajectory::new(10000);
// ... add many actions ...

// `encode_chunked`/`decode_chunked` require the "metadata" and "integrity" features.
let config = CodecConfig::default();
let chunk_size = 1024 * 1024; // 1MB chunks

// Encode and split into chunks for network transmission
let chunks = trajectory.encode_chunked(&config, chunk_size)?;
println!("Split into {} chunks", chunks.len());

// ... transmit chunks over network ...

// Reassemble on the receiving end
let decoded = RelayRLTrajectory::decode_chunked(&chunks, &config)?;
```

### 6. Metadata Tracking

```rust
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use uuid::Uuid;

// Create trajectory with full metadata
let trajectory = RelayRLTrajectory::with_metadata(
    1000,                          // max_length
    Some(Uuid::new_v4()),         // agent_id
    None,                          // env_id
    None,                          // env_label
    Some(42),                      // episode number
    Some(1000),                    // training_step
);

// Check age
println!("Trajectory age: {}s", trajectory.age_seconds());

// Access metadata
if let Some(agent_id) = trajectory.get_agent_id() {
    println!("Agent: {}", agent_id);
}
```

## Codec Pipeline

The encoding pipeline processes data in this order:

```
┌─────────────────┐
│  RelayRLAction  │
│ RelayRLTraject. │
└────────┬────────┘
         │
         ▼
   ┌──────────┐
   │ Bincode  │  Serialize to bytes
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Compress │  LZ4 or Zstd (optional)
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Encrypt  │  ChaCha20-Poly1305 (optional)
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Checksum │  BLAKE3 integrity (optional)
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  Output  │  Final encoded bytes
   └──────────┘
```

Decoding reverses this pipeline with automatic verification.

## ONNX Model Ingestion

`ModelModule::load_from_path` / `ModelModule::from_onnx_bytes` accept **any** ONNX graph that
exposes exactly **one tensor input and one tensor output** — the graph is not required to name
its input `"input"` or its output `"output"`. At load time, `relayrl_types`:

1. Commits the graph with ONNX Runtime and **introspects** the real input/output names,
   element types, and shapes directly from the loaded session (`Outlet`s), instead of assuming
   fixed names.
2. **Validates** the caller-supplied `ModelMetadata` against that discovered signature:
   - `input_dtype`/`output_dtype` must map to the graph's actual ONNX element type exactly.
   - `input_shape`/`output_shape` must have the same rank as the graph's declared shape.
   - Every *fixed* graph dimension (i.e. not a dynamic/symbolic axis) must equal the
     corresponding metadata dimension; dynamic axes (e.g. a symbolic batch dimension) accept
     any concrete size at runtime.
3. At inference time, binds the input tensor and reads the output tensor by their **discovered
   names**, and returns the model's *actual* ONNX Runtime output shape (important for graphs
   with a dynamic batch dimension), rather than a shape reconstructed from metadata.

Graphs with more than one input/output, or with non-tensor I/O (sequences, maps, optionals),
are rejected with a specific `ModelError` at construction time rather than failing later inside
`Session::run`.

`metadata.json` itself is unchanged and remains **required** — RelayRL still needs it for
shapes/dtypes/device defaults and as the cross-process wire contract used by the transport
layer (coordinator bundles, NATS `ModelFilesBundle`, and on-disk directories); it is now
*validated against the graph* instead of being trusted blindly. See
[`crates/relayrl_types/tests/onnx_ingestion.rs`](tests/onnx_ingestion.rs) for worked examples,
including non-standard I/O names, dynamic batch dimensions, `bool` tensors, and each rejection
case.

## Performance Tips

- **LZ4**: Best for real-time inference (3-4 GB/s decompression)
- **Zstd**: Best compression ratio for training data (5-10x reduction)
- **Chunking**: Use for trajectories > 10MB for network transmission
- **Integrity**: Minimal overhead (~50ns per MB with BLAKE3)
- **Encryption**: ~1 GB/s with ChaCha20-Poly1305

## Examples

See the `tests/` directory for more examples:
- Basic action/trajectory usage
- Compression benchmarks
- Encryption examples
- Chunked network transmission

## License

Apache-2.0
