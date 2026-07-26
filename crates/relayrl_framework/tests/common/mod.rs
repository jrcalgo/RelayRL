//! Shared helpers for the offline (local, non-transport) client integration tests.
//!
//! Every helper here is scoped to the local/default client runtime: no ZMQ or NATS transport
//! feature is ever enabled, and every test file that uses this module gates itself with
//! `#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]`.
#![allow(dead_code)]

use burn_ndarray::NdArray;
use relayrl_framework::prelude::network::{
    ActorDataMode, ActorInferenceMode, ActorInfo, AgentBuilder, RelayRLActors, RelayRLAgent,
};
use relayrl_types::data::tensor::DeviceType;
use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelError, ModelFileType, ModelMetadata, ModelModule};
use std::time::Duration;
use tempfile::{TempDir, tempdir};

/// Backend used throughout the offline integration tests.
pub type TestBackend = NdArray<f32>;

/// A tiny single-input/single-output identity ONNX graph (`output = input`), shape `[2]`,
/// `f32` in and out. Shared across every offline test that needs a real, loadable model.
pub const TEST_ONNX_IDENTITY: &[u8] = &[
    0x08, 0x07, 0x12, 0x0d, 0x72, 0x65, 0x6c, 0x61, 0x79, 0x72, 0x6c, 0x2d, 0x74, 0x65, 0x73, 0x74,
    0x73, 0x3a, 0x67, 0x0a, 0x23, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x06, 0x6f, 0x75,
    0x74, 0x70, 0x75, 0x74, 0x1a, 0x08, 0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x22, 0x08,
    0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x15, 0x72, 0x65, 0x6c, 0x61, 0x79, 0x72,
    0x6c, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x5a,
    0x13, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04,
    0x0a, 0x02, 0x08, 0x02, 0x62, 0x14, 0x0a, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x0a,
    0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x02, 0x10, 0x0d,
];

/// Loads [`TEST_ONNX_IDENTITY`] into a real `ModelModule`. Returns `Err` when ONNX Runtime's
/// native binaries are unavailable (for example in a network-restricted sandbox); callers
/// should treat that as "skip this test" rather than a failure, matching the rest of the suite.
pub fn load_test_model_module() -> Result<(TempDir, ModelModule<TestBackend>), ModelError> {
    let model_dir = tempdir().expect("tempdir should be created");
    let metadata = ModelMetadata {
        model_file: "test.onnx".to_string(),
        model_type: ModelFileType::Onnx,
        input_dtype: DType::NdArray(NdArrayDType::F32),
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: vec![2],
        output_shape: vec![2],
        default_device: Some(DeviceType::Cpu),
    };

    let model_module =
        ModelModule::<TestBackend>::from_onnx_bytes(TEST_ONNX_IDENTITY.to_vec(), metadata)?;

    Ok((model_dir, model_module))
}

/// A started `RelayRLAgent` plus the temp-dir guards that must outlive it.
pub struct AgentCtx {
    pub agent: RelayRLAgent<TestBackend>,
    _model_dir: TempDir,
    _config_dir: TempDir,
}

/// Builds and starts an offline agent under `data_mode` (with the default `Client(Independent)`
/// inference mode) and the shared identity model as its default model. Returns `Ok(None)` when
/// ONNX Runtime is unavailable, signalling that the calling test should skip itself
/// (`return Ok(());`) rather than fail.
pub async fn start_offline_agent(
    data_mode: ActorDataMode,
) -> Result<Option<AgentCtx>, Box<dyn std::error::Error>> {
    start_offline_agent_with_modes(ActorInferenceMode::default(), data_mode).await
}

/// Like [`start_offline_agent`], but also lets the caller pick the actor inference mode (for
/// example `ActorInferenceMode::Client(ModelMode::Shared)`).
pub async fn start_offline_agent_with_modes(
    inference_mode: ActorInferenceMode,
    data_mode: ActorDataMode,
) -> Result<Option<AgentCtx>, Box<dyn std::error::Error>> {
    let (model_dir, default_model) = match load_test_model_module() {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("skipping test because ONNX Runtime is unavailable: {err}");
            return Ok(None);
        }
    };

    let config_dir = tempdir()?;
    let config_path = config_dir.path().join("client_config.json");
    std::fs::write(&config_path, "{}")?;

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .modes()
        .actor_inference_mode(inference_mode)
        .actor_data_mode(data_mode)
        .params()
        .default_model(default_model)
        .config_path(config_path)
        .build()
        .await?;

    agent.start(params).await?;

    Ok(Some(AgentCtx {
        agent,
        _model_dir: model_dir,
        _config_dir: config_dir,
    }))
}

/// Polls `agent.get_model_versions([actor])` until it reports something other than `baseline`,
/// or `timeout` elapses. Model-update dispatch is fire-and-forget (the message is enqueued for
/// the actor's own task to process), so version changes land asynchronously relative to
/// `update_models` returning.
pub async fn wait_for_model_version_change(
    agent: &RelayRLAgent<TestBackend>,
    actor: &ActorInfo,
    baseline: i64,
    timeout: Duration,
) -> Option<i64> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Ok(versions) = agent.get_model_versions(std::slice::from_ref(actor)).await
            && let Some((_, version)) = versions.first()
            && *version != baseline
        {
            return Some(*version);
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Polls `check` every 25ms until it returns `true` or `timeout` elapses. Used for assertions on
/// effects that land asynchronously through the router/buffer pipeline (cache pushes, file
/// writes, hot-swapped model versions) instead of a fixed sleep that would either flake under
/// load or waste time when the effect lands immediately.
pub async fn wait_until<F: FnMut() -> bool>(mut check: F, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if check() {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Default poll timeout for async router/buffer side effects in these tests.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(5);
