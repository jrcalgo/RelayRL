use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Float, Tensor, TensorData};
use relayrl_framework::prelude::network::{AgentBuilder, RelayRLActors, RelayRLStepDriven};
use relayrl_framework::prelude::types::tensor::DeviceType;
use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelError, ModelFileType, ModelMetadata, ModelModule};
use std::fs;
use tempfile::tempdir;

type TestBackend = NdArray<f32>;
const TEST_ONNX_IDENTITY: &[u8] = &[
    0x08, 0x07, 0x12, 0x0d, 0x72, 0x65, 0x6c, 0x61, 0x79, 0x72, 0x6c, 0x2d, 0x74, 0x65, 0x73, 0x74,
    0x73, 0x3a, 0x67, 0x0a, 0x23, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x06, 0x6f, 0x75,
    0x74, 0x70, 0x75, 0x74, 0x1a, 0x08, 0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x22, 0x08,
    0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x15, 0x72, 0x65, 0x6c, 0x61, 0x79, 0x72,
    0x6c, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x69, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x5a,
    0x13, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04,
    0x0a, 0x02, 0x08, 0x02, 0x62, 0x14, 0x0a, 0x06, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x12, 0x0a,
    0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x02, 0x10, 0x0d,
];

fn load_test_model_module() -> Result<(tempfile::TempDir, ModelModule<TestBackend>), ModelError> {
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

#[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
#[tokio::test]
async fn local_client_smoke_covers_build_start_request_and_shutdown()
-> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let config_path = temp_dir.path().join("client_config.json");
    fs::write(&config_path, "{}")?;
    let (_model_dir, default_model) = match load_test_model_module() {
        Ok(model) => model,
        Err(err) => {
            eprintln!("skipping ONNX smoke test because ONNX Runtime is unavailable: {err}");
            return Ok(());
        }
    };

    let (mut agent, params) = AgentBuilder::<TestBackend>::builder()
        .params()
        .default_model(default_model)
        .config_path(config_path.clone())
        .build()
        .await?;

    assert_eq!(params.data_routers, 1);
    assert_eq!(params.config_path.as_ref(), Some(&config_path));

    agent.start(params).await?;

    let actor_info = agent
        .new_actor::<1, 1>(
            DeviceType::Cpu,
            1000,
            None,
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
        )
        .await?;

    let observation = Tensor::<TestBackend, 1, Float>::from_data(
        TensorData::new(vec![1.0_f32, 2.0_f32], [2]),
        &NdArrayDevice::default(),
    );
    let action = agent
        .request_action::<1, 1, Float, Float>(
            &actor_info,
            observation,
            None::<Tensor<TestBackend, 1, Float>>,
            1.25,
        )
        .await?;

    assert_eq!(action.get_rew(), 1.25);
    assert_eq!(action.get_agent_id(), Some(&actor_info.id));

    agent.shutdown().await?;
    Ok(())
}
