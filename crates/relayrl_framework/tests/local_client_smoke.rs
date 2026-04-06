use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Float, Tensor, TensorData};
use relayrl_framework::prelude::network::{AgentBuilder, RelayRLAgentActors};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;
use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelFileType, ModelMetadata, ModelModule};
use std::fs;
use tch::{CModule, Device as TchDevice, Kind, Tensor as TchTensor};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

fn load_test_model_module() -> (tempfile::TempDir, ModelModule<TestBackend>) {
    let model_dir = tempdir().expect("tempdir should be created");
    let model_path = model_dir.path().join("test.pt");
    let metadata = ModelMetadata {
        model_file: "test.pt".to_string(),
        model_type: ModelFileType::Pt,
        input_dtype: DType::NdArray(NdArrayDType::F32),
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: vec![2],
        output_shape: vec![2],
        default_device: Some(DeviceType::Cpu),
    };

    let trace_inputs = [TchTensor::zeros([2], (Kind::Float, TchDevice::Cpu))];
    let mut trace_closure =
        |inputs: &[TchTensor]| -> Vec<TchTensor> { vec![inputs[0].shallow_clone()] };
    let traced_module = CModule::create_by_tracing(
        "relayrl_test_module",
        "forward",
        &trace_inputs,
        &mut trace_closure,
    )
    .expect("TorchScript smoke module should be traceable");
    traced_module
        .save(&model_path)
        .expect("TorchScript smoke module should be written");

    metadata
        .save_to_dir(model_dir.path())
        .expect("model metadata should be written");

    let model_module = ModelModule::<TestBackend>::load_from_path(model_dir.path())
        .expect("test TorchScript payload should load through the public model API");

    (model_dir, model_module)
}

#[tokio::test]
async fn local_client_smoke_covers_build_start_request_and_shutdown()
-> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let config_path = temp_dir.path().join("client_config.json");
    fs::write(&config_path, "{}")?;
    let (_model_dir, default_model) = load_test_model_module();

    let (mut agent, params) = AgentBuilder::<TestBackend, 1, 1, Float, Float>::builder()
        .default_device(DeviceType::Cpu)
        .default_model(default_model)
        .config_path(config_path.clone())
        .build()
        .await?;

    assert_eq!(params.actor_count, 1);
    assert_eq!(params.router_scale, 1);
    assert_eq!(params.default_device, DeviceType::Cpu);
    assert_eq!(params.config_path.as_ref(), Some(&config_path));

    agent.start(params).await?;

    let ids = agent.get_actor_ids()?;
    assert_eq!(ids.len(), 1);

    let observation = Tensor::<TestBackend, 1, Float>::from_data(
        TensorData::new(vec![1.0_f32, 2.0_f32], [2]),
        &NdArrayDevice::default(),
    );
    let actions = agent
        .request_action(ids.clone(), observation, None, 1.25)
        .await?;

    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].0, ids[0]);
    assert_eq!(actions[0].1.get_rew(), 1.25);
    assert_eq!(actions[0].1.get_agent_id(), Some(&ids[0]));

    agent.shutdown().await?;
    Ok(())
}
