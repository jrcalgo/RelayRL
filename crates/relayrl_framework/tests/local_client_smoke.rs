#![cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]

mod common;

use burn_ndarray::NdArrayDevice;
use burn_tensor::{Float, Tensor, TensorData};
use common::{TestBackend, try_load_test_model_module};
use relayrl_framework::prelude::network::{AgentBuilder, RelayRLActors, RelayRLStepDriven};
use relayrl_framework::prelude::types::tensor::DeviceType;
use std::fs;
use tempfile::tempdir;

#[tokio::test]
async fn local_client_smoke_covers_build_start_request_and_shutdown()
-> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let config_path = temp_dir.path().join("client_config.json");
    fs::write(&config_path, "{}")?;
    let Some((_model_dir, default_model)) = try_load_test_model_module() else {
        return Ok(());
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
    assert_eq!(action.get_agent_id(), Some(&actor_info.id()));

    agent.shutdown().await?;
    Ok(())
}
