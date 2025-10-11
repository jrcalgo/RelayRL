extern crate relayrl_framework;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use relayrl_framework::bindings::python::network::client::o3_agent::PyRelayRLAgent;
use relayrl_framework::bindings::python::network::server::o3_training_server::PyTrainingServer;
use relayrl_framework::orchestration::tokio::utils::get_or_init_tokio_runtime;
use std::sync::Arc;
use std::time::Duration;

use relayrl_framework::network::client::agent_grpc::RelayRLAgentGrpcTrait;
use relayrl_framework::network::client::agent_zmq::RelayRLAgentZmqTrait;
use tch::{Device, Kind, Tensor};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

const TRAJECTORY_SIZES: [usize; 6] = [10, 50, 100, 250, 500, 1000];
const NEXT_ACTION_WAIT_MILLIS: [u64; 8] = [1000, 750, 500, 333, 250, 100, 50, 25];

/// Benchmark gRPC agent inference
#[cfg(any(feature = "networks", feature = "grpc_network"))]
fn benchmark_grpc_agent_inference(c: &mut Criterion) {
    let rt: Arc<Runtime> = get_or_init_tokio_runtime();
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("gRPC Agent Inference");
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("grpc_agent_inference", |b| {
        b.iter(|| {
            rt.block_on(async {
                let server: Arc<RwLock<PyTrainingServer>> = PyTrainingServer::new(
                    "_BENCHMARK".to_string(),
                    2,
                    2,
                    1000,
                    1,
                    false,
                    false,
                    None,
                    None,
                    None,
                    None,
                    Some("grpc".to_string()),
                    None,
                    None,
                    None,
                )
                .await;

                let mut agent: PyRelayRLAgent =
                    PyRelayRLAgent::new(None, None, Some("grpc".to_string()), None, None, None)
                        .await;

                {
                    if let Some(grpc_agent) = &mut agent.agent_grpc {
                        let obs_tensor: Tensor = Tensor::ones([4], (Kind::Uint8, Device::Cpu));
                        let mask_tensor: Tensor = Tensor::from_slice(&[0.0]);
                        let reward: f32 = 0.0;

                        // Perform inference
                        let _ = grpc_agent
                            .request_for_action(obs_tensor, mask_tensor, reward)
                            .await
                            .expect("Failed to get RelayRLAction");
                    }
                }
            })
        })
    });

    group.finish();
}

/// Benchmark ZMQ agent inference
#[cfg(any(feature = "networks", feature = "zmq_server"))]
fn benchmark_zmq_agent_inference(c: &mut Criterion) {
    let rt: Arc<Runtime> = get_or_init_tokio_runtime();
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("ZMQ Agent Inference");
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("zmq_agent_inference", |b| {
        b.iter(|| {
            rt.block_on(async {
                let server: Arc<RwLock<PyTrainingServer>> = PyTrainingServer::new(
                    "_BENCHMARK".to_string(),
                    2,
                    2,
                    1000,
                    1,
                    false,
                    false,
                    None,
                    None,
                    None,
                    None,
                    Some("zmq".to_string()),
                    None,
                    None,
                    None,
                )
                .await;

                let mut agent: PyRelayRLAgent =
                    PyRelayRLAgent::new(None, None, Some("zmq".to_string()), None, None, None)
                        .await;

                {
                    if let Some(zmq_agent) = &mut agent.agent_zmq {
                        let obs_tensor: Tensor = Tensor::ones([4], (Kind::Uint8, Device::Cpu));
                        let mask_tensor: Tensor = Tensor::from_slice(&[0.0]);
                        let reward: f32 = 0.0;

                        // Perform inference
                        let _ = zmq_agent
                            .request_for_action(&obs_tensor, &mask_tensor, reward)
                            .expect("Failed to get RelayRLAction");
                    }
                }
            })
        })
    });

    group.finish();
}

/// Benchmarks for round-trip latency between gRPC client and server
#[cfg(any(feature = "networks", feature = "grpc_network"))]
fn benchmark_grpc_latency(c: &mut Criterion) {
    let rt: Arc<Runtime> = get_or_init_tokio_runtime();
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("gRPC Network Latency");
    group.measurement_time(Duration::from_secs(10));

    for size in TRAJECTORY_SIZES.iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                rt.block_on(async {
                    let server: Arc<RwLock<PyTrainingServer>> = PyTrainingServer::new(
                        "_BENCHMARK".to_string(),
                        2,
                        2,
                        1000,
                        1,
                        false,
                        false,
                        None,
                        None,
                        None,
                        None,
                        Some("grpc".to_string()),
                        None,
                        None,
                        None,
                    )
                    .await;

                    let mut agent: PyRelayRLAgent =
                        PyRelayRLAgent::new(None, None, Some("grpc".to_string()), None, None, None)
                            .await;

                    {
                        // Send enough trajectories to a trigger a new model update (after the handshake)
                        if let Some(grpc_agent) = &mut agent.agent_grpc {
                            let reward: f32 = 0.0;

                            let mut iter: i32 = 0;
                            loop {
                                // Stop after the first model update
                                if grpc_agent.get_model_version().await > 1 {
                                    break;
                                }
                                // Flag the last action as done, _BENCHMARK will save a new model
                                if iter > size as i32 {
                                    grpc_agent.flag_last_action(reward).await;
                                    iter = -1;
                                }
                                // So long as it's not the last action, iterate
                                if iter >= 0 {
                                    iter += 1
                                }

                                let obs_tensor: Tensor =
                                    Tensor::ones([4], (Kind::Uint8, Device::Cpu));
                                let mask_tensor: Tensor = Tensor::from_slice(&[0.0]);

                                // Request actions (fills trajectory buffer)
                                let _ = grpc_agent
                                    .request_for_action(obs_tensor, mask_tensor, reward)
                                    .await
                                    .expect("Failed to get RelayRLAction");

                                tokio::time::sleep(Duration::from_millis(50)).await;
                            }
                        }
                    }
                })
            })
        });
    }
    group.finish();
}

/// Benchmarks for round-trip latency between ZMQ client and server
#[cfg(any(feature = "networks", feature = "zmq_server"))]
fn benchmark_zmq_latency(c: &mut Criterion) {
    let rt: Arc<Runtime> = get_or_init_tokio_runtime();
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("ZMQ Network Latency");
    group.measurement_time(Duration::from_secs(10));

    for size in TRAJECTORY_SIZES.iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                rt.block_on(async {
                    let server: Arc<RwLock<PyTrainingServer>> = PyTrainingServer::new(
                        "_BENCHMARK".to_string(),
                        2,
                        2,
                        1000,
                        1,
                        false,
                        false,
                        None,
                        None,
                        None,
                        None,
                        Some("zmq".to_string()),
                        None,
                        None,
                        None,
                    )
                    .await;

                    let mut agent: PyRelayRLAgent =
                        PyRelayRLAgent::new(None, None, Some("zmq".to_string()), None, None, None)
                            .await;

                    {
                        // Send enough trajectories to a trigger a new model update (after the handshake)
                        if let Some(zmq_agent) = &mut agent.agent_zmq {
                            let obs_tensor: Tensor = Tensor::ones([4], (Kind::Uint8, Device::Cpu));
                            let mask_tensor: Tensor = Tensor::from_slice(&[0.0]);
                            let reward: f32 = 0.0;

                            let mut iter: i32 = 0;
                            loop {
                                // Stop after the first model update
                                if zmq_agent.get_model_version() > 1 {
                                    break;
                                }
                                // Flag the last action as done, _BENCHMARK will save a new model
                                if iter > size as i32 {
                                    zmq_agent.flag_last_action(reward);
                                    iter = -1;
                                }
                                // So long as it's not the last action, iterate
                                if iter >= 0 {
                                    iter += 1
                                }

                                // Request actions (fills trajectory buffer)
                                let _ = zmq_agent
                                    .request_for_action(&obs_tensor, &mask_tensor, reward)
                                    .expect("Failed to get RelayRLAction");

                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }
                        }
                    }
                })
            })
        });
    }
    group.finish();
}

/// Benchmarks for throughput between gRPC client and server
#[cfg(any(feature = "networks", feature = "grpc_network"))]
fn benchmark_grpc_throughput(c: &mut Criterion) {
    let rt: Arc<Runtime> = get_or_init_tokio_runtime();
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("gRPC Network Throughput");
    group.measurement_time(Duration::from_secs(60));

    for millis in NEXT_ACTION_WAIT_MILLIS.iter() {
        for size in TRAJECTORY_SIZES.iter() {
            group.throughput(Throughput::Bytes(*size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let server: Arc<RwLock<PyTrainingServer>> = PyTrainingServer::new(
                            "_BENCHMARK".to_string(),
                            2,
                            2,
                            1000,
                            1,
                            false,
                            false,
                            None,
                            None,
                            None,
                            None,
                            Some("grpc".to_string()),
                            None,
                            None,
                            None,
                        )
                        .await;

                        let mut agent: PyRelayRLAgent = PyRelayRLAgent::new(
                            None,
                            None,
                            Some("grpc".to_string()),
                            None,
                            None,
                            None,
                        )
                        .await;

                        {
                            // Send enough trajectories to a trigger a new model update (after the handshake)
                            if let Some(grpc_agent) = &mut agent.agent_grpc {
                                let reward: f32 = 0.0;

                                let mut iter: i32 = 0;
                                loop {
                                    // Stop after the first model update
                                    if grpc_agent.get_model_version().await > 1 {
                                        break;
                                    }
                                    // Flag the last action as done, _BENCHMARK will save a new model
                                    if iter > size as i32 {
                                        grpc_agent.flag_last_action(reward).await;
                                        iter = -1;
                                    }
                                    // So long as it's not the last action, iterate
                                    if iter >= 0 {
                                        iter += 1
                                    }

                                    let obs_tensor: Tensor =
                                        Tensor::ones([4], (Kind::Uint8, Device::Cpu));
                                    let mask_tensor: Tensor = Tensor::from_slice(&[0.0]);

                                    // Request actions (fills trajectory buffer)
                                    let _ = grpc_agent
                                        .request_for_action(obs_tensor, mask_tensor, reward)
                                        .await
                                        .expect("Failed to get RelayRLAction")
                                        .get_act();

                                    tokio::time::sleep(Duration::from_millis(*millis)).await;
                                }
                            }
                        }
                    })
                })
            });
        }
    }
}

/// Benchmarks for throughput between ZMQ client and server
#[cfg(any(feature = "networks", feature = "zmq_server"))]
fn benchmark_zmq_throughput(c: &mut Criterion) {
    let rt: Arc<Runtime> = get_or_init_tokio_runtime();
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("ZMQ Network Throughput");
    group.measurement_time(Duration::from_secs(60));

    for millis in NEXT_ACTION_WAIT_MILLIS.iter() {
        for size in TRAJECTORY_SIZES.iter() {
            group.throughput(Throughput::Bytes(*size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let server: Arc<RwLock<PyTrainingServer>> = PyTrainingServer::new(
                            "_BENCHMARK".to_string(),
                            2,
                            2,
                            1000,
                            1,
                            false,
                            false,
                            None,
                            None,
                            None,
                            None,
                            Some("zmq".to_string()),
                            None,
                            None,
                            None,
                        )
                        .await;

                        let mut agent: PyRelayRLAgent = PyRelayRLAgent::new(
                            None,
                            None,
                            Some("zmq".to_string()),
                            None,
                            None,
                            None,
                        )
                        .await;

                        {
                            // Send enough trajectories to a trigger a new model update (after the handshake)
                            if let Some(zmq_agent) = &mut agent.agent_zmq {
                                let obs_tensor: Tensor =
                                    Tensor::ones([4], (Kind::Uint8, Device::Cpu));
                                let mask_tensor: Tensor = Tensor::from_slice(&[0.0]);
                                let reward: f32 = 0.0;

                                let mut iter: i32 = 0;
                                loop {
                                    // Stop after the first model update
                                    if zmq_agent.get_model_version() > 1 {
                                        break;
                                    }
                                    // Flag the last action as done, _BENCHMARK will save a new model
                                    if iter > size as i32 {
                                        zmq_agent.flag_last_action(reward);
                                        iter = -1;
                                    }
                                    // So long as it's not the last action, iterate
                                    if iter >= 0 {
                                        iter += 1
                                    }

                                    // Request actions (fills trajectory buffer)
                                    let _ = zmq_agent
                                        .request_for_action(&obs_tensor, &mask_tensor, reward)
                                        .expect("Failed to get RelayRLAction")
                                        .get_act();

                                    tokio::time::sleep(Duration::from_millis(*millis)).await;
                                }
                            }
                        }
                    })
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    network_benches,
    benchmark_grpc_agent_inference,
    benchmark_zmq_agent_inference,
    benchmark_grpc_latency,
    benchmark_zmq_latency,
    benchmark_grpc_throughput,
    benchmark_zmq_throughput
);
criterion_main!(network_benches);
