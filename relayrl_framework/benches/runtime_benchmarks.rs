extern crate relayrl_framework;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use relayrl_framework::types::action::{RelayRLAction, TensorData};
use relayrl_framework::network::client::agent_wrapper::RelayRLAgent;
use relayrl_framework::network::server::training_server_wrapper::TrainingServer;
use relayrl_framework::orchestration::tokio::utils::get_or_init_tokio_runtime;
use relayrl_framework::types::trajectory::{RelayRLTrajectoryTrait, serialize_trajectory};
use serde_pickle as pickle;
use std::sync::Arc;
use std::time::Duration;
use tch::Tensor;
use tokio::runtime::Runtime;

const TENSOR_SIZES: [u64; 10] = [1, 10, 15, 25, 50, 100, 250, 500, 1000, 10000];

/// Benchmarks the `SafeTensor` serialization of a single tensor
fn benchmark_tensor_serialization(c: &mut Criterion) {
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("SafeTensor Tensor Serialization");
    group.measurement_time(Duration::from_secs(10));

    for size in TENSOR_SIZES.iter() {
        group.throughput(Throughput::Bytes(*size));

        group.bench_with_input(BenchmarkId::new("u8_tensor", size), size, |b, &size| {
            b.iter(|| {
                let u8_tensor: Tensor = Tensor::f_from_slice(&vec![u8::from(1); *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(u8_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });

        group.bench_with_input(BenchmarkId::new("i16_tensor", size), size, |b, &size| {
            b.iter(|| {
                let i16_tensor: Tensor = Tensor::f_from_slice(&vec![1i16; *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(i16_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });

        group.bench_with_input(BenchmarkId::new("i32_tensor", size), size, |b, &size| {
            b.iter(|| {
                let i32_tensor: Tensor = Tensor::f_from_slice(&vec![i32::from(1); *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(i32_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });

        group.bench_with_input(BenchmarkId::new("i64_tensor", size), size, |b, &size| {
            b.iter(|| {
                let i64_tensor: Tensor = Tensor::f_from_slice(&vec![i64::from(1); *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(i64_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });

        group.bench_with_input(BenchmarkId::new("f32_tensor", size), size, |b, &size| {
            b.iter(|| {
                let f32_tensor: Tensor = Tensor::f_from_slice(&vec![f32::from(1.0); *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(f32_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });

        group.bench_with_input(BenchmarkId::new("f64_tensor", size), size, |b, &size| {
            b.iter(|| {
                let f64_tensor: Tensor = Tensor::f_from_slice(&vec![f64::from(1.0); *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(f64_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });

        group.bench_with_input(BenchmarkId::new("bool_tensor", size), size, |b, &size| {
            b.iter(|| {
                let bool_tensor: Tensor = Tensor::f_from_slice(&vec![bool::from(true); *size as usize])
                    .expect("Failed to create tensor");
                let tensordata: TensorData =
                    TensorData::try_from(bool_tensor).expect("Failed to create TensorData");
                let _ = Tensor::try_from(&tensordata).expect("Failed to create tensor");
            });
        });
    }

    group.finish();
}

/// Benchmarks the combined `SafeTensor` and `pickle` serialization of a single action of `n` size tensors
fn benchmark_action_serialization(c: &mut Criterion) {
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("Action JSON Serialization");
    group.measurement_time(Duration::from_secs(10));

    for size in TENSOR_SIZES.iter() {
        group.bench_with_input(
            BenchmarkId::new("Action_all_tensors", size),
            size,
            |b, &size| {
                b.iter(|| {
                    group.throughput(Throughput::Bytes(*size * 3));
                    let obs_tensor: Tensor = Tensor::f_from_slice(&vec![f32::from(1.0); *size as usize])
                        .expect("Failed to create tensor");
                    let act_tensor: Tensor = Tensor::f_from_slice(&vec![f32::from(1.0); *size as usize])
                        .expect("Failed to create tensor");
                    let mask_tensor: Tensor = Tensor::f_from_slice(&vec![f32::from(1.0); *size as usize])
                        .expect("Failed to create tensor");
                    let obs_td: TensorData =
                        TensorData::try_from(obs_tensor).expect("Failed to create TensorData");
                    let act_td: TensorData =
                        TensorData::try_from(act_tensor).expect("Failed to create TensorData");
                    let mask_td: TensorData =
                        TensorData::try_from(mask_tensor).expect("Failed to create TensorData");
                    let action: RelayRLAction = RelayRLAction {
                        obs: Some(obs_td),
                        act: Some(act_td),
                        mask: Some(mask_td),
                        rew: 0.0,
                        data: None,
                        done: false,
                        reward_updated: false,
                    };

                    let serialized: Vec<u8> =
                        serde_json::to_vec(&action).expect("Failed to serialize");
                    let _deserialized: RelayRLAction =
                        serde_json::from_slice(&serialized).expect("Failed to deserialize");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Action_obs_only", size),
            size,
            |b, &size| {
                b.iter(|| {
                    group.throughput(Throughput::Bytes(*size));
                    let obs_tensor: Tensor = Tensor::f_from_slice(&vec![f32::from(1.0); *size as usize])
                        .expect("Failed to create tensor");
                    let obs_td: TensorData =
                        TensorData::try_from(obs_tensor).expect("Failed to create TensorData");
                    let action: RelayRLAction = RelayRLAction {
                        obs: Some(obs_td),
                        act: None,
                        mask: None,
                        rew: 0.0,
                        data: None,
                        done: false,
                        reward_updated: false,
                    };

                    let serialized: Vec<u8> =
                        serde_json::to_vec(&action).expect("Failed to serialize");
                    let _deserialized: RelayRLAction =
                        serde_json::from_slice(&serialized).expect("Failed to deserialize");
                });
            },
        );
    }

    group.finish();
}

/// Benchmarks the combined `SafeTensor` and `pickle` serialization of a single trajectory as `n` number of actions
fn benchmark_trajectory_serialization(c: &mut Criterion) {
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("Trajectory JSON Serialization");
    group.measurement_time(Duration::from_secs(10));

    // Count of actions in a trajectory
    let trajectory_sizes: [u64; 10] = [5, 10, 15, 25, 50, 100, 250, 500, 1000, 10000];

    // for each trajectory buffer of size `n`, use tensors of size `m`
    for tensor_size in TENSOR_SIZES.iter() {
        for traj_size in trajectory_sizes.iter() {
            group.throughput(Throughput::Bytes(*tensor_size * *traj_size));

            group.bench_with_input(
                BenchmarkId::new(format!("Trajectory_tensor_size_{}_traj_size_{}", tensor_size, traj_size), *tensor_size * *traj_size),
                &(*tensor_size, *traj_size),
                |b, &(tensor_size, traj_size)| {
                    b.iter(|| {
                        let mut trajectory = relayrl_framework::trajectory::RelayRLTrajectory::new(
                            Some(traj_size as u32 + 1), // +1 to ensure it fits
                            None,
                        );

                        // Create a tensor to reuse
                        let tensor =
                            Tensor::f_from_slice(&vec![f32::from(1.0); tensor_size as usize])
                                .expect("Failed to create tensor");

                        // Add actions to the trajectory
                        for i in 0..traj_size {
                            let is_terminal = i == traj_size - 1;
                            let action = RelayRLAction::from_tensors(
                                Some(&tensor),
                                Some(&tensor),
                                Some(&tensor),
                                i as f32 / traj_size as f32, // Reward between 0 and 1
                                None,
                                is_terminal,
                                false,
                            )
                            .expect("Failed to create action");

                            trajectory.add_action(&action, false);
                        }

                        // Serialize the trajectory
                        let serialized: Vec<u8> = serialize_trajectory(&trajectory);
                        let _deserialized: Vec<RelayRLAction> =
                            pickle::from_slice(&serialized, Default::default())
                                .expect("Failed to deserialize");
                    });
                },
            );
        }
    }

    group.finish();
}

// TODO: Add benchmark for roundtrip between gRPC server and python channel
/// Benchmarks throughput of roundtrip between gRPC server and python channel
// #[cfg(any(feature = "networks", feature = "grpc_network"))]
// fn benchmark_grpc_python_channel_throughput(c: &mut Criterion) {
//     let rt: Arc<Runtime> = get_or_init_tokio_runtime();
//     let mut group = c.benchmark_group("gRPC Python Channel Throughput");
//
//     group.finish();
// }

// TODO: Add benchmark for roundtrip between ZMQ server and python channel
/// Benchmarks throughput of roundtrip between ZMQ server and python channel
// #[cfg(any(feature = "networks", feature = "zmq_server"))]
// fn benchmark_zmq_python_channel_throughput(c: &mut Criterion) {
//     let rt: Arc<Runtime> = get_or_init_tokio_runtime();
//     let mut group = c.benchmark_group("ZMQ Python Channel Throughput");
//
//     group.finish();
// }

criterion_group!(
    runtime_benches,
    benchmark_tensor_serialization,
    benchmark_action_serialization,
    benchmark_trajectory_serialization,
    // benchmark_grpc_python_channel_throughput,
    // benchmark_zmq_python_channel_throughput
);
criterion_main!(runtime_benches);
