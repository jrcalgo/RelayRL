use crate::network::client::agent::ClientModes;
use crate::network::client::agent::{ActorInferenceMode, ActorTrainingDataMode, ModelMode};
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    InferenceDispatcher, ScalingDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::data::transport_sink::{
    ScalingOperation, SyncClientInferenceTransportOps, SyncClientScalingTransportOps,
    SyncClientTrainingTransportOps, SyncClientTransportInterface, TransportError, TransportUuid,
    ZmqClientError,
};
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::Algorithm;

use active_uuid_registry::interface::reserve_id_with;
use relayrl_types::HyperparameterArgs;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use std::thread::sleep;
use tokio::sync::RwLock as TokioRwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

use super::ops::{ZmqInferenceOps, ZmqPool, ZmqTrainingOps};
use super::policies::{
    BackpressureController, CircuitBreaker, CircuitState, RetryPolicy, ZmqPolicyConfig,
};
use super::{ZmqInferenceExecution, ZmqTrainingExecution};

struct ZmqProtocol {
    circuit_breaker: CircuitBreaker,
    backpressure: BackpressureController,
    config: ZmqPolicyConfig,
}

pub(crate) struct ZmqInterface<B: Backend + BackendMatcher<Backend = B>> {
    zmq_inference_ops: ZmqInferenceOps,
    zmq_training_ops: ZmqTrainingOps,
    inference_protocol: Option<ZmqProtocol>,
    training_protocol: Option<ZmqProtocol>,
    scaling_protocol: Option<ZmqProtocol>,
    _phantom: PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientTransportInterface<B> for ZmqInterface<B> {
    fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
    ) -> Result<Self, TransportError> {
        let transport_id: TransportUuid =
            reserve_id_with(client_namespace.as_ref(), "zmq_transport_client", 42, 100)
                .map_err(TransportError::from)?;
        let zmq_pool = Arc::new(RwLock::new(ZmqPool::new().map_err(ZmqClientError::from)?));
        let zmq_inference_ops = ZmqInferenceOps::new(transport_id, zmq_pool.clone());
        let zmq_training_ops = ZmqTrainingOps::new(transport_id, zmq_pool.clone());

        let inference_protocol = match shared_client_modes.actor_inference_mode {
            ActorInferenceMode::Server(_) => {
                let config = ZmqPolicyConfig::for_inference();
                Some(ZmqProtocol {
                    circuit_breaker: CircuitBreaker::new(
                        config.circuit_breaker_threshold,
                        config.circuit_breaker_duration,
                    ),
                    backpressure: BackpressureController::new(config.max_concurrent_requests),
                    config,
                })
            }
            ActorInferenceMode::Local(_) => None,
        };

        let training_protocol = match shared_client_modes.actor_training_data_mode {
            ActorTrainingDataMode::Online(_) | ActorTrainingDataMode::Hybrid(_, _) => {
                let config = ZmqPolicyConfig::for_training();
                Some(ZmqProtocol {
                    circuit_breaker: CircuitBreaker::new(
                        config.circuit_breaker_threshold,
                        config.circuit_breaker_duration,
                    ),
                    backpressure: BackpressureController::new(config.max_concurrent_requests),
                    config,
                })
            }
            _ => None,
        };

        let scaling_protocol = match (
            &shared_client_modes.actor_inference_mode,
            &shared_client_modes.actor_training_data_mode,
        ) {
            (
                ActorInferenceMode::Local(_),
                ActorTrainingDataMode::Disabled | ActorTrainingDataMode::Offline(_),
            ) => None,
            _ => {
                let config = ZmqPolicyConfig::for_scaling();
                Some(ZmqProtocol {
                    circuit_breaker: CircuitBreaker::new(
                        config.circuit_breaker_threshold,
                        config.circuit_breaker_duration,
                    ),
                    backpressure: BackpressureController::new(config.max_concurrent_requests),
                    config,
                })
            }
        };

        Ok(Self {
            zmq_inference_ops,
            zmq_training_ops,
            inference_protocol,
            training_protocol,
            scaling_protocol,
            _phantom: PhantomData,
        })
    }

    fn shutdown(&self, server_addresses: SharedTransportAddresses) -> Result<(), TransportError> {
        unimplemented!()
    }
}

fn combine_scaling_results(
    result1: Option<Result<(), TransportError>>,
    result2: Option<Result<(), TransportError>>,
) -> Result<(), TransportError> {
    match (result1, result2) {
        (Some(Err(e)), Some(Err(e2))) => Err(TransportError::MultipleErrors(e.to_string(), e2.to_string())),
        (Some(Err(e)), None) => Err(e),
        (None, Some(Err(e))) => Err(e),
        (None, None) => Err(TransportError::InvalidState(
            "Inference and Training servers not initialized, and yet we have a scaling operation. This should never happen.".to_string(),
        )),
        _ => Ok(()),
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientScalingTransportOps<B>
    for ZmqInterface<B>
{
    fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            std::thread::scope(|s| {
                let inference_thread = if let Some(inference_protocol) =
                    self.inference_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = inference_protocol.backpressure.acquire();

                        if inference_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let inference_scaling_server_address: &str = server_addresses
                            .inference_addresses
                            .inference_scaling_server_address
                            .as_ref();

                        let mut attempts = 0;
                        loop {
                            let result =
                                <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_client_ids(
                                    &self,
                                    scaling_id,
                                    client_ids,
                                    inference_scaling_server_address,
                                );

                            match result {
                                Ok(_) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < inference_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    inference_protocol.circuit_breaker.record_failure();
                                    let delay = inference_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                }
                                Err(e) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let training_thread = if let Some(training_protocol) =
                    self.training_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = training_protocol.backpressure.acquire();

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            server_addresses.training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_client_ids(
                                &self,
                                scaling_id,
                                client_ids,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < training_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    training_protocol.circuit_breaker.record_failure();
                                    let delay = training_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                }
                                Err(e) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let inference_result = inference_thread.map(|thread| {
                    thread
                        .join()
                        .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                        .and_then(|r| r)
                });

                let training_result: Option<Result<(), TransportError>> =
                    training_thread.map(|thread| {
                        thread
                            .join()
                            .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                            .and_then(|r| r)
                    });

                combine_scaling_results(inference_result, training_result)
            })
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }

    fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            std::thread::scope(|s| {
                let inference_thread = if let Some(inference_protocol) =
                    self.inference_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = inference_protocol.backpressure.acquire();

                        if inference_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let inference_scaling_server_address: &str =
                            server_addresses.inference_addresses.inference_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_scaling_warning(
                                &self,
                                scaling_id,
                                operation,
                                inference_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < inference_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    inference_protocol.circuit_breaker.record_failure();
                                    let delay = inference_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                    continue;
                                }
                                Err(e) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let training_thread = if let Some(training_protocol) =
                    self.training_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = training_protocol.backpressure.acquire();

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            server_addresses.training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_scaling_warning(
                                &self,
                                scaling_id,
                                operation,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < training_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    training_protocol.circuit_breaker.record_failure();
                                    let delay = training_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                    continue;
                                }
                                Err(e) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let inference_result = inference_thread.map(|thread| {
                    thread
                        .join()
                        .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                        .and_then(|r| r)
                });

                let training_result: Option<Result<(), TransportError>> =
                    training_thread.map(|thread| {
                        thread
                            .join()
                            .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                            .and_then(|r| r)
                    });

                combine_scaling_results(inference_result, training_result)
            })
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }

    fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            std::thread::scope(|s| {
                let inference_thread = if let Some(inference_protocol) =
                    self.inference_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = inference_protocol.backpressure.acquire();

                        if inference_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let inference_scaling_server_address: &str =
                            server_addresses.inference_addresses.inference_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_scaling_complete(
                                &self,
                                scaling_id,
                                operation,
                                inference_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < inference_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    inference_protocol.circuit_breaker.record_failure();
                                    let delay = inference_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                    continue;
                                }
                                Err(e) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let training_thread = if let Some(training_protocol) =
                    self.training_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = training_protocol.backpressure.acquire();

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            server_addresses.training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_scaling_complete(
                                &self,
                                scaling_id,
                                operation,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < training_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    training_protocol.circuit_breaker.record_failure();
                                    let delay = training_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                    continue;
                                }
                                Err(e) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let inference_result = inference_thread.map(|thread| {
                    thread
                        .join()
                        .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                        .and_then(|r| r)
                });

                let training_result: Option<Result<(), TransportError>> =
                    training_thread.map(|thread| {
                        thread
                            .join()
                            .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                            .and_then(|r| r)
                    });

                combine_scaling_results(inference_result, training_result)
            })
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }

    fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            std::thread::scope(|s| {
                let inference_thread = if let Some(inference_protocol) =
                    self.inference_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = inference_protocol.backpressure.acquire();

                        if inference_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let inference_scaling_server_address: &str =
                            server_addresses.inference_addresses.inference_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_shutdown_signal(
                                &self,
                                scaling_id,
                                inference_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < inference_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    inference_protocol.circuit_breaker.record_failure();
                                    let delay = inference_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                    continue;
                                }
                                Err(e) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let training_thread = if let Some(training_protocol) =
                    self.training_protocol.as_ref()
                {
                    Some(s.spawn(|| {
                        let _permit = training_protocol.backpressure.acquire();

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            server_addresses.training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_shutdown_signal(
                                &self,
                                scaling_id,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(e)
                                    if attempts
                                        < training_protocol.config.retry_policy.max_attempts =>
                                {
                                    attempts += 1;
                                    training_protocol.circuit_breaker.record_failure();
                                    let delay = training_protocol
                                        .config
                                        .retry_policy
                                        .delay_for_attempt(attempts);
                                    sleep(delay);
                                }
                                Err(e) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    return Err(TransportError::MaxRetriesExceeded {
                                        cause: e.to_string(),
                                        attempts,
                                    });
                                }
                            }
                        }
                    }))
                } else {
                    None
                };

                let inference_result = inference_thread.map(|thread| {
                    thread
                        .join()
                        .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                        .and_then(|r| r)
                });

                let training_result: Option<Result<(), TransportError>> =
                    training_thread.map(|thread| {
                        thread
                            .join()
                            .map_err(|e| TransportError::JoinError(format!("{:?}", e)))
                            .and_then(|r| r)
                    });

                combine_scaling_results(inference_result, training_result)
            })
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientInferenceTransportOps<B>
    for ZmqInterface<B>
{
    fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        server_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError> {
        if let Some(protocol) = self.inference_protocol.as_ref() {
            let _permit = protocol.backpressure.acquire();

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let inference_server_address: &str = server_addresses
                .inference_addresses
                .inference_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_inference_request(
                    actor_id,
                    obs_bytes,
                    inference_server_address,
                );

                match result {
                    Ok(action) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(action);
                    }
                    Err(e) if attempts < protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        protocol.circuit_breaker.record_failure();
                        let delay = protocol.config.retry_policy.delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Inference protocol not initialized".to_string(),
            ));
        }
    }

    fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.inference_protocol.as_ref() {
            let _permit = protocol.backpressure.acquire();

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let inference_server_address: &str = server_addresses
                .inference_addresses
                .inference_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_flag_last_inference(
                    actor_id,
                    reward,
                    inference_server_address,
                );

                match result {
                    Ok(()) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(e) if attempts < protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        protocol.circuit_breaker.record_failure();
                        let delay = protocol.config.retry_policy.delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Inference protocol not initialized".to_string(),
            ));
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ZmqInferenceExecution for ZmqInterface<B> {
    fn execute_send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_inference_request(
            &self.zmq_inference_ops,
            actor_id,
            obs_bytes,
            inference_server_address,
        )
    }

    fn execute_send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_flag_last_inference(
            &self.zmq_inference_ops,
            actor_id,
            reward,
            inference_server_address,
        )
    }

    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_client_ids(
            &self.zmq_inference_ops,
            scaling_id,
            client_ids,
            inference_scaling_server_address,
        )
    }

    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_scaling_warning(
            &self.zmq_inference_ops,
            scaling_id,
            operation,
            inference_scaling_server_address,
        )
    }

    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_scaling_complete(
            &self.zmq_inference_ops,
            scaling_id,
            operation,
            inference_scaling_server_address,
        )
    }

    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_shutdown_signal(
            &self.zmq_inference_ops,
            scaling_id,
            inference_scaling_server_address,
        )
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientTrainingTransportOps<B>
    for ZmqInterface<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            let _permit = protocol.backpressure.acquire();

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let agent_listener_address: &str = server_addresses
                .training_addresses
                .agent_listener_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_algorithm_init_request(
                    scaling_id,
                    model_mode.clone(),
                    algorithm.clone(),
                    hyperparams.clone(),
                    agent_listener_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(e) if attempts < protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        protocol.circuit_breaker.record_failure();
                        let delay = protocol.config.retry_policy.delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Training protocol not initialized".to_string(),
            ));
        }
    }

    fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        server_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            let _premit = protocol.backpressure.acquire();

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let agent_listener_address: &str = server_addresses
                .training_addresses
                .agent_listener_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_initial_model_handshake(actor_id, agent_listener_address);

                match result {
                    Ok(model) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(model);
                    }
                    Err(e) if attempts < protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        protocol.circuit_breaker.record_failure();
                        let delay = protocol.config.retry_policy.delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Training protocol not initialized".to_string(),
            ));
        }
    }

    fn send_trajectory(
        &self,
        buffer_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            let _permit = protocol.backpressure.acquire();

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let trajectory_server_address: &str = server_addresses
                .training_addresses
                .trajectory_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_trajectory(
                    buffer_id,
                    encoded_trajectory.clone(),
                    trajectory_server_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(e) if attempts < protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        protocol.circuit_breaker.record_failure();
                        let delay = protocol.config.retry_policy.delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Training protocol not initialized".to_string(),
            ));
        }
    }

    fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        server_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let model_server_address = server_addresses
                .training_addresses
                .model_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_listen_for_model(
                    receiver_id,
                    global_dispatcher_tx.clone(),
                    model_server_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(e) if attempts < protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        protocol.circuit_breaker.record_failure();
                        let delay = protocol.config.retry_policy.delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Training protocol not initialized".to_string(),
            ));
        }
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ZmqTrainingExecution<B> for ZmqInterface<B> {
    #[inline]
    fn execute_listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        model_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_listen_for_model(
            &self.zmq_training_ops,
            receiver_id,
            global_dispatcher_tx,
            model_server_address,
        )
    }

    #[inline]
    fn execute_send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_algorithm_init_request(
            &self.zmq_training_ops,
            scaling_id,
            model_mode,
            algorithm,
            hyperparams,
            agent_listener_address,
        )
    }

    #[inline]
    fn execute_initial_model_handshake(
        &self,
        actor_id: &Uuid,
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_initial_model_handshake(
            &self.zmq_training_ops,
            actor_id,
            agent_listener_address,
        )
    }

    #[inline]
    fn execute_send_trajectory(
        &self,
        buffer_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_trajectory(
            &self.zmq_training_ops,
            buffer_id,
            encoded_trajectory,
            trajectory_server_address,
        )
    }

    #[inline]
    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &[(String, Uuid)],
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_client_ids(
            &self.zmq_training_ops,
            scaling_id,
            client_ids,
            training_scaling_server_address,
        )
    }

    #[inline]
    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_scaling_warning(
            &self.zmq_training_ops,
            scaling_id,
            operation,
            training_scaling_server_address,
        )
    }

    #[inline]
    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_scaling_complete(
            &self.zmq_training_ops,
            scaling_id,
            operation,
            training_scaling_server_address,
        )
    }

    #[inline]
    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_shutdown_signal(
            &self.zmq_training_ops,
            scaling_id,
            training_scaling_server_address,
        )
    }
}
