use crate::network::client::agent::ClientModes;
use crate::network::client::agent::{ActorInferenceMode, ActorTrainingDataMode, ModelMode};
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::data::transport_sink::combine_scaling_results;
use crate::network::client::runtime::data::transport_sink::{
    ScalingOperation, SyncClientInferenceTransportOps, SyncClientScalingTransportOps,
    SyncClientTrainingTransportOps, SyncClientTransportInterface, TransportError, TransportUuid,
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
use tokio::sync::mpsc::Sender;

use super::ops::{ZmqInferenceOps, ZmqPool, ZmqTrainingOps};
use super::policies::{BackpressureController, CircuitBreaker, ZmqPolicyConfig};
use super::{ZmqInferenceExecution, ZmqTrainingExecution};

use active_uuid_registry::{ContextString, NamespaceString, registry_uuid::Uuid};

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
        let transport_id: TransportUuid = reserve_id_with(
            client_namespace.as_ref(),
            crate::network::ZMQ_CLIENT_CONTEXT,
            42,
            100,
        )
        .map_err(TransportError::from)?;

        let transport_entry = (
            client_namespace.to_string(),
            crate::network::ZMQ_CLIENT_CONTEXT.to_string(),
            transport_id,
        );

        let zmq_pool = Arc::new(RwLock::new(ZmqPool::new(client_namespace.clone())));
        let zmq_inference_ops = ZmqInferenceOps::new(transport_entry.clone(), zmq_pool.clone());
        let zmq_training_ops = ZmqTrainingOps::new(transport_entry, zmq_pool.clone());

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

    fn shutdown(&self) -> Result<(), TransportError> {
        let training_result = Some(self.zmq_training_ops.shutdown());
        let inference_result = Some(self.zmq_inference_ops.shutdown());
        combine_scaling_results(inference_result, training_result)
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientScalingTransportOps<B>
    for ZmqInterface<B>
{
    fn send_client_ids(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        client_ids: Vec<(NamespaceString, ContextString, Uuid)>,
        replace_context: bool,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            let _permit = scaling_protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if scaling_protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let mut attempts = 0;
            loop {
                let result = std::thread::scope(|s| {
                    let inference_thread =
                        if let Some(inference_protocol) = self.inference_protocol.as_ref() {
                            Some(s.spawn(|| {
                                let _permit = inference_protocol
                                    .backpressure
                                    .acquire()
                                    .map_err(TransportError::from)?;

                                if inference_protocol.circuit_breaker.is_open() {
                                    return Err(TransportError::CircuitOpen);
                                }

                                let inference_scaling_server_address: &str = transport_addresses
                                    .zmq_inference_addresses
                                    .inference_scaling_server_address
                                    .as_ref();

                                let mut attempts = 0;
                                loop {
                                    let result =
                                <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_client_ids(
                                    &self,
                                    &scaling_entry,
                                    &client_ids,
                                    inference_scaling_server_address,
                                );

                                    match result {
                                        Ok(_) => {
                                            inference_protocol.circuit_breaker.record_success();
                                            return Ok(());
                                        }
                                        Err(_)
                                            if attempts
                                                < inference_protocol
                                                    .config
                                                    .retry_policy
                                                    .max_attempts =>
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
                        let _permit = training_protocol.backpressure.acquire().map_err(TransportError::from)?;

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            transport_addresses.zmq_training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_client_ids(
                                &self,
                                &scaling_entry,
                                &client_ids,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(_)
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
                });

                match result {
                    Ok(_) => {
                        scaling_protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        scaling_protocol.circuit_breaker.record_failure();
                    }
                    Err(e) => {
                        scaling_protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }

    fn send_scaling_warning(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            let _permit = scaling_protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if scaling_protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let mut attempts = 0;
            loop {
                let result = std::thread::scope(|s| {
                    let inference_thread = if let Some(inference_protocol) =
                        self.inference_protocol.as_ref()
                    {
                        Some(s.spawn(|| {
                        let _permit = inference_protocol.backpressure.acquire().map_err(TransportError::from)?;

                        if inference_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let inference_scaling_server_address: &str =
                            transport_addresses.zmq_inference_addresses.inference_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_scaling_warning(
                                &self,
                                &scaling_entry,
                                &operation,
                                inference_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(_)
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
                        let _permit = training_protocol.backpressure.acquire().map_err(TransportError::from)?;

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            transport_addresses.zmq_training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_scaling_warning(
                                &self,
                                &scaling_entry,
                                &operation,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(_)
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
                });

                match result {
                    Ok(_) => {
                        scaling_protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        scaling_protocol.circuit_breaker.record_failure();
                    }
                    Err(e) => {
                        scaling_protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }

    fn send_scaling_complete(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            let _permit = scaling_protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if scaling_protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let mut attempts = 0;
            loop {
                let result = std::thread::scope(|s| {
                    let inference_thread = if let Some(inference_protocol) =
                        self.inference_protocol.as_ref()
                    {
                        Some(s.spawn(|| {
                        let _permit = inference_protocol.backpressure.acquire().map_err(TransportError::from)?;

                        if inference_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let inference_scaling_server_address: &str =
                            transport_addresses.zmq_inference_addresses.inference_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_scaling_complete(
                                &self,
                                &scaling_entry,
                                &operation,
                                inference_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(_)
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
                        let _permit = training_protocol.backpressure.acquire().map_err(TransportError::from)?;

                        if training_protocol.circuit_breaker.is_open() {
                            return Err(TransportError::CircuitOpen);
                        }

                        let training_scaling_server_address: &str =
                            transport_addresses.zmq_training_addresses.training_scaling_server_address.as_ref();

                        let mut attempts = 0;
                        loop {
                            let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_scaling_complete(
                                &self,
                                &scaling_entry,
                                &operation,
                                training_scaling_server_address,
                            );

                            match result {
                                Ok(_) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Ok(());
                                }
                                Err(_)
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
                });

                match result {
                    Ok(_) => {
                        scaling_protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        scaling_protocol.circuit_breaker.record_failure();
                    }
                    Err(e) => {
                        scaling_protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        } else {
            return Err(TransportError::InvalidState(
                "Scaling protocol not initialized".to_string(),
            ));
        }
    }

    fn send_shutdown_signal(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(scaling_protocol) = self.scaling_protocol.as_ref() {
            let _permit = scaling_protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if scaling_protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let mut attempts = 0;
            loop {
                let result = std::thread::scope(|s| {
                    let inference_thread = if let Some(inference_protocol) =
                        self.inference_protocol.as_ref()
                    {
                        Some(s.spawn(|| {
                            let _permit = inference_protocol.backpressure.acquire().map_err(TransportError::from)?;

                            if inference_protocol.circuit_breaker.is_open() {
                                return Err(TransportError::CircuitOpen);
                            }

                            let inference_scaling_server_address: &str =
                                transport_addresses.zmq_inference_addresses.inference_scaling_server_address.as_ref();

                            let mut attempts = 0;
                            loop {
                                let result = <ZmqInterface<B> as ZmqInferenceExecution>::execute_send_shutdown_signal(
                                    &self,
                                    &scaling_entry,
                                    inference_scaling_server_address,
                                );

                                match result {
                                    Ok(_) => {
                                        inference_protocol.circuit_breaker.record_success();
                                        return Ok(());
                                    }
                                    Err(_)
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
                            let _permit = training_protocol.backpressure.acquire().map_err(TransportError::from)?;

                            if training_protocol.circuit_breaker.is_open() {
                                return Err(TransportError::CircuitOpen);
                            }

                            let training_scaling_server_address: &str =
                                transport_addresses.zmq_training_addresses.training_scaling_server_address.as_ref();

                            let mut attempts = 0;
                            loop {
                                let result = <ZmqInterface<B> as ZmqTrainingExecution<B>>::execute_send_shutdown_signal(
                                    &self,
                                    &scaling_entry,
                                    training_scaling_server_address,
                                );

                                match result {
                                    Ok(_) => {
                                        training_protocol.circuit_breaker.record_success();
                                        return Ok(());
                                    }
                                    Err(_)
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
                });

                match result {
                    Ok(_) => {
                        scaling_protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                        attempts += 1;
                        scaling_protocol.circuit_breaker.record_failure();
                        let delay = scaling_protocol
                            .config
                            .retry_policy
                            .delay_for_attempt(attempts);
                        sleep(delay);
                    }
                    Err(e) => {
                        scaling_protocol.circuit_breaker.record_failure();
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
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
    fn send_inference_model_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        model_mode: ModelMode,
        model_module: Option<ModelModule<B>>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.inference_protocol.as_ref() {
            let _permit = protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let inference_scaling_server_address: &str = transport_addresses
                .zmq_inference_addresses
                .inference_scaling_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_inference_model_init_request(
                    &scaling_entry,
                    &model_mode,
                    &model_module,
                    inference_scaling_server_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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

    fn send_inference_request(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        obs_bytes: Vec<u8>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError> {
        if let Some(protocol) = self.inference_protocol.as_ref() {
            let _permit = protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let inference_server_address: &str = transport_addresses
                .zmq_inference_addresses
                .inference_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_inference_request(
                    &actor_entry,
                    &obs_bytes,
                    inference_server_address,
                );

                match result {
                    Ok(action) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(action);
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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
        actor_entry: (NamespaceString, ContextString, Uuid),
        reward: f32,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.inference_protocol.as_ref() {
            let _permit = protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let inference_server_address: &str = transport_addresses
                .zmq_inference_addresses
                .inference_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_flag_last_inference(
                    &actor_entry,
                    &reward,
                    inference_server_address,
                );

                match result {
                    Ok(()) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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
        actor_entry: &(NamespaceString, ContextString, Uuid),
        obs_bytes: &[u8],
        inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_inference_request(
            &self.zmq_inference_ops,
            actor_entry,
            obs_bytes,
            inference_server_address,
        )
    }

    fn execute_send_flag_last_inference(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        reward: &f32,
        inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_flag_last_inference(
            &self.zmq_inference_ops,
            actor_entry,
            reward,
            inference_server_address,
        )
    }

    fn execute_send_inference_model_init_request<MB: Backend + BackendMatcher<Backend = MB>>(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        model_mode: &ModelMode,
        model_module: &Option<ModelModule<MB>>,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_inference_model_init_request(
            &self.zmq_inference_ops,
            scaling_entry,
            model_mode,
            model_module,
            inference_scaling_server_address,
        )
    }

    fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_client_ids(
            &self.zmq_inference_ops,
            scaling_entry,
            client_ids,
            inference_scaling_server_address,
        )
    }

    fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_scaling_warning(
            &self.zmq_inference_ops,
            scaling_entry,
            operation,
            inference_scaling_server_address,
        )
    }

    fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_scaling_complete(
            &self.zmq_inference_ops,
            scaling_entry,
            operation,
            inference_scaling_server_address,
        )
    }

    fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        inference_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqInferenceOps as ZmqInferenceExecution>::execute_send_shutdown_signal(
            &self.zmq_inference_ops,
            scaling_entry,
            inference_scaling_server_address,
        )
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncClientTrainingTransportOps<B>
    for ZmqInterface<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            let _permit = protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let agent_listener_address: &str = transport_addresses
                .zmq_training_addresses
                .agent_listener_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_algorithm_init_request(
                    &scaling_entry,
                    &actor_entries,
                    &model_mode,
                    &algorithm,
                    &hyperparams,
                    agent_listener_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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
        actor_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            let _premit = protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let agent_listener_address: &str = transport_addresses
                .zmq_training_addresses
                .agent_listener_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result =
                    self.execute_initial_model_handshake(&actor_entry, agent_listener_address);

                match result {
                    Ok(model) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(model);
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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
        buffer_entry: (NamespaceString, ContextString, Uuid),
        encoded_trajectory: EncodedTrajectory,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            let _permit = protocol
                .backpressure
                .acquire()
                .map_err(TransportError::from)?;

            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let trajectory_server_address: &str = transport_addresses
                .zmq_training_addresses
                .trajectory_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_send_trajectory(
                    &buffer_entry,
                    &encoded_trajectory,
                    trajectory_server_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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
        receiver_entry: (NamespaceString, ContextString, Uuid),
        global_dispatcher_tx: Sender<RoutedMessage>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        if let Some(protocol) = self.training_protocol.as_ref() {
            if protocol.circuit_breaker.is_open() {
                return Err(TransportError::CircuitOpen);
            }

            let model_server_address = transport_addresses
                .zmq_training_addresses
                .model_server_address
                .as_ref();

            let mut attempts = 0;
            loop {
                let result = self.execute_listen_for_model(
                    &receiver_entry,
                    &global_dispatcher_tx.clone(),
                    model_server_address,
                );

                match result {
                    Ok(_) => {
                        protocol.circuit_breaker.record_success();
                        return Ok(());
                    }
                    Err(_) if attempts < protocol.config.retry_policy.max_attempts => {
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

    fn stop_model_listener(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
    ) -> Result<(), TransportError> {
        self.zmq_training_ops.stop_model_listener(&receiver_entry)
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ZmqTrainingExecution<B> for ZmqInterface<B> {
    #[inline]
    fn execute_listen_for_model(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
        global_dispatcher_tx: &Sender<RoutedMessage>,
        model_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_listen_for_model(
            &self.zmq_training_ops,
            receiver_entry,
            global_dispatcher_tx,
            model_server_address,
        )
    }

    #[inline]
    fn execute_send_algorithm_init_request(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        actor_entries: &[(NamespaceString, ContextString, Uuid)],
        model_mode: &ModelMode,
        algorithm: &Algorithm,
        hyperparams: &HashMap<Algorithm, HyperparameterArgs>,
        agent_listener_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_algorithm_init_request(
            &self.zmq_training_ops,
            scaling_entry,
            actor_entries,
            model_mode,
            algorithm,
            hyperparams,
            agent_listener_address,
        )
    }

    #[inline]
    fn execute_initial_model_handshake(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        agent_listener_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_initial_model_handshake(
            &self.zmq_training_ops,
            actor_entry,
            agent_listener_address,
        )
    }

    #[inline]
    fn execute_send_trajectory(
        &self,
        buffer_entry: &(NamespaceString, ContextString, Uuid),
        encoded_trajectory: &EncodedTrajectory,
        trajectory_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_trajectory(
            &self.zmq_training_ops,
            buffer_entry,
            encoded_trajectory,
            trajectory_server_address,
        )
    }

    #[inline]
    fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_client_ids(
            &self.zmq_training_ops,
            scaling_entry,
            client_ids,
            training_scaling_server_address,
        )
    }

    #[inline]
    fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_scaling_warning(
            &self.zmq_training_ops,
            scaling_entry,
            operation,
            training_scaling_server_address,
        )
    }

    #[inline]
    fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_scaling_complete(
            &self.zmq_training_ops,
            scaling_entry,
            operation,
            training_scaling_server_address,
        )
    }

    #[inline]
    fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        training_scaling_server_address: &str,
    ) -> Result<(), TransportError> {
        <ZmqTrainingOps as ZmqTrainingExecution<B>>::execute_send_shutdown_signal(
            &self.zmq_training_ops,
            scaling_entry,
            training_scaling_server_address,
        )
    }
}
