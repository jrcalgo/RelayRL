use crate::network::client::agent::ClientModes;
use crate::network::client::agent::{ActorInferenceMode, ActorTrainingDataMode, ModelMode};
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::data::transport_sink::{
    AsyncClientInferenceTransportOps, AsyncClientScalingTransportOps,
    AsyncClientTrainingTransportOps, AsyncClientTransportInterface, ScalingOperation,
    TransportError, TransportUuid,
};
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::Algorithm;

use active_uuid_registry::interface::reserve_id_with;
use relayrl_types::HyperparameterArgs;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use async_trait::async_trait;
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;

use super::super::combine_scaling_results;
use super::ops::{NatsConnectionManager, NatsInferenceOps, NatsTrainingOps};
use super::policies::{BackpressureController, CircuitBreaker, NatsPolicyConfig};
use super::{NatsInferenceExecution, NatsTrainingExecution};

use active_uuid_registry::{ContextString, NamespaceString, registry_uuid::Uuid};

struct NatsProtocol {
    circuit_breaker: CircuitBreaker,
    backpressure: BackpressureController,
    config: NatsPolicyConfig,
}

pub(crate) struct NatsInterface<B: Backend + BackendMatcher<Backend = B>> {
    nats_inference_ops: NatsInferenceOps,
    nats_training_ops: NatsTrainingOps,
    inference_protocol: Option<NatsProtocol>,
    training_protocol: Option<NatsProtocol>,
    scaling_protocol: Option<NatsProtocol>,
    _phantom: PhantomData<B>,
}

impl<B: Backend + BackendMatcher<Backend = B>> NatsInterface<B> {
    async fn is_shutting_down(&self) -> bool {
        self.nats_training_ops.is_shutting_down().await
    }
}

#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientTransportInterface<B>
    for NatsInterface<B>
{
    async fn new(
        client_namespace: Arc<str>,
        shared_client_modes: Arc<ClientModes>,
    ) -> Result<Self, TransportError> {
        let _transport_id: TransportUuid = reserve_id_with(
            client_namespace.as_ref(),
            crate::network::NATS_CLIENT_CONTEXT,
            42,
            100,
        )
        .map_err(TransportError::from)?;

        let transport_entry = (
            client_namespace.to_string(),
            crate::network::NATS_CLIENT_CONTEXT.to_string(),
        );

        let nats_connection_manager = Arc::new(RwLock::new(NatsConnectionManager::new(
            client_namespace.clone(),
        )));
        let nats_inference_ops =
            NatsInferenceOps::new(transport_entry.clone(), nats_connection_manager.clone());
        let nats_training_ops =
            NatsTrainingOps::new(transport_entry.clone(), nats_connection_manager.clone());

        let inference_protocol = match shared_client_modes.actor_inference_mode {
            ActorInferenceMode::Server(_) => {
                let config = NatsPolicyConfig::for_inference();
                Some(NatsProtocol {
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
            ActorTrainingDataMode::Online(_)
            | ActorTrainingDataMode::OnlineWithFiles(_, _)
            | ActorTrainingDataMode::OnlineWithMemory(_) => {
                let config = NatsPolicyConfig::for_training();
                Some(NatsProtocol {
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
                ActorTrainingDataMode::Disabled
                | ActorTrainingDataMode::OfflineWithFiles(_)
                | ActorTrainingDataMode::OfflineWithMemory(_),
            ) => None,
            _ => {
                let config = NatsPolicyConfig::for_scaling();
                Some(NatsProtocol {
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
            nats_inference_ops,
            nats_training_ops,
            inference_protocol,
            training_protocol,
            scaling_protocol,
            _phantom: PhantomData,
        })
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        self.nats_training_ops.shutdown().await
    }
}

#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientScalingTransportOps<B>
    for NatsInterface<B>
{
    async fn send_client_ids(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        client_ids: Vec<(NamespaceString, ContextString, Uuid)>,
        _replace_context: bool,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let scaling_protocol = self.scaling_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Scaling protocol not initialized".into())
        })?;
        let _permit = scaling_protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if scaling_protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address = transport_addresses.nats_inference_address.clone();
        let nats_training_server_address = transport_addresses.nats_training_address.clone();
        let mut attempts = 0u32;

        loop {
            let (inference_result, training_result) = tokio::join!(
                async {
                    if let Some(inference_protocol) = self.inference_protocol.as_ref() {
                        let _permit = inference_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if inference_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                inference_protocol.config.timeout,
                                <NatsInterface<B> as NatsInferenceExecution>::execute_send_client_ids(
                                    self,
                                    &scaling_entry,
                                    &client_ids,
                                    nats_inference_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                },
                async {
                    if let Some(training_protocol) = self.training_protocol.as_ref() {
                        let _permit = training_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if training_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                training_protocol.config.timeout,
                                <NatsInterface<B> as NatsTrainingExecution<B>>::execute_send_client_ids(
                                    self,
                                    &scaling_entry,
                                    &client_ids,
                                    nats_training_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
            );

            match combine_scaling_results(inference_result, training_result) {
                Ok(()) => {
                    scaling_protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                    attempts += 1;
                    scaling_protocol.circuit_breaker.record_failure();
                    tokio::time::sleep(
                        scaling_protocol
                            .config
                            .retry_policy
                            .delay_for_attempt(attempts),
                    )
                    .await;
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
    }

    async fn send_scaling_warning(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let scaling_protocol = self.scaling_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Scaling protocol not initialized".into())
        })?;
        let _permit = scaling_protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if scaling_protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address = transport_addresses.nats_inference_address.clone();
        let nats_training_server_address = transport_addresses.nats_training_address.clone();
        let mut attempts = 0u32;

        loop {
            let (inference_result, training_result) = tokio::join!(
                async {
                    if let Some(inference_protocol) = self.inference_protocol.as_ref() {
                        let _permit = inference_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if inference_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                inference_protocol.config.timeout,
                                <NatsInterface<B> as NatsInferenceExecution>::execute_send_scaling_warning(
                                    &self,
                                    &scaling_entry,
                                    &operation,
                                    nats_inference_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                },
                async {
                    if let Some(training_protocol) = self.training_protocol.as_ref() {
                        let _permit = training_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if training_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                training_protocol.config.timeout,
                                <NatsInterface<B> as NatsTrainingExecution<B>>::execute_send_scaling_warning(
                                    &self,
                                    &scaling_entry,
                                    &operation,
                                    nats_training_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
            );

            match combine_scaling_results(inference_result, training_result) {
                Ok(()) => {
                    scaling_protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                    attempts += 1;
                    scaling_protocol.circuit_breaker.record_failure();
                    tokio::time::sleep(
                        scaling_protocol
                            .config
                            .retry_policy
                            .delay_for_attempt(attempts),
                    )
                    .await;
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
    }

    async fn send_scaling_complete(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        operation: ScalingOperation,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let scaling_protocol = self.scaling_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Scaling protocol not initialized".into())
        })?;
        let _permit = scaling_protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if scaling_protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address = transport_addresses.nats_inference_address.clone();
        let nats_training_server_address = transport_addresses.nats_training_address.clone();
        let mut attempts = 0u32;

        loop {
            let (inference_result, training_result) = tokio::join!(
                async {
                    if let Some(inference_protocol) = self.inference_protocol.as_ref() {
                        let _permit = inference_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if inference_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                inference_protocol.config.timeout,
                                <NatsInterface<B> as NatsInferenceExecution>::execute_send_scaling_complete(&self,
                                    &scaling_entry,
                                    &operation,
                                    nats_inference_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                },
                async {
                    if let Some(training_protocol) = self.training_protocol.as_ref() {
                        let _permit = training_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if training_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                training_protocol.config.timeout,
                                <NatsInterface<B> as NatsTrainingExecution<B>>::execute_send_scaling_complete(
                                    &self,
                                    &scaling_entry,
                                    &operation,
                                    nats_training_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
            );

            match combine_scaling_results(inference_result, training_result) {
                Ok(()) => {
                    scaling_protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                    attempts += 1;
                    scaling_protocol.circuit_breaker.record_failure();
                    tokio::time::sleep(
                        scaling_protocol
                            .config
                            .retry_policy
                            .delay_for_attempt(attempts),
                    )
                    .await;
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
    }

    async fn send_shutdown_signal(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let scaling_protocol = self.scaling_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Scaling protocol not initialized".into())
        })?;
        let _permit = scaling_protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if scaling_protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address = transport_addresses.nats_inference_address.clone();
        let nats_training_server_address = transport_addresses.nats_training_address.clone();
        let mut attempts = 0u32;

        loop {
            let (inference_result, training_result) = tokio::join!(
                async {
                    if let Some(inference_protocol) = self.inference_protocol.as_ref() {
                        let _permit = inference_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if inference_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                inference_protocol.config.timeout,
                                <NatsInterface<B> as NatsInferenceExecution>::execute_send_shutdown_signal(
                                    &self,
                                    &scaling_entry,
                                    nats_inference_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    inference_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    inference_protocol.circuit_breaker.record_failure();
                                    if attempts < inference_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            inference_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                },
                async {
                    if let Some(training_protocol) = self.training_protocol.as_ref() {
                        let _permit = training_protocol
                            .backpressure
                            .acquire()
                            .await
                            .map_err(|e| TransportError::NatsClientError(e.to_string()));
                        if let Err(e) = _permit {
                            return Some(Err(e));
                        }
                        if training_protocol.circuit_breaker.is_open() {
                            return Some(Err(TransportError::CircuitOpen));
                        }
                        let mut attempts = 0u32;
                        loop {
                            match tokio::time::timeout(
                                training_protocol.config.timeout,
                                <NatsInterface<B> as NatsTrainingExecution<B>>::execute_send_shutdown_signal(
                                    self,
                                    &scaling_entry,
                                    nats_training_server_address.as_ref(),
                                ),
                            )
                            .await
                            {
                                Ok(Ok(())) => {
                                    training_protocol.circuit_breaker.record_success();
                                    return Some(Ok(()));
                                }
                                Ok(Err(e)) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: e.to_string(),
                                            attempts,
                                        }));
                                    }
                                }
                                Err(timeout) => {
                                    training_protocol.circuit_breaker.record_failure();
                                    if attempts < training_protocol.config.retry_policy.max_attempts {
                                        attempts += 1;
                                        tokio::time::sleep(
                                            training_protocol.config.retry_policy.delay_for_attempt(attempts),
                                        )
                                        .await;
                                    } else {
                                        return Some(Err(TransportError::MaxRetriesExceeded {
                                            cause: format!("timeout: {}", timeout),
                                            attempts,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
            );

            match combine_scaling_results(inference_result, training_result) {
                Ok(()) => {
                    scaling_protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(_) if attempts < scaling_protocol.config.retry_policy.max_attempts => {
                    attempts += 1;
                    scaling_protocol.circuit_breaker.record_failure();
                    tokio::time::sleep(
                        scaling_protocol
                            .config
                            .retry_policy
                            .delay_for_attempt(attempts),
                    )
                    .await;
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
    }
}

#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientInferenceTransportOps<B>
    for NatsInterface<B>
{
    async fn send_inference_model_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        model_mode: ModelMode,
        model_module: Option<ModelModule<B>>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let protocol = self.inference_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Inference protocol not initialized".into())
        })?;
        let _permit = protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address: &str =
            transport_addresses.nats_inference_address.as_ref();
        let mut attempts = 0u32;

        loop {
            match tokio::time::timeout(
                protocol.config.timeout,
                self.execute_send_inference_model_init_request(
                    &scaling_entry,
                    &model_mode,
                    &model_module,
                    nats_inference_server_address,
                ),
            )
            .await
            {
                Ok(Ok(())) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Ok(Err(e)) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
                Err(timeout) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: format!("timeout: {}", timeout),
                            attempts,
                        });
                    }
                }
            }
        }
    }

    async fn send_inference_request(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        obs_bytes: Vec<u8>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<RelayRLAction, TransportError> {
        let protocol = self.inference_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Inference protocol not initialized".into())
        })?;
        let _permit = protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address: &str =
            transport_addresses.nats_inference_address.as_ref();
        let mut attempts = 0u32;

        loop {
            match tokio::time::timeout(
                protocol.config.timeout,
                self.execute_send_inference_request(
                    &actor_entry,
                    &obs_bytes,
                    nats_inference_server_address,
                ),
            )
            .await
            {
                Ok(Ok(action)) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(action);
                }
                Ok(Err(e)) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
                Err(timeout) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: format!("timeout: {}", timeout),
                            attempts,
                        });
                    }
                }
            }
        }
    }

    async fn send_flag_last_inference(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        reward: f32,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let protocol = self.inference_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Inference protocol not initialized".into())
        })?;
        let _permit = protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_inference_server_address: &str =
            transport_addresses.nats_inference_address.as_ref();
        let mut attempts = 0u32;

        loop {
            match tokio::time::timeout(
                protocol.config.timeout,
                self.execute_send_flag_last_inference(
                    &actor_entry,
                    &reward,
                    nats_inference_server_address,
                ),
            )
            .await
            {
                Ok(Ok(())) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Ok(Err(e)) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
                Err(timeout) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: format!("timeout: {}", timeout),
                            attempts,
                        });
                    }
                }
            }
        }
    }
}

#[async_trait]
impl<B: Backend + BackendMatcher<Backend = B>> AsyncClientTrainingTransportOps<B>
    for NatsInterface<B>
{
    async fn send_algorithm_init_request(
        &self,
        scaling_entry: (NamespaceString, ContextString, Uuid),
        actor_entries: Vec<(NamespaceString, ContextString, Uuid)>,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let protocol = self.training_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Training protocol not initialized".into())
        })?;
        let _permit = protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_training_server_address: &str = transport_addresses.nats_training_address.as_ref();
        let mut attempts = 0u32;

        loop {
            match tokio::time::timeout(
                protocol.config.timeout,
                self.execute_send_algorithm_init_request(
                    &scaling_entry,
                    &actor_entries,
                    &model_mode,
                    &algorithm,
                    &hyperparams,
                    nats_training_server_address,
                ),
            )
            .await
            {
                Ok(Ok(())) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Ok(Err(e)) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
                Err(timeout) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: format!("timeout: {}", timeout),
                            attempts,
                        });
                    }
                }
            }
        }
    }

    async fn initial_model_handshake(
        &self,
        actor_entry: (NamespaceString, ContextString, Uuid),
        transport_addresses: SharedTransportAddresses,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        let protocol = self.training_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Training protocol not initialized".into())
        })?;
        let _permit = protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_training_server_address: &str = transport_addresses.nats_training_address.as_ref();
        let mut attempts = 0u32;

        loop {
            match tokio::time::timeout(
                protocol.config.timeout,
                self.execute_initial_model_handshake(&actor_entry, nats_training_server_address),
            )
            .await
            {
                Ok(Ok(model)) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(model);
                }
                Ok(Err(e)) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
                Err(timeout) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: format!("timeout: {}", timeout),
                            attempts,
                        });
                    }
                }
            }
        }
    }

    async fn send_trajectory(
        &self,
        buffer_entry: (NamespaceString, ContextString, Uuid),
        encoded_trajectory: EncodedTrajectory,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let protocol = self.training_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Training protocol not initialized".into())
        })?;
        let _permit = protocol
            .backpressure
            .acquire()
            .await
            .map_err(|e| TransportError::NatsClientError(e.to_string()))?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }

        let nats_training_server_address: &str = transport_addresses.nats_training_address.as_ref();
        let mut attempts = 0u32;

        loop {
            match tokio::time::timeout(
                protocol.config.timeout,
                self.execute_send_trajectory(
                    &buffer_entry,
                    &encoded_trajectory,
                    nats_training_server_address,
                ),
            )
            .await
            {
                Ok(Ok(())) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Ok(Err(e)) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
                Err(timeout) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: format!("timeout: {}", timeout),
                            attempts,
                        });
                    }
                }
            }
        }
    }

    async fn listen_for_model(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
        model_update_tx: Sender<RoutedMessage>,
        transport_addresses: SharedTransportAddresses,
    ) -> Result<(), TransportError> {
        let protocol = self.training_protocol.as_ref().ok_or_else(|| {
            TransportError::InvalidState("Training protocol not initialized".into())
        })?;
        if protocol.circuit_breaker.is_open() {
            return Err(TransportError::CircuitOpen);
        }
        if self.is_shutting_down().await {
            return Ok(());
        }

        let nats_training_server_address: &str = transport_addresses.nats_training_address.as_ref();
        let mut attempts = 0u32;

        loop {
            if self.is_shutting_down().await {
                return Ok(());
            }

            match self
                .execute_listen_for_model(
                    &receiver_entry,
                    &model_update_tx,
                    nats_training_server_address,
                )
                .await
            {
                Ok(()) => {
                    protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(_) if self.is_shutting_down().await => {
                    log::info!("[NatsInterface] Model listener stopped during shutdown");
                    protocol.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(e) => {
                    protocol.circuit_breaker.record_failure();
                    if attempts < protocol.config.retry_policy.max_attempts {
                        attempts += 1;
                        tokio::time::sleep(
                            protocol.config.retry_policy.delay_for_attempt(attempts),
                        )
                        .await;
                    } else {
                        return Err(TransportError::MaxRetriesExceeded {
                            cause: e.to_string(),
                            attempts,
                        });
                    }
                }
            }
        }
    }

    async fn stop_model_listener(
        &self,
        receiver_entry: (NamespaceString, ContextString, Uuid),
    ) -> Result<(), TransportError> {
        self.nats_training_ops
            .stop_model_listener(&receiver_entry)
            .await
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> NatsInferenceExecution for NatsInterface<B> {
    #[inline]
    async fn execute_send_inference_request(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        obs_bytes: &[u8],
        nats_inference_server_address: &str,
    ) -> Result<RelayRLAction, TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_inference_request(
            &self.nats_inference_ops,
            actor_entry,
            obs_bytes,
            nats_inference_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_flag_last_inference(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        reward: &f32,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_flag_last_inference(
            &self.nats_inference_ops,
            actor_entry,
            reward,
            nats_inference_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_inference_model_init_request<
        MB: Backend + BackendMatcher<Backend = MB>,
    >(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        model_mode: &ModelMode,
        model_module: &Option<ModelModule<MB>>,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_inference_model_init_request(
            &self.nats_inference_ops,
            scaling_entry,
            model_mode,
            model_module,
            nats_inference_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_client_ids(
            &self.nats_inference_ops,
            scaling_entry,
            client_ids,
            nats_inference_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_scaling_warning(
            &self.nats_inference_ops,
            scaling_entry,
            operation,
            nats_inference_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_scaling_complete(
            &self.nats_inference_ops,
            scaling_entry,
            operation,
            nats_inference_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        nats_inference_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsInferenceOps as NatsInferenceExecution>::execute_send_shutdown_signal(
            &self.nats_inference_ops,
            scaling_entry,
            nats_inference_server_address,
        )
        .await
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> NatsTrainingExecution<B> for NatsInterface<B> {
    #[inline]
    async fn execute_listen_for_model(
        &self,
        receiver_entry: &(NamespaceString, ContextString, Uuid),
        model_update_tx: &Sender<RoutedMessage>,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_listen_for_model(
            &self.nats_training_ops,
            receiver_entry,
            model_update_tx,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_algorithm_init_request(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        actor_entries: &[(NamespaceString, ContextString, Uuid)],
        model_mode: &ModelMode,
        algorithm: &Algorithm,
        hyperparams: &HashMap<Algorithm, HyperparameterArgs>,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_algorithm_init_request(
            &self.nats_training_ops,
            scaling_entry,
            actor_entries,
            model_mode,
            algorithm,
            hyperparams,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_initial_model_handshake(
        &self,
        actor_entry: &(NamespaceString, ContextString, Uuid),
        nats_training_server_address: &str,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_initial_model_handshake(
            &self.nats_training_ops,
            actor_entry,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_trajectory(
        &self,
        buffer_entry: &(NamespaceString, ContextString, Uuid),
        encoded_trajectory: &EncodedTrajectory,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_trajectory(
            &self.nats_training_ops,
            buffer_entry,
            encoded_trajectory,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_client_ids(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        client_ids: &[(NamespaceString, ContextString, Uuid)],
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_client_ids(
            &self.nats_training_ops,
            scaling_entry,
            client_ids,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_scaling_warning(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_scaling_warning(
            &self.nats_training_ops,
            scaling_entry,
            operation,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_scaling_complete(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        operation: &ScalingOperation,
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_scaling_complete(
            &self.nats_training_ops,
            scaling_entry,
            operation,
            nats_training_server_address,
        )
        .await
    }

    #[inline]
    async fn execute_send_shutdown_signal(
        &self,
        scaling_entry: &(NamespaceString, ContextString, Uuid),
        nats_training_server_address: &str,
    ) -> Result<(), TransportError> {
        <NatsTrainingOps as NatsTrainingExecution<B>>::execute_send_shutdown_signal(
            &self.nats_training_ops,
            scaling_entry,
            nats_training_server_address,
        )
        .await
    }
}
