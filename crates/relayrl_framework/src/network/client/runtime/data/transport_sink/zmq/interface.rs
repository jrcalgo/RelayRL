use crate::network::client::agent::ClientCapabilities;
use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    InferenceDispatcher, ScalingDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::data::transport_sink::{
    ScalingOperation, SyncClientTransportInterface, SyncInferenceTransportOps,
    SyncTrainingTransportOps, SyncScalingTransportOps, TransportError, TransportUuid,
};
use crate::network::client::runtime::router::RoutedMessage;
use crate::utilities::configuration::Algorithm;

use active_uuid_registry::interface::reserve_with;
use relayrl_types::HyperparameterArgs;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use tokio::sync::RwLock as TokioRwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

use super::ops::{ZmqInferenceOps, ZmqPool, ZmqTrainingOps};
use super::policies::{BackpressureController, CircuitBreaker, CircuitState, RetryPolicy, ZmqPolicyConfig};
use super::{ZmqInferenceServerExecution, ZmqTrainingServerExecution};

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
    fn new(shared_client_capabilities: Arc<ClientCapabilities>) -> Result<Self, TransportError> {
        let transport_id: TransportUuid =
            reserve_with("zmq_transport_client", 42, 100).map_err(TransportError::from)?;
        let zmq_pool = Arc::new(RwLock::new(ZmqPool::new()?));
        let zmq_inference_ops = ZmqInferenceOps::new(transport_id, zmq_pool.clone())?;
        let zmq_training_ops = ZmqTrainingOps::new(transport_id, zmq_pool.clone())?;

        let inference_protocol = if shared_client_capabilities.server_inference {
            let config = ZmqPolicyConfig::for_inference();
            Some(ZmqProtocol {
                circuit_breaker: CircuitBreaker::new(config.circuit_breaker_threshold, config.circuit_breaker_duration),
                backpressure: BackpressureController::new(config.max_concurrent_requests),
                config: config,
            })
        } else {
            None
        };

        let training_protocol = if !shared_client_capabilities.training_server_mode == ActorServerModelMode::Disabled {
            let config = ZmqPolicyConfig::for_training();
            Some(ZmqProtocol {
                circuit_breaker: CircuitBreaker::new(config.circuit_breaker_threshold, config.circuit_breaker_duration),
                backpressure: BackpressureController::new(config.max_concurrent_requests),
                config: config,
            })
        } else {
            None
        };

        let scaling_protocol = if shared_client_capabilities.server_inference || shared_client_capabilities.training_server_mode == ActorServerModelMode::Disabled {
            let config = ZmqPolicyConfig::for_scaling();
            Some(ZmqProtocol {
                circuit_breaker: CircuitBreaker::new(config.circuit_breaker_threshold, config.circuit_breaker_duration),
                backpressure: BackpressureController::new(config.max_concurrent_requests),
                config: config,
            })
        } else {
            None
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
        let client_id = get()
        self.send_shutdown_signal
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncScalingTransportOps<B> for ZmqInterface<B> {
    fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &Vec<(String, Uuid)>,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {

    }

    fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {

    }

    fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {

    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncInferenceTransportOps<B>
    for ZmqInterface<B>
{
    fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<RelayRLAction, TransportError> {
    }

    fn send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ZmqInferenceExecution for ZmqInterface<B> {
    fn execute_send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<RelayRLAction, TransportError> {
    }

    fn execute_send_flag_last_inference(
        &self,
        actor_id: &Uuid,
        reward: f32,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &Vec<(String, Uuid)>,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> SyncTrainingTransportOps<B>
    for ZmqInterface<B>
{
    fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn initial_model_handshake(
        &self,
        sender_id: &Uuid,
        shared_server_address: Arc<RwLock<ServerAddresses>>,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
    }

    fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        let _permit = self.training_protocol.as_ref().unwrap().backpressure.acquire().await?;
    }

    fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }
}

impl<B: Backend + BackendMatcher<Backend = B>> ZmqTrainingExecution<B> for ZmqInterface<B> {
    fn execute_send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_initial_model_handshake(
        &self,
        actor_id: &Uuid,
        model_server_address: &str,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
    }

    fn execute_send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: &Vec<(String, Uuid)>,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }

    fn execute_send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<TokioRwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
    }
}


