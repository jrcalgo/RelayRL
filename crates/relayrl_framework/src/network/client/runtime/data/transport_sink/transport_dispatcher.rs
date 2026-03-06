use crate::network::client::agent::ModelMode;
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::data::transport_sink::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::{
    ClientTransportInterface, RoutedMessage, TransportError,
};
use crate::utilities::configuration::Algorithm;

use relayrl_types::HyperparameterArgs;
use relayrl_types::prelude::action::RelayRLAction;
use relayrl_types::prelude::model::ModelModule;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;
use relayrl_types::prelude::trajectory::EncodedTrajectory;

use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

pub(crate) struct InferenceDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<ClientTransportInterface<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> InferenceDispatcher<B> {
    pub(crate) fn new(transport: Arc<ClientTransportInterface<B>>) -> Self {
        Self { transport }
    }

    pub(crate) async fn send_inference_request(
        &self,
        actor_entry: (String, String, Uuid),
        obs_bytes: Vec<u8>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<RelayRLAction, TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_inference_request(actor_entry, obs_bytes, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_inference_request(actor_entry, obs_bytes, transport_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_flag_last_inference(
        &self,
        actor_entry: (String, String, Uuid),
        reward: f32,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_flag_last_inference(actor_entry, reward, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_flag_last_inference(actor_entry, reward, transport_addresses)
                    .await
            }
        }
    }
}

pub(crate) struct TrainingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<ClientTransportInterface<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrainingDispatcher<B> {
    pub(crate) fn new(transport: Arc<ClientTransportInterface<B>>) -> Self {
        Self { transport }
    }

    pub(crate) async fn initial_model_handshake(
        &self,
        actor_entry: (String, String, Uuid),
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.initial_model_handshake(actor_entry, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .initial_model_handshake(actor_entry, transport_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn listen_for_model(
        &self,
        receiver_entry: (String, String, Uuid),
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.listen_for_model(receiver_entry, global_dispatcher_tx, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .listen_for_model(receiver_entry, global_dispatcher_tx, transport_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_trajectory(
        &self,
        buffer_entry: (String, String, Uuid),
        encoded_trajectory: EncodedTrajectory,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_trajectory(buffer_entry, encoded_trajectory, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_trajectory(buffer_entry, encoded_trajectory, transport_addresses)
                    .await
            }
        }
    }
}

pub(crate) enum ProcessInitRequest<B: Backend + BackendMatcher<Backend = B>> {
    TrainingAlgorithmInit(ModelMode, Algorithm, HashMap<Algorithm, HyperparameterArgs>),
    InferenceModelInit(ModelMode, Option<ModelModule<B>>),
}

pub(crate) struct ScalingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<ClientTransportInterface<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ScalingDispatcher<B> {
    pub(crate) fn new(transport: Arc<ClientTransportInterface<B>>) -> Self {
        Self { transport }
    }

    pub(crate) async fn send_process_init_request(
        &self,
        scaling_entry: (String, String, Uuid),
        actor_entries: Vec<(String, String, Uuid)>,
        process_init_request: ProcessInitRequest<B>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match process_init_request {
            ProcessInitRequest::TrainingAlgorithmInit(model_mode, algorithm, hyperparams) => {
                match &*self.transport {
                    #[cfg(feature = "zmq-transport")]
                    ClientTransportInterface::Sync(sync_tr) => sync_tr.send_algorithm_init_request(
                        scaling_entry,
                        actor_entries,
                        model_mode,
                        algorithm,
                        hyperparams,
                        transport_addresses,
                    ),
                    #[cfg(feature = "nats-transport")]
                    ClientTransportInterface::Async(async_tr) => {
                        async_tr
                            .send_algorithm_init_request(
                                scaling_entry,
                                actor_entries,
                                model_mode,
                                algorithm,
                                hyperparams,
                                transport_addresses,
                            )
                            .await
                    }
                }
            }
            ProcessInitRequest::InferenceModelInit(model_mode, model_module) => {
                match &*self.transport {
                    #[cfg(feature = "zmq-transport")]
                    ClientTransportInterface::Sync(sync_tr) => sync_tr
                        .send_inference_model_init_request(
                            scaling_entry,
                            model_mode,
                            model_module,
                            transport_addresses,
                        ),
                    #[cfg(feature = "nats-transport")]
                    ClientTransportInterface::Async(async_tr) => {
                        async_tr
                            .send_inference_model_init_request(
                                scaling_entry,
                                &model_mode,
                                &model_module,
                                transport_addresses,
                            )
                            .await
                    }
                }
            }
        }
    }

    pub(crate) async fn send_client_ids(
        &self,
        scaling_entry: (String, String, Uuid),
        client_ids: Vec<(String, String, Uuid)>,
        replace_context: bool,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => sync_tr.send_client_ids(
                scaling_entry,
                client_ids,
                replace_context,
                transport_addresses,
            ),
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_client_ids(
                        scaling_entry,
                        &client_ids,
                        replace_context,
                        transport_addresses,
                    )
                    .await
            }
        }
    }

    pub(crate) async fn send_scaling_warning(
        &self,
        scaling_entry: (String, String, Uuid),
        operation: ScalingOperation,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_scaling_warning(scaling_entry, operation, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_scaling_warning(scaling_entry, operation, transport_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_scaling_complete(
        &self,
        scaling_entry: (String, String, Uuid),
        operation: ScalingOperation,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_scaling_complete(scaling_entry, operation, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_scaling_complete(scaling_entry, operation, transport_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_shutdown_signal(
        &self,
        scaling_entry: (String, String, Uuid),
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let transport_addresses = shared_transport_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_shutdown_signal(scaling_entry, transport_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_shutdown_signal(scaling_entry, transport_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn shutdown_transport(&self) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => sync_tr.shutdown(),
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => async_tr.shutdown().await,
        }
    }
}

#[cfg(test)]
mod tests {}
