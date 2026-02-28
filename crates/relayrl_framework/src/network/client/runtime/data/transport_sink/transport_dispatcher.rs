use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::data::transport_sink::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::{
    ClientTransportInterface, RoutedMessage, TransportError,
};
use crate::network::client::agent::ModelMode;
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
        actor_id: &Uuid,
        obs_bytes: &[u8],
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<RelayRLAction, TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_inference_request(actor_id, obs_bytes, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_inference_request(actor_id, obs_bytes, server_addresses)
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

    pub(crate) async fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        model_mode: ModelMode,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => sync_tr.send_algorithm_init_request(
                scaling_id,
                model_mode,
                algorithm,
                hyperparams,
                server_addresses,
            ),
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_algorithm_init_request(
                        scaling_id,
                        model_mode,
                        algorithm,
                        hyperparams,
                        server_addresses,
                    )
                    .await
            }
        }
    }

    pub(crate) async fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.initial_model_handshake(actor_id, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .initial_model_handshake(actor_id, server_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.listen_for_model(receiver_id, global_dispatcher_tx, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .listen_for_model(receiver_id, global_dispatcher_tx, server_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_trajectory(sender_id, encoded_trajectory, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_trajectory(sender_id, encoded_trajectory, server_addresses)
                    .await
            }
        }
    }
}

pub(crate) struct ScalingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<ClientTransportInterface<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ScalingDispatcher<B> {
    pub(crate) fn new(transport: Arc<ClientTransportInterface<B>>) -> Self {
        Self { transport }
    }

    pub(crate) async fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: Vec<(String, Uuid)>,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_client_ids(scaling_id, &client_ids, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_client_ids(scaling_id, client_ids, server_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_scaling_warning(scaling_id, operation, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_scaling_warning(scaling_id, operation, server_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_scaling_complete(scaling_id, operation, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_scaling_complete(scaling_id, operation, server_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => {
                sync_tr.send_shutdown_signal(scaling_id, server_addresses)
            }
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => {
                async_tr
                    .send_shutdown_signal(scaling_id, server_addresses)
                    .await
            }
        }
    }

    pub(crate) async fn shutdown_transport(
        &self,
        shared_server_addresses: Arc<RwLock<SharedTransportAddresses>>,
    ) -> Result<(), TransportError> {
        let server_addresses = shared_server_addresses.read().await.clone();

        match &*self.transport {
            #[cfg(feature = "zmq-transport")]
            ClientTransportInterface::Sync(sync_tr) => sync_tr.shutdown(server_addresses),
            #[cfg(feature = "nats-transport")]
            ClientTransportInterface::Async(async_tr) => async_tr.shutdown(server_addresses).await,
        }
    }
}

#[cfg(test)]
mod tests {}
