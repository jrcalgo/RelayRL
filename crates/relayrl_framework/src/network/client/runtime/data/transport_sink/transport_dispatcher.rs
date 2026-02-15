use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::data::transport_sink::ScalingOperation;
use crate::network::client::runtime::data::transport_sink::{
    RoutedMessage, TransportClient, TransportError,
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
    transport: Arc<TransportClient<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> InferenceDispatcher<B> {
    pub fn new(transport: Arc<TransportClient<B>>) -> Self {
        Self { transport }
    }

    pub async fn send_inference_request(
        &self,
        actor_id: &Uuid,
        obs_bytes: &[u8],
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<RelayRLAction, TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.send_inference_request(actor_id, obs_bytes, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_inference_request(actor_id, obs_bytes, inference_server_address)
                    .await
            }
        }
    }
}

pub(crate) struct TrainingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<TransportClient<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> TrainingDispatcher<B> {
    pub fn new(transport: Arc<TransportClient<B>>) -> Self {
        Self { transport }
    }

    pub async fn send_algorithm_init_request(
        &self,
        scaling_id: &Uuid,
        algorithm: Algorithm,
        hyperparams: HashMap<Algorithm, HyperparameterArgs>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => sync_tr.send_algorithm_init_request(
                scaling_id,
                algorithm,
                hyperparams,
                shared_server_addresses,
            ),
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_algorithm_init_request(
                        scaling_id,
                        algorithm,
                        hyperparams,
                        shared_server_addresses,
                    )
                    .await
            }
        }
    }

    pub async fn initial_model_handshake(
        &self,
        actor_id: &Uuid,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<Option<ModelModule<B>>, TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.initial_model_handshake(actor_id, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .initial_model_handshake(actor_id, shared_server_addresses)
                    .await
            }
        }
    }

    pub async fn listen_for_model(
        &self,
        receiver_id: &Uuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.listen_for_model(receiver_id, global_dispatcher_tx, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .listen_for_model(receiver_id, global_dispatcher_tx, shared_server_addresses)
                    .await
            }
        }
    }

    pub async fn send_trajectory(
        &self,
        sender_id: &Uuid,
        encoded_trajectory: EncodedTrajectory,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.send_trajectory(sender_id, encoded_trajectory, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_trajectory(sender_id, encoded_trajectory, shared_server_addresses)
                    .await
            }
        }
    }
}

pub(crate) struct ScalingDispatcher<B: Backend + BackendMatcher<Backend = B>> {
    transport: Arc<TransportClient<B>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ScalingDispatcher<B> {
    pub fn new(transport: Arc<TransportClient<B>>) -> Self {
        Self { transport }
    }

    pub async fn send_client_ids(
        &self,
        scaling_id: &Uuid,
        client_ids: Vec<(String, Uuid)>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.send_client_ids(scaling_id, &client_ids, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_client_ids(scaling_id, client_ids, shared_server_addresses)
                    .await
            }
        }
    }

    pub async fn send_scaling_warning(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.send_scaling_warning(scaling_id, operation, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_scaling_warning(scaling_id, operation, shared_server_addresses)
                    .await
            }
        }
    }

    pub async fn send_scaling_complete(
        &self,
        scaling_id: &Uuid,
        operation: ScalingOperation,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.send_scaling_complete(scaling_id, operation, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_scaling_complete(scaling_id, operation, shared_server_addresses)
                    .await
            }
        }
    }

    pub async fn send_shutdown_signal(
        &self,
        scaling_id: &Uuid,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => {
                sync_tr.send_shutdown_signal(scaling_id, shared_server_addresses)
            },
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => {
                async_tr
                    .send_shutdown_signal(scaling_id, shared_server_addresses)
                    .await
            }
        }
    }

    pub async fn shutdown_transport(&self, shared_server_addresses: Arc<RwLock<ServerAddresses>>) -> Result<(), TransportError> {
        match &*self.transport {
            #[cfg(feature = "sync_transport")]
            TransportClient::Sync(sync_tr) => sync_tr.shutdown(shared_server_addresses),
            #[cfg(feature = "async_transport")]
            TransportClient::Async(async_tr) => async_tr.shutdown(shared_server_addresses).await,
        }
    }
}
