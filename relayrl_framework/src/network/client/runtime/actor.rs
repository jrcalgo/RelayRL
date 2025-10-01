use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::AsyncClientTransport;
use crate::network::client::runtime::transport::{TransportClient, client_transport_factory};
use crate::network::{HotReloadableModel, TransportType, validate_model};
use crate::orchestration::tokio::utils::get_or_init_tokio_runtime;
use crate::orchestration::tonic::grpc_utils::deserialize_model;
use crate::types::NetworkParticipant;
use crate::types::action::RL4SysAction;
use crate::types::trajectory::{RL4SysTrajectory, RL4SysTrajectoryTrait};
use crate::utilities::configuration::ClientConfigLoader;
use std::path::PathBuf;
use std::sync::Arc;
use tch::{CModule, Device, TchError};
use tokio::sync::mpsc::{Receiver, Sender};
use uuid::{Timestamp, Uuid};

pub trait ActorEntity: Send + Sync + 'static {
    async fn new(
        id: Uuid,
        device: Device,
        model: Option<CModule>,
        model_path: PathBuf,
        shared_config: Arc<ClientConfigLoader>,
        rx_from_router: Receiver<RoutedMessage>,
        tx_to_sender: Sender<RoutedMessage>,
        transport: Arc<TransportClient>,
    ) -> (Self, bool)
    where
        Self: Sized;
    async fn spawn_loop(&mut self);
    async fn _initial_model_handshake(&self, msg: RoutedMessage);
    fn __request_for_action(&mut self, msg: RoutedMessage);
    fn __flag_last_action(&mut self, msg: RoutedMessage);
    fn __get_model_version(&self, msg: RoutedMessage);
    async fn _refresh_model(&self, msg: RoutedMessage);
    fn __get_actor_statistics(&self, msg: RoutedMessage);
    async fn _handle_shutdown(&self, msg: RoutedMessage);
}

/// Responsible for performing inference with an in-memory TorchScript model
pub(crate) struct Actor {
    id: Uuid,
    model: Option<HotReloadableModel>,
    model_path: PathBuf,
    model_latest_version: i64,
    current_traj: RL4SysTrajectory,
    rx_from_router: Receiver<RoutedMessage>,
    shared_tx_to_sender: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient>>,
    shared_config: Arc<ClientConfigLoader>,
}

impl ActorEntity for Actor {
    async fn new(
        id: Uuid,
        device: Device,
        model: Option<CModule>,
        model_path: PathBuf,
        shared_config: Arc<ClientConfigLoader>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_tx_to_sender: Sender<RoutedMessage>,
        transport: Arc<TransportClient>,
    ) -> (Self, bool)
    where
        Self: Sized,
    {
        let max_length = Some(shared_config.transport_config.max_traj_length);

        let mut actor = Self {
            id,
            model: None,
            model_path,
            model_latest_version: 0,
            current_traj: RL4SysTrajectory::new(
                max_length,
                NetworkParticipant::RL4SysAgent,
                &shared_config.client_config.config_path,
            ),
            rx_from_router,
            shared_tx_to_sender,
            transport: Some(transport),
            shared_config: shared_config.clone(),
        };

        let mut model_init_flag: bool = false;
        match model {
            Some(some_model) => {
                validate_model(&some_model);
                actor.model = Some(
                    HotReloadableModel::new_from_model(some_model, device)
                        .await
                        .ok()
                        .expect("[ActorEntity] New model could not be created..."),
                );
            }
            None => {
                eprintln!(
                    "[ActorEntity] Startup model is None, initial model handshake necessitated..."
                );
                model_init_flag = true;
            }
        }

        (actor, model_init_flag)
    }

    async fn spawn_loop(&mut self) {
        while let Some(msg) = self.rx_from_router.recv().await {
            match msg.protocol {
                RoutingProtocol::ModelHandshake => self._initial_model_handshake(msg).await,
                RoutingProtocol::RequestInference => self.__request_for_action(msg),
                RoutingProtocol::FlagLastInference => self.__flag_last_action(msg),
                RoutingProtocol::ModelVersion => self.__get_model_version(msg),
                RoutingProtocol::ModelUpdate => self._refresh_model(msg).await,
                RoutingProtocol::ActorStatistics => self.__get_actor_statistics(msg),
                RoutingProtocol::Shutdown => self._handle_shutdown(msg).await,
                _ => {}
            }
        }
    }

    async fn _initial_model_handshake(&self, msg: RoutedMessage) {
        if let RoutedPayload::ModelHandshake = msg.payload {
            if self.model.is_none() {
                if let Some(transport) = &self.transport {
                    match &**transport {
                        #[cfg(feature = "grpc_network")]
                        TransportClient::Async(async_tr) => {
                            // Use training server address for model handshake
                            let training_server_address = format!(
                                "{}:{}",
                                self.shared_config
                                    .transport_config
                                    .training_server_address
                                    .host,
                                self.shared_config
                                    .transport_config
                                    .training_server_address
                                    .port
                            );
                            println!(
                                "[Actor {:?}] Starting async model handshake with {}",
                                self.id, training_server_address
                            );

                            if let Ok(Some(model)) = async_tr
                                .initial_model_handshake(&training_server_address)
                                .await
                            {
                                println!(
                                    "[Actor {:?}] Model handshake successful, received model data",
                                    self.id
                                );
                                // Here you would process the received model data
                                // This is a simplified implementation
                            } else {
                                eprintln!(
                                    "[Actor {:?}] Model handshake failed or no model update needed",
                                    self.id
                                );
                            }
                        }
                        TransportClient::Sync(sync_tr) => {
                            // Use agent listener address for model handshake
                            let model_server_address = format!(
                                "{}:{}",
                                self.shared_config
                                    .transport_config
                                    .agent_listener_address
                                    .host,
                                self.shared_config
                                    .transport_config
                                    .agent_listener_address
                                    .port
                            );
                            println!(
                                "[Actor {:?}] Starting sync model handshake with {}",
                                self.id, model_server_address
                            );

                            if let Some(trajectory) =
                                sync_tr.initial_model_handshake(&model_server_address)
                            {
                                println!(
                                    "[Actor {:?}] Model handshake successful, received model data",
                                    self.id
                                );
                                // Here you would process the received model data
                            } else {
                                eprintln!(
                                    "[Actor {:?}] Model handshake failed or no model update needed",
                                    self.id
                                );
                            }
                        }
                    }
                } else {
                    eprintln!(
                        "[Actor {:?}] No transport configured for model handshake",
                        self.id
                    );
                }
            } else {
                println!(
                    "[Actor {:?}] Model already available, handshake not needed",
                    self.id
                );
            }
        }
    }

    fn __request_for_action(&mut self, msg: RoutedMessage) {
        if let Some(model) = &self.model {
            if let RoutedPayload::RequestInference {
                observation,
                mask,
                reward,
                reply_to,
            } = msg.payload
            {
                let rt = get_or_init_tokio_runtime();
                match rt.block_on(async { model.forward(observation, mask, reward).await }) {
                    Ok(r4sa) => {
                        self.current_traj.add_action(&r4sa);
                        reply_to
                            .send(Arc::new(r4sa))
                            .expect("Failed to send inference... ");
                    }
                    Err(e) => {
                        eprintln!(
                            "[ActorEntity {:?}] Failed inference, no inference created or available... {:?}",
                            self.id, e
                        )
                    }
                }
            }
        }
    }

    fn __flag_last_action(&mut self, msg: RoutedMessage) {
        if let RoutedPayload::FlagLastInference { reward } = msg.payload {
            let mut last_action: RL4SysAction =
                RL4SysAction::new(None, None, None, reward, None, true);
            last_action.update_reward(reward);
            self.current_traj.add_action(&last_action);

            let send_traj_msg = {
                let traj_clone = self.current_traj.clone();
                RoutedMessage {
                    actor_id: self.id,
                    protocol: RoutingProtocol::SendTrajectory,
                    payload: RoutedPayload::SendTrajectory {
                        timestamp: Timestamp::now(()),
                        trajectory: traj_clone,
                    },
                }
            };

            let rt = get_or_init_tokio_runtime();
            rt.block_on(async {
                self.shared_tx_to_sender
                    .send(send_traj_msg)
                    .await
                    .expect("TODO: panic message")
            });
        }
    }

    fn __get_model_version(&self, msg: RoutedMessage) {
        if let RoutedPayload::ModelVersion { reply_to } = msg.payload {
            let current_model = &self.model;

            match current_model {
                Some(some_model) => {
                    let version = some_model.version();
                    reply_to
                        .send(version)
                        .expect("[ActorEntity] Failed to send version...")
                }
                None => reply_to
                    .send(-1)
                    .expect("[ActorEntity] Failed to send version..."),
            }
        }
    }

    async fn _refresh_model(&self, msg: RoutedMessage) {
        if let RoutedPayload::ModelUpdate {
            model_bytes,
            version,
        } = msg.payload
        {
            let model: Result<CModule, TchError> = deserialize_model(model_bytes);
            let model_path: PathBuf = self.model_path.clone();
            if model.is_ok() {
                let ok_model: CModule = model.expect("[ActorEntity] Failed model acquisition...");
                validate_model(&ok_model);
                ok_model
                    .save(&model_path)
                    .expect("[ActorEntity] Failed model save...");
            }

            match &self.model {
                Some(model) => match model.reload(model_path, version).await {
                    Ok(_) => (),
                    Err(e) => {
                        eprintln!("[ActorEntity {:?}] Failed reload, {:?}", self.id, e);
                    }
                },
                None => {
                    eprintln!(
                        "[ActorEntity] Model does not exist, no model handshake necessary..."
                    );
                }
            }
        }
    }

    fn __get_actor_statistics(&self, msg: RoutedMessage) {}

    async fn _handle_shutdown(&self, msg: RoutedMessage) {}
}
