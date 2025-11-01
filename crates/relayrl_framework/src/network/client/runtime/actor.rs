use crate::network::HotReloadableModel;
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::TransportClient;
use crate::utilities::configuration::ClientConfigLoader;
use crate::utilities::orchestration::tokio_utils::get_or_init_tokio_runtime;

use relayrl_types::types::model::{ModelError, ModelModule, deserialize_model, validate_model};
use relayrl_types::types::action::RelayRLAction;
use relayrl_types::types::tensor::{BackendMatcher, DeviceType};
use relayrl_types::types::trajectory::{RelayRLTrajectory, RelayRLTrajectoryTrait};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::sync::mpsc::{Receiver, Sender};
use uuid::Uuid;

use burn_tensor::{Tensor, backend::Backend};

pub trait ActorEntity<B: Backend + BackendMatcher, const O: usize, const M: usize>:
    Send + Sync + 'static
{
    async fn new(
        id: Uuid,
        device: DeviceType,
        model: Option<HotReloadableModel<B>>,
        model_path: PathBuf,
        shared_config: Arc<RwLock<ClientConfigLoader>>,
        rx_from_router: Receiver<RoutedMessage>,
        tx_to_sender: Sender<RoutedMessage>,
        transport: Arc<TransportClient<B>>,
    ) -> (Self, bool)
    where
        Self: Sized;
    async fn spawn_loop(&mut self);
    async fn _initial_model_handshake(&mut self, msg: RoutedMessage);
    fn __request_action(&mut self, msg: RoutedMessage);
    fn __flag_last_action(&mut self, msg: RoutedMessage);
    fn __get_model_version(&self, msg: RoutedMessage);
    async fn _refresh_model(&self, msg: RoutedMessage);
    fn __get_actor_statistics(&self, _msg: RoutedMessage);
    async fn _handle_shutdown(&self, _msg: RoutedMessage);
}

/// Responsible for performing inference with an in-memory model
pub(crate) struct Actor<B: Backend + BackendMatcher> {
    id: Uuid,
    model: Option<HotReloadableModel<B>>,
    model_path: PathBuf,
    model_latest_version: i64,
    model_device: DeviceType,
    current_traj: RelayRLTrajectory,
    rx_from_router: Receiver<RoutedMessage>,
    shared_tx_to_sender: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient<B>>>,
    shared_config: Arc<RwLock<ClientConfigLoader>>,
}

impl<B: Backend + BackendMatcher, const O: usize, const M: usize> ActorEntity<B, O, M>
    for Actor<B>
{
    async fn new(
        id: Uuid,
        device: DeviceType,
        model: Option<HotReloadableModel<B>>,
        model_path: PathBuf,
        shared_config: Arc<RwLock<ClientConfigLoader>>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_tx_to_sender: Sender<RoutedMessage>,
        transport: Arc<TransportClient<B>>,
    ) -> (Self, bool)
    where
        Self: Sized,
    {
        let max_length = shared_config.read().await.transport_config.max_traj_length;

        let mut actor = Self {
            id,
            model: None,
            model_path,
            model_latest_version: 0,
            model_device: device,
            current_traj: RelayRLTrajectory::new(max_length as usize),
            rx_from_router,
            shared_tx_to_sender,
            transport: Some(transport),
            shared_config: shared_config.clone(),
        };

        let mut model_init_flag: bool = false;
        match model {
            Some(some_model) => {
                actor.model = Some(some_model);
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
                RoutingProtocol::ModelHandshake => {
                    <Actor<B> as ActorEntity<B, O, M>>::_initial_model_handshake(self, msg).await
                }
                RoutingProtocol::RequestInference => {
                    <Actor<B> as ActorEntity<B, O, M>>::__request_action(self, msg)
                }
                RoutingProtocol::FlagLastInference => {
                    <Actor<B> as ActorEntity<B, O, M>>::__flag_last_action(self, msg)
                }
                RoutingProtocol::ModelVersion => {
                    <Actor<B> as ActorEntity<B, O, M>>::__get_model_version(self, msg)
                }
                RoutingProtocol::ModelUpdate => {
                    <Actor<B> as ActorEntity<B, O, M>>::_refresh_model(self, msg).await
                }
                RoutingProtocol::ActorStatistics => {
                    <Actor<B> as ActorEntity<B, O, M>>::__get_actor_statistics(self, msg)
                }
                RoutingProtocol::Shutdown => {
                    <Actor<B> as ActorEntity<B, O, M>>::_handle_shutdown(self, msg).await;
                    break;
                }
                _ => {}
            }
        }
    }

    async fn _initial_model_handshake(&mut self, msg: RoutedMessage) {
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
                                    .read()
                                    .await
                                    .transport_config
                                    .training_server_address
                                    .host,
                                self.shared_config
                                    .read()
                                    .await
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

                                // Save model to configured path
                                if let Err(e) = model.save(&self.model_path) {
                                    eprintln!("[Actor {:?}] Failed to save model: {}", self.id, e);
                                }
                                let version = self.model_latest_version + 1;

                                match &self.model {
                                    Some(model) => {
                                        let model_version =
                                            model.reload(self.model_path.clone(), version).await;
                                        match model_version {
                                            Ok(_) => (),
                                            Err(e) => {
                                                eprintln!(
                                                    "[Actor {:?}] Failed to reload model: {}",
                                                    self.id, e
                                                );
                                            }
                                        }
                                    }
                                    None => {
                                        self.model = Some(
                                            HotReloadableModel::<B>::new_from_module(
                                                model,
                                                self.model_device,
                                            )
                                            .await
                                            .expect(
                                                "[ActorEntity] New model could not be created...",
                                            ),
                                        );
                                    }
                                }
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
                                    .read()
                                    .await
                                    .transport_config
                                    .agent_listener_address
                                    .host,
                                self.shared_config
                                    .read()
                                    .await
                                    .transport_config
                                    .agent_listener_address
                                    .port
                            );
                            println!(
                                "[Actor {:?}] Starting sync model handshake with {}",
                                self.id, model_server_address
                            );

                            if let Ok(Some(model)) =
                                sync_tr.initial_model_handshake(&model_server_address)
                            {
                                println!(
                                    "[Actor {:?}] Model handshake successful, received model data",
                                    self.id
                                );

                                // Save model to configured path
                                if let Err(e) = model.save(&self.model_path) {
                                    eprintln!("[Actor {:?}] Failed to save model: {}", self.id, e);
                                }
                                let version = self.model_latest_version + 1;

                                match &self.model {
                                    Some(model) => {
                                        let model_version =
                                            model.reload(self.model_path.clone(), version).await;
                                        match model_version {
                                            Ok(_) => (),
                                            Err(e) => {
                                                eprintln!(
                                                    "[Actor {:?}] Failed to reload model: {}",
                                                    self.id, e
                                                );
                                            }
                                        }
                                    }
                                    None => {
                                        self.model =
                                            Some(HotReloadableModel::<B>::new_from_module(
                                                module,
                                                self.model_device,
                                            ));
                                    }
                                }
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

    fn __request_action(&mut self, msg: RoutedMessage) {
        if let Some(model) = &self.model {
            if let RoutedPayload::RequestInference(inference_request) = msg.payload {
                let InferenceRequest {
                    observation,
                    mask,
                    reward,
                    reply_to,
                } = *inference_request; // Dereference the Box to get the inner value

                let observation_tensor: Tensor<B, O> =
                    *observation.downcast::<Tensor<B, O>>().unwrap();
                let mask_tensor: Tensor<B, M> = *mask.downcast::<Tensor<B, M>>().unwrap();

                let actor_id = self.id;
                let rt = get_or_init_tokio_runtime();
                match rt.block_on(async {
                    model
                        .forward::<O, M>(observation_tensor, mask_tensor, reward, actor_id)
                        .await
                }) {
                    Ok(r4sa) => {
                        self.current_traj.add_action(r4sa.clone());
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
            let actor_id = self.id;
            let mut last_action: RelayRLAction =
                RelayRLAction::new(None, None, None, reward, true, None, Some(actor_id));
            last_action.update_reward(reward);
            self.current_traj.add_action(last_action);

            let send_traj_msg = {
                let traj_clone = self.current_traj.clone();
                RoutedMessage {
                    actor_id: self.id,
                    protocol: RoutingProtocol::SendTrajectory,
                    payload: RoutedPayload::SendTrajectory {
                        timestamp: (
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Clock skew")
                                .as_millis(),
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Clock skew")
                                .as_nanos(),
                        ),
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
            // TODO: Minimize creation of model module in memory
            let model: Result<ModelModule<B>, ModelError> =
                deserialize_model::<B>(model_bytes, self.model_device);
            let model_path: PathBuf = self.model_path.clone();
            if model.is_ok() {
                let ok_model: ModelModule<B> =
                    model.expect("[ActorEntity] Failed model deserialization...");

                const INPUT_DIM: usize = ok_model.input_dim;
                const OUTPUT_DIM: usize = ok_model.output_dim;
                validate_model::<B>(&ok_model, input_dim, output_dim)?;
                ok_model
                    .save::<B>(&model_path)
                    .expect("[ActorEntity] Failed model save...");
            } 

            match &self.model {
                Some(model) => match model.reload_from_module(ok_model, version).await {
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

    fn __get_actor_statistics(&self, _msg: RoutedMessage) {}

    async fn _handle_shutdown(&self, _msg: RoutedMessage) {
        if !self.current_traj.actions.is_empty() {
            let send_traj_msg = {
                let traj_clone = self.current_traj.clone();
                RoutedMessage {
                    actor_id: self.id,
                    protocol: RoutingProtocol::SendTrajectory,
                    payload: RoutedPayload::SendTrajectory {
                        timestamp: (
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Clock skew")
                                .as_millis(),
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Clock skew")
                                .as_nanos(),
                        ),
                        trajectory: traj_clone,
                    },
                }
            };

            let _ = self.shared_tx_to_sender.send(send_traj_msg).await;
        }
    }
}
