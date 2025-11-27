use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
#[cfg(feature = "grpc_network")]
use crate::network::client::runtime::transport::TransportClient;
use crate::utilities::configuration::ClientConfigLoader;
use crate::utilities::misc_utils::ServerAddresses;
use crate::utilities::orchestration::tokio_utils::get_or_init_tokio_runtime;

use relayrl_types::prelude::AnyBurnTensor;
use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::tensor::{BackendMatcher, DeviceType};
use relayrl_types::types::data::trajectory::RelayRLTrajectory;
use relayrl_types::types::model::utils::{deserialize_model_module, validate_module};
use relayrl_types::types::model::{HotReloadableModel, ModelError, ModelModule};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use burn_tensor::{Tensor, backend::Backend};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ActorError {
    #[error(transparent)]
    ModelError(#[from] ModelError),
    #[error("Trajectory send failed: {0}")]
    TrajectorySendError(String),
    #[error("Action request failed: {0}")]
    ActionRequestError(String),
    #[error("Message handling failed: {0}")]
    MessageHandlingError(String),
    #[error("Type conversion failed: {0}")]
    TypeConversionError(String),
    #[error("System error: {0}")]
    SystemError(String),
}

pub trait ActorEntity<B: Backend + BackendMatcher<Backend = B>>: Send + Sync + 'static {
    async fn new(
        actor_id: ActorUuid,
        device: DeviceType,
        model: Option<HotReloadableModel<B>>,
        model_path: PathBuf,
        shared_config: Arc<RwLock<ClientConfigLoader>>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
        rx_from_router: Receiver<RoutedMessage>,
        tx_to_sender: Sender<RoutedMessage>,
        transport: Arc<TransportClient<B>>,
    ) -> (Self, bool)
    where
        Self: Sized;
    async fn spawn_loop(&mut self) -> Result<(), ActorError>;
    async fn _initial_model_handshake(&mut self, msg: RoutedMessage) -> Result<(), ActorError>;
    fn __request_action(&mut self, msg: RoutedMessage) -> Result<(), ActorError>;
    fn __flag_last_action(&mut self, msg: RoutedMessage) -> Result<(), ActorError>;
    fn __get_model_version(&self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn _refresh_model(&self, msg: RoutedMessage) -> Result<(), ActorError>;
    fn __get_actor_statistics(&self, _msg: RoutedMessage) -> Result<(), ActorError>;
    async fn _handle_shutdown(&self, _msg: RoutedMessage) -> Result<(), ActorError>;
}

/// Responsible for performing inference with an in-memory model
pub(crate) struct Actor<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    actor_id: ActorUuid,
    model: Option<HotReloadableModel<B>>,
    model_path: PathBuf,
    model_device: DeviceType,
    current_traj: RelayRLTrajectory,
    rx_from_router: Receiver<RoutedMessage>,
    shared_tx_to_sender: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient<B>>>,
    shared_config: Arc<RwLock<ClientConfigLoader>>,
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> ActorEntity<B>
    for Actor<B, D_IN, D_OUT>
{
    async fn new(
        actor_id: ActorUuid,
        device: DeviceType,
        model: Option<HotReloadableModel<B>>,
        model_path: PathBuf,
        shared_config: Arc<RwLock<ClientConfigLoader>>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_tx_to_sender: Sender<RoutedMessage>,
        transport: Arc<TransportClient<B>>,
    ) -> (Self, bool)
    where
        Self: Sized,
    {
        let max_length: u128 = shared_config.read().await.transport_config.max_traj_length;

        let mut actor: Actor<B, D_IN, D_OUT> = Self {
            actor_id,
            model: None,
            model_path,
            model_device: device,
            current_traj: RelayRLTrajectory::new(max_length as usize),
            rx_from_router,
            shared_tx_to_sender,
            transport: Some(transport),
            shared_config: shared_config.clone(),
            shared_server_addresses: shared_server_addresses.clone(),
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

    async fn spawn_loop(&mut self) -> Result<(), ActorError> {
        while let Some(msg) = self.rx_from_router.recv().await {
            match msg.protocol {
                RoutingProtocol::ModelHandshake => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::_initial_model_handshake(self, msg)
                        .await?;
                }
                RoutingProtocol::RequestInference => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::__request_action(self, msg)?;
                }
                RoutingProtocol::FlagLastInference => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::__flag_last_action(self, msg)?;
                }
                RoutingProtocol::ModelVersion => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::__get_model_version(self, msg)?;
                }
                RoutingProtocol::ModelUpdate => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::_refresh_model(self, msg).await?;
                }
                RoutingProtocol::ActorStatistics => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::__get_actor_statistics(self, msg)?;
                }
                RoutingProtocol::Shutdown => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::_handle_shutdown(self, msg).await?;
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }

    async fn _initial_model_handshake(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelHandshake = msg.payload {
            if self.model.is_none() {
                if let Some(transport) = &self.transport {
                    let model_server_address: String = self
                        .shared_server_addresses
                        .read()
                        .await
                        .model_server_address
                        .clone();
                    let agent_listener_address: String = self
                        .shared_server_addresses
                        .read()
                        .await
                        .agent_listener_address
                        .clone();

                    match &**transport {
                        #[cfg(feature = "grpc_network")]
                        TransportClient::Async(async_tr) => {
                            // Use training server address for model handshake
                            println!(
                                "[Actor {:?}] Starting async model handshake with {}",
                                self.actor_id, model_server_address
                            );

                            if let Ok(Some(model)) = async_tr
                                .initial_model_handshake(
                                    &self.actor_id,
                                    &model_server_address,
                                    &agent_listener_address,
                                )
                                .await
                            {
                                println!(
                                    "[Actor {:?}] Model handshake successful, received model data",
                                    self.actor_id
                                );

                                // Save model to configured path
                                if let Err(e) = model.save(&self.model_path) {
                                    eprintln!(
                                        "[Actor {:?}] Failed to save model: {:?}",
                                        self.actor_id, e
                                    );
                                }

                                match &self.model {
                                    Some(model) => {
                                        let model_version = {
                                            let version = model.version() + 1;
                                            model
                                                .reload_from_path(self.model_path.clone(), version)
                                                .await
                                        };

                                        model_version.map_err(|e| {
                                            eprintln!(
                                                "[Actor {:?}] Failed to reload model: {:?}",
                                                self.actor_id, e
                                            );
                                            ActorError::from(e)
                                        })?;
                                    }
                                    None => {
                                        self.model = Some(
                                            HotReloadableModel::<B>::new_from_module(
                                                model,
                                                self.model_device.clone(),
                                            )
                                            .await
                                            .map_err(ActorError::from)?,
                                        );
                                    }
                                }
                            } else {
                                eprintln!(
                                    "[Actor {:?}] Model handshake failed or no model update needed",
                                    self.actor_id
                                );
                            }
                        }
                        TransportClient::Sync(sync_tr) => {
                            // Use agent listener address for model handshake
                            println!(
                                "[Actor {:?}] Starting sync model handshake with {}",
                                self.actor_id, model_server_address
                            );

                            if let Ok(Some(model)) = sync_tr.initial_model_handshake(
                                &self.actor_id,
                                &model_server_address,
                                &agent_listener_address,
                            ) {
                                println!(
                                    "[Actor {:?}] Model handshake successful, received model data",
                                    self.actor_id
                                );

                                // Save model to configured path
                                if let Err(e) = model.save(&self.model_path) {
                                    eprintln!(
                                        "[Actor {:?}] Failed to save model: {:?}",
                                        self.actor_id, e
                                    );
                                }

                                match &self.model {
                                    Some(existing_model) => {
                                        let version = existing_model.version() + 1;
                                        let model_version = existing_model
                                            .reload_from_path(self.model_path.clone(), version)
                                            .await;
                                        model_version.map_err(|e| {
                                            eprintln!(
                                                "[Actor {:?}] Failed to reload model: {:?}",
                                                self.actor_id, e
                                            );
                                            ActorError::from(e)
                                        })?;
                                    }
                                    None => {
                                        self.model = Some(
                                            HotReloadableModel::<B>::new_from_module(
                                                model,
                                                self.model_device.clone(),
                                            )
                                            .await
                                            .map_err(ActorError::from)?,
                                        );
                                    }
                                }
                            } else {
                                eprintln!(
                                    "[Actor {:?}] Model handshake failed or no model update needed",
                                    self.actor_id
                                );
                            }
                        }
                    }
                } else {
                    eprintln!(
                        "[Actor {:?}] No transport configured for model handshake",
                        self.actor_id
                    );
                }
            } else {
                println!(
                    "[Actor {:?}] Model already available, handshake not needed",
                    self.actor_id
                );
            }
        }
        Ok(())
    }

    /// Request action from the model
    ///
    /// # Synchronous Context
    ///
    /// This method is synchronous but needs to call async model operations. It uses `block_on()` to
    /// bridge the sync/async boundary. This is acceptable because:
    /// 1. This method is called from sync context within the actor's message loop
    /// 2. The model forward pass is the primary async operation that needs to complete
    /// 3. The actor's spawn_loop is async, so this doesn't block the async runtime
    ///
    /// If deadlocks occur, consider making the entire ActorEntity trait async.
    fn __request_action(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let Some(model) = &self.model {
            if let RoutedPayload::RequestInference(inference_request) = msg.payload {
                let InferenceRequest {
                    observation,
                    mask,
                    reward,
                    reply_to,
                } = *inference_request; // Dereference the Box to get the inner value

                let observation_tensor: AnyBurnTensor<B, D_IN> = *observation
                    .downcast::<AnyBurnTensor<B, D_IN>>()
                    .map_err(|_| {
                        ActorError::TypeConversionError(
                            "Failed to downcast observation to AnyBurnTensor".to_string(),
                        )
                    })?;
                let mask_tensor: Option<AnyBurnTensor<B, D_OUT>> = *mask
                    .downcast::<Option<AnyBurnTensor<B, D_OUT>>>()
                    .map_err(|_| {
                        ActorError::TypeConversionError(
                            "Failed to downcast mask to Option<AnyBurnTensor>".to_string(),
                        )
                    })?;

                let actor_id = self.actor_id;
                let rt = get_or_init_tokio_runtime();
                let timeout_secs = 30;
                match rt.block_on(async {
                    timeout(Duration::from_secs(timeout_secs), async {
                        model.forward::<D_IN, D_OUT>(
                            observation_tensor,
                            mask_tensor,
                            reward,
                            actor_id,
                        )
                    })
                    .await
                    .map_err(|_| {
                        ActorError::SystemError(format!(
                            "Inference timeout after {} seconds",
                            timeout_secs
                        ))
                    })
                }) {
                    Ok(result) => match result {
                        Ok(r4sa) => {
                            self.current_traj.add_action(r4sa.clone());
                            reply_to.send(Arc::new(r4sa)).map_err(|e| {
                                ActorError::MessageHandlingError(format!(
                                    "Failed to send inference: {:?}",
                                    e
                                ))
                            })?;
                        }
                        Err(e) => {
                            eprintln!(
                                "[ActorEntity {:?}] Failed inference, no inference created or available... {:?}",
                                self.actor_id, e
                            );
                            return Err(ActorError::ActionRequestError(format!("{:?}", e)));
                        }
                    },
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Flag the last action and send trajectory
    ///
    /// # Synchronous Context
    ///
    /// This method is synchronous but needs to send messages via async channels. It uses `block_on()`
    /// to bridge the sync/async boundary. This is acceptable because:
    /// 1. This method is called from sync context within the actor's message loop
    /// 2. The channel send operation needs async execution
    /// 3. The actor's spawn_loop is async, so this doesn't block the async runtime
    ///
    /// If deadlocks occur, consider making the entire ActorEntity trait async.
    fn __flag_last_action(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::FlagLastInference { reward } = msg.payload {
            let actor_id = self.actor_id;
            let mut last_action: RelayRLAction =
                RelayRLAction::new(None, None, None, reward, true, None, Some(actor_id));
            last_action.update_reward(reward);
            self.current_traj.add_action(last_action);

            let send_traj_msg = {
                let traj_clone = self.current_traj.clone();
                let now = SystemTime::now();
                let duration_ms = now
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| ActorError::SystemError(format!("Clock skew: {}", e)))?;
                let duration_ns = now
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| ActorError::SystemError(format!("Clock skew: {}", e)))?;
                RoutedMessage {
                    actor_id: self.actor_id,
                    protocol: RoutingProtocol::SendTrajectory,
                    payload: RoutedPayload::SendTrajectory {
                        timestamp: (duration_ms.as_millis(), duration_ns.as_nanos()),
                        trajectory: traj_clone,
                    },
                }
            };

            let rt = get_or_init_tokio_runtime();
            rt.block_on(async {
                self.shared_tx_to_sender
                    .send(send_traj_msg)
                    .await
                    .map_err(|e| ActorError::TrajectorySendError(format!("{:?}", e)))
            })?;
        }
        Ok(())
    }

    fn __get_model_version(&self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelVersion { reply_to } = msg.payload {
            let current_model = &self.model;

            match current_model {
                Some(some_model) => {
                    let version = some_model.version();
                    reply_to
                        .send(version)
                        .map_err(|e| ActorError::MessageHandlingError(format!("{:?}", e)))?;
                }
                None => {
                    reply_to
                        .send(-1)
                        .map_err(|e| ActorError::MessageHandlingError(format!("{:?}", e)))?;
                }
            }
        }
        Ok(())
    }

    async fn _refresh_model(&self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelUpdate {
            model_bytes,
            version,
        } = msg.payload
        {
            let model: Result<ModelModule<B>, ModelError> =
                deserialize_model_module::<B>(model_bytes, self.model_device.clone());
            let model_path: PathBuf = self.model_path.clone();
            if let Ok(ok_model) = model {
                // Validate the model - it gets dimensions from the model itself
                if let Err(e) = validate_module::<B>(&ok_model).map_err(ActorError::from) {
                    eprintln!(
                        "[ActorEntity {:?}] Failed to validate model: {:?}",
                        self.actor_id, e
                    );
                    return Err(e);
                };

                if let Err(e) = ok_model.save(&model_path).map_err(ActorError::from) {
                    eprintln!(
                        "[ActorEntity {:?}] Failed to save model: {:?}",
                        self.actor_id, e
                    );
                    return Err(e);
                }

                match &self.model {
                    Some(model) => {
                        model
                            .reload_from_module(ok_model.clone(), version)
                            .await
                            .map_err(ActorError::from)?;
                    }
                    None => {
                        eprintln!(
                            "[ActorEntity {:?}] Model does not exist, no model refresh possible...",
                            self.actor_id
                        );
                        return Err(ActorError::ModelError(ModelError::IoError(
                            "Model does not exist in actor instance".to_string(),
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn __get_actor_statistics(&self, _msg: RoutedMessage) -> Result<(), ActorError> {
        Ok(())
    }

    async fn _handle_shutdown(&self, _msg: RoutedMessage) -> Result<(), ActorError> {
        if !self.current_traj.actions.is_empty() {
            let send_traj_msg = {
                let traj_clone = self.current_traj.clone();
                let now = SystemTime::now();
                let duration_ms = now
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| ActorError::SystemError(format!("Clock skew: {}", e)))?;
                let duration_ns = now
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| ActorError::SystemError(format!("Clock skew: {}", e)))?;
                RoutedMessage {
                    actor_id: self.actor_id,
                    protocol: RoutingProtocol::SendTrajectory,
                    payload: RoutedPayload::SendTrajectory {
                        timestamp: (duration_ms.as_millis(), duration_ns.as_nanos()),
                        trajectory: traj_clone,
                    },
                }
            };

            let _ = self.shared_tx_to_sender.send(send_traj_msg).await;
        }
        Ok(())
    }
}
