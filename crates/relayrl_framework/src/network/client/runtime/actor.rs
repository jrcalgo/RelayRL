use crate::network::client::agent::{ActorInferenceMode, ClientModes};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::TransportError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::{
    InferenceDispatcher, TrainingDispatcher,
};
use crate::network::client::runtime::router::{
    InferenceRequest, RoutedMessage, RoutedPayload, RoutingProtocol,
};
use crate::utilities::configuration::ClientConfigLoader;
use crate::utilities::tokio::get_or_init_tokio_runtime;

use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{BackendMatcher, ConversionBurnTensor, DeviceType};
use relayrl_types::data::trajectory::RelayRLTrajectory;
use relayrl_types::model::utils::{deserialize_model_module, validate_module};
use relayrl_types::model::{HotReloadableModel, ModelError, ModelModule};
use relayrl_types::prelude::tensor::relayrl::AnyBurnTensor;

use bincode::config;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;
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
    #[error("Inference request failed: {0}")]
    InferenceRequestError(String),
    #[error("Message handling failed: {0}")]
    MessageHandlingError(String),
    #[error("Type conversion failed: {0}")]
    TypeConversionError(String),
    #[error("System error: {0}")]
    SystemError(String),
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error(transparent)]
    TransportError(#[from] TransportError),
}

pub trait ActorEntity<B: Backend + BackendMatcher<Backend = B>>: Send + Sync + 'static {
    async fn new(
        actor_id: ActorUuid,
        device: DeviceType,
        reloadable_model: Option<HotReloadableModel<B>>,
        shared_local_model_path: Arc<RwLock<PathBuf>>,
        shared_max_traj_length: Arc<RwLock<u128>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_server_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_tx_to_buffer: Sender<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
    ) -> (Self, bool)
    where
        Self: Sized;
    async fn spawn_loop(&mut self) -> Result<(), ActorError>;
    async fn _initial_model_handshake(&mut self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn __get_model_version(&self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn _refresh_model(&self, msg: RoutedMessage) -> Result<(), ActorError>;
    async fn _handle_shutdown(&self, _msg: RoutedMessage) -> Result<(), ActorError>;
}

/// Responsible for performing inference with an in-memory model
pub(crate) struct Actor<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    actor_id: ActorUuid,
    reloadable_model: Option<Arc<HotReloadableModel<B>>>,
    shared_local_model_path: Arc<RwLock<PathBuf>>,
    shared_max_traj_length: Arc<RwLock<u128>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    shared_server_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
    model_device: DeviceType,
    current_traj: RelayRLTrajectory,
    rx_from_router: Receiver<RoutedMessage>,
    shared_tx_to_buffer: Sender<RoutedMessage>,
    shared_client_modes: Arc<ClientModes>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    Actor<B, D_IN, D_OUT>
{
    #[inline(always)]
    fn extract_inference_request(
        msg: RoutedMessage,
    ) -> Result<
        (
            Arc<AnyBurnTensor<B, D_IN>>,
            Option<Arc<AnyBurnTensor<B, D_OUT>>>,
            f32,
            oneshot::Sender<Arc<RelayRLAction>>,
        ),
        ActorError,
    > {
        let RoutedPayload::RequestInference(req) = msg.payload else {
            return Err(ActorError::MessageHandlingError(
                "Expected RequestInference payload".to_string(),
            ));
        };

        let InferenceRequest {
            observation,
            mask,
            reward,
            reply_to,
        } = *req;

        let obs: Arc<AnyBurnTensor<B, D_IN>> = *observation
            .downcast::<Arc<AnyBurnTensor<B, D_IN>>>()
            .map_err(|_| {
                ActorError::TypeConversionError("Failed to downcast observation".into())
            })?;

        let mask: Option<Arc<AnyBurnTensor<B, D_OUT>>> = *mask
            .downcast::<Option<Arc<AnyBurnTensor<B, D_OUT>>>>()
            .map_err(|_| ActorError::TypeConversionError("Failed to downcast mask".into()))?;

        Ok((obs, mask, reward, reply_to))
    }

    #[inline(always)]
    async fn handle_inference_kind(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        match self.shared_client_modes.actor_inference_mode {
            ActorInferenceMode::Local(_) => self.perform_local_inference(msg).await,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            ActorInferenceMode::Server(_) => self.request_server_inference(msg).await,
        }
    }

    async fn perform_local_inference(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        let reloadable_model = self.reloadable_model.as_ref().ok_or_else(|| {
            ActorError::SystemError("Model not loaded/available for actor inference".into())
        })?;
        let (obs, mask, reward, reply_to) = Self::extract_inference_request(msg)?;

        let reloadable_model = Arc::clone(reloadable_model);
        let actor_id = self.actor_id;

        let r4sa = tokio::task::spawn_blocking(move || -> Result<RelayRLAction, ModelError> {
            reloadable_model.forward::<D_IN, D_OUT>(obs, mask, reward, actor_id)
        })
        .await
        .map_err(|e| ActorError::SystemError(format!("spawn_blocking join error: {e}")))?
        .map_err(ActorError::from)?;

        self.current_traj.add_action(r4sa.clone());
        reply_to.send(Arc::new(r4sa)).map_err(|e| {
            ActorError::MessageHandlingError(format!("reply_to send failed: {e:?}"))
        })?;

        Ok(())
    }

    /// Server inference: serialize observation (and optionally mask) and send to server.
    /// Note: if obs/mask live on GPU, you will pay a device->host copy during serialization.
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    async fn request_server_inference(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        // Both the inference_kind and inference_dispatcher initializations are based on
        // the client_capabilities.server_inference flag. Thus, if server_inference is true, the inference_dispatcher will be Some
        // and inference_kind will be InferenceKind::Server. The opposite is true: see request_local_inference for the opposite case.
        // If the inference_dispatcher is None, we will use the local model.
        if let Some(inference_dispatcher) = &self.shared_inference_dispatcher {
            let _reloadable_model = self
                .reloadable_model
                .as_ref()
                .ok_or_else(|| ActorError::SystemError("Model not loaded".into()))?;

            // we assume that the server_addresses are available if the inference_dispatcher is Some
            let shared_server_addresses = self
                .shared_server_addresses
                .as_ref()
                .ok_or_else(|| ActorError::SystemError("Server addresses not available".into()))?
                .clone();

            let (obs, _mask, _reward, reply_to) = Self::extract_inference_request(msg)?;

            let obs_bytes: Vec<u8> = Vec::new();
            let _ = obs; // suppress unused warning

            let actor_id = self.actor_id;

            let r4sa = inference_dispatcher
                .send_inference_request(&actor_id, &obs_bytes, shared_server_addresses)
                .await?;

            self.current_traj.add_action(r4sa.clone());
            reply_to.send(Arc::new(r4sa)).map_err(|e| {
                ActorError::MessageHandlingError(format!("reply_to send failed: {e:?}"))
            })?;

            return Ok(());
        } else {
            // local inference fallback (this should never happen, but just in case)
            return self.perform_local_inference(msg).await;
        }

        Ok(())
    }

    async fn perform_flag_last_action(&mut self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::FlagLastInference { reward } = msg.payload {
            let actor_id = self.actor_id;
            let mut last_action =
                RelayRLAction::new(None, None, None, reward, true, None, Some(actor_id));
            last_action.update_reward(reward);
            self.current_traj.add_action(last_action);

            let traj_clone = self.current_traj.clone();
            let now = SystemTime::now();
            let duration = now
                .duration_since(UNIX_EPOCH)
                .map_err(|e| ActorError::SystemError(format!("Clock skew: {e}")))?;
            let send_traj_msg = RoutedMessage {
                actor_id: self.actor_id,
                protocol: RoutingProtocol::SendTrajectory,
                payload: RoutedPayload::SendTrajectory {
                    timestamp: (duration.as_millis(), duration.as_nanos()),
                    trajectory: traj_clone,
                },
            };

            self.shared_tx_to_buffer
                .send(send_traj_msg)
                .await
                .map_err(|e| ActorError::TrajectorySendError(format!("{e:?}")))?;
        }
        Ok(())
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize> ActorEntity<B>
    for Actor<B, D_IN, D_OUT>
{
    async fn new(
        actor_id: ActorUuid,
        device: DeviceType,
        reloadable_model: Option<HotReloadableModel<B>>,
        shared_local_model_path: Arc<RwLock<PathBuf>>,
        shared_max_traj_length: Arc<RwLock<u128>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_inference_dispatcher: Option<Arc<InferenceDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_training_dispatcher: Option<Arc<TrainingDispatcher<B>>>,
        #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
        shared_server_addresses: Option<Arc<RwLock<SharedTransportAddresses>>>,
        rx_from_router: Receiver<RoutedMessage>,
        shared_tx_to_buffer: Sender<RoutedMessage>,
        shared_client_modes: Arc<ClientModes>,
    ) -> (Self, bool)
    where
        Self: Sized,
    {
        let max_traj_length: u128 = shared_max_traj_length.read().await.clone();

        let mut actor: Actor<B, D_IN, D_OUT> = Self {
            actor_id,
            reloadable_model: None,
            shared_local_model_path,
            shared_max_traj_length,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_inference_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_training_dispatcher,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            shared_server_addresses,
            model_device: device,
            current_traj: RelayRLTrajectory::new(max_traj_length as usize),
            rx_from_router,
            shared_tx_to_buffer,
            shared_client_modes,
        };

        let mut model_init_flag: bool = false;
        match reloadable_model {
            Some(some_model) => {
                actor.reloadable_model = Some(Arc::new(some_model));
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
                    self.handle_inference_kind(msg).await?;
                }
                RoutingProtocol::FlagLastInference => {
                    self.perform_flag_last_action(msg).await?;
                }
                RoutingProtocol::ModelVersion => {
                    self.__get_model_version(msg).await?;
                }
                RoutingProtocol::ModelUpdate => {
                    <Actor<B, D_IN, D_OUT> as ActorEntity<B>>::_refresh_model(self, msg).await?;
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
            if self.reloadable_model.is_none() {
                #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
                if let Some(training_dispatcher) = &self.shared_training_dispatcher {
                    println!(
                        "[Actor {:?}] Starting training model handshake",
                        self.actor_id
                    );

                    let shared_server_addresses = self
                        .shared_server_addresses
                        .as_ref()
                        .ok_or_else(|| {
                            ActorError::SystemError("Server addresses not available".into())
                        })?
                        .clone();

                    if let Ok(Some(model)) = training_dispatcher
                        .initial_model_handshake(&self.actor_id, shared_server_addresses)
                        .await
                    {
                        println!(
                            "[Actor {:?}] Model handshake successful, received model data",
                            self.actor_id
                        );

                        if let Err(e) =
                            model.save(&self.shared_local_model_path.read().await.clone())
                        {
                            eprintln!("[Actor {:?}] Failed to save model: {:?}", self.actor_id, e);
                        }

                        match &self.reloadable_model {
                            Some(existing_reloadable_model) => {
                                let model_version = {
                                    let version = existing_reloadable_model.version() + 1;
                                    existing_reloadable_model
                                        .reload_from_path(
                                            self.shared_local_model_path.read().await.clone(),
                                            version,
                                        )
                                        .await
                                }
                                .map_err(|e| {
                                    eprintln!(
                                        "[Actor {:?}] Failed to reload model: {:?}",
                                        self.actor_id, e
                                    );
                                    ActorError::from(e)
                                })?;
                            }
                            None => {
                                self.reloadable_model = Some(Arc::new(
                                    HotReloadableModel::<B>::new_from_module(
                                        model,
                                        self.model_device.clone(),
                                    )
                                    .await
                                    .map_err(ActorError::from)?,
                                ));
                            }
                        }
                    } else {
                        eprintln!(
                            "[Actor {:?}] Model handshake failed or no model update needed",
                            self.actor_id
                        );
                    }
                } else {
                    eprintln!(
                        "[Actor {:?}] No transport dispatcher configured for model handshake",
                        self.actor_id
                    );
                }
                #[cfg(not(any(feature = "nats-transport", feature = "zmq-transport")))]
                {
                    eprintln!(
                        "[Actor {:?}] No transport dispatcher configured for model handshake",
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

    async fn __get_model_version(&self, msg: RoutedMessage) -> Result<(), ActorError> {
        if let RoutedPayload::ModelVersion { reply_to } = msg.payload {
            let current_model = &self.reloadable_model;

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
            let model_path: PathBuf = self.shared_local_model_path.read().await.clone();
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

                match &self.reloadable_model {
                    Some(existing_reloadable_model) => {
                        existing_reloadable_model
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

            let _ = self.shared_tx_to_buffer.send(send_traj_msg).await;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {}
