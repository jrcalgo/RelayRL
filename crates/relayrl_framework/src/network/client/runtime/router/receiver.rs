use crate::network::client::runtime::coordination::coordinator::CHANNEL_THROUGHPUT;
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::state_manager::{ActorUuid, StateManager};
use crate::network::client::runtime::data::transport_sink::TransportError;
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::TrainingDispatcher;
use crate::network::client::runtime::router::{
    RoutedMessage, RoutedPayload, RouterError, RoutingProtocol,
};

use relayrl_types::prelude::tensor::burn::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::get_context_entries;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::sync::{RwLock, broadcast};
use tokio::time::Duration;

#[derive(Debug, Error)]
pub enum TransportReceiverError {
    #[error(transparent)]
    TransportError(#[from] TransportError),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error("No context entries found")]
    NoEntriesFound,
}

fn prepare_transport_model_update_for_dispatch<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>(
    shared_state: &StateManager<B, D_IN, D_OUT>,
    mut msg: RoutedMessage,
    last_forwarded_model_versions: &mut HashMap<ActorUuid, i64>,
) -> Option<RoutedMessage> {
    let model_version = match (&msg.protocol, &msg.payload) {
        (RoutingProtocol::ModelUpdate, RoutedPayload::ModelUpdate { version, .. }) => *version,
        _ => return Some(msg),
    };

    let canonical_actor_id = shared_state.canonical_model_update_target(msg.actor_id);
    if last_forwarded_model_versions.get(&canonical_actor_id) == Some(&model_version) {
        return None;
    }

    last_forwarded_model_versions.insert(canonical_actor_id, model_version);
    msg.actor_id = canonical_actor_id;

    Some(msg)
}

/// Listens & receives model bytes from a training server. Created once per client runtime.
pub(crate) struct ClientTransportModelReceiver<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    client_namespace: Arc<str>,
    active: AtomicBool,
    global_dispatcher_tx: Sender<RoutedMessage>,
    training_dispatcher: Arc<TrainingDispatcher<B>>,
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientTransportModelReceiver<B, D_IN, D_OUT>
{
    pub fn new(
        client_namespace: Arc<str>,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
        training_dispatcher: Arc<TrainingDispatcher<B>>,
    ) -> Self {
        Self {
            client_namespace,
            active: AtomicBool::new(false),
            global_dispatcher_tx,
            training_dispatcher,
            shared_state,
            shared_transport_addresses,
            shutdown: None,
        }
    }

    pub fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(&mut self) -> Result<(), RouterError> {
        self.active.store(true, Ordering::SeqCst);

        let entries = get_context_entries(
            self.client_namespace.as_ref(),
            crate::network::RECEIVER_CONTEXT,
        )
        .map_err(TransportReceiverError::from)?;
        let receiver_entry = entries
            .first()
            .ok_or(TransportReceiverError::NoEntriesFound)?
            .clone();

        let (model_update_tx, mut model_update_rx) =
            tokio::sync::mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

        let training_dispatcher = self.training_dispatcher.clone();
        let transport_addresses = self.shared_transport_addresses.clone();
        let receiver_entry_for_task = receiver_entry.clone();
        let listener_handle = tokio::spawn(async move {
            loop {
                match training_dispatcher
                    .listen_for_model(
                        receiver_entry_for_task.clone(),
                        model_update_tx.clone(),
                        transport_addresses.clone(),
                    )
                    .await
                {
                    Ok(()) => {
                        log::warn!(
                            "[ClientTransportModelReceiver] Model listener stopped gracefully"
                        );
                        break;
                    }
                    Err(e) => {
                        log::error!(
                            "[ClientTransportModelReceiver] Failed to listen for model: {}",
                            e
                        );
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });
        let mut last_forwarded_model_versions: HashMap<ActorUuid, i64> = HashMap::new();

        while self.active.load(Ordering::SeqCst) {
            tokio::select! {
                biased;

                _ = async {
                    if let Some(rx) = &mut self.shutdown {
                        let _ = rx.recv().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    if let Err(e) = self
                        .training_dispatcher
                        .stop_model_listener(receiver_entry.clone())
                        .await
                    {
                        log::error!(
                            "[ClientTransportModelReceiver] Failed to stop model listener: {}",
                            e
                        );
                    }
                    listener_handle.abort();
                    self.active.store(false, Ordering::SeqCst);
                }

                msg = model_update_rx.recv() => {
                    match msg {
                        Some(msg) => {
                            let msg = {
                                let shared_state = self.shared_state.read().await;
                                prepare_transport_model_update_for_dispatch(
                                    &shared_state,
                                    msg,
                                    &mut last_forwarded_model_versions,
                                )
                            };
                            let Some(msg) = msg else {
                                continue;
                            };

                            if let Err(e) = self.global_dispatcher_tx.send(msg).await {
                                log::error!("[ClientTransportModelReceiver] Failed to send message to global dispatcher: {}", e);
                            }
                        }
                        None => {
                            log::warn!("[ClientTransportModelReceiver] Model update channel closed, shutting down");
                            listener_handle.abort();
                            self.active.store(false, Ordering::SeqCst);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::network::client::agent::{
        ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode,
    };
    use crate::network::client::runtime::coordination::state_manager::StateManager;
    #[cfg(feature = "metrics")]
    use crate::utilities::observability::metrics::MetricsManager;
    use active_uuid_registry::UuidPoolError;
    use active_uuid_registry::registry_uuid::Uuid;
    use burn_ndarray::NdArray;
    use relayrl_types::data::tensor::DeviceType;
    use std::path::PathBuf;
    use tokio::sync::mpsc;

    type TestBackend = NdArray<f32>;
    const D_IN: usize = 4;
    const D_OUT: usize = 1;

    fn independent_modes() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn shared_modes() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Shared),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn make_state_manager(
        modes: Arc<ClientModes>,
    ) -> (
        StateManager<TestBackend, D_IN, D_OUT>,
        tokio::sync::mpsc::Receiver<RoutedMessage>,
    ) {
        let namespace: Arc<str> = Arc::from(format!("test-receiver-{}", Uuid::new_v4()));
        StateManager::<TestBackend, D_IN, D_OUT>::new(
            namespace,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            modes,
            Arc::new(RwLock::new(100)),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            Arc::new(RwLock::new(PathBuf::new())),
            None,
            #[cfg(feature = "metrics")]
            test_metrics(),
        )
    }

    #[cfg(feature = "metrics")]
    fn test_metrics() -> MetricsManager {
        let metrics_args = ("test-transport-receiver".to_string(), String::new());
        MetricsManager::new(
            Arc::new(RwLock::new(metrics_args.clone())),
            metrics_args,
            None,
        )
    }

    fn make_model_update(actor_id: Uuid, version: i64) -> RoutedMessage {
        RoutedMessage {
            actor_id,
            protocol: RoutingProtocol::ModelUpdate,
            payload: RoutedPayload::ModelUpdate {
                model_bytes: vec![1, 2, 3],
                version,
            },
        }
    }

    #[test]
    fn no_entries_found_displays_non_empty_string() {
        let err = TransportReceiverError::NoEntriesFound;
        let s = format!("{}", err);
        assert!(!s.is_empty(), "Display output should be non-empty");
    }

    #[test]
    fn uuid_pool_error_wraps_source() {
        let source = UuidPoolError::FailedToFindUuidInPoolError("test-uuid".to_string());
        let err = TransportReceiverError::from(source.clone());
        assert!(matches!(err, TransportReceiverError::UuidPoolError(_)));
        let display = format!("{}", err);
        assert!(!display.is_empty());
    }

    #[test]
    fn uuid_pool_error_display_contains_source_message() {
        let source = UuidPoolError::FailedToFindUuidInPoolError("my-id".to_string());
        let err = TransportReceiverError::from(source);
        let display = format!("{}", err);
        assert!(display.contains("my-id"));
    }

    #[tokio::test]
    async fn shared_mode_model_updates_remap_to_canonical_actor() {
        let (mut state_manager, _rx) = make_state_manager(shared_modes());
        let actor_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        let (tx_to_buffer, _buffer_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

        for actor_id in &actor_ids {
            state_manager
                .new_actor(
                    *actor_id,
                    Arc::from("router-a"),
                    DeviceType::Cpu,
                    None,
                    tx_to_buffer.clone(),
                )
                .await
                .unwrap();
        }

        let expected_actor_id = actor_ids
            .iter()
            .min_by_key(|actor_id| actor_id.to_string())
            .copied()
            .unwrap();
        let source_actor_id = actor_ids
            .iter()
            .copied()
            .find(|actor_id| *actor_id != expected_actor_id)
            .unwrap();
        let mut last_forwarded_model_versions = HashMap::new();

        let forwarded_message = prepare_transport_model_update_for_dispatch(
            &state_manager,
            make_model_update(source_actor_id, 7),
            &mut last_forwarded_model_versions,
        )
        .expect("expected model update to be forwarded");

        assert_eq!(forwarded_message.actor_id, expected_actor_id);
        assert!(matches!(
            forwarded_message.payload,
            RoutedPayload::ModelUpdate { version, .. } if version == 7
        ));
    }

    #[tokio::test]
    async fn shared_mode_model_updates_deduplicate_same_representative_version() {
        let (mut state_manager, _rx) = make_state_manager(shared_modes());
        let actor_ids: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();
        let (tx_to_buffer, _buffer_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

        for actor_id in &actor_ids {
            state_manager
                .new_actor(
                    *actor_id,
                    Arc::from("router-a"),
                    DeviceType::Cpu,
                    None,
                    tx_to_buffer.clone(),
                )
                .await
                .unwrap();
        }

        let expected_actor_id = actor_ids
            .iter()
            .min_by_key(|actor_id| actor_id.to_string())
            .copied()
            .unwrap();
        let mut last_forwarded_model_versions = HashMap::new();

        let first_forward = prepare_transport_model_update_for_dispatch(
            &state_manager,
            make_model_update(actor_ids[0], 11),
            &mut last_forwarded_model_versions,
        )
        .expect("expected first model update to be forwarded");
        assert_eq!(first_forward.actor_id, expected_actor_id);

        let duplicate_forward = prepare_transport_model_update_for_dispatch(
            &state_manager,
            make_model_update(actor_ids[1], 11),
            &mut last_forwarded_model_versions,
        );
        assert!(duplicate_forward.is_none());

        let newer_forward = prepare_transport_model_update_for_dispatch(
            &state_manager,
            make_model_update(actor_ids[1], 12),
            &mut last_forwarded_model_versions,
        )
        .expect("expected newer model version to be forwarded");
        assert_eq!(newer_forward.actor_id, expected_actor_id);
    }

    #[tokio::test]
    async fn independent_mode_model_updates_keep_distinct_actor_targets() {
        let (mut state_manager, _rx) = make_state_manager(independent_modes());
        let actor_ids: Vec<Uuid> = (0..2).map(|_| Uuid::new_v4()).collect();
        let (tx_to_buffer, _buffer_rx) = mpsc::channel::<RoutedMessage>(CHANNEL_THROUGHPUT);

        for actor_id in &actor_ids {
            state_manager
                .new_actor(
                    *actor_id,
                    Arc::from("router-a"),
                    DeviceType::Cpu,
                    None,
                    tx_to_buffer.clone(),
                )
                .await
                .unwrap();
        }

        let mut last_forwarded_model_versions = HashMap::new();
        let first_forward = prepare_transport_model_update_for_dispatch(
            &state_manager,
            make_model_update(actor_ids[0], 5),
            &mut last_forwarded_model_versions,
        )
        .expect("expected first independent model update to be forwarded");
        let second_forward = prepare_transport_model_update_for_dispatch(
            &state_manager,
            make_model_update(actor_ids[1], 5),
            &mut last_forwarded_model_versions,
        )
        .expect("expected second independent model update to be forwarded");

        assert_eq!(first_forward.actor_id, actor_ids[0]);
        assert_eq!(second_forward.actor_id, actor_ids[1]);
    }
}
