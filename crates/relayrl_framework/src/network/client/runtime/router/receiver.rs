use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::data::transport_sink::TransportError;
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::TrainingDispatcher;
use crate::network::client::runtime::router::{RoutedMessage, RouterError};
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;

use relayrl_types::prelude::tensor::burn::backend::Backend;
use relayrl_types::prelude::tensor::relayrl::BackendMatcher;

use active_uuid_registry::interface::get_context_entries;
use active_uuid_registry::UuidPoolError;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::sync::{broadcast, RwLock};
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

/// Listens & receives model bytes from a training server
pub(crate) struct ClientTransportModelReceiver<B: Backend + BackendMatcher<Backend = B>> {
    associated_router_namespace: RouterNamespace,
    active: AtomicBool,
    global_dispatcher_tx: Sender<RoutedMessage>,
    training_dispatcher: Arc<TrainingDispatcher<B>>,
    shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>> ClientTransportModelReceiver<B> {
    pub fn new(
        associated_router_namespace: RouterNamespace,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
        training_dispatcher: Arc<TrainingDispatcher<B>>,
    ) -> Self {
        Self {
            associated_router_namespace,
            active: AtomicBool::new(false),
            global_dispatcher_tx,
            training_dispatcher,
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
            self.associated_router_namespace.as_ref(),
            crate::network::RECEIVER_CONTEXT,
        )
        .map_err(TransportReceiverError::from)?;
        let receiver_entry = entries
            .first()
            .ok_or(TransportReceiverError::NoEntriesFound)?;

        let global_dispatcher_tx = self.global_dispatcher_tx.clone();

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
                    self.active.store(false, Ordering::SeqCst);
                }

                result = self.training_dispatcher.listen_for_model(receiver_entry.clone(), global_dispatcher_tx.clone(), self.shared_transport_addresses.clone()) => {
                    match result {
                        Ok(()) => {
                            log::warn!("[ClientTransportModelReceiver] Model listener stopped gracefully");
                            self.active.store(false, Ordering::SeqCst);
                        }
                        Err(e) => {
                            log::error!("[ClientTransportModelReceiver] Failed to listen for model: {}", e);
                            tokio::time::sleep(Duration::from_secs(1)).await;
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
    use active_uuid_registry::UuidPoolError;

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
}
