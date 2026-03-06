use super::{RoutedMessage, RouterError};
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::coordination::lifecycle_manager::SharedTransportAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::TransportError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::data::transport_sink::transport_dispatcher::TrainingDispatcher;

use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::get_context_entries;

use burn_tensor::backend::Backend;
use relayrl_types::data::tensor::BackendMatcher;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::sync::{RwLock, broadcast};

#[derive(Debug, Error)]
pub enum TransportReceiverError {
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    #[error(transparent)]
    TransportError(#[from] TransportError),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error("No context entries found")]
    NoEntriesFound,
}

/// Listens & receives model bytes from a training server
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) struct ClientTransportModelReceiver<B: Backend + BackendMatcher<Backend = B>> {
    associated_router_namespace: RouterNamespace,
    active: AtomicBool,
    global_dispatcher_tx: Sender<RoutedMessage>,
    training_dispatcher: Arc<TrainingDispatcher<B>>,
    shared_transport_addresses: Arc<RwLock<SharedTransportAddresses>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
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
                            // this should never happen, but if it does, we need to break the loop
                            eprintln!("[ClientTransportModelReceiver] listen_for_model returned Ok");
                            self.active.store(false, Ordering::SeqCst);
                        }
                        Err(e) => {
                            eprintln!("[ClientTransportModelReceiver] Failed to listen for model: {}", e);
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
mod tests {}
