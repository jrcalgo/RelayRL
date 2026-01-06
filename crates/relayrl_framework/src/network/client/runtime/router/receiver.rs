use super::{RoutedMessage, RouterError};
use crate::network::client::runtime::coordination::lifecycle_manager::ServerAddresses;
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
use crate::network::client::runtime::data::transport::{TransportClient, TransportError};

use burn_tensor::backend::Backend;
use relayrl_types::types::data::tensor::BackendMatcher;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc::Sender;
use tokio::sync::{RwLock, broadcast};

#[derive(Debug, Error)]
pub enum TransportReceiverError {
    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    #[error("Transport error: {0}")]
    TransportError(#[from] TransportError),
}

/// Listens & receives model bytes from a training server
#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
pub(crate) struct ClientTransportModelReceiver<B: Backend + BackendMatcher<Backend = B>> {
    associated_router_id: RouterUuid,
    active: AtomicBool,
    global_dispatcher_tx: Sender<RoutedMessage>,
    transport: Option<Arc<TransportClient<B>>>,
    shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

#[cfg(any(feature = "async_transport", feature = "sync_transport"))]
impl<B: Backend + BackendMatcher<Backend = B>> ClientTransportModelReceiver<B> {
    pub fn new(
        associated_router_id: RouterUuid,
        global_dispatcher_tx: Sender<RoutedMessage>,
        shared_server_addresses: Arc<RwLock<ServerAddresses>>,
    ) -> Self {
        Self {
            associated_router_id,
            active: AtomicBool::new(false),
            global_dispatcher_tx,
            transport: None,
            shared_server_addresses,
            shutdown: None,
        }
    }

    #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
    pub fn with_transport(mut self, transport: Arc<TransportClient<B>>) -> Self {
        self.transport = Some(transport);
        self
    }

    pub fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(&self) -> Result<(), RouterError> {
        self.active.store(true, Ordering::SeqCst);

        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        if let Some(transport) = &self.transport {
            match &**transport {
                #[cfg(feature = "sync_transport")]
                TransportClient::Sync(_) => {
                    while self.active.load(Ordering::SeqCst) {
                        let agent_listener_address = self
                            .shared_server_addresses
                            .read()
                            .await
                            .agent_listener_address
                            .clone();
                        let global_dispatcher_tx: Sender<RoutedMessage> =
                            self.global_dispatcher_tx.clone();
                        let transport_clone: Arc<TransportClient<B>> = transport.clone();
                        let identity: RouterUuid = self.associated_router_id;

                        let transport_handle: tokio::task::JoinHandle<()> =
                            tokio::task::spawn_blocking(move || {
                                if let TransportClient::Sync(sync_tr) = &*transport_clone {
                                    match sync_tr.listen_for_model(
                                        &identity,
                                        agent_listener_address.as_str(),
                                        global_dispatcher_tx.clone(),
                                    ) {
                                        Ok(()) => {}
                                        Err(e) => {
                                            eprintln!(
                                                "[ClientTransportModelReceiver] ZMQ listen error: {}",
                                                e
                                            );
                                            std::thread::sleep(std::time::Duration::from_secs(5));
                                        }
                                    }
                                }
                            });

                        if let Some(mut shutdown_rx) =
                            self.shutdown.as_ref().map(|s| s.resubscribe())
                        {
                            let _ = shutdown_rx.recv().await;
                        } else {
                            std::future::pending::<()>().await;
                        }

                        self.active.store(false, Ordering::SeqCst);
                        transport_handle.abort();
                    }
                }
                #[cfg(feature = "async_transport")]
                TransportClient::Async(async_tr) => {
                    let mut shutdown_rx = self.shutdown.as_ref().map(|s| s.resubscribe());

                    while self.active.load(Ordering::SeqCst) {
                        let agent_listener_address: String = self
                            .shared_server_addresses
                            .read()
                            .await
                            .agent_listener_address
                            .clone();
                        let global_dispatcher_tx: Sender<RoutedMessage> =
                            self.global_dispatcher_tx.clone();
                        let identity: RouterUuid = self.associated_router_id;

                        tokio::select! {
                            result = async_tr.listen_for_model(&identity, agent_listener_address.as_str(), global_dispatcher_tx.clone()) => {
                                match result {
                                    Ok(()) => {
                                        // this should never happen, but if it does, we need to break the loop
                                        eprintln!("[ClientTransportModelReceiver] listen_for_model returned Ok");
                                        break;
                                    }
                                    Err(e) => {
                                        eprintln!("[ClientTransportModelReceiver] Failed to listen for model: {}", e);
                                        tokio::time::sleep(Duration::from_secs(1)).await;
                                    }
                                }
                            }
                            _ = async {
                                match &mut shutdown_rx {
                                    Some(rx) => rx.recv().await.map(|_| ()).map_err(|_| ()),
                                    None => {
                                        std::future::pending::<()>().await;
                                        Ok(())
                                    }
                                }
                            } => {
                                self.active.store(false, Ordering::SeqCst);
                                break;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
