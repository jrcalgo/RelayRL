use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::RoutedMessage;

use thiserror::Error;

use burn_tensor::backend::Backend;
use relayrl_types::types::data::tensor::BackendMatcher;

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::broadcast;
use tokio::sync::mpsc::{Receiver, Sender};

#[derive(Debug, Error)]
pub enum RouterDispatcherError {
    #[error("Failed to dispatch message: {0}")]
    DispatchError(String),
    #[error("Router not found for actor: {0}")]
    RouterNotFoundError(String),
    #[error("Actor not assigned to any router: {0}")]
    ActorNotAssignedError(String),
}

/// Central dispatcher that routes messages from external sources (ExternalReceivers, Coordinator)
/// to the appropriate router's filter based on actor-router assignments.
///
/// # Message Handling
///
/// Messages for actors not yet assigned to a router are dropped with a warning log.
/// This can occur during:
/// - Initial actor startup before router assignment
/// - Scaling operations where actors are being reassigned
/// - Race conditions between actor creation and router assignment
///
/// Callers should ensure actors are assigned to routers before sending messages to them.
pub(crate) struct RouterDispatcher<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    global_rx: Receiver<RoutedMessage>,
    router_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RouterDispatcher<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        global_rx: Receiver<RoutedMessage>,
        router_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Self {
        Self {
            global_rx,
            router_channels,
            shared_state,
            shutdown: None,
        }
    }

    pub(crate) fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    /// Main dispatch loop - reads from global channel and routes to appropriate router
    ///
    /// This loop:
    /// 1. Receives new messages from the global channel
    /// 2. Attempts to dispatch them immediately to the appropriate router
    /// 3. Drops messages for unassigned actors with a warning
    pub(crate) async fn spawn_loop(mut self) -> Result<(), RouterDispatcherError> {
        let mut shutdown = self.shutdown.take();

        loop {
            tokio::select! {
                msg_opt = self.global_rx.recv() => {
                    match msg_opt {
                        Some(msg) => {
                            if let Err(e) = self.dispatch_message(msg).await {
                                // Log errors but continue processing
                                match e {
                                    RouterDispatcherError::ActorNotAssignedError(ref msg) => {
                                        eprintln!("[RouterDispatcher] {}", msg);
                                    }
                                    _ => {
                                        eprintln!("[RouterDispatcher] Dispatch error: {}", e);
                                    }
                                }
                            }
                        }
                        None => {
                            // Channel closed, exit loop
                            println!("[RouterDispatcher] Global channel closed, shutting down");
                            break Ok(());
                        }
                    }
                }
                _ = async {
                    match &mut shutdown {
                        Some(rx) => { let _ = rx.recv().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    println!("[RouterDispatcher] Shutdown signal received");
                    break Ok(());
                }
            }
        }
    }

    /// Dispatch a single message to the appropriate router
    ///
    /// Messages for unassigned actors are dropped with an error.
    async fn dispatch_message(&self, msg: RoutedMessage) -> Result<(), RouterDispatcherError> {
        let actor_id = msg.actor_id;

        // Look up which router this actor is assigned to
        let router_id = {
            let state = self.shared_state.read().await;
            state
                .actor_router_addresses
                .get(&actor_id)
                .map(|entry| *entry.value())
        };

        match router_id {
            Some(router_id) => {
                match self.router_channels.get(&router_id) {
                    Some(tx) => {
                        tx.send(msg).await.map_err(|e| {
                            RouterDispatcherError::DispatchError(format!(
                                "Failed to send message to router {}: {}",
                                router_id, e
                            ))
                        })?;
                        Ok(())
                    }
                    None => Err(RouterDispatcherError::RouterNotFoundError(format!(
                        "Router {} not found for actor {}",
                        router_id, actor_id
                    ))),
                }
            }
            None => {
                // Actor not assigned to any router yet - drop the message
                // This commonly happens during:
                // - Initial actor startup before router assignment
                // - Scaling operations where actors are being reassigned
                Err(RouterDispatcherError::ActorNotAssignedError(format!(
                    "Actor {} not assigned to any router (message dropped)",
                    actor_id
                )))
            }
        }
    }
}
