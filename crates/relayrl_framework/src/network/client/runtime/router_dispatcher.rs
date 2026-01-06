use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::state_manager::ActorUuid;
use crate::network::client::runtime::coordination::state_manager::StateManager;
use crate::network::client::runtime::router::{RoutedMessage, RoutingProtocol};

use thiserror::Error;

use burn_tensor::backend::Backend;
use relayrl_types::types::data::tensor::BackendMatcher;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, MissedTickBehavior, interval};

#[derive(Debug, Error)]
pub enum RouterDispatcherError {
    #[error("Failed to dispatch message: {0}")]
    DispatchError(String),
    #[error("Router not found for actor: {0}")]
    RouterNotFoundError(String),
    #[error("Actor not assigned to any router: {0}")]
    ActorNotAssignedError(String),
}

struct PendingMessage {
    message: RoutedMessage,
    first_attempt: Instant,
    retry_count: u32,
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
    global_dispatcher_rx: Receiver<RoutedMessage>,
    router_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
    shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shutdown: Option<broadcast::Receiver<()>>,
    pending_messages: Arc<Mutex<HashMap<ActorUuid, PendingMessage>>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    RouterDispatcher<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        global_dispatcher_rx: Receiver<RoutedMessage>,
        router_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Self {
        Self {
            global_dispatcher_rx,
            router_channels,
            shared_state,
            shutdown: None,
            pending_messages: Arc::new(Mutex::new(HashMap::new())),
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
    /// 3. Queues messages for unassigned actors and retries them with exponential backoff
    /// 4. Spawns a background task to retry pending messages
    pub(crate) async fn spawn_loop(mut self) -> Result<(), RouterDispatcherError> {
        let mut shutdown = self.shutdown.take();

        // Spawn background retry task
        let pending_messages = self.pending_messages.clone();
        let router_channels = self.router_channels.clone();
        let shared_state = self.shared_state.clone();
        let retry_handle = tokio::spawn(async move {
            Self::retry_pending_messages_loop(pending_messages, router_channels, shared_state)
                .await;
        });

        loop {
            tokio::select! {
                msg_opt = self.global_dispatcher_rx.recv() => {
                    match msg_opt {
                        Some(msg) => {
                            if let Err(e) = self.dispatch_message(msg).await {
                                // Log errors but continue processing
                                match e {
                                    RouterDispatcherError::ActorNotAssignedError(error_message) => {
                                        eprintln!("[RouterDispatcher] {}. Message queued for retry.", error_message);
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
                            retry_handle.abort();
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
                    retry_handle.abort();
                    break Ok(());
                }
            }
        }
    }

    fn get_timeout_for_message_protocol(protocol: &RoutingProtocol) -> Duration {
        match protocol {
            RoutingProtocol::RequestInference => Duration::from_secs(10),
            RoutingProtocol::ModelVersion => Duration::from_secs(15),
            RoutingProtocol::FlagLastInference => Duration::from_secs(20),
            RoutingProtocol::ModelHandshake | RoutingProtocol::SendTrajectory => {
                Duration::from_secs(30)
            }

            RoutingProtocol::ModelUpdate | RoutingProtocol::Shutdown => Duration::from_secs(60),
            _ => Duration::from_secs(30),
        }
    }

    /// Background task that periodically retries pending messages
    async fn retry_pending_messages_loop(
        pending_messages: Arc<Mutex<HashMap<ActorUuid, PendingMessage>>>,
        router_channels: Arc<DashMap<RouterUuid, Sender<RoutedMessage>>>,
        shared_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) {
        const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(100);
        const MAX_RETRY_DELAY: Duration = Duration::from_millis(800);

        let mut interval = tokio::time::interval(Duration::from_millis(50));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            // Batch operation: Lock once, process all actors, then release
            let retry_messages = {
                let mut pending_map = pending_messages.lock().await;

                let mut to_remove: Vec<ActorUuid> = Vec::new();
                let mut to_retry: Vec<(ActorUuid, Instant, u32)> = Vec::new();

                // Process all actors in a single lock
                for (actor_id, pending) in pending_map.iter_mut() {
                    let elapsed = pending.first_attempt.elapsed();

                    let max_retry_duration =
                        Self::get_timeout_for_message_protocol(&pending.message.protocol);

                    // Check if expired
                    if elapsed > max_retry_duration {
                        eprintln!(
                            "[RouterDispatcher] Message for actor {} expired after {}ms (retry count: {})",
                            actor_id,
                            elapsed.as_millis(),
                            pending.retry_count
                        );
                        to_remove.push(*actor_id);
                        continue;
                    }

                    // Calculate exponential backoff delay
                    let retry_delay = INITIAL_RETRY_DELAY
                        .as_millis()
                        .saturating_mul(1 << pending.retry_count.min(3)) // Cap at 800ms
                        .min(MAX_RETRY_DELAY.as_millis());

                    // Check if ready to retry
                    let time_since_first = elapsed.as_millis();
                    let expected_retry_time = retry_delay * (pending.retry_count + 1) as u128;

                    if time_since_first >= expected_retry_time {
                        // Ready to retry - save metadata
                        to_retry.push((*actor_id, pending.first_attempt, pending.retry_count));
                    }
                    // If not ready, leave it in the map (no action needed)
                }

                // Remove expired messages while we still have the lock
                for actor_id in &to_remove {
                    pending_map.remove(actor_id);
                }

                // Extract messages ready to retry (move them out of the map)
                let mut retry_messages = Vec::new();
                for (actor_id, first_attempt, retry_count) in &to_retry {
                    if let Some(pending_msg) = pending_map.remove(actor_id) {
                        retry_messages.push((*actor_id, pending_msg, *first_attempt, *retry_count));
                    }
                }

                // Release lock before async operations
                drop(pending_map);

                retry_messages
            };

            // Now process retries without holding the lock
            for (actor_id, pending_msg, first_attempt, retry_count) in retry_messages {
                // Check router assignment (async operation, no lock needed)
                let router_id = {
                    let state = shared_state.read().await;
                    state
                        .actor_router_addresses
                        .get(&actor_id)
                        .map(|entry| *entry.value())
                };

                match router_id {
                    Some(router_id) => {
                        match router_channels.get(&router_id) {
                            Some(tx) => {
                                // Try to send (no lock needed)
                                match tx.try_send(pending_msg.message) {
                                    Ok(()) => {
                                        println!(
                                            "[RouterDispatcher] Successfully dispatched queued message for actor {} after {} retries",
                                            actor_id, retry_count
                                        );
                                        // Message successfully sent, don't add back to queue
                                    }
                                    Err(tokio::sync::mpsc::error::TrySendError::Full(msg)) => {
                                        // Channel full, put back in queue (need lock for write)
                                        let mut pending_map = pending_messages.lock().await;
                                        pending_map.insert(
                                            actor_id,
                                            PendingMessage {
                                                message: msg,
                                                first_attempt,
                                                retry_count,
                                            },
                                        );
                                    }
                                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                        eprintln!(
                                            "[RouterDispatcher] Router channel closed for actor {}, removing from retry queue",
                                            actor_id
                                        );
                                        // Channel closed, message already removed from queue
                                    }
                                }
                            }
                            None => {
                                // Router not found, put back in queue with incremented retry count
                                let mut pending_map = pending_messages.lock().await;
                                pending_map.insert(
                                    actor_id,
                                    PendingMessage {
                                        message: pending_msg.message,
                                        first_attempt,
                                        retry_count: retry_count + 1,
                                    },
                                );
                            }
                        }
                    }
                    None => {
                        // Actor still not assigned, put back in queue with incremented retry count
                        let mut pending_map = pending_messages.lock().await;
                        pending_map.insert(
                            actor_id,
                            PendingMessage {
                                message: pending_msg.message,
                                first_attempt,
                                retry_count: retry_count + 1,
                            },
                        );
                    }
                }
            }
        }
    }
    /// Dispatch a single message to the appropriate router
    ///
    /// Messages for unassigned actors are queued for retry instead of being dropped.
    async fn dispatch_message(&mut self, msg: RoutedMessage) -> Result<(), RouterDispatcherError> {
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
            Some(router_id) => match self.router_channels.get(&router_id) {
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
            },
            None => {
                // Actor not assigned to any router yet - queue for retry
                let mut pending_map = self.pending_messages.lock().await;
                pending_map.insert(
                    actor_id,
                    PendingMessage {
                        message: msg,
                        first_attempt: Instant::now(),
                        retry_count: 0,
                    },
                );
                Err(RouterDispatcherError::ActorNotAssignedError(format!(
                    "Actor {} not assigned to any router (message queued for retry)",
                    actor_id
                )))
            }
        }
    }
}
