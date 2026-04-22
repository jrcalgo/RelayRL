use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
use crate::network::client::runtime::coordination::state_manager::{ActorUuid, SharedRouterState};
use crate::network::client::runtime::router::{RoutedMessage, RoutingProtocol};
#[cfg(feature = "metrics")]
use crate::utilities::observability::metrics::MetricsManager;

use thiserror::Error;

use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::time::Duration;

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
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
pub(crate) struct RouterDispatcher {
    global_dispatcher_rx: Receiver<RoutedMessage>,
    router_channels: Arc<DashMap<RouterNamespace, Sender<RoutedMessage>>>,
    shared_router_state: Arc<SharedRouterState>,
    shutdown: Option<broadcast::Receiver<()>>,
    pending_messages: Arc<DashMap<ActorUuid, PendingMessage>>,
    #[cfg(feature = "metrics")]
    metrics: MetricsManager,
}

impl RouterDispatcher {
    pub(crate) async fn new(
        global_dispatcher_rx: Receiver<RoutedMessage>,
        router_channels: Arc<DashMap<RouterNamespace, Sender<RoutedMessage>>>,
        shared_router_state: Arc<SharedRouterState>,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
    ) -> Self {
        Self {
            global_dispatcher_rx,
            router_channels,
            shared_router_state,
            shutdown: None,
            pending_messages: Arc::new(DashMap::<ActorUuid, PendingMessage>::new()),
            #[cfg(feature = "metrics")]
            metrics,
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

        // Spawn background retry task;
        let pending_messages = self.pending_messages.clone();
        let router_channels = self.router_channels.clone();
        let shared_router_state = self.shared_router_state.clone();
        #[cfg(feature = "metrics")]
        let metrics = self.metrics.clone();
        let retry_handle = tokio::spawn(async move {
            Self::retry_pending_messages_loop(
                pending_messages,
                router_channels,
                shared_router_state,
                #[cfg(feature = "metrics")]
                metrics,
            )
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
                                        log::error!("[RouterDispatcher] {}. Message queued for retry.", error_message);
                                    }
                                    _ => {
                                        log::error!("[RouterDispatcher] Dispatch error: {}", e);
                                    }
                                }
                            }
                        }
                        None => {
                            // Channel closed, exit loop
                            log::warn!("[RouterDispatcher] Global channel closed, shutting down");
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
                    log::info!("[RouterDispatcher] Shutdown signal received");
                    retry_handle.abort();
                    break Ok(());
                }
            }
        }
    }

    fn get_timeout_for_message_protocol(protocol: &RoutingProtocol) -> Duration {
        match protocol {
            RoutingProtocol::RequestInference | RoutingProtocol::RequestInferenceBatch => {
                Duration::from_secs(10)
            }
            RoutingProtocol::ModelVersion => Duration::from_secs(15),
            RoutingProtocol::FlagLastInference => Duration::from_secs(20),
            RoutingProtocol::ModelHandshake | RoutingProtocol::SendTrajectory => {
                Duration::from_secs(30)
            }

            RoutingProtocol::ModelUpdate | RoutingProtocol::Shutdown => Duration::from_secs(60),
        }
    }

    /// Background task that periodically retries pending messages
    async fn retry_pending_messages_loop(
        pending_messages: Arc<DashMap<ActorUuid, PendingMessage>>,
        router_channels: Arc<DashMap<RouterNamespace, Sender<RoutedMessage>>>,
        shared_router_state: Arc<SharedRouterState>,
        #[cfg(feature = "metrics")] metrics: MetricsManager,
    ) {
        const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(100);
        const MAX_RETRY_DELAY: Duration = Duration::from_millis(800);

        let mut interval = tokio::time::interval(Duration::from_millis(50));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            // Batch operation: Lock once, process all actors, then release
            let retry_results = {
                let mut to_remove: Vec<ActorUuid> = Vec::new();
                let mut to_retry: Vec<(ActorUuid, Instant, u32)> = Vec::new();

                // Process all actors in a single lock
                for entry in pending_messages.iter() {
                    let (actor_id, pending) = entry.pair();
                    let elapsed = pending.first_attempt.elapsed();

                    let max_retry_duration =
                        Self::get_timeout_for_message_protocol(&pending.message.protocol);

                    // Check if expired
                    if elapsed > max_retry_duration {
                        log::warn!(
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
                    pending_messages.remove(actor_id);
                }

                // Extract messages ready to retry (move them out of the map)
                let mut retry_messages = Vec::new();
                for (actor_id, first_attempt, retry_count) in &to_retry {
                    if let Some(pending_msg) = pending_messages.remove(actor_id) {
                        retry_messages.push((
                            *actor_id,
                            pending_msg.1,
                            *first_attempt,
                            *retry_count,
                        ));
                    }
                }

                #[cfg(feature = "metrics")]
                {
                    (retry_messages, to_remove.len() as u64)
                }
                #[cfg(not(feature = "metrics"))]
                {
                    retry_messages
                }
            };

            #[cfg(feature = "metrics")]
            let (retry_messages, expired_count) = retry_results;
            #[cfg(not(feature = "metrics"))]
            let retry_messages = retry_results;

            #[cfg(feature = "metrics")]
            if expired_count > 0 {
                metrics
                    .record_counter("router_messages_expired", expired_count, &[])
                    .await;
            }

            // Now process retries without holding the lock
            for (actor_id, pending_msg, first_attempt, retry_count) in retry_messages {
                // Check router assignment (async operation, no lock needed)
                let router_namespace = {
                    shared_router_state
                        .actor_router_addresses
                        .get(&actor_id)
                        .map(|entry| entry.value().clone())
                };

                match router_namespace {
                    Some(router_namespace) => {
                        match router_channels.get(&router_namespace) {
                            Some(tx) => {
                                // Try to send (no lock needed)
                                match tx.try_send(pending_msg.message) {
                                    Ok(()) => {
                                        log::info!(
                                            "[RouterDispatcher] Successfully dispatched queued message for actor {} after {} retries",
                                            actor_id,
                                            retry_count
                                        );
                                        #[cfg(feature = "metrics")]
                                        metrics
                                            .record_counter("router_messages_dispatched", 1, &[])
                                            .await;
                                        // Message successfully sent, don't add back to queue
                                    }
                                    Err(tokio::sync::mpsc::error::TrySendError::Full(msg)) => {
                                        // Channel full, put back in queue (need lock for write)
                                        pending_messages.insert(
                                            actor_id,
                                            PendingMessage {
                                                message: msg,
                                                first_attempt,
                                                retry_count,
                                            },
                                        );
                                    }
                                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                        log::warn!(
                                            "[RouterDispatcher] Router channel closed for actor {}, removing from retry queue",
                                            actor_id
                                        );
                                        // Channel closed, message already removed from queue
                                    }
                                }
                            }
                            None => {
                                // Router not found, put back in queue with incremented retry count
                                pending_messages.insert(
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
                        pending_messages.insert(
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
        #[cfg(feature = "metrics")]
        let start_time = Instant::now();
        let actor_id = msg.actor_id;

        // Look up which router this actor is assigned to
        let router_namespace = {
            self.shared_router_state
                .actor_router_addresses
                .get(&actor_id)
                .map(|entry| entry.value().clone())
        };

        match router_namespace {
            Some(router_namespace) => match self.router_channels.get(&router_namespace) {
                Some(tx) => {
                    tx.send(msg).await.map_err(|e| {
                        RouterDispatcherError::DispatchError(format!(
                            "Failed to send message to router {}: {}",
                            router_namespace, e
                        ))
                    })?;
                    #[cfg(feature = "metrics")]
                    {
                        let duration = start_time.elapsed().as_secs_f64();
                        self.metrics
                            .record_histogram("router_dispatch_latency", duration, &[])
                            .await;
                        self.metrics
                            .record_counter("router_messages_dispatched", 1, &[])
                            .await;
                    }
                    Ok(())
                }
                None => Err(RouterDispatcherError::RouterNotFoundError(format!(
                    "Router {} not found for actor {}",
                    router_namespace, actor_id
                ))),
            },
            None => {
                // Actor not assigned to any router yet - queue for retry
                {
                    self.pending_messages.insert(
                        actor_id,
                        PendingMessage {
                            message: msg,
                            first_attempt: Instant::now(),
                            retry_count: 0,
                        },
                    );
                }
                #[cfg(feature = "metrics")]
                self.metrics
                    .record_counter("router_messages_queued", 1, &[])
                    .await;
                Err(RouterDispatcherError::ActorNotAssignedError(format!(
                    "Actor {} not assigned to any router (message queued for retry)",
                    actor_id
                )))
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::network::client::agent::{
        ActorInferenceMode, ActorTrainingDataMode, ClientModes, ModelMode,
    };
    use crate::network::client::runtime::coordination::state_manager::StateManager;
    use crate::network::client::runtime::router::{RoutedPayload, RoutingProtocol};
    #[cfg(feature = "metrics")]
    use crate::utilities::observability::metrics::MetricsManager;
    use active_uuid_registry::registry_uuid::Uuid;
    use burn_ndarray::NdArray;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::{RwLock, broadcast, mpsc};

    type TestBackend = NdArray<f32>;
    const D_IN: usize = 4;
    const D_OUT: usize = 1;

    fn disabled_modes() -> Arc<ClientModes> {
        Arc::new(ClientModes {
            actor_inference_mode: ActorInferenceMode::Local(ModelMode::Independent),
            actor_training_data_mode: ActorTrainingDataMode::Disabled,
        })
    }

    fn make_state_manager() -> (
        StateManager<TestBackend, D_IN, D_OUT>,
        mpsc::Receiver<RoutedMessage>,
    ) {
        StateManager::<TestBackend, D_IN, D_OUT>::new(
            Arc::from("test-dispatcher"),
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
            None,
            disabled_modes(),
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
        MetricsManager::new(
            Arc::new(RwLock::new((
                "test-router-dispatcher".to_string(),
                String::new(),
            ))),
            ("test-router-dispatcher".to_string(), String::new()),
            None,
        )
    }

    fn make_routed_message(actor_id: Uuid, protocol: RoutingProtocol) -> RoutedMessage {
        RoutedMessage {
            actor_id,
            protocol,
            payload: RoutedPayload::ModelHandshake,
        }
    }

    /// Build a RouterDispatcher with a pre-wired state manager and router channel map.
    /// Returns: (dispatcher, global_tx, router_channels, shared_state)
    async fn make_dispatcher() -> (
        RouterDispatcher,
        mpsc::Sender<RoutedMessage>,
        Arc<DashMap<RouterNamespace, mpsc::Sender<RoutedMessage>>>,
        Arc<SharedRouterState>,
    ) {
        let (sm, _state_global_rx) = make_state_manager();
        let shared_router_state = sm.shared_router_state.clone();
        let router_channels: Arc<DashMap<RouterNamespace, mpsc::Sender<RoutedMessage>>> =
            Arc::new(DashMap::new());
        // The dispatcher reads from its own channel (not the StateManager's global_rx)
        let (global_tx, global_rx) = mpsc::channel::<RoutedMessage>(32);
        let dispatcher = RouterDispatcher::new(
            global_rx,
            router_channels.clone(),
            shared_router_state.clone(),
            #[cfg(feature = "metrics")]
            test_metrics(),
        )
        .await;
        (dispatcher, global_tx, router_channels, shared_router_state)
    }

    #[test]
    fn get_timeout_for_protocol_correct_values() {
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::RequestInference),
            Duration::from_secs(10)
        );
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::ModelVersion),
            Duration::from_secs(15)
        );
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::FlagLastInference),
            Duration::from_secs(20)
        );
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::ModelHandshake),
            Duration::from_secs(30)
        );
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::SendTrajectory),
            Duration::from_secs(30)
        );
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::ModelUpdate),
            Duration::from_secs(60)
        );
        assert_eq!(
            RouterDispatcher::get_timeout_for_message_protocol(&RoutingProtocol::Shutdown),
            Duration::from_secs(60)
        );
    }

    #[tokio::test]
    async fn dispatches_to_assigned_router() {
        let (dispatcher, global_tx, router_channels, shared_router_state) = make_dispatcher().await;

        let actor_id = Uuid::new_v4();
        let ns: RouterNamespace = Arc::from("router-dispatch-test");

        // Register actor → namespace in state
        shared_router_state
            .actor_router_addresses
            .insert(actor_id, ns.clone());

        // Create router channel and register it
        let (router_tx, mut router_rx) = mpsc::channel::<RoutedMessage>(4);
        router_channels.insert(ns, router_tx);

        // Shutdown after one message to make the test deterministic
        let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
        let dispatcher = dispatcher.with_shutdown(shutdown_rx);

        let _handle = tokio::spawn(async move { dispatcher.spawn_loop().await });

        global_tx
            .send(make_routed_message(
                actor_id,
                RoutingProtocol::ModelHandshake,
            ))
            .await
            .unwrap();

        let received =
            tokio::time::timeout(tokio::time::Duration::from_millis(300), router_rx.recv())
                .await
                .expect("timeout waiting for router to receive message")
                .expect("router rx closed");

        assert_eq!(received.actor_id, actor_id);
        shutdown_tx.send(()).ok();
    }

    #[tokio::test]
    async fn queues_message_for_unassigned_actor() {
        let (mut dispatcher, _tx, _router_channels, _shared_router_state) = make_dispatcher().await;

        let actor_id = Uuid::new_v4();
        // Actor has no router assignment → dispatch_message should queue it
        let msg = make_routed_message(actor_id, RoutingProtocol::ModelHandshake);
        let result = dispatcher.dispatch_message(msg).await;
        assert!(
            matches!(result, Err(RouterDispatcherError::ActorNotAssignedError(_))),
            "Expected ActorNotAssignedError, got {:?}",
            result
        );

        // Verify it ended up in pending_messages
        assert!(
            dispatcher.pending_messages.contains_key(&actor_id),
            "Message should be queued in pending_messages"
        );
    }

    #[tokio::test]
    async fn retries_deliver_message_after_assignment() {
        let (dispatcher, global_tx, router_channels, shared_router_state) = make_dispatcher().await;

        let actor_id = Uuid::new_v4();
        let ns: RouterNamespace = Arc::from("retry-ns");

        // No router assignment yet — send message
        // Create router channel but don't register the router yet
        let (router_tx, mut router_rx) = mpsc::channel::<RoutedMessage>(4);

        let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
        let dispatcher = dispatcher.with_shutdown(shutdown_rx);

        let _handle = tokio::spawn(async move { dispatcher.spawn_loop().await });

        // Send message before actor is assigned → queued
        global_tx
            .send(make_routed_message(
                actor_id,
                RoutingProtocol::ModelHandshake,
            ))
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;

        // Now assign actor to a router
        shared_router_state
            .actor_router_addresses
            .insert(actor_id, ns.clone());
        router_channels.insert(ns, router_tx);

        // Wait for retry loop to deliver (up to 500ms)
        let received =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), router_rx.recv())
                .await
                .expect("timeout: retry did not deliver message")
                .expect("router rx closed");

        assert_eq!(received.actor_id, actor_id);
        shutdown_tx.send(()).ok();
    }

    #[tokio::test]
    async fn dispatcher_exits_on_broadcast_signal() {
        let (dispatcher, _global_tx, _router_channels, _shared_router_state) =
            make_dispatcher().await;
        let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
        let dispatcher = dispatcher.with_shutdown(shutdown_rx);

        let handle = tokio::spawn(async move { dispatcher.spawn_loop().await });

        shutdown_tx.send(()).unwrap();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("dispatcher did not exit in time")
            .expect("join error");

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn dispatcher_exits_on_channel_close() {
        let (dispatcher, global_tx, _router_channels, _shared_router_state) =
            make_dispatcher().await;
        let handle = tokio::spawn(async move { dispatcher.spawn_loop().await });

        drop(global_tx); // closed channel → dispatcher sees None → exits

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("dispatcher did not exit in time")
            .expect("join error");

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn closed_router_channel_does_not_panic() {
        let (dispatcher, global_tx, router_channels, shared_router_state) = make_dispatcher().await;
        let actor_id = Uuid::new_v4();
        let ns: RouterNamespace = Arc::from("closed-router-ns");

        shared_router_state
            .actor_router_addresses
            .insert(actor_id, ns.clone());

        // Insert a router channel, then immediately drop the rx side
        let (router_tx, router_rx) = mpsc::channel::<RoutedMessage>(4);
        router_channels.insert(ns, router_tx);
        drop(router_rx);

        let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
        let dispatcher = dispatcher.with_shutdown(shutdown_rx);
        let handle = tokio::spawn(async move { dispatcher.spawn_loop().await });

        // This should log an error but not panic
        global_tx
            .send(make_routed_message(
                actor_id,
                RoutingProtocol::ModelHandshake,
            ))
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        shutdown_tx.send(()).ok();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("timeout")
            .expect("join error");

        assert!(
            result.is_ok(),
            "Dispatcher should not panic on closed router channel"
        );
    }
}
