use super::{RoutedMessage, RouterError, RoutingProtocol};
use crate::network::client::runtime::coordination::scale_manager::RouterNamespace;
use crate::network::client::runtime::coordination::state_manager::{
    ActorRoute, SharedRouterState, StateManager,
};

use active_uuid_registry::registry_uuid::Uuid;

use burn_tensor::backend::Backend;
use relayrl_types::data::tensor::BackendMatcher;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{RwLock, broadcast};

#[derive(Debug, Error)]
pub enum FilterError {
    #[error("Filter routing error: {0}")]
    RoutingError(String),
}

/// Intermediary routing process/filter for routing received models and requests to specified ActorEntity
pub(crate) struct ClientCentralFilter<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    associated_router_namespace: RouterNamespace,
    rx_from_receiver: Receiver<RoutedMessage>,
    shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientCentralFilter<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        associated_router_namespace: RouterNamespace,
        rx_from_receiver: Receiver<RoutedMessage>,
        shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Self {
        Self {
            associated_router_namespace,
            rx_from_receiver,
            shared_agent_state,
            shutdown: None,
        }
    }

    pub(crate) fn with_shutdown(mut self, rx: broadcast::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    pub(crate) async fn spawn_loop(mut self) -> Result<(), RouterError> {
        let mut shutdown: Option<broadcast::Receiver<()>> = self.shutdown.take();
        let mut rx: Receiver<RoutedMessage> = self.rx_from_receiver;
        let this_router_namespace: RouterNamespace = self.associated_router_namespace.clone();
        let shared_router_state: Arc<SharedRouterState> = self
            .shared_agent_state
            .read()
            .await
            .shared_router_state
            .clone();

        loop {
            tokio::select! {
                msg_opt = rx.recv() => {
                    match msg_opt {
                        Some(msg) => {
                            if let RoutingProtocol::Shutdown = msg.protocol {
                                Self::route_message(msg, &this_router_namespace, &shared_router_state).await?;
                                break Ok(());
                            }
                            Self::route_message(msg, &this_router_namespace, &shared_router_state).await?;
                        }
                        None => break Ok(()),
                    }
                }
                _ = async {
                    match &mut shutdown {
                        Some(rx) => { let _ = rx.recv().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    break Ok(());
                }
            }
        }
    }

    async fn route_message(
        msg: RoutedMessage,
        router_namespace: &RouterNamespace,
        shared_router_state: &Arc<SharedRouterState>,
    ) -> Result<(), RouterError> {
        let actor_id: Uuid = msg.actor_id;

        let route = shared_router_state
            .actor_routes
            .get(&actor_id)
            .map(|entry| entry.value().clone());

        match route {
            Some(ActorRoute {
                router_namespace: Some(assigned_router_namespace),
                inbox,
            }) if assigned_router_namespace == *router_namespace => {
                if let Err(e) = inbox.send(msg).await {
                    return Err(RouterError::FilterError(FilterError::RoutingError(
                        format!("Cannot send message to actor: {}", e),
                    )));
                }

                Ok(())
            }
            Some(ActorRoute {
                router_namespace: Some(other_router_namespace),
                ..
            }) => Err(RouterError::FilterError(FilterError::RoutingError(
                format!(
                    "Actor {} is assigned to router {:?}, but message is for router {}",
                    actor_id, other_router_namespace, router_namespace
                ),
            ))),
            Some(ActorRoute {
                router_namespace: None,
                ..
            })
            | None => Err(RouterError::FilterError(FilterError::RoutingError(
                format!(
                    "Actor {} is not assigned to any router or does not exist",
                    actor_id
                ),
            ))),
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
    use crate::network::client::runtime::router::{RoutedMessage, RoutedPayload, RoutingProtocol};
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

    /// Create a minimal StateManager (no transport, no model).
    fn make_state_manager() -> (
        StateManager<TestBackend, D_IN, D_OUT>,
        tokio::sync::mpsc::Receiver<RoutedMessage>,
    ) {
        StateManager::<TestBackend, D_IN, D_OUT>::new(
            Arc::from("test-filter-ns"),
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
            Arc::new(RwLock::new(("test-filter".to_string(), String::new()))),
            ("test-filter".to_string(), String::new()),
            None,
        )
    }

    /// Build a shared StateManager with one actor pre-registered in the given namespace.
    /// Returns (shared_state, actor_id, actor_inbox_rx).
    fn make_shared_state_with_actor(
        namespace: RouterNamespace,
    ) -> (
        Arc<RwLock<StateManager<TestBackend, D_IN, D_OUT>>>,
        Uuid,
        mpsc::Receiver<RoutedMessage>,
    ) {
        let (sm, _global_rx) = make_state_manager();
        let actor_id = Uuid::new_v4();
        let (actor_tx, actor_rx) = mpsc::channel::<RoutedMessage>(16);
        sm.shared_router_state.actor_routes.insert(
            actor_id,
            ActorRoute {
                router_namespace: Some(namespace),
                inbox: actor_tx,
            },
        );
        (Arc::new(RwLock::new(sm)), actor_id, actor_rx)
    }

    fn make_msg(
        actor_id: Uuid,
        protocol: RoutingProtocol,
        payload: RoutedPayload,
    ) -> RoutedMessage {
        RoutedMessage {
            actor_id,
            protocol,
            payload,
        }
    }

    fn make_filter(
        ns: RouterNamespace,
        rx: mpsc::Receiver<RoutedMessage>,
        shared: Arc<RwLock<StateManager<TestBackend, D_IN, D_OUT>>>,
    ) -> ClientCentralFilter<TestBackend, D_IN, D_OUT> {
        ClientCentralFilter::<TestBackend, D_IN, D_OUT>::new(ns, rx, shared)
    }

    #[tokio::test]
    async fn routes_message_to_correct_actor_inbox() {
        let ns: RouterNamespace = Arc::from("router-1");
        let (shared, actor_id, mut actor_rx) = make_shared_state_with_actor(ns.clone());

        let (filter_tx, filter_rx) = mpsc::channel::<RoutedMessage>(16);
        let (_shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1000);
        let filter = make_filter(ns.clone(), filter_rx, shared).with_shutdown(shutdown_rx);

        let _handle = tokio::spawn(async move {
            let _ = filter.spawn_loop().await;
        });

        filter_tx
            .send(make_msg(
                actor_id,
                RoutingProtocol::ModelHandshake,
                RoutedPayload::ModelHandshake,
            ))
            .await
            .unwrap();

        let received = tokio::time::timeout(tokio::time::Duration::from_secs(30), actor_rx.recv())
            .await
            .expect("timeout waiting for actor message")
            .expect("actor rx closed");

        assert!(matches!(received.protocol, RoutingProtocol::ModelHandshake));
    }

    #[tokio::test]
    async fn error_if_actor_assigned_to_different_namespace() {
        let ns_a: RouterNamespace = Arc::from("ns-a-wrong");
        let ns_b: RouterNamespace = Arc::from("ns-b-wrong");
        let (shared, actor_id, _actor_rx) = make_shared_state_with_actor(ns_a.clone());

        let (filter_tx, filter_rx) = mpsc::channel::<RoutedMessage>(4);
        // Filter uses namespace B, actor is assigned to A, error expected
        let filter = make_filter(ns_b, filter_rx, shared);
        let handle = tokio::spawn(async move { filter.spawn_loop().await });

        filter_tx
            .send(make_msg(
                actor_id,
                RoutingProtocol::ModelHandshake,
                RoutedPayload::ModelHandshake,
            ))
            .await
            .unwrap();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("timeout")
            .expect("join error");

        assert!(result.is_err(), "Filter should error on wrong namespace");
    }

    #[tokio::test]
    async fn error_if_actor_not_registered() {
        let ns: RouterNamespace = Arc::from("ns-unreg");
        let (shared, _actor_id, _actor_rx) = make_shared_state_with_actor(ns.clone());

        let unknown = Uuid::new_v4();
        let (filter_tx, filter_rx) = mpsc::channel::<RoutedMessage>(4);
        let filter = make_filter(ns, filter_rx, shared);
        let handle = tokio::spawn(async move { filter.spawn_loop().await });

        filter_tx
            .send(make_msg(
                unknown,
                RoutingProtocol::ModelHandshake,
                RoutedPayload::ModelHandshake,
            ))
            .await
            .unwrap();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("timeout")
            .expect("join error");

        assert!(result.is_err(), "Filter should error on unknown actor");
    }

    #[tokio::test]
    async fn shuts_down_on_broadcast_signal() {
        let ns: RouterNamespace = Arc::from("ns-shutdown-sig");
        let (shared, _actor_id, _actor_rx) = make_shared_state_with_actor(ns.clone());

        let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
        let (_, filter_rx) = mpsc::channel::<RoutedMessage>(4);
        let filter = make_filter(ns, filter_rx, shared).with_shutdown(shutdown_rx);
        let handle = tokio::spawn(async move { filter.spawn_loop().await });

        shutdown_tx.send(()).unwrap();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("filter did not exit in time")
            .expect("join error");

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn shuts_down_when_input_channel_closed() {
        let ns: RouterNamespace = Arc::from("ns-chan-closed");
        let (shared, _actor_id, _actor_rx) = make_shared_state_with_actor(ns.clone());

        let (filter_tx, filter_rx) = mpsc::channel::<RoutedMessage>(4);
        let filter = make_filter(ns, filter_rx, shared);
        let handle = tokio::spawn(async move { filter.spawn_loop().await });

        drop(filter_tx); // closed channel, filter receives None, exits

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("filter did not exit in time")
            .expect("join error");

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn shutdown_message_routed_then_loop_exits() {
        let ns: RouterNamespace = Arc::from("ns-shutdown-msg2");
        let (shared, actor_id, mut actor_rx) = make_shared_state_with_actor(ns.clone());

        let (filter_tx, filter_rx) = mpsc::channel::<RoutedMessage>(4);
        let filter = make_filter(ns, filter_rx, shared);
        let handle = tokio::spawn(async move { filter.spawn_loop().await });

        filter_tx
            .send(make_msg(
                actor_id,
                RoutingProtocol::Shutdown,
                RoutedPayload::Shutdown,
            ))
            .await
            .unwrap();

        let msg = tokio::time::timeout(tokio::time::Duration::from_millis(200), actor_rx.recv())
            .await
            .expect("timeout waiting for shutdown msg")
            .expect("actor rx closed");

        assert!(matches!(msg.protocol, RoutingProtocol::Shutdown));

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("filter did not exit after Shutdown msg")
            .expect("join error");

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn filter_error_on_actor_inbox_send_failure() {
        let ns: RouterNamespace = Arc::from("ns-inbox-fail2");
        let (shared, actor_id, actor_rx) = make_shared_state_with_actor(ns.clone());

        drop(actor_rx); // closing actor's rx makes tx.send fail

        let (filter_tx, filter_rx) = mpsc::channel::<RoutedMessage>(4);
        let filter = make_filter(ns, filter_rx, shared);
        let handle = tokio::spawn(async move { filter.spawn_loop().await });

        filter_tx
            .send(make_msg(
                actor_id,
                RoutingProtocol::ModelHandshake,
                RoutedPayload::ModelHandshake,
            ))
            .await
            .unwrap();

        let result = tokio::time::timeout(tokio::time::Duration::from_millis(300), handle)
            .await
            .expect("timeout")
            .expect("join error");

        assert!(
            result.is_err(),
            "Filter should error when actor inbox is closed"
        );
    }
}
