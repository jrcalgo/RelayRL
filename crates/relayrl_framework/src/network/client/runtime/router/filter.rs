use super::{RoutedMessage, RouterError, RoutingProtocol};
use crate::network::client::runtime::coordination::scale_manager::RouterUuid;
use crate::network::client::runtime::coordination::state_manager::StateManager;

use burn_tensor::backend::Backend;
use relayrl_types::data::tensor::BackendMatcher;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc::Receiver;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

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
    associated_router_id: RouterUuid,
    rx_from_receiver: Receiver<RoutedMessage>,
    shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    shutdown: Option<broadcast::Receiver<()>>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    ClientCentralFilter<B, D_IN, D_OUT>
{
    pub(crate) fn new(
        associated_router_id: RouterUuid,
        rx_from_receiver: Receiver<RoutedMessage>,
        shared_agent_state: Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Self {
        Self {
            associated_router_id,
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
        let this_router_id: RouterUuid = self.associated_router_id;
        let shared_agent_state = self.shared_agent_state.clone();

        loop {
            tokio::select! {
                msg_opt = rx.recv() => {
                    match msg_opt {
                        Some(msg) => {
                            if let RoutingProtocol::Shutdown = msg.protocol {
                                Self::route_message(msg, &this_router_id, &shared_agent_state).await?;
                                break Ok(());
                            }
                            Self::route_message(msg, &this_router_id, &shared_agent_state).await?;
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
        router_id: &RouterUuid,
        shared_agent_state: &Arc<RwLock<StateManager<B, D_IN, D_OUT>>>,
    ) -> Result<(), RouterError> {
        let actor_id: Uuid = msg.actor_id;
        let shared_state = shared_agent_state.read().await;

        match shared_state.actor_router_addresses.get(&actor_id) {
            Some(assigned_router_id) if *assigned_router_id == *router_id => {
                match shared_state.actor_inboxes.get(&actor_id) {
                    Some(tx) => {
                        if let Err(e) = tx.send(msg).await {
                            return Err(RouterError::FilterError(FilterError::RoutingError(
                                format!("Cannot send message to actor: {}", e),
                            )));
                        }
                        Ok(())
                    }
                    None => Err(RouterError::FilterError(FilterError::RoutingError(
                        format!("Actor inbox not found: {}", actor_id),
                    ))),
                }
            }
            Some(other_router_id) => Err(RouterError::FilterError(FilterError::RoutingError(
                format!(
                    "Actor {} is assigned to router {:?}, but message is for router {}",
                    actor_id, other_router_id, router_id
                ),
            ))),
            None => Err(RouterError::FilterError(FilterError::RoutingError(
                format!(
                    "Actor {} is not assigned to any router or does not exist",
                    actor_id
                ),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {}
