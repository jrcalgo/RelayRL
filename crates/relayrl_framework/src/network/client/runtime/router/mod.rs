use crate::network::client::runtime::router::buffer::TrajectorySinkError;
use crate::network::client::runtime::router::filter::FilterError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::router::receiver::TransportReceiverError;

use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::trajectory::RelayRLTrajectory;

use active_uuid_registry::registry_uuid::Uuid;

use std::any::Any;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::oneshot;

pub(crate) mod buffer;
pub(crate) mod filter;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) mod receiver;
pub(crate) mod router_dispatcher;

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum RouterError {
    #[error(transparent)]
    FilterError(#[from] FilterError),
    #[error(transparent)]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    TransportReceiverError(#[from] TransportReceiverError),
    #[error(transparent)]
    TrajectorySinkError(#[from] TrajectorySinkError),
}

pub(crate) struct RoutedMessage {
    pub actor_id: Uuid,
    pub protocol: RoutingProtocol,
    pub payload: RoutedPayload,
}

pub(crate) enum RoutingProtocol {
    ModelHandshake,
    RequestInference,
    FlagLastInference,
    ModelVersion,
    ModelUpdate,
    SendTrajectory,
    Shutdown,
}

pub(crate) enum RoutedPayload {
    ModelHandshake,
    RequestInference(Box<InferenceRequest>),
    FlagLastInference {
        reward: f32,
        env_id: Option<Uuid>,
        env_label: Option<String>,
    },
    ModelVersion {
        reply_to: oneshot::Sender<i64>,
    },
    ModelUpdate {
        model_bytes: Vec<u8>,
        version: i64,
    },
    SendTrajectory {
        timestamp: (u128, u128),
        trajectory: RelayRLTrajectory,
    },
    Shutdown,
}

/// observation and mask are Arc<AnyBurnTensor<B, D_IN>> and Arc<Option<AnyBurnTensor<B, D_OUT>>> respectively
///
/// Using Box<dyn Any + Send + Sync> to avoid adding generic parameters to this struct.
/// This is (probably) safe because InferenceRequest is only sent to the actor from the coordinator layer, both of which are unavailable to the user.
pub(crate) struct InferenceRequest {
    pub(crate) observation: Box<dyn Any + Send + Sync>,
    pub(crate) mask: Box<dyn Any + Send + Sync>,
    pub(crate) reward: f32,
    pub(crate) reply_to: oneshot::Sender<Arc<RelayRLAction>>,
}
