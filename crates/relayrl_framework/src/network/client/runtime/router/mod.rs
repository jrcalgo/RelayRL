use relayrl_types::types::data::action::RelayRLAction;
use relayrl_types::types::data::trajectory::RelayRLTrajectory;

use crate::network::client::runtime::router::buffer::TrajectorySinkError;
use crate::network::client::runtime::router::filter::FilterError;
use crate::network::client::runtime::router::receiver::TransportReceiverError;

use std::any::Any;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::oneshot;
use uuid::Uuid;

pub(crate) mod buffer;
pub(crate) mod filter;
pub(crate) mod receiver;

#[derive(Debug, Error)]
pub enum RouterError {
    #[error(transparent)]
    FilterError(#[from] FilterError),
    #[error(transparent)]
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
/// Using Box<dyn Any + Send> to avoid adding generic parameters to this struct.
/// This is (probably) safe because InferenceRequest is only sent to the actor from the coordinator layer, both of which are unavailable to the user.
pub(crate) struct InferenceRequest {
    pub(crate) observation: Box<dyn Any + Send>,
    pub(crate) mask: Box<dyn Any + Send>,
    pub(crate) reward: f32,
    pub(crate) reply_to: oneshot::Sender<Arc<RelayRLAction>>,
}
