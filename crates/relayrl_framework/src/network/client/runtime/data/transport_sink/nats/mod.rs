pub(crate) mod interface;
pub(super) mod ops;
pub(super) mod policies;

#[derive(Debug, Error, Clone)]
pub enum NatsClientError {
    #[error("NATS transport error: {0}")]
    NatsTransportError(String),
    #[error("Task join error: {0}")]
    JoinError(String),
}

pub(super) trait NatsInferenceExecution {}

pub(super) trait NatsTrainingExecution {}

#[cfg(test)]
mod tests {}
