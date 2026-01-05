use active_uuid_registry::UuidPoolError;
use active_uuid_registry::interface::get;
use uuid::Uuid;

pub mod agent;
pub(crate) mod runtime {
    pub(crate) mod actor;
    pub(crate) mod coordination {
        pub(crate) mod coordinator;
        pub(crate) mod lifecycle_manager;
        pub(crate) mod scale_manager;
        pub(crate) mod state_manager;
    }
    pub(crate) mod router;
    pub(crate) mod router_dispatcher;
    pub(crate) mod data {
        #[cfg(any(feature = "async_transport", feature = "sync_transport"))]
        pub(crate) mod transport;
        #[cfg(any(feature = "postgres_db", feature = "sqlite_db"))]
        pub(crate) mod database;
    }
}

#[inline(always)]
pub(crate) fn get_all_client_ids() -> Result<Vec<(String, Uuid)>, UuidPoolError> {
    let actor_ids = get("actor")?;
    let scale_manager_ids = get("scale_manager")?;
    let external_sender_ids = get("external_sender")?;
    let zmq_transport_client_ids = get("zmq_transport_client")?;
    Ok(actor_ids
        .iter()
        .chain(scale_manager_ids.iter())
        .chain(external_sender_ids.iter())
        .chain(zmq_transport_client_ids.iter())
        .map(|(id, name)| (id.clone(), name.clone()))
        .collect::<Vec<(String, Uuid)>>())
}