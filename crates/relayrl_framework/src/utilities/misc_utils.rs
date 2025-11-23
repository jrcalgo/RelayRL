use std::any::TypeId;
use std::mem::size_of;

use crate::network::TransportType;
use crate::prelude::config::TransportConfigParams;
use crate::utilities::configuration::NetworkParams;

pub(crate) fn round_to_8_decimals<N>(num: N) -> N
where
    N: Copy + 'static,
{
    let size: usize = size_of::<N>();
    if TypeId::of::<N>() == TypeId::of::<f32>() && size == size_of::<f32>() {
        // Convert the generic N to f32.
        let n: f32 = unsafe { *(&num as *const N as *const f32) };
        let factor: f32 = 100_000_000.0_f32;
        let rounded: f32 = (n * factor).round() / factor;
        // Convert back to N.
        unsafe { *(&rounded as *const f32 as *const N) }
    } else if TypeId::of::<N>() == TypeId::of::<f64>() && size == size_of::<f64>() {
        // Convert the generic N to f64.
        let n: f64 = unsafe { *(&num as *const N as *const f64) };
        let factor: f64 = 100_000_000.0_f64;
        let rounded: f64 = (n * factor).round() / factor;
        // Convert back to N.
        unsafe { *(&rounded as *const f64 as *const N) }
    } else {
        panic!("Unsupported type. Only f32 and f64 are allowed.");
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ServerAddresses {
    pub(crate) agent_listener_address: String,
    pub(crate) model_server_address: String,
    pub(crate) trajectory_server_address: String,
}

pub(crate) fn construct_server_addresses(
    transport_config: &TransportConfigParams,
    transport_type: &TransportType,
) -> ServerAddresses {
    fn construct_address(transport_type: &TransportType, network_params: &NetworkParams) -> String {
        match *transport_type {
            TransportType::GRPC => network_params.host.clone() + ":" + &network_params.port.clone(),
            TransportType::ZMQ => {
                network_params.prefix.clone() + &network_params.host.clone() + ":" + &network_params.port.clone()
            }
        }
    }

    ServerAddresses {
        agent_listener_address: construct_address(
            &transport_type,
            &transport_config.agent_listener_address,
        ),
        model_server_address: construct_address(
            &transport_type,
            &transport_config.model_server_address,
        ),
        trajectory_server_address: construct_address(
            &transport_type,
            &transport_config.trajectory_server_address,
        ),
    }
}
