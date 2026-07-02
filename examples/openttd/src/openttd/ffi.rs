use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct RelayRlOpenTtd {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RelayRlOpenTtdSnapshot {
    pub tick: u64,
    pub company_balance: f32,
    pub profit_delta: f32,
    pub cargo_backlog: f32,
    pub delivery_latency: f32,
    pub vehicle_idle_ratio: f32,
    pub congestion_index: f32,
    pub station_throughput: f32,
    pub demand_pressure: f32,
    pub cashflow_volatility: f32,
    pub industry_input_shortage: f32,
    pub industry_output_saturation: f32,
    pub wasted_production: f32,
    pub town_population_growth: f32,
    pub town_station_rating: f32,
    pub local_cargo_satisfaction: f32,
    pub construction_cost_pressure: f32,
    pub topology_connectivity: f32,
    pub bottleneck_pressure: f32,
    pub network_redundancy: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RelayRlOpenTtdCommand {
    pub subsystem: u32,
    pub action_index: u32,
    pub intensity: f32,
}

unsafe extern "C" {
    pub fn relayrl_openttd_create(
        source_dir: *const c_char,
        build_dir: *const c_char,
        seed: u64,
        map_size: u32,
        max_ticks: u64,
    ) -> *mut RelayRlOpenTtd;
    pub fn relayrl_openttd_destroy(handle: *mut RelayRlOpenTtd);
    pub fn relayrl_openttd_reset(
        handle: *mut RelayRlOpenTtd,
        out_snapshot: *mut RelayRlOpenTtdSnapshot,
    ) -> c_int;
    pub fn relayrl_openttd_step(
        handle: *mut RelayRlOpenTtd,
        out_snapshot: *mut RelayRlOpenTtdSnapshot,
    ) -> c_int;
    pub fn relayrl_openttd_snapshot(
        handle: *mut RelayRlOpenTtd,
        out_snapshot: *mut RelayRlOpenTtdSnapshot,
    ) -> c_int;
    pub fn relayrl_openttd_apply_command(
        handle: *mut RelayRlOpenTtd,
        command: RelayRlOpenTtdCommand,
    ) -> c_int;
}
