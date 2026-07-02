use super::ffi;
use super::{OpenTtdBridgeError, OpenTtdCommand, OpenTtdConfig, OpenTtdSnapshot};
use std::ffi::CString;
use std::ptr::NonNull;

#[derive(Debug)]
pub struct OpenTtdBridge {
    handle: NonNull<ffi::RelayRlOpenTtd>,
}

unsafe impl Send for OpenTtdBridge {}
unsafe impl Sync for OpenTtdBridge {}

impl OpenTtdBridge {
    pub fn new(config: OpenTtdConfig) -> Result<Self, OpenTtdBridgeError> {
        let source_dir = config
            .source_dir
            .as_ref()
            .ok_or_else(|| OpenTtdBridgeError::new("OPENTTD_SOURCE_DIR is required"))?;
        let build_dir = config
            .build_dir
            .as_ref()
            .ok_or_else(|| OpenTtdBridgeError::new("OPENTTD_BUILD_DIR is required"))?;
        let source_dir = CString::new(source_dir.to_string_lossy().as_bytes())
            .map_err(|e| OpenTtdBridgeError::new(e.to_string()))?;
        let build_dir = CString::new(build_dir.to_string_lossy().as_bytes())
            .map_err(|e| OpenTtdBridgeError::new(e.to_string()))?;

        let raw = unsafe {
            ffi::relayrl_openttd_create(
                source_dir.as_ptr(),
                build_dir.as_ptr(),
                config.seed,
                config.map_size,
                config.max_ticks,
            )
        };
        let handle = NonNull::new(raw)
            .ok_or_else(|| OpenTtdBridgeError::new("relayrl_openttd_create returned null"))?;
        Ok(Self { handle })
    }

    pub fn reset(&mut self) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        self.read_snapshot_with(ffi::relayrl_openttd_reset)
    }

    pub fn snapshot(&self) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        self.read_snapshot_with(ffi::relayrl_openttd_snapshot)
    }

    pub fn apply_command(&mut self, command: &OpenTtdCommand) -> Result<(), OpenTtdBridgeError> {
        let command = ffi::RelayRlOpenTtdCommand {
            subsystem: command.subsystem.ffi_code(),
            action_index: command.action_index as u32,
            intensity: command.intensity,
        };
        let status = unsafe { ffi::relayrl_openttd_apply_command(self.handle.as_ptr(), command) };
        status_to_result(status, "relayrl_openttd_apply_command").map(|_| ())
    }

    pub fn step(&mut self) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        self.read_snapshot_with(ffi::relayrl_openttd_step)
    }

    fn read_snapshot_with(
        &self,
        f: unsafe extern "C" fn(
            *mut ffi::RelayRlOpenTtd,
            *mut ffi::RelayRlOpenTtdSnapshot,
        ) -> std::os::raw::c_int,
    ) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        let mut raw = ffi::RelayRlOpenTtdSnapshot::default();
        let status = unsafe { f(self.handle.as_ptr(), &mut raw) };
        status_to_result(status, "OpenTTD snapshot call")?;
        Ok(raw.into())
    }
}

impl Drop for OpenTtdBridge {
    fn drop(&mut self) {
        unsafe {
            ffi::relayrl_openttd_destroy(self.handle.as_ptr());
        }
    }
}

fn status_to_result(status: i32, operation: &str) -> Result<(), OpenTtdBridgeError> {
    if status == 0 {
        Ok(())
    } else {
        Err(OpenTtdBridgeError::new(format!(
            "{operation} failed with status {status}"
        )))
    }
}

impl From<ffi::RelayRlOpenTtdSnapshot> for OpenTtdSnapshot {
    fn from(raw: ffi::RelayRlOpenTtdSnapshot) -> Self {
        Self {
            tick: raw.tick,
            company_balance: raw.company_balance,
            profit_delta: raw.profit_delta,
            cargo_backlog: raw.cargo_backlog,
            delivery_latency: raw.delivery_latency,
            vehicle_idle_ratio: raw.vehicle_idle_ratio,
            congestion_index: raw.congestion_index,
            station_throughput: raw.station_throughput,
            demand_pressure: raw.demand_pressure,
            cashflow_volatility: raw.cashflow_volatility,
            industry_input_shortage: raw.industry_input_shortage,
            industry_output_saturation: raw.industry_output_saturation,
            wasted_production: raw.wasted_production,
            town_population_growth: raw.town_population_growth,
            town_station_rating: raw.town_station_rating,
            local_cargo_satisfaction: raw.local_cargo_satisfaction,
            construction_cost_pressure: raw.construction_cost_pressure,
            topology_connectivity: raw.topology_connectivity,
            bottleneck_pressure: raw.bottleneck_pressure,
            network_redundancy: raw.network_redundancy,
        }
    }
}
