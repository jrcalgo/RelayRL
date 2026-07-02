use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GameSubsystem {
    Transport,
    Economy,
    Industry,
    TownGrowth,
    Infrastructure,
}

impl GameSubsystem {
    pub const ALL: [Self; 5] = [
        Self::Transport,
        Self::Economy,
        Self::Industry,
        Self::TownGrowth,
        Self::Infrastructure,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::Economy => "economy",
            Self::Industry => "industry",
            Self::TownGrowth => "town-growth",
            Self::Infrastructure => "infrastructure",
        }
    }

    pub fn ffi_code(self) -> u32 {
        match self {
            Self::Transport => 0,
            Self::Economy => 1,
            Self::Industry => 2,
            Self::TownGrowth => 3,
            Self::Infrastructure => 4,
        }
    }
}

impl fmt::Display for GameSubsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTtdConfig {
    pub source_dir: Option<PathBuf>,
    pub build_dir: Option<PathBuf>,
    pub seed: u64,
    pub map_size: u32,
    pub max_ticks: u64,
}

impl Default for OpenTtdConfig {
    fn default() -> Self {
        Self {
            source_dir: std::env::var_os("OPENTTD_SOURCE_DIR").map(PathBuf::from),
            build_dir: std::env::var_os("OPENTTD_BUILD_DIR").map(PathBuf::from),
            seed: 0x15_03,
            map_size: 256,
            max_ticks: 10_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTtdCommand {
    pub subsystem: GameSubsystem,
    pub actor_name: &'static str,
    pub action_index: usize,
    pub intensity: f32,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTtdSnapshot {
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

impl Default for OpenTtdSnapshot {
    fn default() -> Self {
        Self {
            tick: 0,
            company_balance: 100_000.0,
            profit_delta: 0.0,
            cargo_backlog: 0.25,
            delivery_latency: 0.35,
            vehicle_idle_ratio: 0.20,
            congestion_index: 0.20,
            station_throughput: 0.50,
            demand_pressure: 0.40,
            cashflow_volatility: 0.10,
            industry_input_shortage: 0.30,
            industry_output_saturation: 0.25,
            wasted_production: 0.20,
            town_population_growth: 0.05,
            town_station_rating: 0.65,
            local_cargo_satisfaction: 0.55,
            construction_cost_pressure: 0.35,
            topology_connectivity: 0.45,
            bottleneck_pressure: 0.30,
            network_redundancy: 0.25,
        }
    }
}

impl OpenTtdSnapshot {
    pub fn done(&self, config: &OpenTtdConfig) -> bool {
        self.tick >= config.max_ticks
    }
}

#[derive(Debug, Clone)]
pub struct OpenTtdBridgeError {
    message: String,
}

impl OpenTtdBridgeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for OpenTtdBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OpenTtdBridgeError {}
