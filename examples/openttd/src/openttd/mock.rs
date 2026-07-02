use super::{GameSubsystem, OpenTtdBridgeError, OpenTtdCommand, OpenTtdConfig, OpenTtdSnapshot};

#[derive(Debug, Clone)]
pub struct OpenTtdBridge {
    config: OpenTtdConfig,
    snapshot: OpenTtdSnapshot,
    pending_profit: f32,
}

impl OpenTtdBridge {
    pub fn new(config: OpenTtdConfig) -> Result<Self, OpenTtdBridgeError> {
        Ok(Self {
            config,
            snapshot: OpenTtdSnapshot::default(),
            pending_profit: 0.0,
        })
    }

    pub fn reset(&mut self) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        self.snapshot = OpenTtdSnapshot::default();
        self.pending_profit = 0.0;
        Ok(self.snapshot.clone())
    }

    pub fn snapshot(&self) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        Ok(self.snapshot.clone())
    }

    pub fn apply_command(&mut self, command: &OpenTtdCommand) -> Result<(), OpenTtdBridgeError> {
        let intensity = command.intensity.clamp(0.0, 1.0);
        match command.subsystem {
            GameSubsystem::Transport => {
                self.snapshot.delivery_latency =
                    improve(self.snapshot.delivery_latency, 0.05 * intensity);
                self.snapshot.vehicle_idle_ratio =
                    improve(self.snapshot.vehicle_idle_ratio, 0.04 * intensity);
                self.snapshot.congestion_index =
                    improve(self.snapshot.congestion_index, 0.03 * intensity);
                self.snapshot.station_throughput =
                    improve_positive(self.snapshot.station_throughput, 0.04 * intensity);
                self.pending_profit += 80.0 * intensity;
            }
            GameSubsystem::Economy => {
                self.snapshot.cashflow_volatility =
                    improve(self.snapshot.cashflow_volatility, 0.04 * intensity);
                self.snapshot.demand_pressure =
                    improve(self.snapshot.demand_pressure, 0.03 * intensity);
                self.pending_profit += 120.0 * intensity;
            }
            GameSubsystem::Industry => {
                self.snapshot.industry_input_shortage =
                    improve(self.snapshot.industry_input_shortage, 0.04 * intensity);
                self.snapshot.industry_output_saturation =
                    improve(self.snapshot.industry_output_saturation, 0.03 * intensity);
                self.snapshot.wasted_production =
                    improve(self.snapshot.wasted_production, 0.04 * intensity);
                self.pending_profit += 65.0 * intensity;
            }
            GameSubsystem::TownGrowth => {
                self.snapshot.town_population_growth =
                    improve_positive(self.snapshot.town_population_growth, 0.02 * intensity);
                self.snapshot.town_station_rating =
                    improve_positive(self.snapshot.town_station_rating, 0.03 * intensity);
                self.snapshot.local_cargo_satisfaction =
                    improve_positive(self.snapshot.local_cargo_satisfaction, 0.04 * intensity);
            }
            GameSubsystem::Infrastructure => {
                self.snapshot.construction_cost_pressure =
                    improve(self.snapshot.construction_cost_pressure, 0.02 * intensity);
                self.snapshot.topology_connectivity =
                    improve_positive(self.snapshot.topology_connectivity, 0.04 * intensity);
                self.snapshot.network_redundancy =
                    improve_positive(self.snapshot.network_redundancy, 0.03 * intensity);
                self.snapshot.bottleneck_pressure =
                    improve(self.snapshot.bottleneck_pressure, 0.04 * intensity);
                self.pending_profit -= 40.0 * intensity;
            }
        }
        Ok(())
    }

    pub fn step(&mut self) -> Result<OpenTtdSnapshot, OpenTtdBridgeError> {
        self.snapshot.tick += 1;
        let seasonal = ((self.snapshot.tick % 97) as f32 / 97.0) * 0.02;
        self.snapshot.cargo_backlog = (self.snapshot.cargo_backlog + seasonal
            - self.snapshot.station_throughput * 0.01)
            .clamp(0.0, 1.0);
        self.snapshot.bottleneck_pressure = (self.snapshot.bottleneck_pressure
            + self.snapshot.congestion_index * 0.02
            - self.snapshot.network_redundancy * 0.01)
            .clamp(0.0, 1.0);
        self.snapshot.profit_delta =
            self.pending_profit - 15.0 * self.snapshot.congestion_index - 8.0 * seasonal;
        self.snapshot.company_balance += self.snapshot.profit_delta;
        self.pending_profit = 0.0;

        if self.snapshot.tick >= self.config.max_ticks {
            self.snapshot.profit_delta -= 50.0;
        }

        Ok(self.snapshot.clone())
    }
}

fn improve(value: f32, amount: f32) -> f32 {
    (value - amount).clamp(0.0, 1.0)
}

fn improve_positive(value: f32, amount: f32) -> f32 {
    (value + amount).clamp(0.0, 1.0)
}
