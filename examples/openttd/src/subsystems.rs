use crate::openttd::{GameSubsystem, OpenTtdCommand, OpenTtdSnapshot};
use crate::{ACTION_DIM, OBSERVATION_DIM};

#[derive(Debug, Clone, Copy)]
pub struct SubsystemActorSpec {
    pub subsystem: GameSubsystem,
    pub actor_name: &'static str,
    pub responsibility: &'static str,
    pub optimizes: &'static str,
    pub command_descriptions: [&'static str; ACTION_DIM],
}

pub const TRANSPORT_ROUTE_PLANNER: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Transport,
    actor_name: "transport.route-planner",
    responsibility: "propose feasible vehicle paths across the network",
    optimizes: "delivery latency and route feasibility",
    command_descriptions: [
        "keep existing routes",
        "prefer low-latency rail path",
        "prefer road feeder route",
        "reroute around congested tiles",
        "increase express service frequency",
        "defer low-priority cargo",
        "send overloaded vehicle to depot",
        "rebalance multimodal transfers",
    ],
};

pub const TRANSPORT_DISPATCH: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Transport,
    actor_name: "transport.dispatch",
    responsibility: "schedule vehicles and reduce idle time",
    optimizes: "vehicle utilization and idle-time reduction",
    command_descriptions: [
        "hold schedule",
        "increase departures",
        "decrease departures",
        "stagger station arrivals",
        "prioritize full loads",
        "release depot queue",
        "pause low-margin service",
        "balance train and road cadence",
    ],
};

pub const TRANSPORT_CONGESTION: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Transport,
    actor_name: "transport.congestion",
    responsibility: "detect bottlenecks and bias routing away from them",
    optimizes: "network throughput and bottleneck avoidance",
    command_descriptions: [
        "accept current congestion",
        "penalize saturated junction",
        "penalize overloaded station",
        "favor bypass route",
        "favor depot relief",
        "shift cargo mode",
        "spread loading windows",
        "reserve capacity for priority cargo",
    ],
};

pub const TRANSPORT_VEHICLE_ALLOCATION: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Transport,
    actor_name: "transport.vehicle-allocation",
    responsibility: "match vehicles to cargo backlogs and route classes",
    optimizes: "cargo backlog reduction per vehicle class",
    command_descriptions: [
        "keep fleet allocation",
        "add trains to bulk cargo",
        "add buses to town service",
        "add trucks to feeder service",
        "retire idle vehicles",
        "move vehicle to high-backlog station",
        "increase ship/air capacity",
        "rebalance cargo class mix",
    ],
};

pub const ECONOMY_PROFIT_MODEL: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Economy,
    actor_name: "economy.profit-model",
    responsibility: "estimate marginal profit from current flows",
    optimizes: "rolling profit delta and cashflow stability",
    command_descriptions: [
        "hold investment posture",
        "favor high-margin routes",
        "trim low-margin service",
        "delay capital expense",
        "accelerate profitable expansion",
        "stabilize cash reserve",
        "rebalance route cluster budget",
        "increase maintenance reserve",
    ],
};

pub const ECONOMY_DEMAND_ESTIMATOR: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Economy,
    actor_name: "economy.demand-estimator",
    responsibility: "forecast demand pressure from cargo and station signals",
    optimizes: "demand forecast accuracy",
    command_descriptions: [
        "hold demand estimate",
        "raise town demand signal",
        "raise industry demand signal",
        "lower stale demand signal",
        "smooth demand spike",
        "prioritize unmet cargo",
        "detect subsidy opportunity",
        "rebalance region demand",
    ],
};

pub const ECONOMY_INVESTMENT_PRIORITY: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Economy,
    actor_name: "economy.investment-priority",
    responsibility: "rank routes and regions for capital allocation",
    optimizes: "capital allocation across routes and regions",
    command_descriptions: [
        "hold capital allocation",
        "prioritize core routes",
        "prioritize growth towns",
        "prioritize industry chains",
        "reserve capital",
        "fund congestion relief",
        "fund new station coverage",
        "fund topology redundancy",
    ],
};

pub const INDUSTRY_SUPPLY_CHAIN: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Industry,
    actor_name: "industry.supply-chain",
    responsibility: "prioritize industry inputs and output pickup",
    optimizes: "input-shortage reduction and output pickup priority",
    command_descriptions: [
        "hold supply-chain priority",
        "prioritize raw inputs",
        "prioritize factory outputs",
        "redirect cargo to starved industry",
        "buffer excess output",
        "smooth production cycle",
        "shift transport mode",
        "trigger route expansion signal",
    ],
};

pub const INDUSTRY_PRODUCTION_BALANCER: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Industry,
    actor_name: "industry.production-balancer",
    responsibility: "balance production throughput against transport capacity",
    optimizes: "throughput efficiency and waste reduction",
    command_descriptions: [
        "hold production balance",
        "reduce wasted output",
        "increase pickup cadence",
        "lower saturated source priority",
        "raise bottlenecked source priority",
        "stabilize inventory buffer",
        "favor complete chains",
        "signal new depot/storage need",
    ],
};

pub const TOWN_GROWTH_PREDICTOR: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::TownGrowth,
    actor_name: "town.growth-predictor",
    responsibility: "estimate town growth from service and accessibility",
    optimizes: "population-growth and accessibility forecasts",
    command_descriptions: [
        "hold growth forecast",
        "prioritize growing town",
        "boost commuter coverage",
        "boost mail/passenger service",
        "rebalance local accessibility",
        "protect underserved town",
        "defer saturated town",
        "signal station upgrade need",
    ],
};

pub const TOWN_RATING_OPTIMIZER: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::TownGrowth,
    actor_name: "town.rating-optimizer",
    responsibility: "improve station ratings and cargo satisfaction",
    optimizes: "station ratings and local cargo satisfaction",
    command_descriptions: [
        "hold town service",
        "increase station pickup",
        "reduce waiting cargo",
        "add local feeder service",
        "favor high-rating station",
        "repair weak station rating",
        "smooth passenger bursts",
        "signal infrastructure upgrade",
    ],
};

pub const INFRA_EXPANSION_PLANNER: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Infrastructure,
    actor_name: "infrastructure.expansion-planner",
    responsibility: "select long-horizon construction targets",
    optimizes: "long-horizon bottleneck relief",
    command_descriptions: [
        "hold expansion plan",
        "build bypass rail",
        "build new station",
        "expand road feeder",
        "expand depot capacity",
        "add port/airport capacity",
        "remove inefficient segment",
        "reserve corridor for future route",
    ],
};

pub const INFRA_COST_OPTIMIZER: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Infrastructure,
    actor_name: "infrastructure.cost-optimizer",
    responsibility: "control construction cost against expected transport gain",
    optimizes: "construction cost per expected transport gain",
    command_descriptions: [
        "hold cost posture",
        "prefer cheap terrain",
        "delay expensive build",
        "reuse existing corridor",
        "favor small upgrade",
        "approve high-return spend",
        "avoid demolition",
        "batch construction work",
    ],
};

pub const INFRA_TOPOLOGY: SubsystemActorSpec = SubsystemActorSpec {
    subsystem: GameSubsystem::Infrastructure,
    actor_name: "infrastructure.topology",
    responsibility: "maintain network graph connectivity and redundancy",
    optimizes: "graph connectivity and redundancy",
    command_descriptions: [
        "hold topology",
        "increase graph connectivity",
        "increase redundancy",
        "separate conflicting flows",
        "add transfer edge",
        "repair fragile component",
        "simplify overbuilt branch",
        "protect critical junction",
    ],
};

pub const ACTOR_SPECS: &[SubsystemActorSpec] = &[
    TRANSPORT_ROUTE_PLANNER,
    TRANSPORT_DISPATCH,
    TRANSPORT_CONGESTION,
    TRANSPORT_VEHICLE_ALLOCATION,
    ECONOMY_PROFIT_MODEL,
    ECONOMY_DEMAND_ESTIMATOR,
    ECONOMY_INVESTMENT_PRIORITY,
    INDUSTRY_SUPPLY_CHAIN,
    INDUSTRY_PRODUCTION_BALANCER,
    TOWN_GROWTH_PREDICTOR,
    TOWN_RATING_OPTIMIZER,
    INFRA_EXPANSION_PLANNER,
    INFRA_COST_OPTIMIZER,
    INFRA_TOPOLOGY,
];

pub fn specs_for_subsystem(subsystem: GameSubsystem) -> impl Iterator<Item = SubsystemActorSpec> {
    ACTOR_SPECS
        .iter()
        .copied()
        .filter(move |spec| spec.subsystem == subsystem)
}

pub fn project_observation(
    snapshot: &OpenTtdSnapshot,
    spec: &SubsystemActorSpec,
) -> [f32; OBSERVATION_DIM] {
    let mut obs = common_features(snapshot);
    match spec.subsystem {
        GameSubsystem::Transport => {
            obs[20] = snapshot.delivery_latency;
            obs[21] = snapshot.vehicle_idle_ratio;
            obs[22] = snapshot.congestion_index;
            obs[23] = snapshot.cargo_backlog;
        }
        GameSubsystem::Economy => {
            obs[20] = scaled_balance(snapshot.company_balance);
            obs[21] = snapshot.profit_delta / 1_000.0;
            obs[22] = snapshot.cashflow_volatility;
            obs[23] = snapshot.demand_pressure;
        }
        GameSubsystem::Industry => {
            obs[20] = snapshot.industry_input_shortage;
            obs[21] = snapshot.industry_output_saturation;
            obs[22] = snapshot.wasted_production;
            obs[23] = snapshot.station_throughput;
        }
        GameSubsystem::TownGrowth => {
            obs[20] = snapshot.town_population_growth;
            obs[21] = snapshot.town_station_rating;
            obs[22] = snapshot.local_cargo_satisfaction;
            obs[23] = snapshot.demand_pressure;
        }
        GameSubsystem::Infrastructure => {
            obs[20] = snapshot.construction_cost_pressure;
            obs[21] = snapshot.topology_connectivity;
            obs[22] = snapshot.bottleneck_pressure;
            obs[23] = snapshot.network_redundancy;
        }
    }

    obs[24] = actor_index(spec.actor_name) as f32 / ACTOR_SPECS.len() as f32;
    obs[25] = subsystem_index(spec.subsystem) as f32 / GameSubsystem::ALL.len() as f32;
    obs[26] = objective_pressure(snapshot, spec);
    obs[27] = 1.0 - objective_pressure(snapshot, spec);
    obs[28] = snapshot.tick as f32 / 10_000.0;
    obs[29] = snapshot.profit_delta.signum();
    obs[30] = (snapshot.cargo_backlog + snapshot.demand_pressure) * 0.5;
    obs[31] = 1.0;
    obs
}

pub fn reward_for_transition(
    before: &OpenTtdSnapshot,
    after: &OpenTtdSnapshot,
    spec: &SubsystemActorSpec,
) -> f32 {
    match spec.actor_name {
        "transport.route-planner" => before.delivery_latency - after.delivery_latency,
        "transport.dispatch" => before.vehicle_idle_ratio - after.vehicle_idle_ratio,
        "transport.congestion" => before.congestion_index - after.congestion_index,
        "transport.vehicle-allocation" => before.cargo_backlog - after.cargo_backlog,
        "economy.profit-model" => (after.profit_delta - before.profit_delta) / 1_000.0,
        "economy.demand-estimator" => before.demand_pressure - after.demand_pressure,
        "economy.investment-priority" => {
            (after.company_balance - before.company_balance) / 100_000.0
                - after.cashflow_volatility * 0.05
        }
        "industry.supply-chain" => before.industry_input_shortage - after.industry_input_shortage,
        "industry.production-balancer" => before.wasted_production - after.wasted_production,
        "town.growth-predictor" => {
            after.town_population_growth - before.town_population_growth
                + (after.local_cargo_satisfaction - before.local_cargo_satisfaction) * 0.25
        }
        "town.rating-optimizer" => {
            after.town_station_rating - before.town_station_rating
                + (after.local_cargo_satisfaction - before.local_cargo_satisfaction) * 0.5
        }
        "infrastructure.expansion-planner" => {
            before.bottleneck_pressure - after.bottleneck_pressure
        }
        "infrastructure.cost-optimizer" => {
            before.construction_cost_pressure - after.construction_cost_pressure
        }
        "infrastructure.topology" => {
            after.topology_connectivity - before.topology_connectivity
                + (after.network_redundancy - before.network_redundancy) * 0.5
        }
        _ => 0.0,
    }
}

pub fn command_from_action(
    spec: &SubsystemActorSpec,
    action_index: usize,
    confidence: f32,
) -> OpenTtdCommand {
    let action_index = action_index.min(ACTION_DIM - 1);
    OpenTtdCommand {
        subsystem: spec.subsystem,
        actor_name: spec.actor_name,
        action_index,
        intensity: confidence.clamp(0.0, 1.0),
        description: spec.command_descriptions[action_index],
    }
}

pub fn action_mask_bytes() -> Vec<u8> {
    let values = [1.0_f32; ACTION_DIM];
    f32_slice_to_bytes(&values)
}

pub fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn common_features(snapshot: &OpenTtdSnapshot) -> [f32; OBSERVATION_DIM] {
    let mut obs = [0.0; OBSERVATION_DIM];
    obs[0] = snapshot.tick as f32 / 10_000.0;
    obs[1] = scaled_balance(snapshot.company_balance);
    obs[2] = snapshot.profit_delta / 1_000.0;
    obs[3] = snapshot.cargo_backlog;
    obs[4] = snapshot.delivery_latency;
    obs[5] = snapshot.vehicle_idle_ratio;
    obs[6] = snapshot.congestion_index;
    obs[7] = snapshot.station_throughput;
    obs[8] = snapshot.demand_pressure;
    obs[9] = snapshot.cashflow_volatility;
    obs[10] = snapshot.industry_input_shortage;
    obs[11] = snapshot.industry_output_saturation;
    obs[12] = snapshot.wasted_production;
    obs[13] = snapshot.town_population_growth;
    obs[14] = snapshot.town_station_rating;
    obs[15] = snapshot.local_cargo_satisfaction;
    obs[16] = snapshot.construction_cost_pressure;
    obs[17] = snapshot.topology_connectivity;
    obs[18] = snapshot.bottleneck_pressure;
    obs[19] = snapshot.network_redundancy;
    obs
}

fn objective_pressure(snapshot: &OpenTtdSnapshot, spec: &SubsystemActorSpec) -> f32 {
    match spec.actor_name {
        "transport.route-planner" => snapshot.delivery_latency,
        "transport.dispatch" => snapshot.vehicle_idle_ratio,
        "transport.congestion" => snapshot.congestion_index,
        "transport.vehicle-allocation" => snapshot.cargo_backlog,
        "economy.profit-model" => 1.0 - (snapshot.profit_delta / 1_000.0).clamp(-1.0, 1.0),
        "economy.demand-estimator" => snapshot.demand_pressure,
        "economy.investment-priority" => snapshot.cashflow_volatility,
        "industry.supply-chain" => snapshot.industry_input_shortage,
        "industry.production-balancer" => snapshot.wasted_production,
        "town.growth-predictor" => 1.0 - snapshot.town_population_growth,
        "town.rating-optimizer" => 1.0 - snapshot.town_station_rating,
        "infrastructure.expansion-planner" => snapshot.bottleneck_pressure,
        "infrastructure.cost-optimizer" => snapshot.construction_cost_pressure,
        "infrastructure.topology" => 1.0 - snapshot.topology_connectivity,
        _ => 0.0,
    }
    .clamp(0.0, 1.0)
}

fn actor_index(actor_name: &str) -> usize {
    ACTOR_SPECS
        .iter()
        .position(|spec| spec.actor_name == actor_name)
        .unwrap_or_default()
}

fn subsystem_index(subsystem: GameSubsystem) -> usize {
    GameSubsystem::ALL
        .iter()
        .position(|candidate| *candidate == subsystem)
        .unwrap_or_default()
}

fn scaled_balance(balance: f32) -> f32 {
    (balance / 1_000_000.0).clamp(-1.0, 1.0)
}
