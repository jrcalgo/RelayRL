#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RelayRlOpenTtd RelayRlOpenTtd;

typedef struct RelayRlOpenTtdSnapshot {
    uint64_t tick;
    float company_balance;
    float profit_delta;
    float cargo_backlog;
    float delivery_latency;
    float vehicle_idle_ratio;
    float congestion_index;
    float station_throughput;
    float demand_pressure;
    float cashflow_volatility;
    float industry_input_shortage;
    float industry_output_saturation;
    float wasted_production;
    float town_population_growth;
    float town_station_rating;
    float local_cargo_satisfaction;
    float construction_cost_pressure;
    float topology_connectivity;
    float bottleneck_pressure;
    float network_redundancy;
} RelayRlOpenTtdSnapshot;

typedef struct RelayRlOpenTtdCommand {
    uint32_t subsystem;
    uint32_t action_index;
    float intensity;
} RelayRlOpenTtdCommand;

RelayRlOpenTtd *relayrl_openttd_create(
    const char *source_dir,
    const char *build_dir,
    uint64_t seed,
    uint32_t map_size,
    uint64_t max_ticks);
void relayrl_openttd_destroy(RelayRlOpenTtd *handle);
int relayrl_openttd_reset(RelayRlOpenTtd *handle, RelayRlOpenTtdSnapshot *out_snapshot);
int relayrl_openttd_step(RelayRlOpenTtd *handle, RelayRlOpenTtdSnapshot *out_snapshot);
int relayrl_openttd_snapshot(RelayRlOpenTtd *handle, RelayRlOpenTtdSnapshot *out_snapshot);
int relayrl_openttd_apply_command(RelayRlOpenTtd *handle, RelayRlOpenTtdCommand command);

#ifdef __cplusplus
}
#endif
