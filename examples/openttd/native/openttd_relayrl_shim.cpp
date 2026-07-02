#include "openttd_relayrl_shim.h"

// This file is the narrow ABI RelayRL expects from an OpenTTD 15.3 integration
// build. Upstream OpenTTD is a C++ application and does not expose a stable C
// ABI, so production integrations should implement these functions inside an
// OpenTTD fork/plugin boundary where the game-state APIs are available.

struct RelayRlOpenTtd {
    uint64_t tick;
    uint64_t max_ticks;
};

RelayRlOpenTtd *relayrl_openttd_create(
    const char * /* source_dir */,
    const char * /* build_dir */,
    uint64_t /* seed */,
    uint32_t /* map_size */,
    uint64_t max_ticks)
{
    RelayRlOpenTtd *handle = new RelayRlOpenTtd();
    handle->tick = 0;
    handle->max_ticks = max_ticks;
    return handle;
}

void relayrl_openttd_destroy(RelayRlOpenTtd *handle)
{
    delete handle;
}

int relayrl_openttd_reset(RelayRlOpenTtd *handle, RelayRlOpenTtdSnapshot *out_snapshot)
{
    if (handle == nullptr || out_snapshot == nullptr) return 1;
    handle->tick = 0;
    return relayrl_openttd_snapshot(handle, out_snapshot);
}

int relayrl_openttd_step(RelayRlOpenTtd *handle, RelayRlOpenTtdSnapshot *out_snapshot)
{
    if (handle == nullptr || out_snapshot == nullptr) return 1;
    handle->tick += 1;
    return relayrl_openttd_snapshot(handle, out_snapshot);
}

int relayrl_openttd_snapshot(RelayRlOpenTtd *handle, RelayRlOpenTtdSnapshot *out_snapshot)
{
    if (handle == nullptr || out_snapshot == nullptr) return 1;
    *out_snapshot = RelayRlOpenTtdSnapshot{};
    out_snapshot->tick = handle->tick;
    out_snapshot->company_balance = 100000.0f;
    out_snapshot->station_throughput = 0.5f;
    out_snapshot->town_station_rating = 0.65f;
    out_snapshot->topology_connectivity = 0.45f;
    return 0;
}

int relayrl_openttd_apply_command(RelayRlOpenTtd *handle, RelayRlOpenTtdCommand /* command */)
{
    if (handle == nullptr) return 1;
    return 0;
}
