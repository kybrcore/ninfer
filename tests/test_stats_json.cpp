#include "serve/stats_json.h"

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::MemorySummary make_memory() {
    ninfer::MemorySummary memory;
    memory.device                        = 0;
    memory.max_context                   = 524288;
    memory.kv_cache                      = ninfer::KvCacheStorage::Nvfp4Group16;
    memory.kv_capacity_mode              = ninfer::KvCapacityMode::Explicit;
    memory.kv_capacity                   = 524288;
    memory.kv_capacity_page_groups       = 8192;
    memory.kv_capacity_max_page_groups   = 24576;
    memory.kv_payload_bytes              = 9663676416;
    memory.planned_slack_bytes           = 479314688;
    memory.host_kv_capacity_bytes        = 34359738368;
    memory.host_state_capacity_slots     = 8;
    memory.available_after_startup_bytes = 1868234752;
    memory.weights   = ninfer::ArenaMemorySummary{19317748736, 19317748736, 19317748736};
    memory.sequence  = ninfer::ArenaMemorySummary{10961894656, 10961894656, 10961894656};
    memory.workspace = ninfer::ArenaMemorySummary{866648064, 0, 159997952};
    return memory;
}

StatsSnapshot make_snapshot(bool with_memory) {
    StatsSnapshot snapshot;
    snapshot.timestamp_unix_ms  = 1'789'000'000'000ULL;
    snapshot.available          = true;
    snapshot.server_instance_id = "serve-1234-5678";
    if (with_memory) { snapshot.memory = make_memory(); }

    ninfer::RuntimeStats& stats           = snapshot.stats;
    stats.running_requests                = 2;
    stats.prefilling_requests             = 1;
    stats.decode_ready_requests           = 1;
    stats.waiting_requests                = 3;
    stats.computed_prefill_tokens         = 2932;
    stats.committed_decode_tokens         = 2363;
    stats.decode_rounds                   = 611;
    stats.decode_row_rounds               = 634;
    stats.active_captures_completed       = 24;
    stats.main_kv_d2h_pages               = 8;
    stats.main_kv_d2h_bytes               = 9437184;
    stats.main_kv_d2h_seconds             = 0.001071;
    stats.state_d2h_count                 = 25;
    stats.state_h2d_bytes                 = 4096;
    stats.pressure_spill_pages            = 0;
    stats.pressure_checkpoints_dropped    = 32;
    stats.pressure_searches               = 28;
    stats.historical_fork_hits            = 0;
    stats.root_selections                 = 25;
    stats.reused_prompt_tokens            = 831;
    stats.last_selected_frontier_tokens   = 0;
    stats.device_state_occupied_slots     = 6;
    stats.host_state_occupied_slots       = 8;
    stats.device_main_kv_occupied_pages   = 38;
    stats.host_kv_occupied_bytes          = 0;
    stats.shared_active_references        = 0;
    stats.actual_context_transfer_seconds = 0.103376;

    snapshot.load.model_id       = "qwen3.8-27b";
    snapshot.load.target         = "qwen3_8_27b";
    snapshot.load.weights_id     = "nvfp4";
    snapshot.load.load_seconds   = 13.8347;
    snapshot.load.tensor_count   = 1060;
    snapshot.load.resource_count = 6;
    snapshot.in_flight           = 3;
    snapshot.max_in_flight       = 11;
    return snapshot;
}

} // namespace

int main() {
    int failures     = 0;
    const Json stats = Json::parse(format_stats_json(make_snapshot(true)));

    failures += check(stats.at("schema") == "ninfer_serve_stats", "schema name changed");
    failures += check(stats.at("schema_version") == 1, "schema version changed");
    failures += check(stats.at("available") == true, "availability not reported");
    failures += check(stats.at("server_instance_id") == "serve-1234-5678",
                      "server instance id not reported");
    failures +=
        check(stats.at("timestamp_unix_ms") == 1789000000000ULL, "render timestamp not reported");

    // Gauges, counters, and HTTP depth.
    failures += check(stats.at("scheduler").at("running") == 2, "running gauge missing");
    failures += check(stats.at("scheduler").at("waiting") == 3, "waiting gauge missing");
    failures +=
        check(stats.at("counters").at("committed_decode_tokens") == 2363, "decode counter missing");
    failures +=
        check(stats.at("counters").at("decode_row_rounds") == 634, "row-round counter missing");
    failures += check(stats.at("http").at("in_flight") == 3, "in-flight depth missing");
    failures += check(stats.at("http").at("max_in_flight") == 11, "in-flight ceiling missing");

    // Transfers and pressure keep their RuntimeStats names and units.
    failures += check(stats.at("kv_transfers").at("main_kv").at("d2h").at("pages") == 8,
                      "KV d2h pages missing");
    failures += check(stats.at("kv_transfers").at("main_kv").at("d2h").at("bytes") == 9437184,
                      "KV d2h bytes missing");
    failures +=
        check(stats.at("state_transfers").at("h2d").at("bytes") == 4096, "state h2d bytes missing");
    failures +=
        check(stats.at("pressure").at("checkpoints_dropped") == 32, "checkpoint drops missing");
    failures +=
        check(stats.at("pressure").at("historical_fork_hits") == 0, "historical fork hits missing");
    failures += check(stats.at("cache_reuse").at("reused_prompt_tokens") == 831,
                      "reused prompt tokens missing");
    failures += check(stats.at("occupancy").at("device_main_kv_pages") == 38,
                      "device KV occupancy missing");

    // Memory renders capacities and arenas, and never duplicates the occupancy gauges.
    failures += check(stats.at("memory").at("kv_cache") == "nvfp4", "KV storage name missing");
    failures +=
        check(stats.at("memory").at("kv_capacity_mode") == "explicit", "KV capacity mode missing");
    failures += check(stats.at("memory").at("kv_capacity") == 524288, "KV capacity missing");
    failures +=
        check(stats.at("memory").at("workspace").at("used_bytes") == 0, "workspace arena missing");
    failures += check(stats.at("memory").at("workspace").at("peak_used_bytes") == 159997952,
                      "workspace peak missing");
    failures += check(!stats.at("memory").contains("host_kv_occupied_bytes"),
                      "memory duplicates an occupancy gauge");
    failures += check(stats.at("memory").at("vision_workspace").is_null(),
                      "disabled vision must render null");
    failures += check(stats.at("load").at("model_id") == "qwen3.8-27b", "load identity missing");

    // ?memory=0 trims the payload: every other field survives and the memory block is explicit.
    const Json cheap = Json::parse(format_stats_json(make_snapshot(false)));
    failures += check(cheap.at("memory").is_null(), "skipped memory read must render null");
    failures += check(cheap.at("counters").at("committed_decode_tokens") == 2363,
                      "skipping memory dropped the counters");
    failures += check(cheap.at("occupancy").at("device_main_kv_pages") == 38,
                      "skipping memory dropped the occupancy gauges");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
