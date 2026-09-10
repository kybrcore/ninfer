#include "serve/stats_json.h"

#include <nlohmann/json.hpp>

#include <optional>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

const char* kv_cache_name(ninfer::KvCacheStorage storage) {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-group64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

Json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json vision_workspace_json(const std::optional<ninfer::VisionWorkspaceMemorySummary>& vision) {
    if (!vision) { return nullptr; }
    return Json{{"aggregate_prompt_tokens", vision->aggregate_prompt_tokens},
                {"max_item_tokens", vision->max_item_tokens},
                {"general_capacity_bytes", vision->general_capacity_bytes},
                {"encode_peak_bytes", vision->encode_peak_bytes},
                {"handoff_offset_bytes", vision->handoff_offset_bytes},
                {"handoff_capacity_bytes", vision->handoff_capacity_bytes},
                {"handoff_active_bytes", vision->handoff_active_bytes},
                {"handoff_peak_bytes", vision->handoff_peak_bytes}};
}

Json memory_json(const std::optional<ninfer::MemorySummary>& value) {
    if (!value) { return nullptr; }
    const ninfer::MemorySummary& memory = *value;
    return Json{{"device", memory.device},
                {"max_context", memory.max_context},
                {"kv_cache", kv_cache_name(memory.kv_cache)},
                {"kv_capacity_mode", kv_capacity_mode_name(memory.kv_capacity_mode)},
                {"kv_capacity", memory.kv_capacity},
                {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
                {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
                {"kv_payload_bytes", memory.kv_payload_bytes},
                {"planned_slack_bytes", memory.planned_slack_bytes},
                {"workspace_logical_peak_bytes", memory.workspace_logical_peak_bytes},
                {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
                {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
                {"kv_capacity_increment_bytes", memory.kv_capacity_increment_bytes},
                {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
                {"available_after_weights_bytes", memory.available_after_weights_bytes},
                {"available_after_startup_bytes", memory.available_after_startup_bytes},
                {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes},
                {"host_kv_capacity_bytes", memory.host_kv_capacity_bytes},
                {"host_state_capacity_slots", memory.host_state_capacity_slots},
                {"weights", arena_json(memory.weights)},
                {"sequence", arena_json(memory.sequence)},
                {"workspace", arena_json(memory.workspace)},
                {"vision_workspace", vision_workspace_json(memory.vision_workspace)}};
}

Json transfer_json(std::uint64_t first, std::uint64_t second, double seconds, const char* first_key,
                   const char* second_key) {
    return Json{{first_key, first}, {second_key, second}, {"seconds", seconds}};
}

} // namespace

std::string format_stats_json(const StatsSnapshot& snapshot) {
    const ninfer::RuntimeStats& stats = snapshot.stats;
    const ninfer::LoadSummary& load   = snapshot.load;

    const Json out = Json{
        {"schema", "ninfer_serve_stats"},
        {"schema_version", 1},
        {"timestamp_unix_ms", snapshot.timestamp_unix_ms},
        {"stats_published_at_unix_ms", snapshot.stats_published_at_unix_ms},
        {"available", snapshot.available},
        {"server_instance_id", snapshot.server_instance_id},
        // Current gauges of the scheduler as of the publication boundary.
        {"scheduler", Json{{"running", stats.running_requests},
                           {"prefilling", stats.prefilling_requests},
                           {"decode_ready", stats.decode_ready_requests},
                           {"waiting", stats.waiting_requests},
                           {"materializing", stats.materializing_requests},
                           {"capture_pending", stats.capture_pending_requests},
                           {"terminal_pending", stats.terminal_pending_requests}}},
        // Monotonic counters since process start. Consumers derive interval rates by subtracting
        // two snapshots; the 5s throughput records remain the aggregation source for rates.
        {"counters", Json{{"computed_prefill_tokens", stats.computed_prefill_tokens},
                          {"committed_decode_tokens", stats.committed_decode_tokens},
                          {"decode_rounds", stats.decode_rounds},
                          {"decode_row_rounds", stats.decode_row_rounds},
                          {"active_captures_completed", stats.active_captures_completed},
                          {"active_captures_aborted", stats.active_captures_aborted}}},
        {"http",
         Json{{"in_flight", snapshot.in_flight}, {"max_in_flight", snapshot.max_in_flight}}},
        {"kv_transfers",
         Json{{"main_kv",
               Json{{"d2h", transfer_json(stats.main_kv_d2h_pages, stats.main_kv_d2h_bytes,
                                          stats.main_kv_d2h_seconds, "pages", "bytes")},
                    {"h2d", transfer_json(stats.main_kv_h2d_pages, stats.main_kv_h2d_bytes,
                                          stats.main_kv_h2d_seconds, "pages", "bytes")},
                    {"d2d", transfer_json(stats.main_kv_d2d_pages, stats.main_kv_d2d_bytes,
                                          stats.main_kv_d2d_seconds, "pages", "bytes")}}},
              {"backend_kv",
               Json{{"d2h", transfer_json(stats.backend_kv_d2h_pages, stats.backend_kv_d2h_bytes,
                                          stats.backend_kv_d2h_seconds, "pages", "bytes")},
                    {"h2d", transfer_json(stats.backend_kv_h2d_pages, stats.backend_kv_h2d_bytes,
                                          stats.backend_kv_h2d_seconds, "pages", "bytes")},
                    {"d2d", transfer_json(stats.backend_kv_d2d_pages, stats.backend_kv_d2d_bytes,
                                          stats.backend_kv_d2d_seconds, "pages", "bytes")}}}}},
        {"state_transfers",
         Json{{"d2h", transfer_json(stats.state_d2h_count, stats.state_d2h_bytes,
                                    stats.state_d2h_seconds, "count", "bytes")},
              {"h2d", transfer_json(stats.state_h2d_count, stats.state_h2d_bytes,
                                    stats.state_h2d_seconds, "count", "bytes")},
              {"d2d", transfer_json(stats.state_d2d_count, stats.state_d2d_bytes,
                                    stats.state_d2d_seconds, "count", "bytes")}}},
        {"state_operations", Json{{"moves", stats.state_moves},
                                  {"forks", stats.state_forks},
                                  {"restores", stats.state_restores}}},
        {"pressure",
         Json{{"spill_pages", stats.pressure_spill_pages},
              {"partial_tail_cow_pages", stats.partial_tail_cow_pages},
              {"private_owners_degraded", stats.pressure_private_owners_degraded},
              {"private_owners_evicted", stats.pressure_private_owners_evicted},
              {"shared_owners_degraded", stats.pressure_shared_owners_degraded},
              {"shared_owners_evicted", stats.pressure_shared_owners_evicted},
              {"checkpoints_dropped", stats.pressure_checkpoints_dropped},
              {"searches", stats.pressure_searches},
              {"search_budget_exhaustions", stats.pressure_search_budget_exhaustions},
              {"maximal_fallback_selections", stats.pressure_maximal_fallback_selections},
              {"historical_fork_hits", stats.historical_fork_hits}}},
        {"cache_reuse",
         Json{{"root", stats.root_selections},
              {"private_endpoint", stats.private_endpoint_selections},
              {"private_turn_closure", stats.private_turn_closure_selections},
              {"private_response_replay", stats.private_response_replay_selections},
              {"private_long_anchor", stats.private_long_anchor_selections},
              {"shared_stable_prefix", stats.shared_stable_prefix_selections},
              {"reused_prompt_tokens", stats.reused_prompt_tokens},
              {"last_selected_frontier_tokens", stats.last_selected_frontier_tokens}}},
        {"occupancy", Json{{"device_state_slots", stats.device_state_occupied_slots},
                           {"host_state_slots", stats.host_state_occupied_slots},
                           {"device_main_kv_pages", stats.device_main_kv_occupied_pages},
                           {"device_backend_kv_pages", stats.device_backend_kv_occupied_pages},
                           {"host_kv_bytes", stats.host_kv_occupied_bytes},
                           {"shared_active_references", stats.shared_active_references}}},
        {"actual_context_transfer_seconds", stats.actual_context_transfer_seconds},
        {"load", Json{{"model_id", load.model_id},
                      {"target", load.target},
                      {"weights_id", load.weights_id},
                      {"load_seconds", load.load_seconds},
                      {"upload_seconds", load.upload_seconds},
                      {"artifact_bytes_read", load.artifact_bytes_read},
                      {"host_to_device_bytes", load.host_to_device_bytes},
                      {"peak_staging_bytes", load.peak_staging_bytes},
                      {"tensor_count", load.tensor_count},
                      {"resource_count", load.resource_count}}},
        // Capacities and arenas are fixed after startup; occupancy gauges live under "occupancy".
        // Null when the caller skipped the live read with ?memory=0.
        {"memory", memory_json(snapshot.memory)},
    };
    return out.dump();
}

} // namespace ninfer::serve
