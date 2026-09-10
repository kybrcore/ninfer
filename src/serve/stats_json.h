#pragma once

// Read-only JSON rendering for GET /stats. HttpServer fills the snapshot from values the Engine
// already exposes; this header owns serialization only and never reaches into engine internals.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::serve {

struct StatsSnapshot {
    // Time the response was rendered (never earlier than stats_published_at_unix_ms).
    std::uint64_t timestamp_unix_ms = 0;
    // Boundary the counters, gauges, and memory below describe. Zero until the worker publishes
    // for the first time.
    std::uint64_t stats_published_at_unix_ms = 0;
    // Engine health verdict, the same one /health turns into a status code.
    bool available = false;
    std::string server_instance_id;
    // Counters, gauges, and memory from one worker boundary, so no two fields describe different
    // instants and no read waits on the execution lock.
    ninfer::RuntimeStats stats;
    // Absent when the caller trims the payload with ?memory=0.
    std::optional<ninfer::MemorySummary> memory;
    ninfer::LoadSummary load;
    // HTTP-layer admission depth: read live on the request, independent of the boundary snapshot.
    std::size_t in_flight     = 0;
    std::size_t max_in_flight = 0;
};

// One complete JSON object without a trailing newline. Field names follow RuntimeStats,
// MemorySummary, and the structured request log so operators can line the endpoint up with what
// the 5s throughput records already report. Every field except `http`, `load`, and
// `timestamp_unix_ms` comes from the same worker boundary.
[[nodiscard]] std::string format_stats_json(const StatsSnapshot& snapshot);

} // namespace ninfer::serve
