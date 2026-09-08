#pragma once

#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <cstddef>
#include <string>

namespace ninfer::targets::qwen3_6::frontend_internal {

// JSON-format counterpart of parse_qwen_tool_call_output, selected when
// ToolCallOutputContract::json_format is set (froggeric v22.5 requests only). It lives in its own
// module so tool_call_parser.cpp carries the upstream XML path with a single dispatch line.
[[nodiscard]] ParsedToolCallOutput parse_json_tool_call_output(const std::string& text,
                                                              std::size_t max_tool_name_length,
                                                              const ToolCallOutputContract& contract);

} // namespace ninfer::targets::qwen3_6::frontend_internal
