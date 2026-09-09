#pragma once

#include <cstdint>
#include <optional>

namespace ninfer {

// Tool-call history/instruction serialization for the compiled chat renderers. XML is the
// default; JSON is a non-strict alternate format (no JSON Schema guarantees).
enum class ToolCallFormat : std::uint8_t {
    Xml  = 0,
    Json,
};

// froggeric-v22.5 request surface. The Artifact renderer deliberately ignores the whole struct,
// so the defaults keep every existing prompt byte-identical. Kept in its own header so a new
// v22.5 option never touches the core types.h / request.h / chat_template.h seam.
struct FroggericV225Options {
    // Alias of preserve_thinking; when set, conflicts with preserve_thinking are an error.
    std::optional<bool> preserve_reasoning;
    bool auto_disable_thinking_with_tools = false;
    ToolCallFormat tool_call_format       = ToolCallFormat::Xml;
    std::uint32_t max_tool_arg_chars      = 0;
    std::uint32_t max_tool_response_chars = 0;
};

} // namespace ninfer
