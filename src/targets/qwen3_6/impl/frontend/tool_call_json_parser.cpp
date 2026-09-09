#include "targets/qwen3_6/impl/frontend/tool_call_json_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Json           = nlohmann::json;
using Contract       = ToolCallOutputContract;
using FallbackReason = ToolCallParseFallbackReason;

// Protocol markers; must stay in sync with kToolOpen/kToolClose in tool_call_parser.cpp.
constexpr std::string_view kToolOpen  = "<tool_call>";
constexpr std::string_view kToolClose = "</tool_call>";

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

} // namespace

// froggeric v22.5 JSON format: each <tool_call> block carries a single JSON object
// {"name": ..., "arguments": ...}; consecutive blocks are separated by format whitespace.
// The OpenAI wrapper {"function": {"name": ..., "arguments": ...}} is accepted and normalized
// to the same call; the native shape wins when both are present. A call without arguments
// carries "{}" like the XML path rather than an empty string.
ParsedToolCallOutput parse_json_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract) {
    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content                 = rtrim_format_whitespace(std::string_view(text).substr(0, first));
    out.diagnostics.marker_seen = true;

    struct JsonCall {
        std::string name;
        std::string arguments;
    };
    std::vector<JsonCall> calls;
    std::size_t cursor = first;
    while (cursor < text.size()) {
        const std::size_t open = text.find(kToolOpen, cursor);
        if (open == std::string::npos) { break; }
        const std::size_t body_begin = open + kToolOpen.size();
        const std::size_t close      = text.find(kToolClose, body_begin);
        if (close == std::string::npos) {
            out.diagnostics.fallback_reason = FallbackReason::MalformedStructure;
            return fallback(text, out.diagnostics);
        }
        std::string_view body = std::string_view(text).substr(body_begin, close - body_begin);
        const auto body_trim_begin = body.find_first_not_of(" \t\r\n");
        if (body_trim_begin == std::string_view::npos) {
            out.diagnostics.fallback_reason = FallbackReason::MalformedStructure;
            return fallback(text, out.diagnostics);
        }
        body.remove_prefix(body_trim_begin);
        const auto body_trim_end = body.find_last_not_of(" \t\r\n");
        body.remove_suffix(body.size() - (body_trim_end + 1));

        const Json parsed = Json::parse(body, nullptr, false);
        if (!parsed.is_object()) {
            out.diagnostics.fallback_reason = FallbackReason::MalformedStructure;
            return fallback(text, out.diagnostics);
        }
        const Json* call_object = &parsed;
        if (!parsed.contains("name")) {
            const auto function = parsed.find("function");
            if (function != parsed.end()) {
                if (!function->is_object()) {
                    out.diagnostics.fallback_reason = FallbackReason::MalformedStructure;
                    return fallback(text, out.diagnostics);
                }
                call_object = &*function;
            }
        }
        const Json& name = call_object->contains("name") ? call_object->at("name") : Json{};
        if (!name.is_string() || name.get_ref<const std::string&>().empty() ||
            name.get_ref<const std::string&>().size() > max_tool_name_length) {
            out.diagnostics.fallback_reason = FallbackReason::InvalidToolName;
            return fallback(text, out.diagnostics);
        }
        JsonCall call;
        call.name      = name.get_ref<const std::string&>();
        call.arguments = "{}";
        if (call_object->contains("arguments")) {
            const Json& arguments = call_object->at("arguments");
            if (arguments.is_string()) {
                call.arguments = arguments.get<std::string>();
            } else if (!arguments.is_null()) {
                call.arguments = arguments.dump();
            }
        }
        calls.push_back(std::move(call));
        cursor = close + kToolClose.size();
        std::size_t gap = cursor;
        while (gap < text.size() && is_format_whitespace(text[gap])) { ++gap; }
        if (gap >= text.size()) { break; }
        if (text.compare(gap, kToolOpen.size(), kToolOpen) != 0) {
            out.diagnostics.fallback_reason = FallbackReason::TrailingContent;
            return fallback(text, out.diagnostics);
        }
        cursor = gap;
    }
    if (calls.empty()) {
        out.diagnostics.fallback_reason = FallbackReason::MalformedStructure;
        return fallback(text, out.diagnostics);
    }

    out.tool_calls.reserve(calls.size());
    for (JsonCall& call : calls) {
        if (contract.enforce_declared_names &&
            find_tool_contract(contract, call.name) == nullptr) {
            out.diagnostics.fallback_reason = FallbackReason::UndeclaredTool;
            return fallback(text, out.diagnostics);
        }
        out.tool_calls.push_back(
            GeneratedToolCall{.name = std::move(call.name), .arguments_json = std::move(call.arguments)});
    }
    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
    return out;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
