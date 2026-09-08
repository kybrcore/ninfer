#include "serve/froggeric_v225_request.h"

#include "serve/request_validation.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

// Parse-time view: every field optional so absent and explicit null stay distinguishable.
struct FroggericV225TemplateOptions {
    std::optional<bool> preserve_reasoning;
    std::optional<bool> auto_disable_thinking_with_tools;
    std::optional<bool> json_tool_format;
    std::optional<std::uint32_t> max_tool_arg_chars;
    std::optional<std::uint32_t> max_tool_response_chars;
};

[[nodiscard]] bool is_froggeric_v225_template_key(std::string_view key) noexcept {
    return key == "preserve_reasoning" || key == "auto_disable_thinking_with_tools" ||
           key == "tool_call_format" || key == "max_tool_arg_chars" ||
           key == "max_tool_response_chars";
}

FroggericV225TemplateOptions parse_froggeric_v225_template_options(const RequestJson& kwargs) {
    FroggericV225TemplateOptions out;
    if (kwargs.contains("preserve_reasoning") && !kwargs.at("preserve_reasoning").is_null()) {
        if (!kwargs.at("preserve_reasoning").is_boolean()) {
            bad_request("chat_template_kwargs.preserve_reasoning must be a boolean or null",
                        "chat_template_kwargs");
        }
        out.preserve_reasoning = kwargs.at("preserve_reasoning").get<bool>();
    }
    if (kwargs.contains("auto_disable_thinking_with_tools") &&
        !kwargs.at("auto_disable_thinking_with_tools").is_null()) {
        if (!kwargs.at("auto_disable_thinking_with_tools").is_boolean()) {
            bad_request("chat_template_kwargs.auto_disable_thinking_with_tools must be a boolean",
                        "chat_template_kwargs");
        }
        out.auto_disable_thinking_with_tools =
            kwargs.at("auto_disable_thinking_with_tools").get<bool>();
    }
    if (kwargs.contains("tool_call_format") && !kwargs.at("tool_call_format").is_null()) {
        const RequestJson& format = kwargs.at("tool_call_format");
        if (!format.is_string()) {
            bad_request("chat_template_kwargs.tool_call_format must be \"xml\" or \"json\"",
                        "chat_template_kwargs");
        }
        const std::string value = format.get<std::string>();
        if (value == "xml") {
            out.json_tool_format = false;
        } else if (value == "json") {
            out.json_tool_format = true;
        } else {
            bad_request("chat_template_kwargs.tool_call_format must be \"xml\" or \"json\"",
                        "chat_template_kwargs");
        }
    }
    for (const char* key : {"max_tool_arg_chars", "max_tool_response_chars"}) {
        if (!kwargs.contains(key) || kwargs.at(key).is_null()) { continue; }
        const RequestJson& value = kwargs.at(key);
        if (!value.is_number_integer()) {
            bad_request(std::string("chat_template_kwargs.") + key +
                            " must be a non-negative integer",
                        "chat_template_kwargs");
        }
        // Parsed JSON stores non-negative integers as unsigned, but callers may also build
        // the object programmatically; accept both representations and reject negatives.
        std::uint64_t parsed = 0;
        if (value.is_number_unsigned()) {
            parsed = value.get<std::uint64_t>();
        } else {
            const std::int64_t signed_value = value.get<std::int64_t>();
            if (signed_value < 0) {
                bad_request(std::string("chat_template_kwargs.") + key +
                                " must be a non-negative integer",
                            "chat_template_kwargs");
            }
            parsed = static_cast<std::uint64_t>(signed_value);
        }
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            bad_request(std::string("chat_template_kwargs.") + key + " exceeds uint32",
                        "chat_template_kwargs");
        }
        if (std::string(key) == "max_tool_arg_chars") {
            out.max_tool_arg_chars = static_cast<std::uint32_t>(parsed);
        } else {
            out.max_tool_response_chars = static_cast<std::uint32_t>(parsed);
        }
    }
    return out;
}

// The compiled renderer prefers preserve_reasoning, so a conflicting explicit pair is a 400.
void reject_conflicting_preserve_options(const std::optional<bool>& preserve_reasoning,
                                         const std::optional<bool>& preserve_thinking) {
    if (preserve_reasoning && preserve_thinking && *preserve_reasoning != *preserve_thinking) {
        bad_request("conflicting preserve_reasoning and preserve_thinking values",
                    "chat_template_kwargs", "conflicting_template_option");
    }
}

} // namespace

ninfer::FroggericV225Options decode_froggeric_v225_kwargs(
    ninfer::ChatStyle style, const RequestJson& kwargs, bool accept_enable_thinking,
    const std::optional<bool>& merged_preserve_thinking) {
    for (auto iterator = kwargs.begin(); iterator != kwargs.end(); ++iterator) {
        const bool is_base = iterator.key() == "preserve_thinking" ||
                             (accept_enable_thinking && iterator.key() == "enable_thinking");
        const bool is_v225 = is_froggeric_v225_template_key(iterator.key());
        // Null values are neutral: accepted for unknown keys (matching the pre-existing
        // behavior) and for known keys alike; only non-null values are validated.
        const bool accepted = iterator.value().is_null() || is_base ||
                              (is_v225 && style == ninfer::ChatStyle::FroggericV225);
        if (!accepted) {
            if (is_v225 && !iterator.value().is_null()) {
                bad_request("chat_template_kwargs." + iterator.key() +
                                " is only supported with --chat-style froggeric-v22.5",
                            "chat_template_kwargs", "chat_template_option_not_supported");
            }
            bad_request("chat_template_kwargs." + iterator.key() + " is not supported",
                        "chat_template_kwargs", "chat_template_option_not_supported");
        }
    }
    if (style != ninfer::ChatStyle::FroggericV225) { return {}; }

    const FroggericV225TemplateOptions v225 = parse_froggeric_v225_template_options(kwargs);
    ninfer::FroggericV225Options out;
    out.preserve_reasoning = v225.preserve_reasoning;
    out.auto_disable_thinking_with_tools =
        v225.auto_disable_thinking_with_tools.value_or(false);
    if (v225.json_tool_format.value_or(false)) {
        out.tool_call_format = ninfer::ToolCallFormat::Json;
    }
    out.max_tool_arg_chars      = v225.max_tool_arg_chars.value_or(0);
    out.max_tool_response_chars = v225.max_tool_response_chars.value_or(0);
    reject_conflicting_preserve_options(out.preserve_reasoning, merged_preserve_thinking);
    return out;
}

std::optional<ninfer::ReasoningEffort> froggeric_v225_effort_alias(
    ninfer::ChatStyle style, RequestedReasoningEffort requested) {
    if (style != ninfer::ChatStyle::FroggericV225) { return std::nullopt; }
    switch (requested) {
    case RequestedReasoningEffort::Minimal: return ninfer::ReasoningEffort::Low;
    case RequestedReasoningEffort::High:
    case RequestedReasoningEffort::Max: return ninfer::ReasoningEffort::XHigh;
    default: return std::nullopt;
    }
}

} // namespace ninfer::serve
