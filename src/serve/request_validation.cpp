#include "serve/request_validation.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

bool is_froggeric_v225_template_key(std::string_view key) noexcept {
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

void reject_conflicting_preserve_options(const std::optional<bool>& preserve_reasoning,
                                         const std::optional<bool>& preserve_thinking) {
    if (preserve_reasoning && preserve_thinking && *preserve_reasoning != *preserve_thinking) {
        bad_request("conflicting preserve_reasoning and preserve_thinking values",
                    "chat_template_kwargs", "conflicting_template_option");
    }
}

} // namespace ninfer::serve
