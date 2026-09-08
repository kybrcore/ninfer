#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// froggeric-v22.5-only chat_template_kwargs. Callers gate the engine style and the kwargs
// whitelist; this helper only validates and decodes the values.
struct FroggericV225TemplateOptions {
    std::optional<bool> preserve_reasoning;
    std::optional<bool> auto_disable_thinking_with_tools;
    std::optional<bool> json_tool_format;
    std::optional<std::uint32_t> max_tool_arg_chars;
    std::optional<std::uint32_t> max_tool_response_chars;
};

[[nodiscard]] bool is_froggeric_v225_template_key(std::string_view key) noexcept;

[[nodiscard]] FroggericV225TemplateOptions
parse_froggeric_v225_template_options(const RequestJson& kwargs);

// The compiled renderer prefers preserve_reasoning, so a conflicting explicit pair is a 400.
void reject_conflicting_preserve_options(const std::optional<bool>& preserve_reasoning,
                                         const std::optional<bool>& preserve_thinking);

} // namespace ninfer::serve
