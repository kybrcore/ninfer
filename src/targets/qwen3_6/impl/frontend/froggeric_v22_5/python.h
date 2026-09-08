#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {

using OracleJson = nlohmann::ordered_json;

bool contains(std::string_view text, std::string_view needle) noexcept;

// Simulated Python semantics (Python 3.11). This is the single authoritative statement of the
// emulation scope; the user-facing deviations live in docs/serving.md "Known boundaries".
//   - str: codepoint-based len/slice, Unicode White_Space trim, ASCII lower (the template only
//     lowercases ASCII tool-format tokens);
//   - json.dumps(ensure_ascii=True, sort_keys=True) with default separators, then jinja's
//     htmlsafe pass escaping < > & ' as \u003c \u003e \u0026 \u0027 in the same single pass;
//   - known deviations: integers beyond uint64 lose precision (the JSON parser stores them as
//     doubles); a few float spellings differ from Python's repr; malformed UTF-8 is rejected
//     instead of decoded with surrogatepass.
// Kept separate from the renderer so every helper can be unit-tested against the pinned oracle.
std::uint32_t utf8_next(std::string_view text, std::size_t& index);
bool py_isspace(std::uint32_t codepoint) noexcept;
std::size_t py_strip_begin(std::string_view text);
std::size_t py_strip_end(std::string_view text);
// Python str.strip() byte bounds: text.substr(begin, end - begin) is the stripped text. An
// all-whitespace input reports the empty range {0, 0} so begin <= end always holds.
std::pair<std::size_t, std::size_t> py_trim_bounds(std::string_view text);
std::size_t py_len(std::string_view text);
std::string py_slice(std::string_view text, std::size_t begin_cp, std::optional<std::size_t> end_cp);
std::string py_lower(std::string_view text);
std::size_t py_prefix_bytes(std::string_view text, std::size_t max_chars);
std::string truncation_notice(std::string_view text);
std::string truncate_python(std::string_view text, std::uint32_t max_chars);

// jinja2 3.1.6 `| tojson`: json.dumps(ensure_ascii=True, sort_keys=True) with default
// separators, then htmlsafe_json_dumps escaping < > & '.
std::string tojson_oracle(const OracleJson& value);

} // namespace ninfer::targets::qwen3_6::frontend_internal
