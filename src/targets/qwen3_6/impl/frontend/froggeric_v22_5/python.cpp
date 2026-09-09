#include "targets/qwen3_6/impl/frontend/froggeric_v22_5/python.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using njson = OracleJson;

std::string hex4(std::uint32_t value) {
    constexpr char digits[] = "0123456789abcdef";
    return std::string{digits[(value >> 12) & 0xF], digits[(value >> 8) & 0xF],
                        digits[(value >> 4) & 0xF], digits[value & 0xF]};
}

std::string py_escape_string(std::string_view text) {
    std::string out;
    std::size_t index = 0;
    while (index < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[index]);
        switch (c) {
        case '"': out += "\\\""; ++index; continue;
        case '\\': out += "\\\\"; ++index; continue;
        case '\b': out += "\\b"; ++index; continue;
        case '\t': out += "\\t"; ++index; continue;
        case '\n': out += "\\n"; ++index; continue;
        case '\f': out += "\\f"; ++index; continue;
        case '\r': out += "\\r"; ++index; continue;
        // jinja's htmlsafe pass, emitted in one pass instead of four whole-string rewrites.
        case '<': out += "\\u003c"; ++index; continue;
        case '>': out += "\\u003e"; ++index; continue;
        case '&': out += "\\u0026"; ++index; continue;
        case '\'': out += "\\u0027"; ++index; continue;
        default: break;
        }
        // Python json.dumps(ensure_ascii=True) escapes C0 controls and DEL (U+007F).
        if (c < 0x20U || c == 0x7FU) { out += "\\u00" + hex4(c).substr(2); ++index; continue; }
        if (c < 0x80U) { out += static_cast<char>(c); ++index; continue; }
        std::size_t next = index;
        const std::uint32_t cp = utf8_next(text, next);
        if (cp <= 0xFFFFU) {
            out += "\\u" + hex4(cp);
        } else {
            const std::uint32_t value = cp - 0x10000U;
            out += "\\u" + hex4(0xD800U + (value >> 10)) + "\\u" + hex4(0xDC00U + (value & 0x3FFU));
        }
        index = next;
    }
    return out;
}

} // namespace

std::uint32_t utf8_next(std::string_view text, std::size_t& index) {
    const auto byte = [](std::string_view text, std::size_t i) {
        return static_cast<std::uint8_t>(text[i]);
    };
    const std::uint8_t first = byte(text, index);
    if (first < 0x80U) { ++index; return first; }
    const int extra = first < 0xE0U ? 1 : first < 0xF0U ? 2 : 3;
    if (index + static_cast<std::size_t>(extra) + 1U > text.size()) {
        throw std::logic_error("froggeric v22.5: malformed UTF-8 in prompt text");
    }
    std::uint32_t value = first & (0x7FU >> (extra + 1));
    for (int i = 1; i <= extra; ++i) {
        const std::uint8_t continuation = byte(text, index + static_cast<std::size_t>(i));
        if ((continuation & 0xC0U) != 0x80U) {
            throw std::logic_error("froggeric v22.5: malformed UTF-8 in prompt text");
        }
        value = (value << 6) | (continuation & 0x3FU);
    }
    index += static_cast<std::size_t>(extra) + 1U;
    return value;
}

// Python str.isspace() (Unicode White_Space; U+FEFF is not whitespace in Python).
bool py_isspace(std::uint32_t cp) noexcept {
    return cp == 0x09U || cp == 0x0AU || cp == 0x0BU || cp == 0x0CU || cp == 0x0DU ||
           (cp >= 0x1CU && cp <= 0x1FU) || cp == 0x20U || cp == 0x85U || cp == 0xA0U ||
           cp == 0x1680U || (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U || cp == 0x2029U ||
           cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}

std::size_t py_strip_begin(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size()) {
        std::size_t next = index;
        if (!py_isspace(utf8_next(text, next))) { break; }
        index = next;
    }
    return index;
}

std::size_t py_strip_end(std::string_view text) {
    std::size_t end = text.size();
    while (end > 0) {
        // Walk back over continuation bytes to the lead byte so 3- and 4-byte codepoints
        // (for example U+3000 or U+1F600) are decoded whole.
        std::size_t start = end - 1;
        while (start > 0 && (static_cast<std::uint8_t>(text[start]) & 0xC0U) == 0x80U) { --start; }
        std::size_t next       = start;
        const std::uint32_t cp = utf8_next(text, next);
        if (next != end) {
            throw std::logic_error("froggeric v22.5: malformed UTF-8 in prompt text");
        }
        if (!py_isspace(cp)) { return end; }
        end = start;
    }
    return end;
}

std::pair<std::size_t, std::size_t> py_trim_bounds(std::string_view text) {
    const std::size_t begin = py_strip_begin(text);
    const std::size_t end   = py_strip_end(text);
    // Python's str.strip() of an all-whitespace string is empty. py_strip_begin returns the end
    // of the text and py_strip_end returns zero in that case, so report the empty range instead
    // of an inverted one: every caller uses text.substr(begin, end - begin) or
    // slice_fragment(fragment, begin, end) and would otherwise throw or erase out of bounds.
    if (begin > end) { return {0, 0}; }
    return {begin, end};
}

std::size_t py_len(std::string_view text) {
    std::size_t count = 0, index = 0;
    while (index < text.size()) { utf8_next(text, index); ++count; }
    return count;
}

// Python text[:end_cp] / text[begin_cp:end_cp] in codepoints.
std::string py_slice(std::string_view text, std::size_t begin_cp, std::optional<std::size_t> end_cp) {
    std::size_t index = 0, seen = 0;
    while (seen < begin_cp && index < text.size()) { utf8_next(text, index); ++seen; }
    const std::size_t start = index;
    if (!end_cp.has_value()) { return std::string(text.substr(start)); }
    while (seen < *end_cp && index < text.size()) { utf8_next(text, index); ++seen; }
    return std::string(text.substr(start, index - start));
}

std::string py_lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return out;
}

std::string truncation_notice(std::string_view text) {
    return "\n[TRUNCATED - original length " + std::to_string(py_len(text)) + " chars]";
}

std::size_t py_prefix_bytes(std::string_view text, std::size_t max_chars) {
    std::size_t index = 0, seen = 0;
    while (seen < max_chars && index < text.size()) { utf8_next(text, index); ++seen; }
    return index;
}

std::string truncate_python(std::string_view text, std::uint32_t max_chars) {
    return py_slice(text, 0, max_chars) + truncation_notice(text);
}

// ---------------------------------------------------------------------------

std::string tojson_oracle(const njson& value) {
    std::string out;
    if (value.is_null()) {
        out = "null";
    } else if (value.is_boolean()) {
        out = value.get<bool>() ? "true" : "false";
    } else if (value.is_number_integer() || value.is_number_unsigned()) {
        out = value.dump();
    } else if (value.is_number_float()) {
        out = value.dump();
        if (out.find_first_of(".eE") == std::string::npos) { out += ".0"; }
        if (out == "-0") { out = "-0.0"; }
    } else if (value.is_string()) {
        out = "\"" + py_escape_string(value.get_ref<const std::string&>()) + "\"";
    } else if (value.is_array()) {
        out = "[";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i != 0) { out += ", "; }
            out += tojson_oracle(value[i]);
        }
        out += "]";
    } else { // object, sort_keys=True
        std::vector<std::pair<std::string, const njson*>> entries;
        for (auto it = value.begin(); it != value.end(); ++it) {
            entries.emplace_back(it.key(), &it.value());
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        out = "{";
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (i != 0) { out += ", "; }
            out += "\"" + py_escape_string(entries[i].first) + "\": ";
            out += tojson_oracle(*entries[i].second);
        }
        out += "}";
    }
    return out;
}

bool contains(std::string_view text, std::string_view needle) noexcept {
    return text.find(needle) != std::string_view::npos;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
