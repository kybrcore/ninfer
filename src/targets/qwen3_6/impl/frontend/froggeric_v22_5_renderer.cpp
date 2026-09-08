// Compiled qwen3.8-froggeric-v22.5 chat renderer.
//
// Byte parity is defined against the pinned upstream Jinja oracle:
//   froggeric/Qwen-Fixed-Chat-Templates @ 855bffc49448e299789730ff92c9b8d834d6cc14
// (chat_template.jinja / chat_template_oneline.txt). See
// tests/fixtures/frontend/froggeric_v22_5/PROVENANCE.md and tools/oracle_froggeric_v22_5/.
// Where Python string semantics matter (codepoint slices, unicode whitespace, tojson key
// sorting / ensure_ascii / html-safe escaping) they are mirrored exactly here.

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using njson = nlohmann::ordered_json;


void append_span(std::vector<ByteSpan>& spans, ByteSpan span) {
    if (span.begin == span.end) { return; }
    if (!spans.empty() && spans.back().end == span.begin) {
        spans.back().end = span.end;
        return;
    }
    if (!spans.empty() && spans.back().end > span.begin) {
        throw std::logic_error("froggeric v22.5: rendered literal byte spans overlap");
    }
    spans.push_back(span);
}

// ---------------------------------------------------------------------------
// UTF-8 / Python string semantics
// ---------------------------------------------------------------------------

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
        const std::uint8_t last = static_cast<std::uint8_t>(text[end - 1]);
        const int codepoint_bytes = last < 0x80U ? 1 : last < 0xE0U ? 2 : last < 0xF0U ? 3 : 4;
        if (static_cast<std::size_t>(codepoint_bytes) > end) {
            throw std::logic_error("froggeric v22.5: malformed UTF-8 in prompt text");
        }
        std::size_t next = 0;
        const std::uint32_t cp = utf8_next(text.substr(end - static_cast<std::size_t>(codepoint_bytes)),
                                           next);
        if (!py_isspace(cp)) { return end; }
        end -= static_cast<std::size_t>(codepoint_bytes);
    }
    return end;
}

std::pair<std::size_t, std::size_t> py_trim_bounds(std::string_view text) {
    return {py_strip_begin(text), py_strip_end(text)};
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
    if (end_cp.has_value()) {
        while (seen < *end_cp && index < text.size()) { utf8_next(text, index); ++seen; }
    }
    return std::string(text.substr(start, index - start));
}

std::string py_lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return out;
}

std::string truncate_python(std::string_view text, std::uint32_t max_chars) {
    return py_slice(text, 0, max_chars) + "\n[TRUNCATED - original length " +
           std::to_string(py_len(text)) + " chars]";
}

// ---------------------------------------------------------------------------
// Jinja `| tojson` parity (jinja2 3.1.6: json.dumps with ensure_ascii=True,
// sort_keys=True, default separators; then htmlsafe_json_dumps escapes
// < > & ' as \uXXXX).
// ---------------------------------------------------------------------------

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
        default: break;
        }
        if (c < 0x20U) { out += "\\u00" + hex4(c).substr(2); ++index; continue; }
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
    for (const auto& [from, to] :
         std::vector<std::pair<std::string, std::string>>{
             {std::string("<"), std::string("\\u003c")},
             {std::string(">"), std::string("\\u003e")},
             {std::string("&"), std::string("\\u0026")},
             {std::string("'"), std::string("\\u0027")}}) {
        std::string replaced;
        std::size_t index = 0;
        while (true) {
            const std::size_t found = out.find(from, index);
            if (found == std::string::npos) { replaced += out.substr(index); break; }
            replaced += out.substr(index, found - index) + to;
            index = found + from.size();
        }
        out = std::move(replaced);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Inline control tags
// ---------------------------------------------------------------------------

constexpr std::string_view kThinkOff     = "<|think_off|>";
constexpr std::string_view kThinkOn      = "<|think_on|>";
constexpr std::string_view kThinkXHigh   = "<|think_xhigh|>";
constexpr std::string_view kThinkHigh    = "<|think_high|>";
constexpr std::string_view kThinkUltra   = "<|think_ultracode|>";
constexpr std::string_view kThinkExtreme = "<|think_extreme|>";
constexpr std::string_view kThinkMax     = "<|think_max|>";
constexpr std::string_view kThinkLow     = "<|think_low|>";
constexpr std::string_view kThinkMinimal = "<|think_minimal|>";
constexpr std::string_view kThinkMedium  = "<|think_medium|>";

bool contains(std::string_view text, std::string_view needle) noexcept {
    return text.find(needle) != std::string_view::npos;
}

// Template pre-scan: one if/elif chain per text item, in item order.
void apply_tag_state(const std::string& text, bool& thinking, ReasoningEffort& effort) {
    if (text.empty()) { return; }
    if (contains(text, kThinkOff)) {
        thinking = false;
    } else if (contains(text, kThinkOn)) {
        thinking = true;
    } else if (contains(text, kThinkXHigh) || contains(text, kThinkHigh) ||
               contains(text, kThinkUltra) || contains(text, kThinkExtreme) ||
               contains(text, kThinkMax)) {
        thinking = true;
        effort   = ReasoningEffort::XHigh;
    } else if (contains(text, kThinkLow) || contains(text, kThinkMinimal)) {
        thinking = true;
        effort   = ReasoningEffort::Low;
    } else if (contains(text, kThinkMedium)) {
        thinking = true;
        effort   = ReasoningEffort::Medium;
    }
}

// Main-loop stripping: for each of the ten tags, if present, erase all occurrences and
// re-trim (jinja: content.split(tag) | join('') | trim).
std::string strip_inline_tags(const std::string& rendered) {
    std::string content = rendered;
    const std::string_view tags[] = {kThinkOff,     kThinkOn,      kThinkXHigh,
                                     kThinkHigh,    kThinkUltra,   kThinkExtreme,
                                     kThinkMax,     kThinkLow,     kThinkMinimal,
                                     kThinkMedium};
    for (const std::string_view tag : tags) {
        if (!contains(content, tag)) { continue; }
        std::string stripped;
        std::size_t index = 0;
        while (true) {
            const std::size_t found = content.find(tag, index);
            if (found == std::string::npos) { stripped += content.substr(index); break; }
            stripped += content.substr(index, found - index);
            index = found + tag.size();
        }
        const auto [begin, end] = py_trim_bounds(stripped);
        content.assign(stripped.substr(begin, end - begin));
    }
    return content;
}

std::string_view reasoning_instructions_for(bool thinking, ReasoningEffort effort) noexcept {
    if (thinking && effort == ReasoningEffort::Low) {
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
               "directly to the conclusion without unnecessary elaboration.";
    }
    if (thinking && effort == ReasoningEffort::XHigh) {
        return "Reasoning effort is set to xhigh. Please think carefully through the task, "
               "validate key assumptions, consider plausible alternatives, and prioritize "
               "correctness, consistency, and clarity in the final answer.";
    }
    return "";
}

// ---------------------------------------------------------------------------
// Prompt assembly
// ---------------------------------------------------------------------------

struct PromptBuilder {
    std::string text;
    std::vector<ByteSpan> literals;
    std::vector<MediaPlaceholderByteSpec> media;
    std::vector<std::size_t> tool_bounds;
    std::vector<std::size_t> execution_boundaries;

    void append_template(std::string_view chunk) { text += chunk; }

    void append_literal(std::string_view chunk) {
        if (chunk.empty()) { return; }
        const std::size_t begin = text.size();
        text += chunk;
        append_span(literals, ByteSpan{begin, text.size()});
    }

    void append_content(const std::string& content_text, const std::vector<ByteSpan>& spans,
                        const std::vector<MediaPlaceholderByteSpec>& placeholders) {
        const std::size_t base = text.size();
        text += content_text;
        for (const ByteSpan span : spans) {
            append_span(literals, ByteSpan{base + span.begin, base + span.end});
        }
        for (MediaPlaceholderByteSpec placeholder : placeholders) {
            placeholder.bytes.begin += base;
            placeholder.bytes.end += base;
            media.push_back(std::move(placeholder));
        }
    }

    void mark_execution_boundary() {
        if (execution_boundaries.empty() || execution_boundaries.back() != text.size()) {
            execution_boundaries.push_back(text.size());
        }
    }

    std::size_t size() const noexcept { return text.size(); }
};

// Rendered + trimmed message content with provenance.
struct ContentBlock {
    std::string text;                          // trimmed view
    std::vector<ByteSpan> literals;            // trimmed view
    std::vector<MediaPlaceholderByteSpec> media; // trimmed view
    std::size_t trim_begin = 0, trim_end = 0;  // raw layout
    std::vector<std::size_t> part_bounds_raw;  // raw layout
};

ContentBlock render_content(const ChatMessage& message, bool add_vision_id, int* image_count,
                            int* video_count, std::size_t* media_count) {
    std::string raw;
    std::vector<ByteSpan> raw_literals;
    std::vector<MediaPlaceholderByteSpec> raw_media;
    std::vector<std::size_t> part_bounds;
    for (const ChatPart& part : message.parts) {
        if (part.kind == ChatPartKind::Text) {
            if (!part.text.empty()) {
                const std::size_t begin = raw.size();
                raw += part.text;
                append_span(raw_literals, ByteSpan{begin, raw.size()});
            }
        } else if (part.kind == ChatPartKind::Image) {
            if (image_count != nullptr) { ++*image_count; }
            if (add_vision_id && image_count != nullptr) {
                raw += "Picture " + std::to_string(*image_count) + ": ";
            }
            raw += "<|vision_start|>";
            // The placeholder span covers only the pad token: the processor replaces exactly
            // that text with the expanded vision embeddings (artifact renderer convention).
            const std::size_t begin = raw.size();
            raw += "<|image_pad|>";
            raw_media.push_back(MediaPlaceholderByteSpec{
                .bytes      = ByteSpan{begin, raw.size()},
                .modality   = Modality::Image,
                .item_index = media_count != nullptr ? (*media_count)++ : 0});
            raw += "<|vision_end|>";
        } else {
            if (video_count != nullptr) { ++*video_count; }
            if (add_vision_id && video_count != nullptr) {
                raw += "Video " + std::to_string(*video_count) + ": ";
            }
            raw += "<|vision_start|>";
            const std::size_t begin = raw.size();
            raw += "<|video_pad|>";
            raw_media.push_back(MediaPlaceholderByteSpec{
                .bytes      = ByteSpan{begin, raw.size()},
                .modality   = Modality::Video,
                .item_index = media_count != nullptr ? (*media_count)++ : 0});
            raw += "<|vision_end|>";
        }
        part_bounds.push_back(raw.size());
    }
    const auto [begin, end] = py_trim_bounds(raw);
    ContentBlock block;
    block.text            = raw.substr(begin, end - begin);
    block.trim_begin      = begin;
    block.trim_end        = end;
    block.part_bounds_raw = std::move(part_bounds);
    for (const ByteSpan span : raw_literals) {
        const std::size_t clipped_begin = std::max(span.begin, begin);
        const std::size_t clipped_end   = std::min(span.end, end);
        if (clipped_begin < clipped_end) {
            append_span(block.literals, ByteSpan{clipped_begin - begin, clipped_end - begin});
        }
    }
    for (const MediaPlaceholderByteSpec& placeholder : raw_media) {
        if (placeholder.bytes.end <= begin || placeholder.bytes.begin >= end) { continue; }
        if (placeholder.bytes.begin < begin || placeholder.bytes.end > end) {
            throw std::logic_error("froggeric v22.5: trim crossed a media placeholder");
        }
        MediaPlaceholderByteSpec shifted = placeholder;
        shifted.bytes.begin -= begin;
        shifted.bytes.end -= begin;
        block.media.push_back(std::move(shifted));
    }
    return block;
}

void validate_no_system_media(const ChatMessage& message) {
    // jinja checks per content item in order; the first media item decides the error.
    for (const ChatPart& part : message.parts) {
        if (part.kind == ChatPartKind::Image) {
            throw std::invalid_argument("System message cannot contain images.");
        }
        if (part.kind == ChatPartKind::Video) {
            throw std::invalid_argument("System message cannot contain videos.");
        }
    }
}

struct ThinkExtraction {
    std::string reasoning;
    std::string body;
};

// jinja split(marker)[0] / split(marker)[-1] use the FIRST occurrence.
ThinkExtraction extract_think(const std::string& explicit_reasoning, std::string body) {
    ThinkExtraction out{explicit_reasoning, body};
    if (!out.reasoning.empty()) {
        std::string_view lead_end;
        if (body.rfind("<think>", 0) == 0 && body.find("</think>") != std::string::npos) {
            lead_end = "</think>";
        } else if (body.rfind("<thinking>", 0) == 0 &&
                   body.find("</thinking>") != std::string::npos) {
            lead_end = "</thinking>";
        } else if (body.rfind("</think>", 0) == 0) {
            lead_end = "</think>";
        } else if (body.rfind("</thinking>", 0) == 0) {
            lead_end = "</thinking>";
        }
        if (!lead_end.empty()) {
            const std::size_t found = body.find(lead_end);
            out.body = body.substr(found + lead_end.size());
            while (!out.body.empty() && out.body.front() == '\n') { out.body.erase(out.body.begin()); }
        }
        const auto [r_begin, r_end] = py_trim_bounds(out.reasoning);
        out.reasoning               = out.reasoning.substr(r_begin, r_end - r_begin);
        return out;
    }
    std::string_view think_end;
    if (body.rfind("</think>", 0) == 0) {
        think_end = "</think>";
    } else if (body.rfind("</thinking>", 0) == 0) {
        think_end = "</thinking>";
    } else if (body.find("\n</think>") != std::string::npos) {
        think_end = "\n</think>";
    } else if (body.find("\n</thinking>") != std::string::npos) {
        think_end = "\n</thinking>";
    } else if (body.find("\n</ think>") != std::string::npos) {
        think_end = "\n</ think>";
    } else if (body.find("\n</think >") != std::string::npos) {
        think_end = "\n</think >";
    } else if (body.rfind("<think>", 0) == 0 && body.find("</think>") != std::string::npos) {
        think_end = "</think>";
    } else if (body.rfind("<thinking>", 0) == 0 && body.find("</thinking>") != std::string::npos) {
        think_end = "</thinking>";
    }
    if (!think_end.empty()) {
        const std::string think_start =
            think_end.find("thinking") != std::string_view::npos ? "<thinking>" : "<think>";
        const std::size_t found = body.find(think_end);
        std::string before = body.substr(0, found);
        while (!before.empty() && before.back() == '\n') { before.pop_back(); }
        if (before.find(think_start) != std::string::npos) {
            const std::size_t start = before.rfind(think_start);
            before = before.substr(start + think_start.size());
            while (!before.empty() && before.front() == '\n') { before.erase(before.begin()); }
        }
        out.reasoning = before;
        out.body      = body.substr(found + think_end.size());
        while (!out.body.empty() && out.body.front() == '\n') { out.body.erase(out.body.begin()); }
    }
    const auto [r_begin, r_end] = py_trim_bounds(out.reasoning);
    out.reasoning               = out.reasoning.substr(r_begin, r_end - r_begin);
    return out;
}

} // namespace

RenderedChat render_froggeric_v225(const std::vector<ChatMessage>& messages,
                                   const ChatRenderOptions& options) {
    if (messages.empty()) { throw std::invalid_argument("No messages provided."); }

    const bool has_tools  = !options.tool_jsons.empty();
    bool thinking         = options.enable_thinking;
    ReasoningEffort effort = options.reasoning_effort.value_or(ReasoningEffort::Medium);
    if (options.auto_disable_thinking_with_tools && has_tools) { thinking = false; }
    const bool preserve_thinking =
        options.preserve_reasoning.has_value() ? *options.preserve_reasoning
        : options.preserve_thinking.has_value() ? *options.preserve_thinking
                                               : true;

    // Pre-scan for inline tags (system/developer/user text parts, in order).
    for (const ChatMessage& message : messages) {
        if (message.role != ChatRole::System && message.role != ChatRole::Developer &&
            message.role != ChatRole::User) {
            continue;
        }
        for (const ChatPart& part : message.parts) {
            if (part.kind == ChatPartKind::Text) { apply_tag_state(part.text, thinking, effort); }
        }
    }
    const std::string_view reasoning_instructions = reasoning_instructions_for(thinking, effort);

    // Leading system/developer run, merged per template (trim, strip tags, join "\n\n").
    std::size_t head_count = 0;
    while (head_count < messages.size() && (messages[head_count].role == ChatRole::System ||
                                           messages[head_count].role == ChatRole::Developer)) {
        ++head_count;
    }
    std::string merged_leading;
    std::size_t first_leading_part_length = 0;
    for (std::size_t i = 0; i < head_count; ++i) {
        validate_no_system_media(messages[i]);
        std::string text;
        for (const ChatPart& part : messages[i].parts) {
            if (part.kind == ChatPartKind::Text) { text += part.text; }
        }
        const auto [begin, end] = py_trim_bounds(text);
        std::string part_text = strip_inline_tags(text.substr(begin, end - begin));
        if (i == 0) { first_leading_part_length = part_text.size(); }
        if (part_text.empty()) { continue; }
        if (!merged_leading.empty()) { merged_leading += "\n\n"; }
        merged_leading += part_text;
    }

    const bool continue_final_assistant =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (continue_final_assistant) {
        const ChatMessage& final = messages.back();
        if (final.role != ChatRole::Assistant || final.parts.empty() ||
            !final.reasoning_content.empty() || !final.tool_calls.empty() || final.has_media()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (options.enable_thinking) {
            throw std::invalid_argument("assistant continuation cannot start in thinking mode");
        }
    }

    // Template multi_step_tool rule, in _msgs coordinates.
    const std::size_t first_body = head_count;
    const long body_count        = static_cast<long>(messages.size()) -
                                   static_cast<long>(head_count);
    long last_query_index        = 0;
    {
        bool multi_step_tool = true;
        for (long i = body_count - 1; i >= 0 && multi_step_tool; --i) {
            const ChatMessage& message = messages[first_body + static_cast<std::size_t>(i)];
            if (message.role != ChatRole::User) { continue; }
            std::string raw;
            for (const ChatPart& part : message.parts) {
                if (part.kind == ChatPartKind::Text) { raw += part.text; }
                if (part.kind == ChatPartKind::Image) {
                    raw += "<|vision_start|><|image_pad|><|vision_end|>";
                } else { raw += "<|vision_start|><|video_pad|><|vision_end|>"; }
            }
            const auto [begin, end] = py_trim_bounds(raw);
            const std::string_view rendered = std::string_view(raw).substr(begin, end - begin);
            const bool is_tool_query =
                rendered.size() >= std::string_view("<tool_response>").size() +
                                   std::string_view("</tool_response>").size() &&
                rendered.substr(0, std::string_view("<tool_response>").size()) ==
                    "<tool_response>" &&
                rendered.substr(rendered.size() - std::string_view("</tool_response>").size()) ==
                    "</tool_response>";
            if (!is_tool_query) {
                multi_step_tool  = false;
                last_query_index = i;
            }
        }
        if (multi_step_tool) { last_query_index = (body_count - 1) > 50 ? body_count - 1 : 0; }
    }

    // ---- system block ----
    PromptBuilder out;
    std::size_t leading_instruction_begin = 0;
    const std::string kToolsHeader =
        "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    const std::string kXmlInstructionsThinking =
        "\n\nIf you choose to call a function ONLY reply in the following format with NO "
        "suffix:\n\n<think>\nBrief explanation of tool call\n</think>\n<tool_call>\n"
        "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
        "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
        "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n"
        "- You can use the <think></think> block to plan your next tool call OR to synthesize "
        "data and formulate your final response to the user.\n"
        "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> "
        "block.\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> "
        "block must be nested within <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
        "thinking, with NO conversational text before it.\n"
        "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with "
        "NO spaces or indentation before them.\n"
        "- To call multiple functions, output a separate, completely closed "
        "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
        "- If you have all necessary data, provide your final answer directly to the user "
        "without any tool call.\n</IMPORTANT>";
    const std::string kXmlInstructionsOff =
        "\n\nIf you choose to call a function ONLY reply in the following format with NO "
        "suffix:\n\n<tool_call>\n<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> "
        "block must be nested within <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with "
        "NO conversational text before it.\n"
        "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with "
        "NO spaces or indentation before them.\n"
        "- To call multiple functions, output a separate, completely closed "
        "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
        "- If you have all necessary data, provide your final answer directly to the user "
        "without any tool call.\n</IMPORTANT>";
    const std::string kJsonInstructionsThinking =
        "\n\nIf you choose to call a function ONLY reply in the following format with NO "
        "suffix:\n\n<think>\nBrief explanation of tool call\n</think>\n<tool_call>\n"
        "{\"name\": \"example_function_name\", \"arguments\": {\"example_parameter_1\": "
        "\"value_1\", \"example_parameter_2\": \"This is the value for the second parameter\"}}"
        "\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
        "- You can use the <think></think> block to plan your next tool call OR to synthesize "
        "data and formulate your final response to the user.\n"
        "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> "
        "block.\n"
        "- Function calls MUST follow the specified format: a single JSON object with \"name\" "
        "and \"arguments\" keys inside <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
        "thinking, with NO conversational text before it.\n"
        "- The <tool_call> tag MUST be at the very beginning of a new line, with NO spaces or "
        "indentation before it.\n"
        "- To call multiple functions, output a separate, completely closed "
        "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
        "- If you have all necessary data, provide your final answer directly to the user "
        "without any tool call.\n</IMPORTANT>";
    const std::string kJsonInstructionsOff =
        "\n\nIf you choose to call a function ONLY reply in the following format with NO "
        "suffix:\n\n<tool_call>\n{\"name\": \"example_function_name\", \"arguments\": "
        "{\"example_parameter_1\": \"value_1\", \"example_parameter_2\": \"This is the value "
        "for the second parameter\"}}\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: a single JSON object with \"name\" "
        "and \"arguments\" keys inside <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with "
        "NO conversational text before it.\n"
        "- The <tool_call> tag MUST be at the very beginning of a new line, with NO spaces or "
        "indentation before it.\n"
        "- To call multiple functions, output a separate, completely closed "
        "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
        "- If you have all necessary data, provide your final answer directly to the user "
        "without any tool call.\n</IMPORTANT>";
    if (has_tools) {
        out.append_template("<|im_start|>system\n");
        if (!reasoning_instructions.empty()) {
            out.append_template(reasoning_instructions);
            out.append_template("\n\n");
        }
        out.append_template(kToolsHeader);
        for (const std::string& tool_json : options.tool_jsons) {
            out.append_template("\n");
            out.append_literal(tojson_oracle(njson::parse(tool_json)));
            out.tool_bounds.push_back(out.size());
        }
        out.append_template("\n</tools>");
        out.append_template(options.tool_call_format == ToolCallFormat::Json
                               ? (thinking ? kJsonInstructionsThinking : kJsonInstructionsOff)
                               : (thinking ? kXmlInstructionsThinking : kXmlInstructionsOff));
        if (!merged_leading.empty()) {
            out.append_template("\n\n");
            leading_instruction_begin = out.size();
            out.append_literal(merged_leading);
        }
        out.append_template("<|im_end|>\n");
    } else if (!merged_leading.empty()) {
        out.append_template("<|im_start|>system\n");
        if (!reasoning_instructions.empty()) {
            out.append_template(reasoning_instructions);
            out.append_template("\n\n");
        }
        leading_instruction_begin = out.size();
        out.append_literal(merged_leading);
        out.append_template("<|im_end|>\n");
    } else if (!reasoning_instructions.empty()) {
        out.append_template("<|im_start|>system\n");
        out.append_template(reasoning_instructions);
        out.append_template("<|im_end|>\n");
    }

    std::vector<std::optional<std::size_t>> message_boundaries(messages.size() + 1U);
    if (head_count == 0) {
        message_boundaries[0] = 0;
    } else {
        message_boundaries[head_count] = out.size();
    }

    // ---- tool-call history ----
    const auto render_tool_call = [&](const ToolCall& call, bool first, bool body_has_text) {
        if (options.tool_call_format == ToolCallFormat::Json) {
            if (first) {
                if (body_has_text) { out.append_template("\n\n"); }
            } else {
                out.append_template("\n");
            }
            std::string args = "{}";
            if (!call.arguments_json.empty()) {
                // jinja: mapping -> tojson; string -> verbatim. The wire model always carries a
                // JSON string, so an object parses to the mapping branch and anything else
                // (arrays, scalars, non-JSON) stays verbatim exactly like the string branch.
                try {
                    const njson parsed = njson::parse(call.arguments_json);
                    if (parsed.is_object()) { args = tojson_oracle(parsed); }
                    else { args = call.arguments_json; }
                } catch (const njson::exception&) {
                    args = call.arguments_json;
                }
            }
            out.append_template("<tool_call>\n{\"name\": ");
            out.append_literal(tojson_oracle(njson(call.name)));
            out.append_template(", \"arguments\": ");
            out.append_literal(args);
            out.append_template("}\n</tool_call>");
            return;
        }
        if (first) {
            if (body_has_text) { out.append_template("\n\n"); }
        } else {
            out.append_template("\n");
        }
        out.append_template("<tool_call>\n<function=");
        out.append_literal(call.name);
        out.append_template(">\n");
        const std::string& raw_args = call.arguments_json;
        std::vector<std::pair<std::string, njson>> ordered_params;
        if (!raw_args.empty()) {
            // NInfer's wire contract carries tool arguments as a JSON object string (the
            // artifact renderer enforces the same shape); the jinja raw-string fallback for
            // non-object arguments is not representable and is rejected explicitly.
            njson parsed;
            try {
                parsed = njson::parse(raw_args);
            } catch (const njson::exception&) {
                throw std::invalid_argument("tool call arguments must be a JSON object");
            }
            if (!parsed.is_object()) {
                throw std::invalid_argument("tool call arguments must be a JSON object");
            }
            for (auto it = parsed.begin(); it != parsed.end(); ++it) {
                ordered_params.emplace_back(it.key(), it.value());
            }
        }
        for (const auto& [param_name, param_value] : ordered_params) {
            out.append_template("<parameter=");
            out.append_literal(param_name);
            out.append_template(">\n");
            const std::string value_text =
                param_value.is_string() ? param_value.get<std::string>()
                                        : tojson_oracle(param_value);
            if (options.max_tool_arg_chars > 0 &&
                py_len(value_text) > options.max_tool_arg_chars) {
                out.append_literal(truncate_python(value_text, options.max_tool_arg_chars));
            } else {
                out.append_literal(value_text);
            }
            out.append_template("\n</parameter>\n");
        }
        out.append_template("</function>\n</tool_call>");
    };

    // ---- main loop ----
    int image_count = 0;
    int video_count = 0;
    std::size_t media_count = 0;
    long consecutive_failures = 0;
    bool prev_was_tool        = false;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::optional<std::size_t>> cache_boundaries(options.cache_markers.size());

    for (std::size_t i = first_body; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        const long body_index      = static_cast<long>(i) - static_cast<long>(first_body);
        const bool is_instruction  = message.role == ChatRole::System ||
                                     message.role == ChatRole::Developer;
        if (is_instruction) { validate_no_system_media(message); }
        ContentBlock content =
            render_content(message, options.add_vision_id, &image_count, &video_count,
                           &media_count);
        if (is_instruction || message.role == ChatRole::User) {
            content.text = strip_inline_tags(content.text);
        }
        const auto emit = [&]() {
            out.append_content(content.text, content.literals, content.media);
        };
        const auto resolve_part_boundaries = [&](std::size_t content_begin) {
            for (std::size_t marker_index = 0; marker_index < options.cache_markers.size();
                 ++marker_index) {
                const PromptCacheMarker& marker = options.cache_markers[marker_index];
                if (marker.location != PromptCacheMarkerLocation::MessagePartBoundary ||
                    marker.after_message_count != i + 1U || marker.after_message_part_count == 0 ||
                    marker.after_message_part_count > content.part_bounds_raw.size()) {
                    continue;
                }
                const std::size_t raw =
                    content.part_bounds_raw[marker.after_message_part_count - 1U];
                const std::size_t clamped = std::clamp(raw, content.trim_begin, content.trim_end);
                cache_boundaries[marker_index] = content_begin + clamped - content.trim_begin;
            }
        };

        if (is_instruction) {
            out.append_template("<|im_start|>system\n");
            const std::size_t content_begin = out.size();
            emit();
            resolve_part_boundaries(content_begin);
            out.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = out.size();
            prev_was_tool = false;
            continue;
        }
        if (message.role == ChatRole::User) {
            consecutive_failures = 0;
            out.append_template("<|im_start|>user\n");
            const std::size_t content_begin = out.size();
            emit();
            resolve_part_boundaries(content_begin);
            out.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = out.size();
            prev_was_tool = false;
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const std::string lower = py_lower(content.text);
            const std::string head  = py_slice(lower, 0, 120);
            const bool is_code_or_grep =
                contains(lower, "throw new ") || contains(lower, "throw error") ||
                contains(lower, "console.error") || contains(lower, "logger.error") ||
                contains(lower, "logging.error") || contains(head, "import ") ||
                contains(head, "def ") || contains(head, "function ");
            const bool exit_code_zero =
                contains(head, "exit code: 0") || contains(head, "process exited with code 0");
            const bool error_field_ok =
                contains(head, "\"error\": null") || contains(head, "\"error\":null") ||
                contains(head, "\"error\": false") || contains(head, "\"error\":false") ||
                contains(head, "\"error\": \"\"") || contains(head, "\"error\":\"\"");
            const bool strong_error =
                (contains(head, "\"error\":") && !error_field_ok) ||
                contains(head, "\"status\": \"error\"") || contains(head, "\"status\":\"error\"") ||
                contains(head, "traceback (most recent call last):") ||
                contains(head, "command not found") || contains(head, "invalid syntax") ||
                contains(head, "fatal:") ||
                ((contains(head, "exit code: ") || contains(head, "process exited with code")) &&
                 !exit_code_zero) ||
                head.rfind("exception:", 0) == 0 || head.rfind("failed to ", 0) == 0;
            const bool weak_error      = contains(head, "error:") || contains(head, "err!");
            const bool weak_suppressed = contains(head, "$ ") || contains(head, "took ") ||
                                         py_len(content.text) >= 600;
            if (!is_code_or_grep && (strong_error || (weak_error && !weak_suppressed))) {
                ++consecutive_failures;
            } else {
                consecutive_failures = 0;
            }

            if (!prev_was_tool) { out.append_template("<|im_start|>user"); }
            std::string tool_text = content.text;
            const bool is_json_payload =
                options.tool_call_format == ToolCallFormat::Json && !tool_text.empty() &&
                (tool_text.front() == '{' || tool_text.front() == '[');
            if (!is_json_payload && options.max_tool_response_chars > 0 &&
                py_len(tool_text) > options.max_tool_response_chars) {
                tool_text = truncate_python(tool_text, options.max_tool_response_chars);
            }
            out.append_template("\n<tool_response>\n");
            out.append_literal(tool_text);
            if (consecutive_failures >= 2) {
                out.append_template("\n\n⚠️ SYSTEM WARNING: ");
                out.append_template(std::to_string(consecutive_failures));
                out.append_template(" consecutive tool errors detected. Your previous approach "
                                    "is incorrect. You MUST use a fundamentally different "
                                    "approach or corrected arguments.");
            } else if (consecutive_failures == 1) {
                out.append_template("\n\n⚠️ SYSTEM WARNING: The previous tool call returned an "
                                    "error. Diagnose the failure and retry with completely "
                                    "corrected arguments.");
            }
            out.append_template("\n</tool_response>");
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (closes_group) { out.append_template("<|im_end|>\n"); }
            message_boundaries[i + 1U] = out.size();
            prev_was_tool = true;
            continue;
        }
        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // ---- assistant ----
        if (continue_final_assistant && i + 1U == messages.size()) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind   = RewriteCheckpointKind::ResponseReplay,
                .offset = static_cast<std::uint32_t>(out.size())};
            out.append_template("<|im_start|>assistant\n");
            out.mark_execution_boundary();
            emit();
            message_boundaries[i + 1U] = out.size();
            prev_was_tool = false;
            continue;
        }
        const ThinkExtraction parts = extract_think(message.reasoning_content, content.text);
        const bool keep_thinking    = preserve_thinking || body_index > last_query_index;
        if (!preserve_thinking && !rewrite_checkpoint && body_index > last_query_index) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind   = RewriteCheckpointKind::TurnClosure,
                .offset = static_cast<std::uint32_t>(out.size())};
        }
        out.append_template("<|im_start|>assistant\n");
        out.mark_execution_boundary();
        if (keep_thinking) {
            out.append_template("<think>\n");
            out.mark_execution_boundary();
            out.append_literal(parts.reasoning);
            out.append_template("\n</think>\n\n");
            out.mark_execution_boundary();
        }
        out.append_literal(parts.body);
        const auto body_trimmed = [&] {
            const auto [b, e] = py_trim_bounds(parts.body);
            return e > b;
        }();
        for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
            render_tool_call(message.tool_calls[call_index], call_index == 0, body_trimmed);
        }
        out.append_template("<|im_end|>\n");
        message_boundaries[i + 1U] = out.size();
        prev_was_tool = false;
    }

    // ---- generation suffix ----
    if (!continue_final_assistant && options.add_generation_prompt) {
        const std::uint32_t generation_begin = static_cast<std::uint32_t>(out.size());
        if (preserve_thinking) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind   = RewriteCheckpointKind::ResponseReplay,
                .offset = generation_begin};
        } else if (!rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind   = RewriteCheckpointKind::TurnClosure,
                .offset = generation_begin};
        }
        out.append_template("<|im_start|>assistant\n");
        out.mark_execution_boundary();
        out.append_template("<think>\n");
        out.mark_execution_boundary();
        if (!thinking) {
            out.append_template("\n</think>\n\n");
            out.mark_execution_boundary();
        }
    }

    // ---- cache markers ----
    for (std::size_t index = 0; index < options.cache_markers.size(); ++index) {
        const PromptCacheMarker& marker = options.cache_markers[index];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < message_boundaries.size()) {
                cache_boundaries[index] = message_boundaries[marker.after_message_count];
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count != 0 && marker.after_tool_count <= out.tool_bounds.size()) {
                cache_boundaries[index] = out.tool_bounds[marker.after_tool_count - 1U];
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (leading_instruction_begin != 0 && head_count > 0) {
                const std::size_t clamped =
                    std::clamp<std::size_t>(marker.leading_instruction_bytes, 0U,
                                                first_leading_part_length);
                cache_boundaries[index] = leading_instruction_begin + clamped;
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            break; // resolved during the main loop.
        }
    }

    return RenderedChat{.text                         = std::move(out.text),
                        .literal_spans                = std::move(out.literals),
                        .media_placeholders           = std::move(out.media),
                        .rewrite_checkpoint           = rewrite_checkpoint,
                        .rewrite_execution_boundaries = std::move(out.execution_boundaries),
                        .message_boundaries           = std::move(message_boundaries),
                        .cache_boundaries             = std::move(cache_boundaries)};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
