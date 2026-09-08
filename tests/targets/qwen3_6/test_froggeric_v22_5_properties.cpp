// Deterministic property tests for the compiled froggeric v22.5 renderer.
//
// The golden fixtures pin the bytes of hand-picked inputs. These properties pin the *invariants*
// over generated inputs, so a refactor that keeps every fixture green but breaks a cache or
// provenance contract still fails. Properties (design.md §8):
//
//   1. determinism        the same input renders byte-identically twice;
//   2. provenance sanity  boundaries, literal spans and media placeholders stay ordered, inside
//                         the text, and mutually consistent;
//   3. generation state   the emitted generation suffix agrees with generation_starts_in_thinking;
//   4. prefix extension   preserve_thinking=true: once the assistant reply is appended, the
//                         previous prompt is a byte prefix of the next prompt;
//   5. checkpoint stable  every published rewrite checkpoint is a prefix shared by the full
//                         render (the cache-reuse safety invariant);
//   6. branch exactness   deterministic cases pin that a new user branch diverges at the
//                         published checkpoint (or immediately after the assistant opener when
//                         the checkpoint sits before a tool-group assistant).
//
// Documented exceptions (upstream template semantics), skipped by the generator:
//   - a prefix that ends inside a still-open leading system/developer run merges differently
//     when another leading instruction is appended;
//   - a prefix that ends inside a consecutive tool-result batch closes the batch differently
//     when another tool result is appended;
//   - an inline control tag in a *later* system/developer/user message changes the pre-scanned
//     thinking/effort state and can rewrite the system block before the checkpoint.
// Properties 4 and 5 therefore only consider boundaries after the last inline tag, and skip the
// two batch boundaries above. The deterministic branch tests cover the non-skipped shapes.
//
// The generator is seeded (NINFER_FROGGERIC_PROP_SEED, default 0x5eed1234) and deterministic
// across platforms (splitmix64, no <random>). A failure prints the seed and a message-minimized
// input; NINFER_FROGGERIC_PROP_CASES (default 500) controls the iteration count.

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_tags.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fj = ninfer::targets::qwen3_6::frontend_internal;

// ---------------------------------------------------------------------------------------------
// Deterministic pseudo-random generator (splitmix64).
// ---------------------------------------------------------------------------------------------

class Rng {
public:
    explicit Rng(std::uint64_t seed) noexcept
        : state_(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed) {}

    std::uint64_t next() noexcept {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    std::size_t below(std::size_t bound) noexcept {
        return bound == 0 ? 0 : static_cast<std::size_t>(next() % bound);
    }

    bool chance(unsigned percent) noexcept { return below(100) < percent; }

    template <typename T>
    const T& pick(const std::vector<T>& values) {
        return values[below(values.size())];
    }

private:
    std::uint64_t state_;
};

// ---------------------------------------------------------------------------------------------
// Generated case and vocabularies.
// ---------------------------------------------------------------------------------------------

struct Case {
    std::vector<fj::ChatMessage> messages;
    fj::ChatRenderOptions options;
};

const std::vector<std::string>& user_texts() {
    static const std::vector<std::string> values = {
        "hello",
        "q1",
        "line1\nline2",
        "  padded  ",
        "emoji 😀 tail",
        "中文问题",
        "",
        "one two three four five six seven eight nine ten",
        "<tool_response>compat result</tool_response>",
        "<tool_response>error: nope</tool_response>",
        "  \u00a0\u3000 ",
        "trailing emoji 🚀",
    };
    return values;
}

const std::vector<std::string>& assistant_bodies() {
    static const std::vector<std::string> values = {
        "answer",
        "",
        "line1\nline2",
        "  trimmed body  ",
        "done ✅",
        "多行\n答案",
    };
    return values;
}

const std::vector<std::string>& reasoning_texts() {
    static const std::vector<std::string> values = {
        "short thought",
        "",
        "step one\nstep two",
        "  spaced reasoning  ",
        "thought with 😀",
    };
    return values;
}

const std::vector<std::string>& tool_result_texts() {
    static const std::vector<std::string> values = {
        "ok",
        "result one",
        "error: command failed",
        "Traceback (most recent call last):\nValueError: bad",
        "Exit code: 1",
        "Exit code: 0",
        "$ rm -rf /",
        "error: long output " + std::string(120, 'x'),
        "",
    };
    return values;
}

const std::vector<std::string>& system_texts() {
    static const std::vector<std::string> values = {
        "you are a helpful assistant",
        "be terse",
        "  leading spaces  ",
        "指令：保持简洁",
        "",
    };
    return values;
}

const std::vector<std::string>& argument_forms() {
    static const std::vector<std::string> values = {
        R"({})",
        R"({"city":"Paris"})",
        R"({"city": "Paris", "units": "c"})",
        R"("raw string")",
        R"(42)",
        "",
        R"({"nested":{"a":[1,2,3]}})",
        R"({"unicode":"中文"})",
    };
    return values;
}

std::string_view random_tag(Rng& rng) {
    return fj::kInlineTags[rng.below(std::size(fj::kInlineTags))];
}

std::string tool_json(std::size_t index) {
    return index == 0
               ? R"({"type":"function","function":{"name":"f","description":"first tool","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}})"
               : R"({"type":"function","function":{"name":"g","description":"second tool","parameters":{"type":"object"}}})";
}

fj::ChatMessage message(ninfer::ChatRole role, std::string text) {
    fj::ChatMessage out;
    out.role = role;
    out.parts.push_back(fj::ChatPart::text_part(std::move(text)));
    return out;
}

void inject_tag(Rng& rng, fj::ChatMessage& target) {
    const std::string tag(random_tag(rng));
    if (rng.chance(50)) {
        target.parts.insert(target.parts.begin(),
                            fj::ChatPart::text_part("pre " + tag + " post"));
        return;
    }
    // Split the tag across two text parts: the stripper must remap provenance exactly like the
    // template's concatenated split/join.
    const std::size_t cut = 1 + rng.below(tag.size() - 1);
    target.parts.push_back(fj::ChatPart::text_part(tag.substr(0, cut)));
    target.parts.push_back(fj::ChatPart::text_part(tag.substr(cut)));
}

fj::ChatMessage user_message(Rng& rng) {
    fj::ChatMessage out;
    out.role = ninfer::ChatRole::User;
    const std::size_t parts = 1 + rng.below(3);
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t roll = rng.below(10);
        if (roll == 0) {
            out.parts.push_back(fj::ChatPart::image(fj::MediaData{}));
        } else if (roll == 1) {
            out.parts.push_back(fj::ChatPart::video(fj::MediaData{}));
        } else {
            out.parts.push_back(fj::ChatPart::text_part(rng.pick(user_texts())));
        }
    }
    if (rng.chance(30)) { inject_tag(rng, out); }
    return out;
}

fj::ChatMessage system_message(Rng& rng, ninfer::ChatRole role) {
    fj::ChatMessage out;
    out.role = role;
    out.parts.push_back(fj::ChatPart::text_part(rng.pick(system_texts())));
    if (rng.chance(20)) { out.parts.push_back(fj::ChatPart::text_part(rng.pick(user_texts()))); }
    if (rng.chance(30)) { inject_tag(rng, out); }
    return out;
}

fj::ChatMessage assistant_message(Rng& rng, bool allow_tools) {
    fj::ChatMessage out;
    out.role = ninfer::ChatRole::Assistant;
    if (rng.chance(85)) {
        out.parts.push_back(fj::ChatPart::text_part(rng.pick(assistant_bodies())));
    }
    if (rng.chance(15)) { out.parts.push_back(fj::ChatPart::image(fj::MediaData{})); }
    if (rng.chance(40)) {
        if (rng.chance(50)) {
            out.reasoning_content = rng.pick(reasoning_texts());
        } else {
            out.parts.insert(out.parts.begin(),
                             fj::ChatPart::text_part("<think>" + rng.pick(reasoning_texts()) +
                                                     "</think>"));
        }
    } else if (rng.chance(15)) {
        out.parts.insert(out.parts.begin(),
                         fj::ChatPart::text_part("prefix</think>tail"));
    }
    if (allow_tools && rng.chance(40)) {
        const std::size_t calls = 1 + rng.below(2);
        for (std::size_t i = 0; i < calls; ++i) {
            out.tool_calls.push_back(
                fj::ToolCall{.name = rng.chance(50) ? "f" : "g",
                             .arguments_json = rng.pick(argument_forms())});
        }
    }
    return out;
}

fj::ChatMessage tool_message(Rng& rng) {
    fj::ChatMessage out;
    out.role = ninfer::ChatRole::Tool;
    out.parts.push_back(fj::ChatPart::text_part(rng.pick(tool_result_texts())));
    if (rng.chance(12)) { out.parts.push_back(fj::ChatPart::image(fj::MediaData{})); }
    return out;
}

bool message_has_media(const std::vector<fj::ChatMessage>& messages) {
    return std::any_of(messages.begin(), messages.end(),
                       [](const fj::ChatMessage& value) { return value.has_media(); });
}

Case generate(Rng& rng) {
    Case out;
    const std::size_t tool_count = rng.below(3);
    for (std::size_t i = 0; i < tool_count; ++i) { out.options.tool_jsons.push_back(tool_json(i)); }

    out.options.enable_thinking = rng.chance(75);
    switch (rng.below(3)) {
    case 0: break; // template default (true)
    case 1: out.options.preserve_thinking = true; break;
    default: out.options.preserve_thinking = false; break;
    }
    if (rng.chance(15)) { out.options.froggeric_v225.preserve_reasoning = rng.chance(50); }
    switch (rng.below(4)) {
    case 0: break; // template default (medium)
    case 1: out.options.reasoning_effort = ninfer::ReasoningEffort::Low; break;
    case 2: out.options.reasoning_effort = ninfer::ReasoningEffort::Medium; break;
    default: out.options.reasoning_effort = ninfer::ReasoningEffort::XHigh; break;
    }
    out.options.froggeric_v225.auto_disable_thinking_with_tools = rng.chance(15);
    out.options.froggeric_v225.tool_call_format =
        rng.chance(25) ? ninfer::ToolCallFormat::Json : ninfer::ToolCallFormat::Xml;
    out.options.add_generation_prompt = rng.chance(80);

    const std::size_t marker_count = rng.below(4);
    for (std::size_t i = 0; i < marker_count; ++i) {
        ninfer::PromptCacheMarker marker;
        switch (rng.below(4)) {
        case 0:
            marker.location             = ninfer::PromptCacheMarkerLocation::MessageBoundary;
            marker.after_message_count  = static_cast<std::uint32_t>(rng.below(8));
            break;
        case 1:
            marker.location        = ninfer::PromptCacheMarkerLocation::ToolBoundary;
            marker.after_tool_count = static_cast<std::uint32_t>(rng.below(tool_count + 1));
            break;
        case 2:
            marker.location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary;
            marker.leading_instruction_bytes = static_cast<std::uint32_t>(rng.below(64));
            break;
        default:
            marker.location = ninfer::PromptCacheMarkerLocation::MessagePartBoundary;
            marker.after_message_count      = static_cast<std::uint32_t>(rng.below(8));
            marker.after_message_part_count = static_cast<std::uint32_t>(rng.below(4));
            break;
        }
        out.options.cache_markers.push_back(marker);
    }

    const std::size_t leading = rng.below(3);
    for (std::size_t i = 0; i < leading; ++i) {
        out.messages.push_back(system_message(
            rng, rng.chance(50) ? ninfer::ChatRole::System : ninfer::ChatRole::Developer));
    }

    if (out.messages.empty() && rng.chance(10)) {
        out.messages.push_back(assistant_message(rng, tool_count > 0));
    } else {
        out.messages.push_back(user_message(rng));
    }

    const std::size_t turns = 1 + rng.below(4);
    for (std::size_t turn = 0; turn < turns && out.messages.size() < 14; ++turn) {
        const ninfer::ChatRole last = out.messages.back().role;
        if (last == ninfer::ChatRole::Assistant) {
            if (rng.chance(30)) { out.messages.push_back(tool_message(rng)); }
            out.messages.push_back(user_message(rng));
            continue;
        }
        if (rng.chance(85)) {
            fj::ChatMessage assistant = assistant_message(rng, tool_count > 0);
            const bool has_calls      = !assistant.tool_calls.empty();
            out.messages.push_back(std::move(assistant));
            if (has_calls) {
                const std::size_t results = 1 + rng.below(2);
                for (std::size_t r = 0; r < results && out.messages.size() < 14; ++r) {
                    out.messages.push_back(tool_message(rng));
                }
                if (out.messages.size() < 14 && rng.chance(70)) {
                    out.messages.push_back(assistant_message(rng, false));
                }
            }
        } else if (rng.chance(50)) {
            out.messages.push_back(system_message(
                rng, rng.chance(50) ? ninfer::ChatRole::System : ninfer::ChatRole::Developer));
        }
        if (out.messages.size() < 14) { out.messages.push_back(user_message(rng)); }
    }

    // A truncation cap that crosses a media placeholder is an explicit invalid_argument. Keep
    // those cases out of the property surface; the oracle fixtures cover the error paths.
    if (message_has_media(out.messages)) {
        out.options.froggeric_v225.max_tool_arg_chars      = 0;
        out.options.froggeric_v225.max_tool_response_chars = 0;
    } else {
        if (rng.chance(30)) {
            out.options.froggeric_v225.max_tool_arg_chars = 1 + rng.below(30);
        }
        if (rng.chance(30)) {
            out.options.froggeric_v225.max_tool_response_chars = 1 + rng.below(30);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Rendering and comparison helpers.
// ---------------------------------------------------------------------------------------------

enum class RenderStatus { Ok, Rejected, Error };

struct RenderResult {
    RenderStatus status = RenderStatus::Error;
    fj::RenderedChat rendered;
    std::string error;
};

RenderResult render_checked(const fj::CompiledChatTemplate& renderer,
                            const std::vector<fj::ChatMessage>& messages,
                            const fj::ChatRenderOptions& options) {
    try {
        return RenderResult{.status = RenderStatus::Ok,
                            .rendered = renderer.render(messages, options),
                            .error = {}};
    } catch (const std::invalid_argument& error) {
        return RenderResult{.status = RenderStatus::Rejected, .rendered = {}, .error = error.what()};
    } catch (const std::exception& error) {
        return RenderResult{.status = RenderStatus::Error, .rendered = {}, .error = error.what()};
    }
}

bool same_rendered(const fj::RenderedChat& left, const fj::RenderedChat& right) {
    if (left.text != right.text || left.literal_spans.size() != right.literal_spans.size() ||
        left.media_placeholders.size() != right.media_placeholders.size() ||
        left.media_token_runs.size() != right.media_token_runs.size() ||
        left.rewrite_execution_boundaries != right.rewrite_execution_boundaries ||
        left.message_boundaries != right.message_boundaries ||
        left.cache_boundaries != right.cache_boundaries ||
        left.generation_starts_in_thinking != right.generation_starts_in_thinking) {
        return false;
    }
    for (std::size_t i = 0; i < left.literal_spans.size(); ++i) {
        if (left.literal_spans[i].begin != right.literal_spans[i].begin ||
            left.literal_spans[i].end != right.literal_spans[i].end) {
            return false;
        }
    }
    for (std::size_t i = 0; i < left.media_placeholders.size(); ++i) {
        const auto& a = left.media_placeholders[i];
        const auto& b = right.media_placeholders[i];
        if (a.bytes.begin != b.bytes.begin || a.bytes.end != b.bytes.end ||
            a.modality != b.modality || a.item_index != b.item_index) {
            return false;
        }
    }
    if (left.rewrite_checkpoint.has_value() != right.rewrite_checkpoint.has_value()) {
        return false;
    }
    if (left.rewrite_checkpoint) {
        if (left.rewrite_checkpoint->kind != right.rewrite_checkpoint->kind ||
            left.rewrite_checkpoint->offset != right.rewrite_checkpoint->offset) {
            return false;
        }
    }
    return true;
}

bool is_instruction(ninfer::ChatRole role) {
    return role == ninfer::ChatRole::System || role == ninfer::ChatRole::Developer;
}

// Last message index that carries an inline control tag in a pre-scanned role; tags in the
// continuation can rewrite the system block, so prefix properties skip boundaries before it.
std::optional<std::size_t> last_inline_tag_index(const std::vector<fj::ChatMessage>& messages) {
    std::optional<std::size_t> last;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (!is_instruction(messages[i].role) && messages[i].role != ninfer::ChatRole::User) {
            continue;
        }
        for (const fj::ChatPart& part : messages[i].parts) {
            if (part.kind != fj::ChatPartKind::Text) { continue; }
            for (const std::string_view tag : fj::kInlineTags) {
                if (part.text.find(tag) != std::string::npos) {
                    last = i;
                    break;
                }
            }
        }
    }
    return last;
}

std::string truncate(std::string text, std::size_t limit = 24) {
    for (char& byte : text) {
        if (byte == '\n') { byte = ' '; }
    }
    if (text.size() > limit) {
        text.resize(limit);
        text += "...";
    }
    return text;
}

std::string describe(const Case& value) {
    std::string out = "[";
    for (std::size_t i = 0; i < value.messages.size(); ++i) {
        const fj::ChatMessage& message = value.messages[i];
        if (i != 0) { out += " | "; }
        switch (message.role) {
        case ninfer::ChatRole::System: out += "system:"; break;
        case ninfer::ChatRole::Developer: out += "developer:"; break;
        case ninfer::ChatRole::User: out += "user:"; break;
        case ninfer::ChatRole::Assistant: out += "assistant:"; break;
        case ninfer::ChatRole::Tool: out += "tool:"; break;
        }
        bool first = true;
        for (const fj::ChatPart& part : message.parts) {
            if (!first) { out += "+"; }
            first = false;
            if (part.kind == fj::ChatPartKind::Image) {
                out += "<img>";
            } else if (part.kind == fj::ChatPartKind::Video) {
                out += "<video>";
            } else {
                out += "\"" + truncate(part.text) + "\"";
            }
        }
        if (!message.reasoning_content.empty()) {
            out += " reasoning=\"" + truncate(message.reasoning_content) + "\"";
        }
        if (!message.tool_calls.empty()) {
            out += " calls=" + std::to_string(message.tool_calls.size());
        }
    }
    out += "] opts{think=" + std::to_string(value.options.enable_thinking ? 1 : 0);
    const char* preserve = "default";
    if (value.options.froggeric_v225.preserve_reasoning) {
        preserve = *value.options.froggeric_v225.preserve_reasoning ? "reasoning:1" : "reasoning:0";
    } else if (value.options.preserve_thinking) {
        preserve = *value.options.preserve_thinking ? "1" : "0";
    }
    out += std::string(" preserve=") + preserve;
    out += " effort=" +
           (value.options.reasoning_effort
                ? std::to_string(static_cast<int>(*value.options.reasoning_effort))
                : std::string("default"));
    out += " auto=" +
           std::to_string(value.options.froggeric_v225.auto_disable_thinking_with_tools ? 1 : 0);
    out += " fmt=" +
           std::string(value.options.froggeric_v225.tool_call_format ==
                               ninfer::ToolCallFormat::Json
                           ? "json"
                           : "xml");
    out += " arg=" + std::to_string(value.options.froggeric_v225.max_tool_arg_chars);
    out += " resp=" + std::to_string(value.options.froggeric_v225.max_tool_response_chars);
    out += " gen=" + std::to_string(value.options.add_generation_prompt ? 1 : 0);
    out += " tools=" + std::to_string(value.options.tool_jsons.size());
    out += " markers=" + std::to_string(value.options.cache_markers.size());
    out += "}";
    return out;
}

// ---------------------------------------------------------------------------------------------
// Properties.
// ---------------------------------------------------------------------------------------------

using Property = std::function<std::optional<std::string>(const Case&)>;

std::optional<std::string> check_determinism(const Case& value,
                                             const fj::CompiledChatTemplate& renderer) {
    const RenderResult first  = render_checked(renderer, value.messages, value.options);
    const RenderResult second = render_checked(renderer, value.messages, value.options);
    if (first.status != RenderStatus::Ok || second.status != RenderStatus::Ok) {
        return "determinism render did not succeed";
    }
    if (!same_rendered(first.rendered, second.rendered)) {
        return "two renders of the same input differ";
    }
    return std::nullopt;
}

std::optional<std::string> check_provenance(const Case& value, const fj::RenderedChat& rendered) {
    const std::size_t size = rendered.text.size();
    for (std::size_t i = 0; i < rendered.literal_spans.size(); ++i) {
        const fj::ByteSpan span = rendered.literal_spans[i];
        if (span.begin > span.end || span.end > size) { return "literal span out of range"; }
        if (span.begin == span.end) { return "empty literal span"; }
        if (i != 0 && rendered.literal_spans[i - 1].end > span.begin) {
            return "literal spans overlap or are unsorted";
        }
    }
    for (const fj::MediaPlaceholderByteSpec& placeholder : rendered.media_placeholders) {
        if (placeholder.bytes.begin > placeholder.bytes.end || placeholder.bytes.end > size) {
            return "media placeholder out of range";
        }
        const std::string_view bytes(rendered.text.data() + placeholder.bytes.begin,
                                     placeholder.bytes.end - placeholder.bytes.begin);
        if (bytes != "<|image_pad|>" && bytes != "<|video_pad|>") {
            return "media placeholder does not cover a pad token";
        }
    }
    for (const fj::MediaPlaceholderByteSpec& placeholder : rendered.media_placeholders) {
        for (const fj::ByteSpan span : rendered.literal_spans) {
            if (span.begin < placeholder.bytes.end && placeholder.bytes.begin < span.end) {
                return "literal span overlaps a media placeholder";
            }
        }
    }
    if (rendered.message_boundaries.size() != value.messages.size() + 1U) {
        return "message boundary count does not match the message count";
    }
    std::size_t previous = 0;
    for (const std::optional<std::size_t> boundary : rendered.message_boundaries) {
        if (!boundary) { continue; }
        if (*boundary > size) { return "message boundary out of range"; }
        if (*boundary < previous) { return "message boundaries decrease"; }
        previous = *boundary;
    }
    if (rendered.cache_boundaries.size() != value.options.cache_markers.size()) {
        return "cache boundary count does not match the marker count";
    }
    for (const std::optional<std::size_t> boundary : rendered.cache_boundaries) {
        if (boundary && *boundary > size) { return "cache boundary out of range"; }
    }
    previous = 0;
    for (const std::size_t boundary : rendered.rewrite_execution_boundaries) {
        if (boundary > size) { return "rewrite execution boundary out of range"; }
        if (boundary <= previous) { return "rewrite execution boundaries are not increasing"; }
        previous = boundary;
    }
    if (rendered.rewrite_checkpoint && rendered.rewrite_checkpoint->offset > size) {
        return "rewrite checkpoint out of range";
    }
    return std::nullopt;
}

std::optional<std::string> check_generation_state(const Case& value,
                                                  const fj::RenderedChat& rendered) {
    if (!value.options.add_generation_prompt ||
        value.options.continuation != ninfer::PromptContinuationMode::NewAssistantTurn) {
        return std::nullopt;
    }
    const std::string_view text = rendered.text;
    if (rendered.generation_starts_in_thinking) {
        if (!text.ends_with("<|im_start|>assistant\n<think>\n")) {
            return "generation_starts_in_thinking=true but the suffix does not open thinking";
        }
    } else if (!text.ends_with("<think>\n\n</think>\n\n")) {
        return "generation_starts_in_thinking=false but the suffix does not close thinking";
    }
    return std::nullopt;
}

std::optional<std::string> check_prefix_extension(const Case& value,
                                                  const fj::CompiledChatTemplate& renderer) {
    const bool preserve = value.options.froggeric_v225.preserve_reasoning
                              ? *value.options.froggeric_v225.preserve_reasoning
                              : value.options.preserve_thinking.value_or(true);
    if (!preserve || !value.options.add_generation_prompt ||
        value.options.continuation != ninfer::PromptContinuationMode::NewAssistantTurn) {
        return std::nullopt;
    }
    const std::optional<std::size_t> last_tag = last_inline_tag_index(value.messages);
    for (std::size_t k = 1; k < value.messages.size(); ++k) {
        if (value.messages[k].role != ninfer::ChatRole::Assistant) { continue; }
        if (last_tag && k <= *last_tag) { continue; }
        const std::vector<fj::ChatMessage> prefix(value.messages.begin(),
                                                  value.messages.begin() + static_cast<long>(k));
        const RenderResult left = render_checked(renderer, prefix, value.options);
        if (left.status != RenderStatus::Ok || !left.rendered.generation_starts_in_thinking) {
            continue;
        }
        const std::vector<fj::ChatMessage> extended(
            value.messages.begin(), value.messages.begin() + static_cast<long>(k) + 1);
        const RenderResult right = render_checked(renderer, extended, value.options);
        if (right.status != RenderStatus::Ok) { continue; }
        if (!right.rendered.text.starts_with(left.rendered.text)) {
            return "prompt before the assistant reply is not a prefix at message " +
                   std::to_string(k);
        }
    }
    return std::nullopt;
}

std::optional<std::string> check_checkpoint_stable(const Case& value,
                                                   const fj::CompiledChatTemplate& renderer) {
    if (value.messages.size() < 2) { return std::nullopt; }
    const RenderResult full = render_checked(renderer, value.messages, value.options);
    if (full.status != RenderStatus::Ok) { return std::nullopt; }
    const std::optional<std::size_t> last_tag = last_inline_tag_index(value.messages);
    for (std::size_t k = 1; k < value.messages.size(); ++k) {
        const ninfer::ChatRole before = value.messages[k - 1].role;
        const ninfer::ChatRole after  = value.messages[k].role;
        if (is_instruction(before) && is_instruction(after)) { continue; }
        if (before == ninfer::ChatRole::Tool && after == ninfer::ChatRole::Tool) { continue; }
        if (last_tag && k <= *last_tag) { continue; }
        fj::ChatRenderOptions prefix_options = value.options;
        prefix_options.add_generation_prompt = true;
        const std::vector<fj::ChatMessage> prefix(value.messages.begin(),
                                                  value.messages.begin() + static_cast<long>(k));
        const RenderResult prefix_render = render_checked(renderer, prefix, prefix_options);
        if (prefix_render.status != RenderStatus::Ok ||
            !prefix_render.rendered.rewrite_checkpoint) {
            continue;
        }
        const std::size_t checkpoint = prefix_render.rendered.rewrite_checkpoint->offset;
        const std::string stable = prefix_render.rendered.text.substr(0, checkpoint);
        if (!full.rendered.text.starts_with(stable)) {
            return "published rewrite checkpoint is not a stable prefix at message " +
                   std::to_string(k);
        }
    }
    return std::nullopt;
}

std::vector<std::pair<std::string, Property>> properties(
    const fj::CompiledChatTemplate& renderer) {
    std::vector<std::pair<std::string, Property>> out;
    out.emplace_back("determinism", [&renderer](const Case& value) {
        return check_determinism(value, renderer);
    });
    out.emplace_back("provenance", [](const Case& value) {
        const RenderResult rendered = render_checked(
            fj::CompiledChatTemplate::froggeric_v225(), value.messages, value.options);
        if (rendered.status != RenderStatus::Ok) { return std::optional<std::string>{}; }
        return check_provenance(value, rendered.rendered);
    });
    out.emplace_back("generation-state", [](const Case& value) {
        const RenderResult rendered = render_checked(
            fj::CompiledChatTemplate::froggeric_v225(), value.messages, value.options);
        if (rendered.status != RenderStatus::Ok) { return std::optional<std::string>{}; }
        return check_generation_state(value, rendered.rendered);
    });
    out.emplace_back("prefix-extension", [&renderer](const Case& value) {
        return check_prefix_extension(value, renderer);
    });
    out.emplace_back("checkpoint-stable", [&renderer](const Case& value) {
        return check_checkpoint_stable(value, renderer);
    });
    return out;
}

// ---------------------------------------------------------------------------------------------
// Deterministic branch exactness cases.
// ---------------------------------------------------------------------------------------------

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cout << "FAIL " << message << '\n';
    return 1;
}

int test_branch_exactness() {
    int failures                              = 0;
    const fj::CompiledChatTemplate renderer   = fj::CompiledChatTemplate::froggeric_v225();
    const std::string_view opener             = "<|im_start|>assistant\n";
    const std::string_view think_opener       = "<think>\n";

    for (const bool preserve : {true, false}) {
        Case value;
        value.messages.push_back(message(ninfer::ChatRole::User, "question"));
        value.options.preserve_thinking = preserve;
        const RenderResult base = render_checked(renderer, value.messages, value.options);
        failures += check(base.status == RenderStatus::Ok && base.rendered.rewrite_checkpoint,
                          "branch base publishes a rewrite checkpoint");
        if (base.status != RenderStatus::Ok || !base.rendered.rewrite_checkpoint) { continue; }
        const std::size_t checkpoint = base.rendered.rewrite_checkpoint->offset;
        std::vector<fj::ChatMessage> branched = value.messages;
        branched.push_back(message(ninfer::ChatRole::User, "follow-up"));
        const RenderResult branch = render_checked(renderer, branched, value.options);
        failures += check(branch.status == RenderStatus::Ok &&
                              branch.rendered.text.starts_with(
                                  base.rendered.text.substr(0, checkpoint)),
                          "branch keeps the stable checkpoint prefix");
        failures += check(branch.status == RenderStatus::Ok &&
                              !branch.rendered.text.starts_with(base.rendered.text.substr(
                                  0, checkpoint + opener.size())),
                          "branch diverges exactly at the generation checkpoint");
    }

    // Tool-group tail: preserve_thinking=false keeps the current turn's assistant thinking and
    // checkpoints before its opener; a later user branch drops that thinking, so the divergence
    // is immediately after the opener.
    Case tool_tail;
    tool_tail.messages.push_back(message(ninfer::ChatRole::User, "call the tool"));
    fj::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.tool_calls.push_back(fj::ToolCall{.name = "f", .arguments_json = "{}"});
    tool_tail.messages.push_back(std::move(assistant));
    tool_tail.messages.push_back(message(ninfer::ChatRole::Tool, "result"));
    tool_tail.options.preserve_thinking = false;
    tool_tail.options.tool_jsons.push_back(tool_json(0));
    const RenderResult base = render_checked(renderer, tool_tail.messages, tool_tail.options);
    failures += check(base.status == RenderStatus::Ok && base.rendered.rewrite_checkpoint &&
                          base.rendered.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
                      "tool-group tail publishes a TurnClosure checkpoint");
    if (base.status == RenderStatus::Ok && base.rendered.rewrite_checkpoint) {
        const std::size_t checkpoint = base.rendered.rewrite_checkpoint->offset;
        failures += check(base.rendered.text.compare(checkpoint, opener.size(), opener) == 0,
                          "tool-group checkpoint sits before the assistant opener");
        std::vector<fj::ChatMessage> branched = tool_tail.messages;
        branched.push_back(message(ninfer::ChatRole::User, "new question"));
        const RenderResult branch = render_checked(renderer, branched, tool_tail.options);
        failures += check(
            branch.status == RenderStatus::Ok &&
                branch.rendered.text.starts_with(base.rendered.text.substr(
                    0, checkpoint + opener.size())),
            "tool-group branch keeps the assistant opener");
        failures += check(
            branch.status == RenderStatus::Ok &&
                !branch.rendered.text.starts_with(base.rendered.text.substr(
                    0, checkpoint + opener.size() + think_opener.size())),
            "tool-group branch drops the unpreserved thinking after the opener");
    }
    return failures;
}

int test_all_whitespace_content() {
    int failures                            = 0;
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();
    const std::string empty_user =
        "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const std::string closed_empty_user = "<|im_start|>user\n<|im_end|>\n"
                                          "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    for (const std::string& text :
         {std::string("   "), std::string("\u00a0\u3000"), std::string("\t\n")}) {
        Case value;
        value.messages.push_back(message(ninfer::ChatRole::User, text));
        const RenderResult rendered = render_checked(renderer, value.messages, value.options);
        failures += check(rendered.status == RenderStatus::Ok &&
                              rendered.rendered.text == empty_user,
                          "all-whitespace user content trims to an empty turn");
    }
    Case tagged;
    tagged.messages.push_back(message(ninfer::ChatRole::User, "<|think_off|>   "));
    const RenderResult rendered = render_checked(renderer, tagged.messages, tagged.options);
    failures += check(rendered.status == RenderStatus::Ok &&
                          rendered.rendered.text == closed_empty_user,
                      "tag-only whitespace content trims to an empty closed turn");
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Driver and shrinker.
// ---------------------------------------------------------------------------------------------

Case shrink(const Case& original, const Property& fails) {
    Case best = original;
    bool changed = true;
    while (changed && best.messages.size() > 1) {
        changed = false;
        for (std::size_t i = 0; i < best.messages.size(); ++i) {
            Case candidate = best;
            candidate.messages.erase(candidate.messages.begin() + static_cast<long>(i));
            if (fails(candidate)) {
                best    = std::move(candidate);
                changed = true;
                break;
            }
        }
    }
    return best;
}

std::size_t env_size(const char* name, std::size_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    return static_cast<std::size_t>(std::strtoull(text, nullptr, 10));
}

std::uint64_t env_seed(const char* name, std::uint64_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') { return fallback; }
    return static_cast<std::uint64_t>(std::strtoull(text, nullptr, 0));
}

} // namespace

int main() {
    const std::uint64_t seed  = env_seed("NINFER_FROGGERIC_PROP_SEED", 0x5eed1234ULL);
    const std::size_t cases   = env_size("NINFER_FROGGERIC_PROP_CASES", 500);
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();

    int failures = test_branch_exactness();
    failures += test_all_whitespace_content();
    std::size_t rendered = 0;
    std::size_t rejected = 0;
    Rng rng(seed);
    for (std::size_t index = 0; index < cases; ++index) {
        const Case value = generate(rng);
        const RenderResult base = render_checked(renderer, value.messages, value.options);
        if (base.status == RenderStatus::Rejected) {
            ++rejected;
            continue;
        }
        if (base.status != RenderStatus::Ok) {
            std::cout << "FAIL case " << index << " render error: " << base.error << '\n';
            std::cout << "  seed=" << seed << " cases=" << cases << '\n';
            std::cout << "  input=" << describe(value) << '\n';
            ++failures;
            continue;
        }
        ++rendered;
        for (const auto& [name, property] : properties(renderer)) {
            const std::optional<std::string> failure = property(value);
            if (!failure) { continue; }
            ++failures;
            std::cout << "FAIL case " << index << " property " << name << ": " << *failure
                      << '\n';
            std::cout << "  seed=" << seed << " cases=" << cases << '\n';
            std::cout << "  input=" << describe(value) << '\n';
            const Case minimal = shrink(value, property);
            std::cout << "  minimal=" << describe(minimal) << '\n';
        }
    }

    std::cout << "froggeric v22.5 properties: " << rendered << " rendered, " << rejected
              << " rejected, " << failures << " failure(s) over " << cases << " cases (seed "
              << seed << ")\n";
    return failures == 0 ? 0 : 1;
}
