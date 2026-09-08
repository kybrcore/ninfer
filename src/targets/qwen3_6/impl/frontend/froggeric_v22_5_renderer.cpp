// Compiled qwen3.8-froggeric-v22.5 chat renderer.
//
// Byte parity is defined against the pinned upstream Jinja oracle:
//   froggeric/Qwen-Fixed-Chat-Templates @ 855bffc49448e299789730ff92c9b8d834d6cc14
// (chat_template.jinja / chat_template_oneline.txt). See
// tests/fixtures/frontend/froggeric_v22_5/PROVENANCE.md and tools/oracle_froggeric_v22_5/.
//
// This file is the orchestration only: Python string/tojson semantics live in
// froggeric_v22_5_python.*, inline-tag handling in froggeric_v22_5_tags.*, think extraction
// in froggeric_v22_5_think.*, and the pinned instruction texts in froggeric_v22_5_prompts.h.
// The shared render-fragment contract (literal spans, media placeholders, slicing) is in
// render_fragment.h and is also used by the artifact renderer.

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_prompts.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_python.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_tags.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_think.h"
#include "targets/qwen3_6/impl/frontend/render_fragment.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using njson = OracleJson;

// Rendered message content plus the raw layout needed to resolve cache markers after
// trimming and inline-tag removal.
struct ContentBlock {
    RenderedFragment fragment;                 // trimmed view
    std::size_t trim_begin = 0, trim_end = 0;  // raw layout
    std::vector<std::size_t> part_bounds_raw;  // raw layout
};

ContentBlock slice_block(const ContentBlock& source, std::size_t begin, std::size_t end) {
    ContentBlock out;
    out.fragment = slice_fragment(source.fragment, begin, end);
    return out;
}

ContentBlock render_content(const ChatMessage& message, bool add_vision_id, int* image_count,
                            int* video_count, std::size_t* media_count, bool strip_tags) {
    RenderBuilder raw;
    std::vector<std::size_t> part_bounds;
    for (const ChatPart& part : message.parts) {
        if (part.kind == ChatPartKind::Text) {
            raw.append_literal(part.text);
        } else if (part.kind == ChatPartKind::Image) {
            if (image_count != nullptr) { ++*image_count; }
            if (add_vision_id && image_count != nullptr) {
                raw.append_template("Picture " + std::to_string(*image_count) + ": ");
            }
            raw.append_template("<|vision_start|>");
            raw.append_media_placeholder("<|image_pad|>", Modality::Image,
                                         media_count != nullptr ? (*media_count)++ : 0);
            raw.append_template("<|vision_end|>");
        } else {
            if (video_count != nullptr) { ++*video_count; }
            if (add_vision_id && video_count != nullptr) {
                raw.append_template("Video " + std::to_string(*video_count) + ": ");
            }
            raw.append_template("<|vision_start|>");
            raw.append_media_placeholder("<|video_pad|>", Modality::Video,
                                         media_count != nullptr ? (*media_count)++ : 0);
            raw.append_template("<|vision_end|>");
        }
        part_bounds.push_back(raw.size());
    }
    RenderedFragment fragment = std::move(raw).release();

    // jinja trims the concatenated render, then strips inline tags on that view (system,
    // developer and user messages only). The stripper carries a source-offset map so the
    // rendered provenance stays exact even when a tag spans two text parts.
    std::size_t begin = 0, end = fragment.text.size();
    if (strip_tags && contains(fragment.text, "<|think_")) {
        TagStripper stripper(std::move(fragment.text));
        stripper.apply();
        std::vector<ByteSpan> mapped_literals;
        for (const ByteSpan span : fragment.literal_spans) {
            if (const std::optional<ByteSpan> mapped = stripper.map_span(span)) {
                append_literal_span(mapped_literals, *mapped);
            }
        }
        std::vector<MediaPlaceholderByteSpec> mapped_media;
        for (MediaPlaceholderByteSpec placeholder : fragment.media_placeholders) {
            const std::optional<ByteSpan> mapped = stripper.map_span(placeholder.bytes);
            if (!mapped || mapped->end - mapped->begin !=
                               placeholder.bytes.end - placeholder.bytes.begin) {
                throw std::logic_error("froggeric v22.5: inline tag removal crossed a media "
                                       "placeholder");
            }
            placeholder.bytes = *mapped;
            mapped_media.push_back(std::move(placeholder));
        }
        for (std::size_t& bound : part_bounds) { bound = stripper.map_position(bound); }
        fragment.text               = std::move(stripper).take_text();
        fragment.literal_spans      = std::move(mapped_literals);
        fragment.media_placeholders = std::move(mapped_media);
        end                         = fragment.text.size();
    } else {
        const auto [trim_begin, trim_end] = py_trim_bounds(fragment.text);
        begin                              = trim_begin;
        end                                = trim_end;
    }
    ContentBlock block;
    block.fragment        = slice_fragment(fragment, begin, end);
    block.trim_begin      = begin;
    block.trim_end        = end;
    block.part_bounds_raw = std::move(part_bounds);
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

// ---- render state and per-message handlers ---------------------------------------------

struct RenderState {
    RenderState(const ChatRenderOptions& render_options, std::size_t message_count)
        : options(render_options),
          message_boundaries(message_count + 1U),
          cache_boundaries(render_options.cache_markers.size()) {}

    const ChatRenderOptions& options;

    bool has_tools                = false;
    bool thinking                 = true;
    ReasoningEffort effort        = ReasoningEffort::Medium;
    bool preserve_thinking        = true;
    std::string_view reasoning_instructions;
    bool continue_final_assistant = false;

    std::size_t head_count                = 0;
    std::string merged_leading;
    std::size_t first_leading_part_length = 0;
    std::size_t first_body                = 0;
    long last_query_index                 = 0;
    std::size_t leading_instruction_begin = 0;

    RenderBuilder out;
    std::vector<std::size_t> tool_bounds;
    std::vector<std::size_t> execution_boundaries;
    std::vector<std::optional<std::size_t>> message_boundaries;
    std::vector<std::optional<std::size_t>> cache_boundaries;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;

    int image_count           = 0;
    int video_count           = 0;
    std::size_t media_count   = 0;
    long consecutive_failures = 0;
    bool prev_was_tool        = false;

    void mark_execution_boundary() {
        if (execution_boundaries.empty() || execution_boundaries.back() != out.size()) {
            execution_boundaries.push_back(out.size());
        }
    }
};

void resolve_render_state(const std::vector<ChatMessage>& messages, RenderState& state) {
    state.has_tools = !state.options.tool_jsons.empty();
    state.thinking  = state.options.enable_thinking;
    state.effort    = state.options.reasoning_effort.value_or(ReasoningEffort::Medium);
    if (state.options.froggeric_v225.auto_disable_thinking_with_tools && state.has_tools) {
        state.thinking = false;
    }
    state.preserve_thinking =
        state.options.froggeric_v225.preserve_reasoning.has_value()
            ? *state.options.froggeric_v225.preserve_reasoning
            : state.options.preserve_thinking.has_value() ? *state.options.preserve_thinking
                                                          : true;

    // Pre-scan for inline tags (system/developer/user text parts, in order).
    for (const ChatMessage& message : messages) {
        if (message.role != ChatRole::System && message.role != ChatRole::Developer &&
            message.role != ChatRole::User) {
            continue;
        }
        for (const ChatPart& part : message.parts) {
            if (part.kind == ChatPartKind::Text) {
                apply_tag_state(part.text, state.thinking, state.effort);
            }
        }
    }
    state.reasoning_instructions = reasoning_instructions_for(state.thinking, state.effort);
}

void merge_leading_instructions(const std::vector<ChatMessage>& messages, RenderState& state) {
    while (state.head_count < messages.size() &&
           (messages[state.head_count].role == ChatRole::System ||
            messages[state.head_count].role == ChatRole::Developer)) {
        ++state.head_count;
    }
    for (std::size_t i = 0; i < state.head_count; ++i) {
        validate_no_system_media(messages[i]);
        std::string text;
        for (const ChatPart& part : messages[i].parts) {
            if (part.kind == ChatPartKind::Text) { text += part.text; }
        }
        const auto [begin, end] = py_trim_bounds(text);
        std::string part_text   = strip_inline_tags(text.substr(begin, end - begin));
        if (i == 0) { state.first_leading_part_length = part_text.size(); }
        if (part_text.empty()) { continue; }
        if (!state.merged_leading.empty()) { state.merged_leading += "\n\n"; }
        state.merged_leading += part_text;
    }
    state.first_body = state.head_count;
}

void validate_continuation(const std::vector<ChatMessage>& messages, RenderState& state) {
    state.continue_final_assistant =
        state.options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (!state.continue_final_assistant) { return; }
    const ChatMessage& final = messages.back();
    if (final.role != ChatRole::Assistant || final.parts.empty() ||
        !final.reasoning_content.empty() || !final.tool_calls.empty() || final.has_media()) {
        throw std::invalid_argument(
            "assistant continuation requires a final text-only assistant message");
    }
    if (state.options.enable_thinking) {
        throw std::invalid_argument("assistant continuation cannot start in thinking mode");
    }
}

void compute_last_query_index(const std::vector<ChatMessage>& messages, RenderState& state) {
    const long body_count = static_cast<long>(messages.size()) -
                            static_cast<long>(state.first_body);
    bool multi_step_tool = true;
    for (long i = body_count - 1; i >= 0 && multi_step_tool; --i) {
        const ChatMessage& message = messages[state.first_body + static_cast<std::size_t>(i)];
        if (message.role != ChatRole::User) { continue; }
        std::string raw;
        for (const ChatPart& part : message.parts) {
            if (part.kind == ChatPartKind::Text) {
                raw += part.text;
            } else if (part.kind == ChatPartKind::Image) {
                raw += "<|vision_start|><|image_pad|><|vision_end|>";
            } else {
                raw += "<|vision_start|><|video_pad|><|vision_end|>";
            }
        }
        const auto [begin, end] = py_trim_bounds(raw);
        const std::string_view rendered = std::string_view(raw).substr(begin, end - begin);
        const bool is_tool_query =
            rendered.size() >= std::string_view("<tool_response>").size() +
                                   std::string_view("</tool_response>").size() &&
            rendered.substr(0, std::string_view("<tool_response>").size()) == "<tool_response>" &&
            rendered.substr(rendered.size() - std::string_view("</tool_response>").size()) ==
                "</tool_response>";
        if (!is_tool_query) {
            multi_step_tool        = false;
            state.last_query_index = i;
        }
    }
    if (multi_step_tool) {
        state.last_query_index = (body_count - 1) > 50 ? body_count - 1 : 0;
    }
}

void render_system_block(RenderState& state) {
    if (state.has_tools) {
        state.out.append_template("<|im_start|>system\n");
        if (!state.reasoning_instructions.empty()) {
            state.out.append_template(state.reasoning_instructions);
            state.out.append_template("\n\n");
        }
        state.out.append_template(kToolsHeader);
        for (const std::string& tool_json : state.options.tool_jsons) {
            state.out.append_template("\n");
            state.out.append_literal(tojson_oracle(njson::parse(tool_json)));
            state.tool_bounds.push_back(state.out.size());
        }
        state.out.append_template("\n</tools>");
        state.out.append_template(
            state.options.froggeric_v225.tool_call_format == ToolCallFormat::Json
                ? (state.thinking ? kJsonInstructionsThinking : kJsonInstructionsOff)
                : (state.thinking ? kXmlInstructionsThinking : kXmlInstructionsOff));
        if (!state.merged_leading.empty()) {
            state.out.append_template("\n\n");
            state.leading_instruction_begin = state.out.size();
            state.out.append_literal(state.merged_leading);
        }
        state.out.append_template("<|im_end|>\n");
        return;
    }
    if (!state.merged_leading.empty()) {
        state.out.append_template("<|im_start|>system\n");
        if (!state.reasoning_instructions.empty()) {
            state.out.append_template(state.reasoning_instructions);
            state.out.append_template("\n\n");
        }
        state.leading_instruction_begin = state.out.size();
        state.out.append_literal(state.merged_leading);
        state.out.append_template("<|im_end|>\n");
        return;
    }
    if (!state.reasoning_instructions.empty()) {
        state.out.append_template("<|im_start|>system\n");
        state.out.append_template(state.reasoning_instructions);
        state.out.append_template("<|im_end|>\n");
    }
}

void render_tool_call(RenderState& state, const ToolCall& call, bool first, bool body_has_text) {
    if (state.options.froggeric_v225.tool_call_format == ToolCallFormat::Json) {
        if (first) {
            if (body_has_text) { state.out.append_template("\n\n"); }
        } else {
            state.out.append_template("\n");
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
            } catch (const njson::exception&) { args = call.arguments_json; }
        }
        state.out.append_template("<tool_call>\n{\"name\": ");
        state.out.append_literal(tojson_oracle(njson(call.name)));
        state.out.append_template(", \"arguments\": ");
        state.out.append_literal(args);
        state.out.append_template("}\n</tool_call>");
        return;
    }
    if (first) {
        if (body_has_text) { state.out.append_template("\n\n"); }
    } else {
        state.out.append_template("\n");
    }
    state.out.append_template("<tool_call>\n<function=");
    state.out.append_literal(call.name);
    state.out.append_template(">\n");
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
        state.out.append_template("<parameter=");
        state.out.append_literal(param_name);
        state.out.append_template(">\n");
        const std::string value_text = param_value.is_string() ? param_value.get<std::string>()
                                                               : tojson_oracle(param_value);
        if (state.options.froggeric_v225.max_tool_arg_chars > 0 &&
            py_len(value_text) > state.options.froggeric_v225.max_tool_arg_chars) {
            state.out.append_literal(
                truncate_python(value_text, state.options.froggeric_v225.max_tool_arg_chars));
        } else {
            state.out.append_literal(value_text);
        }
        state.out.append_template("\n</parameter>\n");
    }
    state.out.append_template("</function>\n</tool_call>");
}

void resolve_part_boundaries(RenderState& state, std::size_t message_index,
                             const ContentBlock& content, std::size_t content_begin) {
    for (std::size_t marker_index = 0; marker_index < state.options.cache_markers.size();
         ++marker_index) {
        const PromptCacheMarker& marker = state.options.cache_markers[marker_index];
        if (marker.location != PromptCacheMarkerLocation::MessagePartBoundary ||
            marker.after_message_count != message_index + 1U ||
            marker.after_message_part_count == 0 ||
            marker.after_message_part_count > content.part_bounds_raw.size()) {
            continue;
        }
        const std::size_t raw = content.part_bounds_raw[marker.after_message_part_count - 1U];
        const std::size_t clamped = std::clamp(raw, content.trim_begin, content.trim_end);
        state.cache_boundaries[marker_index] = content_begin + clamped - content.trim_begin;
    }
}

void render_instruction(RenderState& state, std::size_t i, const ContentBlock& content) {
    state.out.append_template("<|im_start|>system\n");
    const std::size_t content_begin = state.out.size();
    state.out.append(content.fragment);
    resolve_part_boundaries(state, i, content, content_begin);
    state.out.append_template("<|im_end|>\n");
    state.message_boundaries[i + 1U] = state.out.size();
    state.prev_was_tool              = false;
}

void render_user(RenderState& state, std::size_t i, const ContentBlock& content) {
    state.consecutive_failures = 0;
    state.out.append_template("<|im_start|>user\n");
    const std::size_t content_begin = state.out.size();
    state.out.append(content.fragment);
    resolve_part_boundaries(state, i, content, content_begin);
    state.out.append_template("<|im_end|>\n");
    state.message_boundaries[i + 1U] = state.out.size();
    state.prev_was_tool              = false;
}

void update_tool_failures(const ContentBlock& content, RenderState& state) {
    const std::string lower = py_lower(content.fragment.text);
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
    const bool weak_error = contains(head, "error:") || contains(head, "err!");
    const bool weak_suppressed =
        contains(head, "$ ") || contains(head, "took ") || py_len(content.fragment.text) >= 600;
    if (!is_code_or_grep && (strong_error || (weak_error && !weak_suppressed))) {
        ++state.consecutive_failures;
    } else {
        state.consecutive_failures = 0;
    }
}

void render_tool(const std::vector<ChatMessage>& messages, RenderState& state, std::size_t i,
                 const ContentBlock& content) {
    update_tool_failures(content, state);

    if (!state.prev_was_tool) { state.out.append_template("<|im_start|>user"); }
    const bool is_json_payload =
        state.options.froggeric_v225.tool_call_format == ToolCallFormat::Json &&
        !content.fragment.text.empty() &&
        (content.fragment.text.front() == '{' || content.fragment.text.front() == '[');
    const bool truncate =
        !is_json_payload && state.options.froggeric_v225.max_tool_response_chars > 0 &&
        py_len(content.fragment.text) > state.options.froggeric_v225.max_tool_response_chars;
    state.out.append_template("\n<tool_response>\n");
    if (truncate) {
        const std::size_t keep =
            py_prefix_bytes(content.fragment.text,
                            state.options.froggeric_v225.max_tool_response_chars);
        for (const MediaPlaceholderByteSpec& placeholder : content.fragment.media_placeholders) {
            if (placeholder.bytes.end > keep) {
                throw std::invalid_argument(
                    "max_tool_response_chars truncates a media placeholder");
            }
        }
        state.out.append(slice_block(content, 0, keep).fragment);
        state.out.append_template(truncation_notice(content.fragment.text));
    } else {
        state.out.append(content.fragment);
    }
    if (state.consecutive_failures >= 2) {
        state.out.append_template("\n\n⚠️ SYSTEM WARNING: ");
        state.out.append_template(std::to_string(state.consecutive_failures));
        state.out.append_template(" consecutive tool errors detected. Your previous approach "
                                  "is incorrect. You MUST use a fundamentally different "
                                  "approach or corrected arguments.");
    } else if (state.consecutive_failures == 1) {
        state.out.append_template("\n\n⚠️ SYSTEM WARNING: The previous tool call returned an "
                                  "error. Diagnose the failure and retry with completely "
                                  "corrected arguments.");
    }
    state.out.append_template("\n</tool_response>");
    const bool closes_group = i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
    if (closes_group) { state.out.append_template("<|im_end|>\n"); }
    state.message_boundaries[i + 1U] = state.out.size();
    state.prev_was_tool              = true;
}

void render_assistant(const std::vector<ChatMessage>& messages, RenderState& state, std::size_t i,
                      const ContentBlock& content) {
    const long body_index = static_cast<long>(i) - static_cast<long>(state.first_body);
    if (state.continue_final_assistant && i + 1U == messages.size()) {
        state.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind   = RewriteCheckpointKind::ResponseReplay,
            .offset = static_cast<std::uint32_t>(state.out.size())};
        state.out.append_template("<|im_start|>assistant\n");
        state.mark_execution_boundary();
        state.out.append(content.fragment);
        state.message_boundaries[i + 1U] = state.out.size();
        state.prev_was_tool              = false;
        return;
    }

    const ThinkExtraction think =
        extract_think(messages[i].reasoning_content, content.fragment.text);
    const bool keep_thinking = state.preserve_thinking || body_index > state.last_query_index;
    if (!state.preserve_thinking && !state.rewrite_checkpoint &&
        body_index > state.last_query_index) {
        state.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind   = RewriteCheckpointKind::TurnClosure,
            .offset = static_cast<std::uint32_t>(state.out.size())};
    }
    state.out.append_template("<|im_start|>assistant\n");
    state.mark_execution_boundary();
    if (keep_thinking) {
        state.out.append_template("<think>\n");
        state.mark_execution_boundary();
        if (!think.explicit_reasoning.empty()) {
            state.out.append_literal(think.explicit_reasoning);
        } else {
            state.out.append(slice_block(content, think.reasoning_begin, think.reasoning_end)
                                 .fragment);
        }
        state.out.append_template("\n</think>\n\n");
        state.mark_execution_boundary();
    }
    const ContentBlock body =
        slice_block(content, think.body_begin, content.fragment.text.size());
    state.out.append(body.fragment);
    const auto [body_begin, body_end] = py_trim_bounds(body.fragment.text);
    const bool body_has_text          = body_end > body_begin;
    for (std::size_t call_index = 0; call_index < messages[i].tool_calls.size(); ++call_index) {
        render_tool_call(state, messages[i].tool_calls[call_index], call_index == 0, body_has_text);
    }
    state.out.append_template("<|im_end|>\n");
    state.message_boundaries[i + 1U] = state.out.size();
    state.prev_was_tool              = false;
}

void render_message(const std::vector<ChatMessage>& messages, RenderState& state, std::size_t i) {
    const ChatMessage& message = messages[i];
    const bool is_instruction  = message.role == ChatRole::System ||
                                 message.role == ChatRole::Developer;
    if (is_instruction) { validate_no_system_media(message); }
    const ContentBlock content = render_content(
        message, state.options.add_vision_id, &state.image_count, &state.video_count,
        &state.media_count, is_instruction || message.role == ChatRole::User);

    if (is_instruction) { render_instruction(state, i, content); return; }
    if (message.role == ChatRole::User) { render_user(state, i, content); return; }
    if (message.role == ChatRole::Tool) {
        render_tool(messages, state, i, content);
        return;
    }
    if (message.role != ChatRole::Assistant) {
        throw std::invalid_argument("unsupported chat role value");
    }
    render_assistant(messages, state, i, content);
}

void render_generation_suffix(RenderState& state) {
    const std::uint32_t generation_begin = static_cast<std::uint32_t>(state.out.size());
    if (state.preserve_thinking) {
        state.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
    } else if (!state.rewrite_checkpoint) {
        state.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind = RewriteCheckpointKind::TurnClosure, .offset = generation_begin};
    }
    state.out.append_template("<|im_start|>assistant\n");
    state.mark_execution_boundary();
    state.out.append_template("<think>\n");
    state.mark_execution_boundary();
    if (!state.thinking) {
        state.out.append_template("\n</think>\n\n");
        state.mark_execution_boundary();
    }
}

void resolve_cache_markers(RenderState& state) {
    for (std::size_t index = 0; index < state.options.cache_markers.size(); ++index) {
        const PromptCacheMarker& marker = state.options.cache_markers[index];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < state.message_boundaries.size()) {
                state.cache_boundaries[index] =
                    state.message_boundaries[marker.after_message_count];
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count != 0 && marker.after_tool_count <= state.tool_bounds.size()) {
                state.cache_boundaries[index] = state.tool_bounds[marker.after_tool_count - 1U];
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (state.leading_instruction_begin != 0 && state.head_count > 0) {
                const std::size_t clamped = std::clamp<std::size_t>(
                    marker.leading_instruction_bytes, 0U, state.first_leading_part_length);
                state.cache_boundaries[index] = state.leading_instruction_begin + clamped;
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            break; // resolved during the main loop.
        }
    }
}

} // namespace

RenderedChat render_froggeric_v225(const std::vector<ChatMessage>& messages,
                                   const ChatRenderOptions& options) {
    if (messages.empty()) { throw std::invalid_argument("No messages provided."); }

    RenderState state(options, messages.size());
    resolve_render_state(messages, state);
    merge_leading_instructions(messages, state);
    validate_continuation(messages, state);
    compute_last_query_index(messages, state);

    render_system_block(state);
    if (state.head_count == 0) {
        state.message_boundaries[0] = 0;
    } else {
        state.message_boundaries[state.head_count] = state.out.size();
    }

    for (std::size_t i = state.first_body; i < messages.size(); ++i) {
        render_message(messages, state, i);
    }
    if (!state.continue_final_assistant && options.add_generation_prompt) {
        render_generation_suffix(state);
    }
    resolve_cache_markers(state);

    RenderedFragment final = std::move(state.out).release();
    return RenderedChat{.text                         = std::move(final.text),
                        .literal_spans                = std::move(final.literal_spans),
                        .media_placeholders           = std::move(final.media_placeholders),
                        .rewrite_checkpoint           = state.rewrite_checkpoint,
                        .rewrite_execution_boundaries = std::move(state.execution_boundaries),
                        .message_boundaries           = std::move(state.message_boundaries),
                        .cache_boundaries             = std::move(state.cache_boundaries),
                        .generation_starts_in_thinking = state.thinking};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
