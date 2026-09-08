#pragma once

#include "targets/qwen3_6/impl/frontend/froggeric_v22_5/python.h"
#include "targets/qwen3_6/impl/frontend/render_fragment.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

inline constexpr std::string_view kThinkOff     = "<|think_off|>";
inline constexpr std::string_view kThinkOn      = "<|think_on|>";
inline constexpr std::string_view kThinkXHigh   = "<|think_xhigh|>";
inline constexpr std::string_view kThinkHigh    = "<|think_high|>";
inline constexpr std::string_view kThinkUltra   = "<|think_ultracode|>";
inline constexpr std::string_view kThinkExtreme = "<|think_extreme|>";
inline constexpr std::string_view kThinkMax     = "<|think_max|>";
inline constexpr std::string_view kThinkLow     = "<|think_low|>";
inline constexpr std::string_view kThinkMinimal = "<|think_minimal|>";
inline constexpr std::string_view kThinkMedium  = "<|think_medium|>";

// Template order (the pre-scan priority chain in apply_tag_state is separate, exactly like
// the jinja).
inline constexpr std::string_view kInlineTags[] = {kThinkOff,  kThinkOn,     kThinkXHigh,
                                                   kThinkHigh, kThinkUltra,  kThinkExtreme,
                                                   kThinkMax,  kThinkMedium, kThinkLow,
                                                   kThinkMinimal};

// Template pre-scan: one if/elif chain per text item, in item order.
void apply_tag_state(const std::string& text, bool& thinking, ReasoningEffort& effort);

// jinja `content.split(tag) | join('') | trim` for every tag, in template order. The
// origin map records each surviving byte's offset in the input, so literal spans, media
// placeholders and part boundaries can be remapped after the removal.
//
// Per-byte map cost (bench_froggeric_v22_5.cpp, 2026-09-08, clang -O2, Apple M-series): a tagged
// 2 MiB input peaks at ~36 MiB (18x input) and renders in ~8.3 ms, versus ~8 MiB (4x) and
// ~0.8 ms untagged. The 100 KiB production-sized case is ~0.4 ms / ~1.8 MiB (18x). The per-byte
// origin vector (8 B/byte) plus remove_all's kept_origin copy dominates both. The measured bound
// is recorded instead of a run-based map: the absolute cost is negligible at production sizes,
// and a segmented rewrite would touch the provenance path that the property tests protect.
// Revisit if multi-megabyte tagged prompts become a workload.
class TagStripper {
public:
    explicit TagStripper(std::string text) : text_(std::move(text)) {
        origin_.resize(text_.size());
        for (std::size_t i = 0; i < origin_.size(); ++i) { origin_[i] = i; }
    }

    void apply() {
        for (const std::string_view tag : kInlineTags) {
            if (!contains(text_, tag)) { continue; }
            remove_all(tag);
            trim();
        }
    }

    [[nodiscard]] std::string take_text() && { return std::move(text_); }

    [[nodiscard]] std::size_t map_position(std::size_t source) const {
        return static_cast<std::size_t>(
            std::lower_bound(origin_.begin(), origin_.end(), source) - origin_.begin());
    }

    [[nodiscard]] std::optional<ByteSpan> map_span(ByteSpan span) const {
        const std::size_t begin = map_position(span.begin);
        const std::size_t end   = map_position(span.end);
        if (begin == end) { return std::nullopt; }
        return ByteSpan{begin, end};
    }

private:
    // One left-to-right pass, like str.split(tag) | join(''): a junction created by the
    // removal is not rescanned for the same tag (a later tag may still match it).
    void remove_all(std::string_view tag) {
        std::string kept;
        std::vector<std::size_t> kept_origin;
        kept.reserve(text_.size());
        kept_origin.reserve(origin_.size());
        std::size_t index = 0;
        while (index < text_.size()) {
            if (std::string_view(text_).substr(index, tag.size()) == tag) {
                index += tag.size();
                continue;
            }
            kept.push_back(text_[index]);
            kept_origin.push_back(origin_[index]);
            ++index;
        }
        text_   = std::move(kept);
        origin_ = std::move(kept_origin);
    }

    void trim() {
        const auto [begin, end] = py_trim_bounds(text_);
        text_.erase(end);
        origin_.erase(origin_.begin() + static_cast<std::ptrdiff_t>(end), origin_.end());
        text_.erase(0, begin);
        origin_.erase(origin_.begin(), origin_.begin() + static_cast<std::ptrdiff_t>(begin));
    }

    std::string text_;
    std::vector<std::size_t> origin_;
};

std::string strip_inline_tags(std::string rendered);

} // namespace ninfer::targets::qwen3_6::frontend_internal
