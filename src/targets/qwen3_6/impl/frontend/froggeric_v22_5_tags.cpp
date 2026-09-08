#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_tags.h"

#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {

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

std::string strip_inline_tags(std::string rendered) {
    if (!contains(rendered, "<|think_")) { return rendered; }
    TagStripper stripper(std::move(rendered));
    stripper.apply();
    return std::move(stripper).take_text();
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
