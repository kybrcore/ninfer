#pragma once

#include <cstddef>
#include <string>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Assistant think-block extraction. The ranges are offsets into the trimmed rendered
// content so the caller can slice the block and keep its literal/media provenance.
struct ThinkExtraction {
    std::string explicit_reasoning; // trimmed message.reasoning_content, empty when derived
    std::size_t reasoning_begin = 0;
    std::size_t reasoning_end   = 0;
    std::size_t body_begin      = 0;
};

ThinkExtraction extract_think(const std::string& explicit_reasoning, const std::string& body);

} // namespace ninfer::targets::qwen3_6::frontend_internal
