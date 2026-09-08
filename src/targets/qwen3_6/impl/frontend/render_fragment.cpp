#include "targets/qwen3_6/impl/frontend/render_fragment.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::frontend_internal {

void append_literal_span(std::vector<ByteSpan>& spans, ByteSpan span) {
    if (span.begin == span.end) { return; }
    if (!spans.empty() && spans.back().end == span.begin) {
        spans.back().end = span.end;
        return;
    }
    if (!spans.empty() && spans.back().end > span.begin) {
        throw std::logic_error("rendered literal byte spans overlap");
    }
    spans.push_back(span);
}

RenderedFragment literal_fragment(std::string text) {
    RenderBuilder builder;
    builder.append_literal(text);
    return std::move(builder).release();
}

std::pair<std::size_t, std::size_t> trim_ascii_whitespace_bounds(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return {begin, end};
}

RenderedFragment slice_fragment(const RenderedFragment& source, std::size_t begin,
                                std::size_t end) {
    if (begin > end || end > source.text.size()) {
        throw std::logic_error("rendered fragment slice is out of range");
    }
    RenderedFragment result;
    result.text = source.text.substr(begin, end - begin);
    for (const ByteSpan span : source.literal_spans) {
        const std::size_t clipped_begin = std::max(span.begin, begin);
        const std::size_t clipped_end   = std::min(span.end, end);
        if (clipped_begin < clipped_end) {
            append_literal_span(result.literal_spans,
                                ByteSpan{clipped_begin - begin, clipped_end - begin});
        }
    }
    for (MediaPlaceholderByteSpec placeholder : source.media_placeholders) {
        if (placeholder.bytes.end <= begin || placeholder.bytes.begin >= end) { continue; }
        if (placeholder.bytes.begin < begin || placeholder.bytes.end > end) {
            throw std::logic_error("rendered fragment slice intersects a media placeholder");
        }
        placeholder.bytes.begin -= begin;
        placeholder.bytes.end -= begin;
        result.media_placeholders.push_back(placeholder);
    }
    return result;
}

RenderedFragment trim_ascii_whitespace(const RenderedFragment& fragment) {
    const auto [begin, end] = trim_ascii_whitespace_bounds(fragment.text);
    return slice_fragment(fragment, begin, end);
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
