#pragma once

#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct MediaPlaceholderByteSpec {
    ByteSpan bytes;
    Modality modality      = Modality::Image;
    std::size_t item_index = 0;
};

// Rendered prompt bytes with client-origin provenance. Shared by the artifact renderer and the
// compiled froggeric v22.5 renderer: both append template text, literal content and media
// placeholders through RenderBuilder so the Processor and the context cache see one contract.
struct RenderedFragment {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
};

void append_literal_span(std::vector<ByteSpan>& spans, ByteSpan span);

RenderedFragment literal_fragment(std::string text);

std::pair<std::size_t, std::size_t> trim_ascii_whitespace_bounds(std::string_view text);

// Slices a fragment and rebases its provenance. A slice may not cut a media placeholder: the
// Processor replaces exactly that byte range.
RenderedFragment slice_fragment(const RenderedFragment& source, std::size_t begin, std::size_t end);

RenderedFragment trim_ascii_whitespace(const RenderedFragment& fragment);

class RenderBuilder {
public:
    void append_template(std::string_view text) { fragment_.text += text; }

    void append_literal(std::string_view text) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        append_literal_span(fragment_.literal_spans, ByteSpan{begin, fragment_.text.size()});
    }

    void append_media_placeholder(std::string_view text, Modality modality,
                                  std::size_t item_index) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        fragment_.media_placeholders.push_back(MediaPlaceholderByteSpec{
            .bytes      = ByteSpan{begin, fragment_.text.size()},
            .modality   = modality,
            .item_index = item_index,
        });
    }

    void append(RenderedFragment fragment) {
        const std::size_t offset = fragment_.text.size();
        fragment_.text += fragment.text;
        for (const ByteSpan span : fragment.literal_spans) {
            append_literal_span(fragment_.literal_spans,
                                ByteSpan{offset + span.begin, offset + span.end});
        }
        for (MediaPlaceholderByteSpec placeholder : fragment.media_placeholders) {
            placeholder.bytes.begin += offset;
            placeholder.bytes.end += offset;
            fragment_.media_placeholders.push_back(placeholder);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return fragment_.text.size(); }

    [[nodiscard]] RenderedFragment release() && { return std::move(fragment_); }

private:
    RenderedFragment fragment_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
