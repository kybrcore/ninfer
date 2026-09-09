// Manual benchmark for the compiled froggeric v22.5 renderer.
//
// Not registered with ctest: build it with the test suite and run it directly.
//
//   cmake --build build --target ninfer_qwen3_6_froggeric_v22_5_bench
//   ./build/tests/ninfer_qwen3_6_froggeric_v22_5_bench
//
// It reports median/max render time, peak heap allocation during one render, and rendered bytes
// for 1K/100K/500K/2M inputs, with and without inline tags, plus image and tool variants. Peak
// allocation is a proxy measured with global new/delete counters in this single-threaded process;
// the per-byte TagStripper origin map is 8 bytes per input byte whenever a tag is present. The
// measured upper bound and the segmented-map decision are recorded in
// docs/maintainer/frontend-chat-rendering.md.

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kAlign = alignof(std::max_align_t);

struct AllocationHeader {
    void* raw = nullptr;
    std::size_t size = 0;
};

std::size_t g_current = 0;
std::size_t g_peak = 0;

void* allocate(std::size_t size) {
    const std::size_t total = size + sizeof(AllocationHeader) + kAlign;
    void* raw = std::malloc(total);
    if (raw == nullptr) { throw std::bad_alloc(); }
    const auto base = reinterpret_cast<std::uintptr_t>(raw) + sizeof(AllocationHeader);
    const auto aligned = (base + kAlign - 1) & ~(static_cast<std::uintptr_t>(kAlign) - 1);
    auto* header = reinterpret_cast<AllocationHeader*>(aligned - sizeof(AllocationHeader));
    header->raw = raw;
    header->size = size;
    g_current += size;
    g_peak = std::max(g_peak, g_current);
    return reinterpret_cast<void*>(aligned);
}

void release(void* ptr) noexcept {
    if (ptr == nullptr) { return; }
    auto* header = reinterpret_cast<AllocationHeader*>(
        reinterpret_cast<std::uintptr_t>(ptr) - sizeof(AllocationHeader));
    g_current -= header->size;
    std::free(header->raw);
}

} // namespace

void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void operator delete(void* ptr) noexcept { release(ptr); }
void operator delete[](void* ptr) noexcept { release(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { release(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { release(ptr); }

namespace {

namespace fj = ninfer::targets::qwen3_6::frontend_internal;

const std::string kTestToolJson =
    R"({"type":"function","function":{"name":"f","description":"benchmark tool","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}})";

std::string filler(std::size_t bytes, bool with_tags) {
    static const std::string_view kSeed =
        "The quick brown fox jumps over the lazy dog. 0123456789 abcdefghijklmnopqrstuvwxyz. ";
    std::string out;
    out.reserve(bytes + 64);
    while (out.size() < bytes) { out += kSeed; }
    out.resize(bytes);
    if (with_tags) {
        // Ten inline tags spread through the text: enough to force the TagStripper path.
        const std::size_t step = bytes / 11;
        for (std::size_t i = 1; i <= 10; ++i) {
            const std::size_t at = i * step;
            if (at + 13 <= out.size()) { out.insert(at, "<|think_low|>"); }
        }
    }
    return out;
}

struct Case {
    std::string name;
    std::vector<fj::ChatMessage> messages;
    fj::ChatRenderOptions options;
};

Case text_case(std::string name, std::size_t bytes, bool with_tags) {
    Case out;
    out.name = std::move(name);
    fj::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(fj::ChatPart::text_part(filler(bytes, with_tags)));
    out.messages.push_back(std::move(user));
    return out;
}

std::vector<Case> build_cases() {
    std::vector<Case> cases;
    for (const std::size_t bytes :
         {std::size_t{1} << 10, std::size_t{100} << 10, std::size_t{500} << 10,
          std::size_t{2} << 20}) {
        const std::string suffix = std::to_string(bytes / 1024) + "k";
        cases.push_back(text_case("plain-" + suffix, bytes, false));
        cases.push_back(text_case("tags-" + suffix, bytes, true));
    }

    Case image = text_case("image-tags-100k", std::size_t{100} << 10, true);
    image.messages.front().parts.push_back(fj::ChatPart::image(fj::MediaData{}));
    cases.push_back(std::move(image));

    Case xml = text_case("tools-xml-100k", std::size_t{100} << 10, true);
    xml.options.tool_jsons.push_back(kTestToolJson);
    cases.push_back(std::move(xml));

    Case json = text_case("tools-json-100k", std::size_t{100} << 10, true);
    json.options.tool_jsons.push_back(kTestToolJson);
    json.options.froggeric_v225.tool_call_format = ninfer::ToolCallFormat::Json;
    cases.push_back(std::move(json));

    Case preserve_off = text_case("preserve-off-100k", std::size_t{100} << 10, true);
    preserve_off.options.preserve_thinking = false;
    cases.push_back(std::move(preserve_off));
    return cases;
}

void run(const fj::CompiledChatTemplate& renderer, const Case& benchmark) {
    const std::size_t input_bytes =
        benchmark.messages.front().parts.front().text.size();
    const std::size_t target_bytes = 64U << 20; // ~64 MiB of rendered input per case
    const std::size_t iterations = std::clamp<std::size_t>(
        target_bytes / std::max<std::size_t>(input_bytes, 1), 5, 500);

    // Warm up caches and allocator state.
    std::size_t rendered_bytes = 0;
    for (int i = 0; i < 3; ++i) {
        rendered_bytes = renderer.render(benchmark.messages, benchmark.options).text.size();
    }

    g_peak = g_current;
    const std::size_t before = g_current;
    const std::size_t peak_before = g_peak;
    std::vector<double> samples;
    samples.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        rendered_bytes = renderer.render(benchmark.messages, benchmark.options).text.size();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
    }
    const std::size_t peak = g_peak - before;
    (void)peak_before;
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double worst  = samples.back();

    std::cout << benchmark.name;
    for (std::size_t pad = benchmark.name.size(); pad < 20; ++pad) { std::cout << ' '; }
    std::cout << " input=" << input_bytes << "B rendered=" << rendered_bytes
              << "B median=" << median << "us worst=" << worst << "us peak=" << (peak >> 10)
              << "KiB peak/input=" << (static_cast<double>(peak) / input_bytes) << "x\n";
}

} // namespace

int main() {
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();
    std::cout << "froggeric v22.5 render benchmark (single-threaded, global new/delete proxy)\n";
    for (const Case& benchmark : build_cases()) { run(renderer, benchmark); }
    return 0;
}
