// Froggeric v22.5 renderer parity test: byte-for-byte comparison of the compiled renderer
// against the goldens produced by the pinned upstream Jinja oracle (see
// tests/fixtures/frontend/froggeric_v22_5/PROVENANCE.md and tools/oracle_froggeric_v22_5/).
//
// Inputs are the oracle-facing JSON documents (OpenAI wire shapes). The harness maps them onto
// the C++ wire model; inputs marked "oracle_only" exercise boundaries that model cannot express
// (closed ChatRole enum, raw non-object XML argument strings) and are skipped here but remain
// part of the upstream oracle surface.

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fj = ninfer::targets::qwen3_6::frontend_internal;
using njson  = nlohmann::ordered_json;

constexpr std::array<std::uint8_t, 32> kPrettyFixtureSha{
    0xe5, 0x76, 0x84, 0xba, 0xe4, 0x15, 0x62, 0x11, 0xa5, 0x54, 0x73, 0xc5, 0xa6, 0x3b, 0xe9, 0x76,
    0xa4, 0x05, 0xa3, 0x7a, 0xb5, 0xbe, 0x5a, 0xe0, 0xe5, 0xab, 0xf1, 0xdf, 0x53, 0x49, 0xc4, 0xb2};
constexpr std::array<std::uint8_t, 32> kOnelineFixtureSha{
    0xee, 0xca, 0xe0, 0xe0, 0x68, 0xe6, 0x0f, 0x9c, 0x86, 0x65, 0xf0, 0x08, 0x5b, 0x58, 0x9d, 0x3e,
    0x1c, 0x41, 0x50, 0x8d, 0x35, 0x97, 0x76, 0xc6, 0x20, 0x18, 0x51, 0x2c, 0x40, 0xb5, 0x87, 0x9b};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

bool sha256_file(const std::filesystem::path& path, const std::array<std::uint8_t, 32>& expected,
                 int& failures) {
    const std::string bytes = read_file(path);
    const fj::Sha256Digest digest =
        fj::sha256(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                                 bytes.size()));
    if (digest != expected) {
        std::cout << "FIXTURE DRIFT: " << path.filename().string() << " sha256 "
                  << fj::sha256_hex(digest) << '\n';
        ++failures;
        return false;
    }
    return true;
}

const njson& sub(const njson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) {
        throw std::runtime_error(std::string("oracle input missing key ") + key);
    }
    return object.at(key);
}

ninfer::ChatRole parse_role(const std::string& text) {
    if (text == "system") { return ninfer::ChatRole::System; }
    if (text == "developer") { return ninfer::ChatRole::Developer; }
    if (text == "user") { return ninfer::ChatRole::User; }
    if (text == "assistant") { return ninfer::ChatRole::Assistant; }
    if (text == "tool") { return ninfer::ChatRole::Tool; }
    throw std::runtime_error("oracle input uses a role outside the C++ wire model: " + text);
}

fj::ChatMessage parse_message(const njson& message) {
    fj::ChatMessage out;
    out.role = parse_role(sub(message, "role").get<std::string>());

    if (message.contains("content")) {
        const njson& content = message.at("content");
        if (!content.is_null()) {
            if (content.is_string()) {
                out.parts.push_back(fj::ChatPart::text_part(content.get<std::string>()));
            } else if (content.is_array()) {
                for (const auto& item : content) {
                    if (item.is_string()) {
                        out.parts.push_back(fj::ChatPart::text_part(item.get<std::string>()));
                        continue;
                    }
                    if (item.is_object() && item.contains("type")) {
                        const std::string type = item.at("type").get<std::string>();
                        if (type == "image") {
                            out.parts.push_back(fj::ChatPart::image(fj::MediaData{}));
                            continue;
                        }
                        if (type == "video") {
                            out.parts.push_back(fj::ChatPart::video(fj::MediaData{}));
                            continue;
                        }
                    }
                    if (item.is_object() && item.contains("text") &&
                        item.at("text").is_string()) {
                        out.parts.push_back(
                            fj::ChatPart::text_part(item.at("text").get<std::string>()));
                        continue;
                    }
                    throw std::runtime_error("oracle input part is not representable");
                }
            } else {
                throw std::runtime_error("oracle input content has an unsupported shape");
            }
        }
    }
    for (const char* key : {"reasoning_content", "thinking", "reasoning"}) {
        if (message.contains(key) && message.at(key).is_string()) {
            out.reasoning_content = message.at(key).get<std::string>();
            break;
        }
    }
    if (message.contains("tool_call_id") && message.at("tool_call_id").is_string()) {
        out.tool_call_id = message.at("tool_call_id").get<std::string>();
    }
    if (message.contains("tool_calls") && message.at("tool_calls").is_array()) {
        for (const auto& raw_call : message.at("tool_calls")) {
            const njson function = raw_call.contains("function") && raw_call.at("function").is_object()
                                       ? raw_call.at("function")
                                       : raw_call;
            fj::ToolCall call;
            call.name = function.value("name", "");
            if (function.contains("arguments")) {
                const njson& arguments = function.at("arguments");
                if (arguments.is_string()) {
                    call.arguments_json = arguments.get<std::string>();
                } else if (!arguments.is_null()) {
                    call.arguments_json = arguments.dump(); // document order
                }
            }
            out.tool_calls.push_back(std::move(call));
        }
    }
    return out;
}

fj::ChatRenderOptions parse_options(const njson& spec) {
    fj::ChatRenderOptions out;
    const njson kwargs = spec.value("kwargs", njson::object());
    out.add_generation_prompt = kwargs.value("add_generation_prompt", true);
    out.enable_thinking       = kwargs.value("enable_thinking", true);
    if (kwargs.contains("reasoning_effort") && kwargs.at("reasoning_effort").is_string()) {
        const std::string effort = kwargs.at("reasoning_effort").get<std::string>();
        if (effort == "none" || effort == "off") {
            out.enable_thinking    = false;
            out.reasoning_effort   = ninfer::ReasoningEffort::Medium;
        } else if (effort == "minimal" || effort == "low") {
            out.reasoning_effort   = ninfer::ReasoningEffort::Low;
        } else if (effort == "high" || effort == "xhigh" || effort == "max" ||
                   effort == "ultracode" || effort == "extreme") {
            out.reasoning_effort   = ninfer::ReasoningEffort::XHigh;
        } else {
            out.reasoning_effort   = ninfer::ReasoningEffort::Medium; // template fallback
        }
    }
    if (kwargs.contains("preserve_thinking") &&
        kwargs.at("preserve_thinking").is_boolean()) {
        out.preserve_thinking = kwargs.at("preserve_thinking").get<bool>();
    }
    if (kwargs.contains("preserve_reasoning") &&
        kwargs.at("preserve_reasoning").is_boolean()) {
        out.froggeric_v225.preserve_reasoning = kwargs.at("preserve_reasoning").get<bool>();
    }
    out.add_vision_id                     = kwargs.value("add_vision_id", false);
    out.froggeric_v225.auto_disable_thinking_with_tools =
        kwargs.value("auto_disable_thinking_with_tools", false);
    const std::string tool_format = kwargs.value("tool_call_format", std::string("xml"));
    if (tool_format == "json") {
        out.froggeric_v225.tool_call_format = ninfer::ToolCallFormat::Json;
    } else if (tool_format != "xml") {
        throw std::runtime_error("oracle input uses an unsupported tool_call_format: " +
                                 tool_format);
    }
    out.froggeric_v225.max_tool_arg_chars      = kwargs.value("max_tool_arg_chars", 0U);
    out.froggeric_v225.max_tool_response_chars = kwargs.value("max_tool_response_chars", 0U);
    if (spec.contains("tools") && spec.at("tools").is_array()) {
        for (const auto& tool : spec.at("tools")) {
            if (!tool.is_object()) { throw std::runtime_error("oracle tools must be objects"); }
            out.tool_jsons.push_back(tool.dump());
        }
    }
    return out;
}

void report_first_diff(const std::string& actual, const std::string& expected,
                       const std::string& name) {
    const std::size_t length = std::min(actual.size(), expected.size());
    std::size_t offset       = 0;
    while (offset < length && actual[offset] == expected[offset]) { ++offset; }
    if (offset == length && actual.size() != expected.size()) { offset = length; }
    const auto ctx = [](const std::string& text, std::size_t at) {
        const std::size_t begin = at > 40 ? at - 40 : 0;
        return std::string(text.substr(begin, std::min<std::size_t>(80, text.size() - begin)));
    };
    std::cout << "DIFF " << name << " at byte " << offset << " (actual " << actual.size()
              << " bytes, golden " << expected.size() << ")\n"
              << "  golden: ...\"" << ctx(expected, offset) << "\"\n"
              << "  actual: ...\"" << ctx(actual, offset) << "\"\n";
}

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cout << "FAIL " << message << '\n';
    return 1;
}

fj::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    fj::ChatMessage message;
    message.role = role;
    message.parts.push_back(fj::ChatPart::text_part(std::move(text)));
    return message;
}

fj::ChatMessage tool_image_result() {
    fj::ChatMessage tool;
    tool.role = ninfer::ChatRole::Tool;
    tool.parts.push_back(fj::ChatPart::text_part("captured "));
    tool.parts.push_back(fj::ChatPart::image(fj::MediaData{}));
    return tool;
}

const std::string kTestToolJson =
    R"({"type":"function","function":{"name":"f","parameters":{"type":"object"}}})";

bool overlaps_placeholder(const fj::RenderedChat& rendered, fj::ByteSpan span) {
    return std::any_of(
        rendered.media_placeholders.begin(), rendered.media_placeholders.end(),
        [&](const fj::MediaPlaceholderByteSpec& placeholder) {
            return span.begin < placeholder.bytes.end && placeholder.bytes.begin < span.end;
        });
}

// Tool and assistant media must keep their placeholder metadata: the Processor expands exactly
// the pad token and rejects a rendered chat whose placeholder count does not match its media
// items, so dropping the metadata turns a supported tool-image result into a 400.
int test_media_placeholder_provenance() {
    int failures                            = 0;
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();

    std::vector<fj::ChatMessage> messages;
    messages.push_back(text_message(ninfer::ChatRole::User, "capture"));
    fj::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.tool_calls.push_back(fj::ToolCall{.name = "f", .arguments_json = "{}"});
    messages.push_back(std::move(assistant));
    messages.push_back(tool_image_result());

    fj::ChatRenderOptions options;
    options.tool_jsons.push_back(kTestToolJson);
    const fj::RenderedChat rendered = renderer.render(messages, options);
    failures += check(rendered.media_placeholders.size() == 1,
                      "tool image publishes one media placeholder");
    if (rendered.media_placeholders.size() == 1) {
        const fj::MediaPlaceholderByteSpec& placeholder = rendered.media_placeholders.front();
        failures += check(placeholder.modality == fj::Modality::Image &&
                              placeholder.item_index == 0,
                          "tool image placeholder modality and item index");
        failures += check(rendered.text.substr(placeholder.bytes.begin,
                                               placeholder.bytes.end - placeholder.bytes.begin) ==
                              "<|image_pad|>",
                          "tool image placeholder covers only the pad token");
    }
    failures += check(
        std::none_of(rendered.literal_spans.begin(), rendered.literal_spans.end(),
                     [&](fj::ByteSpan span) { return overlaps_placeholder(rendered, span); }),
        "tool image placeholder is not inside a literal span");

    std::vector<fj::ChatMessage> assistant_media;
    assistant_media.push_back(text_message(ninfer::ChatRole::User, "hi"));
    fj::ChatMessage media_assistant;
    media_assistant.role = ninfer::ChatRole::Assistant;
    media_assistant.parts.push_back(fj::ChatPart::text_part("see "));
    media_assistant.parts.push_back(fj::ChatPart::image(fj::MediaData{}));
    assistant_media.push_back(std::move(media_assistant));
    fj::ChatRenderOptions vision_options;
    vision_options.add_vision_id = true;
    const fj::RenderedChat assistant_rendered =
        renderer.render(assistant_media, vision_options);
    failures += check(assistant_rendered.media_placeholders.size() == 1 &&
                          assistant_rendered.text.find("Picture 1: <|vision_start|><|image_pad|>") !=
                              std::string::npos,
                      "assistant image publishes its placeholder and vision id");

    fj::ChatRenderOptions truncating = options;
    truncating.froggeric_v225.max_tool_response_chars = 3;
    bool rejected                      = false;
    try {
        (void)renderer.render(messages, truncating);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("truncates a media placeholder") !=
                   std::string::npos;
    }
    failures += check(rejected, "tool media truncation is rejected explicitly");
    return failures;
}

// Inline tags are stripped from the concatenated render, so part boundaries must be remapped
// through the removal instead of being computed on the raw parts.
int test_tag_stripping_provenance() {
    int failures                            = 0;
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();

    std::vector<fj::ChatMessage> messages;
    fj::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(fj::ChatPart::text_part("<|think_off|>ab"));
    user.parts.push_back(fj::ChatPart::text_part("cd"));
    messages.push_back(std::move(user));

    fj::ChatRenderOptions options;
    options.cache_markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count      = 1,
        .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 1});
    options.cache_markers.push_back(ninfer::PromptCacheMarker{
        .after_message_count      = 1,
        .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 2});
    const fj::RenderedChat rendered = renderer.render(messages, options);
    const std::size_t content       = rendered.text.find("abcd");
    failures += check(content != std::string::npos, "tag-stripped user content is abcd");
    failures += check(rendered.cache_boundaries.size() == 2 && rendered.cache_boundaries[0] &&
                          rendered.cache_boundaries[1],
                      "tag-stripped part boundaries resolve");
    if (content != std::string::npos && rendered.cache_boundaries.size() == 2 &&
        rendered.cache_boundaries[0] && rendered.cache_boundaries[1]) {
        failures += check(*rendered.cache_boundaries[0] == content + 2 &&
                              *rendered.cache_boundaries[1] == content + 4,
                          "part boundaries follow the surviving bytes");
    }
    return failures;
}

// NInfer carries tool arguments as a JSON string; object strings are normalized through the
// template's mapping branch (documented behavior, not byte parity with the raw string branch).
int test_tool_argument_string_normalization() {
    int failures                            = 0;
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();

    std::vector<fj::ChatMessage> messages;
    messages.push_back(text_message(ninfer::ChatRole::User, "q"));
    fj::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.tool_calls.push_back(
        fj::ToolCall{.name = "f", .arguments_json = R"({"city":"Paris","units":"c"})"});
    messages.push_back(std::move(assistant));

    fj::ChatRenderOptions xml_options;
    xml_options.tool_jsons.push_back(kTestToolJson);
    const fj::RenderedChat xml = renderer.render(messages, xml_options);
    failures += check(xml.text.find("<parameter=city>\nParis\n</parameter>\n"
                                    "<parameter=units>\nc\n</parameter>") != std::string::npos,
                      "XML object-string arguments normalize to parameters");

    fj::ChatRenderOptions json_options = xml_options;
    json_options.froggeric_v225.tool_call_format      = ninfer::ToolCallFormat::Json;
    const fj::RenderedChat json        = renderer.render(messages, json_options);
    failures += check(json.text.find(R"({"name": "f", "arguments": {"city": "Paris", )"
                                     R"("units": "c"}})") != std::string::npos,
                      "JSON object-string arguments normalize to a sorted object");
    return failures;
}

// The generation suffix and the output session must agree on the thinking state left by inline
// control tags, not only on the request-level enable_thinking flag.
int test_generation_thinking_state() {
    int failures                            = 0;
    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();

    std::vector<fj::ChatMessage> messages;
    messages.push_back(text_message(ninfer::ChatRole::User, "<|think_off|>hi"));
    fj::ChatRenderOptions options;
    options.enable_thinking        = true;
    const fj::RenderedChat closed  = renderer.render(messages, options);
    failures += check(!closed.generation_starts_in_thinking &&
                          closed.text.ends_with("<think>\n\n</think>\n\n"),
                      "think_off closes the generation prompt");

    messages[0]                    = text_message(ninfer::ChatRole::User, "<|think_on|>hi");
    options.enable_thinking        = false;
    const fj::RenderedChat opened  = renderer.render(messages, options);
    failures += check(opened.generation_starts_in_thinking &&
                          opened.text.ends_with("<|im_start|>assistant\n<think>\n"),
                      "think_on opens the generation prompt");
    return failures;
}

int test_v225_capabilities() {
    const fj::CompiledChatTemplate renderer        = fj::CompiledChatTemplate::froggeric_v225();
    const ninfer::PromptCapabilities capabilities = renderer.capabilities();
    int failures                                   = 0;
    // Service policy: the froggeric style aligns its omitted-effort default with the artifact
    // (reasoning-effort) style instead of the pinned template's medium (chat_template.cpp).
    failures += check(capabilities.enable_thinking && capabilities.reasoning_effort.low &&
                          capabilities.reasoning_effort.medium &&
                          capabilities.reasoning_effort.xhigh &&
                          capabilities.reasoning_effort.default_effort ==
                              ninfer::ReasoningEffort::XHigh,
                      "v22.5 capabilities report low/medium/xhigh with an xhigh service default");

    // Template parity: the compiled renderer keeps the pinned template's medium fallback, so an
    // omitted effort renders exactly like an explicit medium and differs from xhigh. Keep both
    // halves: the first pins parity with the pinned Jinja template, the second proves the renderer
    // fallback and the service default are distinct values rather than one value under two names.
    std::vector<fj::ChatMessage> messages;
    messages.push_back(text_message(ninfer::ChatRole::User, "hi"));
    const fj::ChatRenderOptions implicit_options;
    fj::ChatRenderOptions medium_options;
    medium_options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    fj::ChatRenderOptions xhigh_options;
    xhigh_options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    const std::string implicit_text = renderer.render(messages, implicit_options).text;
    failures += check(implicit_text == renderer.render(messages, medium_options).text,
                      "omitted effort keeps the v22.5 template medium fallback");
    failures += check(implicit_text != renderer.render(messages, xhigh_options).text,
                      "the v22.5 template fallback is distinct from the xhigh service default");
    return failures;
}

} // namespace

int main() {
    const std::filesystem::path fixture_dir =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests" / "fixtures" / "frontend" /
        "froggeric_v22_5";
    int failures = 0;

    if (!sha256_file(fixture_dir / "chat_template.jinja", kPrettyFixtureSha, failures)) {}
    if (!sha256_file(fixture_dir / "chat_template_oneline.txt", kOnelineFixtureSha, failures)) {}
    failures += test_media_placeholder_provenance();
    failures += test_tag_stripping_provenance();
    failures += test_tool_argument_string_normalization();
    failures += test_generation_thinking_state();
    failures += test_v225_capabilities();

    std::vector<std::filesystem::path> inputs;
    for (const auto& entry :
         std::filesystem::directory_iterator(fixture_dir / "inputs")) {
        if (entry.path().extension() == ".json") { inputs.push_back(entry.path()); }
    }
    std::sort(inputs.begin(), inputs.end());
    if (inputs.empty()) {
        std::cout << "FAIL: no oracle inputs under " << (fixture_dir / "inputs") << "\n";
        return 1;
    }

    const fj::CompiledChatTemplate renderer = fj::CompiledChatTemplate::froggeric_v225();
    int rendered = 0, skipped = 0, error_cases = 0;
    for (const auto& input_path : inputs) {
        const njson spec = njson::parse(read_file(input_path));
        const std::string name = input_path.stem().string();
        if (spec.value("oracle_only", false)) {
            ++skipped;
            std::cout << "SKIP " << name << " (oracle_only)\n";
            continue;
        }
        const std::string expect_error = spec.value("expect_error", std::string());
        fj::ChatRenderOptions options = parse_options(spec);
        std::vector<fj::ChatMessage> messages;
        if (spec.contains("messages")) {
            for (const auto& raw_message : spec.at("messages")) {
                messages.push_back(parse_message(raw_message));
            }
        }
        const std::filesystem::path golden_path =
            fixture_dir / "golden" / (name + ".expected");
        if (expect_error.empty()) {
            if (!std::filesystem::exists(golden_path)) {
                std::cout << "FAIL " << name << ": no golden\n";
                ++failures;
                continue;
            }
            const fj::RenderedChat rendered_chat = renderer.render(messages, options);
            const std::string expected = read_file(golden_path);
            if (rendered_chat.text != expected) {
                report_first_diff(rendered_chat.text, expected, name);
                ++failures;
                continue;
            }
            ++rendered;
            std::cout << "PASS " << name << " (" << rendered_chat.text.size() << " bytes)\n";
        } else {
            try {
                (void)renderer.render(messages, options);
                std::cout << "FAIL " << name << ": expected error \"" << expect_error
                          << "\", render succeeded\n";
                ++failures;
            } catch (const std::invalid_argument& error) {
                if (std::string(error.what()).find(expect_error) == std::string::npos) {
                    std::cout << "FAIL " << name << ": expected error \"" << expect_error
                              << "\", got \"" << error.what() << "\"\n";
                    ++failures;
                    continue;
                }
                ++error_cases;
                std::cout << "PASS " << name << " (expected error)\n";
            }
        }
    }
    std::cout << "froggeric v22.5: " << rendered << " rendered, " << error_cases
              << " error cases, " << skipped << " oracle-only, " << failures
              << " failure(s) over " << inputs.size() << " inputs\n";
    return failures == 0 ? 0 : 1;
}
