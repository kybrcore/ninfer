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
        out.preserve_reasoning = kwargs.at("preserve_reasoning").get<bool>();
    }
    out.add_vision_id                     = kwargs.value("add_vision_id", false);
    out.auto_disable_thinking_with_tools  = kwargs.value("auto_disable_thinking_with_tools", false);
    const std::string tool_format = kwargs.value("tool_call_format", std::string("xml"));
    if (tool_format == "json") {
        out.tool_call_format = ninfer::ToolCallFormat::Json;
    } else if (tool_format != "xml") {
        throw std::runtime_error("oracle input uses an unsupported tool_call_format: " +
                                 tool_format);
    }
    out.max_tool_arg_chars      = kwargs.value("max_tool_arg_chars", 0U);
    out.max_tool_response_chars = kwargs.value("max_tool_response_chars", 0U);
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

} // namespace

int main() {
    const std::filesystem::path fixture_dir =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests" / "fixtures" / "frontend" /
        "froggeric_v22_5";
    int failures = 0;

    if (!sha256_file(fixture_dir / "chat_template.jinja", kPrettyFixtureSha, failures)) {}
    if (!sha256_file(fixture_dir / "chat_template_oneline.txt", kOnelineFixtureSha, failures)) {}

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
