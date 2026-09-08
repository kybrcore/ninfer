// Direct unit tests for the froggeric v22.5 helper modules. These pin the Python string and
// tojson semantics, the inline-tag offset mapping, and think extraction independently of the
// end-to-end oracle fixtures, so a helper regression fails with a precise assertion.

#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_prompts.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_python.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_tags.h"
#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_think.h"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace fj = ninfer::targets::qwen3_6::frontend_internal;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cout << "FAIL " << message << '\n';
    return 1;
}

int test_python_string_semantics() {
    int failures = 0;
    failures += check(fj::py_len("aé😀b") == 4, "py_len counts codepoints");
    failures += check(fj::py_slice("aé😀b", 1, 3) == "é😀", "py_slice uses codepoint offsets");
    failures += check(fj::py_slice("aé😀b", 3, std::nullopt) == "b", "py_slice open-ended");
    failures += check(fj::py_slice("abc", 5, std::nullopt).empty(), "py_slice past the end");
    failures += check(fj::py_prefix_bytes("aé😀b", 3) == std::string_view("aé😀").size(),
                      "py_prefix_bytes returns a byte count");

    const std::string padded = "\u00a0\u3000x\u2028";
    const auto [begin, end] = fj::py_trim_bounds(padded);
    failures += check(padded.substr(begin, end - begin) == "x",
                      "py_trim_bounds strips Unicode whitespace");
    const auto [empty_begin, empty_end] = fj::py_trim_bounds("   ");
    failures += check(empty_begin == 0 && empty_end == 0,
                      "py_trim_bounds collapses all-whitespace to the empty range");
    const auto [wide_begin, wide_end] = fj::py_trim_bounds("\u00a0\u3000");
    failures += check(wide_begin == 0 && wide_end == 0,
                      "py_trim_bounds collapses Unicode all-whitespace to the empty range");
    failures += check(fj::py_isspace(0x7f) == false, "U+007F is not whitespace");
    failures += check(fj::py_isspace(0x20) && fj::py_isspace(0xa0) && fj::py_isspace(0x3000),
                      "Python whitespace set includes space, NBSP, ideographic space");
    return failures;
}

int test_tojson_oracle() {
    int failures = 0;
    const fj::OracleJson value = fj::OracleJson::parse(
        R"({"b":1,"a":"<x>&'","del":"\u007f","ctrl":"\u0001","emoji":"\ud83d\ude00","f":1.0,"i":7,"n":null,"t":true})");
    const std::string expected =
        R"({"a": "\u003cx\u003e\u0026\u0027", "b": 1, "ctrl": "\u0001", "del": "\u007f", "emoji": "\ud83d\ude00", "f": 1.0, "i": 7, "n": null, "t": true})";
    failures += check(fj::tojson_oracle(value) == expected,
                      "tojson_oracle matches json.dumps(ensure_ascii, sort_keys) + htmlsafe");
    failures += check(fj::tojson_oracle(fj::OracleJson::parse(R"([1,"two",false,null])")) ==
                          R"([1, "two", false, null])",
                      "tojson_oracle array separators");
    failures += check(fj::tojson_oracle(fj::OracleJson::parse("-0.0")) == "-0.0",
                      "tojson_oracle keeps negative zero");
    return failures;
}

int test_tag_stripper_mapping() {
    int failures = 0;
    {
        const std::string raw = "x<|think_off|>y";
        fj::TagStripper stripper(raw);
        stripper.apply();
        failures += check(stripper.map_position(0) == 0 && stripper.map_position(1) == 1 &&
                              stripper.map_position(raw.size()) == 2,
                          "tag removal maps boundaries to surviving bytes");
        const std::optional<fj::ByteSpan> head = stripper.map_span(fj::ByteSpan{0, 1});
        const std::optional<fj::ByteSpan> tail = stripper.map_span(fj::ByteSpan{14, 15});
        failures += check(head && head->begin == 0 && head->end == 1 &&
                              tail && tail->begin == 1 && tail->end == 2,
                          "literal/media spans are rebased after tag removal");
        failures += check(!stripper.map_span(fj::ByteSpan{1, 14}),
                          "a span fully inside the tag maps to nothing");
        failures += check(std::move(stripper).take_text() == "xy", "tag text is removed");
    }
    {
        fj::TagStripper stripper("  <|think_off|> a ");
        stripper.apply();
        failures += check(std::move(stripper).take_text() == "a",
                          "trim happens after each tag removal");
    }
    failures += check(fj::strip_inline_tags("<|think_xhigh|>hi") == "hi",
                      "strip_inline_tags keeps unrelated text");
    return failures;
}

int test_extract_think() {
    int failures = 0;
    {
        const std::string body = "<think>x</think>mid</think>tail";
        const fj::ThinkExtraction think = fj::extract_think("R", body);
        failures += check(think.explicit_reasoning == "R" &&
                              body.substr(think.body_begin) == "tail",
                          "explicit reasoning body starts after the LAST close marker");
    }
    {
        const std::string body = "a\n</think>b";
        const fj::ThinkExtraction think = fj::extract_think("", body);
        failures += check(think.explicit_reasoning.empty() &&
                              body.substr(think.reasoning_begin,
                                          think.reasoning_end - think.reasoning_begin) == "a" &&
                              body.substr(think.body_begin) == "b",
                          "derived reasoning and body split on the close marker");
    }
    {
        const fj::ThinkExtraction think = fj::extract_think("", "hello");
        failures += check(think.body_begin == 0 && think.reasoning_begin == think.reasoning_end,
                          "no marker keeps the whole body and empty reasoning");
    }
    return failures;
}

int test_prompt_constants() {
    int failures = 0;
    failures += check(fj::reasoning_instructions_for(true, ninfer::ReasoningEffort::Low) ==
                          fj::kLowReasoningInstructions &&
                          fj::reasoning_instructions_for(true, ninfer::ReasoningEffort::XHigh) ==
                              fj::kXHighReasoningInstructions,
                      "effort instructions are selected only while thinking");
    failures += check(fj::reasoning_instructions_for(false, ninfer::ReasoningEffort::Low).empty() &&
                          fj::reasoning_instructions_for(true, ninfer::ReasoningEffort::Medium)
                              .empty(),
                      "non-thinking and medium inject no instruction");
    failures += check(fj::kToolsHeader.starts_with("# Tools") &&
                          fj::kXmlInstructionsThinking.starts_with("\n\nIf you choose") &&
                          fj::kJsonInstructionsOff.starts_with("\n\nIf you choose"),
                      "pinned instruction constants keep their anchors");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_python_string_semantics();
    failures += test_tojson_oracle();
    failures += test_tag_stripper_mapping();
    failures += test_extract_think();
    failures += test_prompt_constants();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
