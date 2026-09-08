#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;

const fi::ToolCallOutputContract kLegacyContract;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), true);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninfer::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_basic_legacy_parsing() {
    const auto parsed = fi::parse_qwen_tool_call_output("Calling weather.\n"
                                                        "<tool_call>\n"
                                                        "<function=get_weather>\n"
                                                        "<parameter=city>\nParis\n</parameter>\n"
                                                        "<parameter=days>\n2\n</parameter>\n"
                                                        "</function>\n"
                                                        "</tool_call>",
                                                        64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "legacy call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "legacy call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "legacy string inference changed");
    failures += check(args.at("days") == 2, "legacy JSON inference changed");
    return failures;
}

int test_multiple_calls() {
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = fi::parse_qwen_tool_call_output(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

int check_decoder_chunk_independent(const fi::ToolCallOutputContract& contract,
                                    const std::string& text, std::string_view message) {
    auto shared = std::make_shared<fi::ToolCallOutputContract>(contract);
    std::string reference;
    bool stable = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(shared, 64);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        const auto terminal   = decoder.finish();
        std::string signature = visible + "|" + terminal.content;
        for (const auto& call : terminal.tool_calls) {
            signature += "|" + call.name + ":" + call.arguments_json;
        }
        if (split == 0) {
            reference = signature;
        } else if (signature != reference) {
            stable = false;
            break;
        }
    }
    return check(stable, std::string(message));
}

int test_embedded_parameter_delimiters_are_preserved() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string unmatched_open =
        tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}});
    const std::string standalone_close = tool_call("bash", {{"command", "echo '</parameter>'"}});

    int failures      = 0;
    const auto opened = fi::parse_qwen_tool_call_output(unmatched_open, 64, contract);
    failures += check(opened.is_tool_call_response && opened.tool_calls.size() == 1,
                      "unbalanced nested parameter open was not rescued");
    if (opened.tool_calls.size() == 1) {
        failures += check(Json::parse(opened.tool_calls.front().arguments_json).at("command") ==
                              "echo '<parameter=unterminated>'",
                          "unbalanced nested parameter open value changed");
    }
    const auto closed = fi::parse_qwen_tool_call_output(standalone_close, 64, contract);
    failures += check(closed.is_tool_call_response && closed.tool_calls.size() == 1,
                      "standalone parameter close was not rescued");
    if (closed.tool_calls.size() == 1) {
        failures += check(Json::parse(closed.tool_calls.front().arguments_json).at("command") ==
                              "echo '</parameter>'",
                          "standalone parameter close value changed");
    }

    // Truncated output must stay a fallback: no close can complete the region.
    const std::string truncated = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la";
    failures +=
        check_rejected(truncated, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "truncated parameter was silently repaired");
    const std::string unterminated_function =
        "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>";
    failures += check_rejected(unterminated_function, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "missing function close was silently repaired");
    return failures;
}

int test_xml_candidate_anchor() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});

    int failures             = 0;
    const std::string prose  = "You saw the raw `<tool_call>` block. Here is the real call:\n";
    const std::string quoted = prose + tool_call("bash", {{"command", "ls -la"}});
    const auto parsed        = fi::parse_qwen_tool_call_output(quoted, 64, contract);
    failures +=
        check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                  parsed.content == "You saw the raw `<tool_call>` block. Here is the real call:" &&
                  Json::parse(parsed.tool_calls.front().arguments_json).at("command") == "ls -la",
              "quoted tool-call marker before the real call was not skipped");
    failures += check_decoder_chunk_independent(contract, quoted,
                                                "quoted-marker decoding depends on chunking");

    const std::string many = "First `<tool_call>`, again `<tool_call>`, and once more "
                             "`<tool_call>`:\n" +
                             tool_call("bash", {{"command", "pwd"}});
    const auto many_parsed = fi::parse_qwen_tool_call_output(many, 64, contract);
    failures +=
        check(many_parsed.is_tool_call_response && many_parsed.tool_calls.size() == 1 &&
                  Json::parse(many_parsed.tool_calls.front().arguments_json).at("command") == "pwd",
              "multiple quoted markers before the real call were not skipped");

    const std::string mention = "Just a mention of `<tool_call>` in prose.";
    failures +=
        check_rejected(mention, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "a marker without a call must still fall back");

    const std::string separated = tool_call("bash", {{"command", "first"}}) + "\nand also\n" +
                                  tool_call("bash", {{"command", "second"}});
    const auto separated_parsed = fi::parse_qwen_tool_call_output(separated, 64, contract);
    failures +=
        check(separated_parsed.is_tool_call_response && separated_parsed.tool_calls.size() == 1 &&
                  Json::parse(separated_parsed.tool_calls.front().arguments_json).at("command") ==
                      "second",
              "prose between two calls did not select the trailing call");
    return failures;
}

int test_xml_parameter_close_disambiguation() {
    const auto bash  = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const auto write = contract_for(
        "write", Json{{"content", Json{{"type", "string"}}}, {"path", Json{{"type", "string"}}}});
    int failures = 0;

    // pi session row 380/382: an unmatched nested open inside the value.
    const std::string grep = "<tool_call>\n"
                             "<function=bash>\n"
                             "<parameter=command>\n"
                             "grep -o '<parameter=msg>[^<]*' file.txt\n"
                             "</parameter>\n"
                             "</function>\n"
                             "</tool_call>";
    const auto grep_parsed = fi::parse_qwen_tool_call_output(grep, 64, bash);
    failures += check(grep_parsed.is_tool_call_response && grep_parsed.tool_calls.size() == 1,
                      "unmatched parameter open in a value broke the call");
    if (grep_parsed.tool_calls.size() == 1) {
        failures +=
            check(Json::parse(grep_parsed.tool_calls.front().arguments_json).at("command") ==
                      "grep -o '<parameter=msg>[^<]*' file.txt",
                  "grep command bytes changed");
    }
    failures += check_decoder_chunk_independent(bash, grep, "grep decoding depends on chunking");

    // A literal close at the end of the value.
    const std::string literal_close = tool_call("bash", {{"command", "printf '</parameter>'"}});
    const auto close_parsed         = fi::parse_qwen_tool_call_output(literal_close, 64, bash);
    failures +=
        check(close_parsed.is_tool_call_response && close_parsed.tool_calls.size() == 1 &&
                  Json::parse(close_parsed.tool_calls.front().arguments_json).at("command") ==
                      "printf '</parameter>'",
              "literal close inside a value was not preserved");

    // Write-file shape: the value embeds a complete parameter block; the sibling must survive.
    const std::string write_text = "<tool_call>\n"
                                   "<function=write>\n"
                                   "<parameter=content>\n"
                                   "before <parameter=path>example</parameter> after\n"
                                   "</parameter>\n"
                                   "<parameter=path>\n"
                                   "/tmp/example.txt\n"
                                   "</parameter>\n"
                                   "</function>\n"
                                   "</tool_call>";
    const auto write_parsed      = fi::parse_qwen_tool_call_output(write_text, 64, write);
    failures += check(write_parsed.is_tool_call_response && write_parsed.tool_calls.size() == 1,
                      "write-file shape with embedded parameter markup broke");
    if (write_parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(write_parsed.tool_calls.front().arguments_json);
        failures +=
            check(args.size() == 2 &&
                      args.at("content") == "before <parameter=path>example</parameter> after" &&
                      args.at("path") == "/tmp/example.txt",
                  "embedded parameter block swallowed the sibling parameter");
    }
    failures += check_decoder_chunk_independent(write, write_text,
                                                "write-file decoding depends on chunking");

    // Unmatched open in the value plus a real sibling: the sibling must not be swallowed.
    const std::string mixed = "<tool_call>\n"
                              "<function=write>\n"
                              "<parameter=content>\n"
                              "grep -o '<parameter=msg>[^<]*' file.txt\n"
                              "</parameter>\n"
                              "<parameter=path>\n"
                              "/tmp/mixed.txt\n"
                              "</parameter>\n"
                              "</function>\n"
                              "</tool_call>";
    const auto mixed_parsed = fi::parse_qwen_tool_call_output(mixed, 64, write);
    failures += check(mixed_parsed.is_tool_call_response && mixed_parsed.tool_calls.size() == 1,
                      "unmatched open with a sibling parameter broke the call");
    if (mixed_parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(mixed_parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 2 &&
                              args.at("content") == "grep -o '<parameter=msg>[^<]*' file.txt" &&
                              args.at("path") == "/tmp/mixed.txt",
                          "unmatched open value swallowed or truncated the sibling parameter");
    }

    // Parameter order is preserved exactly.
    const auto ordered = contract_for("configure", Json{{"alpha", Json{{"type", "string"}}},
                                                        {"beta", Json{{"type", "string"}}},
                                                        {"gamma", Json{{"type", "string"}}}});
    const auto ordered_parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"alpha", "1"}, {"beta", "2"}, {"gamma", "3"}}), 64, ordered);
    failures +=
        check(ordered_parsed.is_tool_call_response && ordered_parsed.tool_calls.size() == 1 &&
                  ordered_parsed.tool_calls.front().arguments_json ==
                      "{\"alpha\":\"1\",\"beta\":\"2\",\"gamma\":\"3\"}",
              "parameter document order changed");
    return failures;
}

int test_xml_parameter_close_budget() {
    const auto write = contract_for(
        "write", Json{{"content", Json{{"type", "string"}}}, {"path", Json{{"type", "string"}}}});
    std::string payload     = "head <parameter=msg>\n";
    const std::string chunk = std::string(80, 'x') + " </parameter>\n";
    for (int index = 0; index < 1000; ++index) { payload += chunk; }
    payload += "tail";
    const std::string text = tool_call("write", {{"content", payload}, {"path", "/tmp/x"}});

    const auto started = std::chrono::steady_clock::now();
    const auto parsed  = fi::parse_qwen_tool_call_output(text, 64, write);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "pathological payload was not parsed");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures +=
            check(args.size() == 2 && args.at("content") == payload && args.at("path") == "/tmp/x",
                  "pathological payload lost bytes");
    }
    failures += check(elapsed < 50, "pathological payload exceeded the time budget");
    if (elapsed >= 50) { std::cerr << "  elapsed_ms=" << elapsed << '\n'; }
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = fi::parse_qwen_tool_call_output(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = fi::parse_qwen_tool_call_output(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish();
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (const char byte : text) { bytewise_visible += bytewise.feed(std::string_view(&byte, 1)); }
    auto bytewise_terminal = bytewise.finish();
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
    return failures;
}

int test_strict_structure_and_active_tool_set() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string malformed = "<tool_call>\n<function=configure>\n";
    failures +=
        check_rejected(malformed, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "missing structural tags were accepted");

    const std::string suffix = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    failures +=
        check_rejected(suffix, contract, ninfer::ToolCallParseFallbackReason::TrailingContent,
                       "non-whitespace suffix was accepted");

    const std::string missing_parameter_close =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
        "</function>\n</tool_call>";
    failures += check_rejected(missing_parameter_close, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "missing parameter close was repaired");

    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    failures +=
        check_rejected(duplicate, contract, ninfer::ToolCallParseFallbackReason::DuplicateParameter,
                       "duplicate parameter was silently overwritten");

    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    failures +=
        check_rejected(unknown_tool, contract, ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                       "undeclared tool name was accepted");

    const std::string invalid_name = tool_call("bad.name", {{"value", "x"}});
    failures += check_rejected(invalid_name, kLegacyContract,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "invalid function-name character was accepted");
    return failures;
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const std::string text          = tool_call(name);
    const auto anthropic            = fi::parse_qwen_tool_call_output(text, 128, kLegacyContract);
    const auto openai               = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = fi::parse_qwen_tool_call_output(too_long_text, 128, kLegacyContract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check(!openai.is_tool_call_response, "128-character OpenAI tool name was accepted");
    failures +=
        check(!too_long.is_tool_call_response, "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = fi::parse_qwen_tool_call_output(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_conflicting_duplicate_tool_contracts_use_legacy_normalization() {
    const std::string integer_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    const std::string string_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "string"}}}});

    const std::vector<std::string> identical_definitions = {integer_definition, integer_definition};
    const auto identical = contract_from_definitions(identical_definitions);
    const auto accepted =
        fi::parse_qwen_tool_call_output(tool_call("configure", {{"value", "7"}}), 64, *identical);

    const std::vector<std::string> conflicting_definitions = {integer_definition,
                                                              string_definition};
    const auto conflicting      = contract_from_definitions(conflicting_definitions);
    const std::string ambiguous = tool_call("configure", {{"value", "7"}});
    const auto ambiguous_parsed = fi::parse_qwen_tool_call_output(ambiguous, 64, *conflicting);

    int failures = 0;
    failures += check(accepted.is_tool_call_response && accepted.tool_calls.size() == 1,
                      "identical duplicate tool contracts became ambiguous");
    failures +=
        check(ambiguous_parsed.is_tool_call_response && ambiguous_parsed.tool_calls.size() == 1 &&
                  ambiguous_parsed.tool_calls.front().arguments_json == "{\"value\":7}" &&
                  ambiguous_parsed.diagnostics.schema_mismatch_arguments == 0,
              "conflicting duplicate tool contracts did not use legacy normalization");
    return failures;
}

int test_all_or_nothing_structural_commit() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    return check_rejected(text, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          "partially valid tool-call region was partially committed");
}

std::shared_ptr<const fi::ToolCallOutputContract>
json_contract(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), true, true);
}

int test_json_tool_calls() {
    const auto contract =
        json_contract({tool_definition("f", Json{{"a", Json{{"type", "integer"}}}}),
                       tool_definition("g", Json{{"b", Json{{"type", "string"}}}})});
    int failures = 0;

    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "hello<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":1}}</tool_call>", 64,
            *contract);
        failures += check(parsed.is_tool_call_response && parsed.content == "hello" &&
                              parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls[0].name == "f" &&
                              parsed.tool_calls[0].arguments_json == "{\"a\":1}",
                          "json single call with leading content");
    }
    {
        const std::string text =
            "<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":1}}</tool_call>\n "
            "<tool_call>{\"name\":\"g\",\"arguments\":{\"b\":\"x\"}}</tool_call>";
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2 &&
                              parsed.tool_calls[1].name == "g" &&
                              parsed.tool_calls[1].arguments_json == "{\"b\":\"x\"}",
                          "json consecutive calls");
    }
    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}</tool_call>"
            "<tool_call>{\"name\":\"g\"}</tool_call>",
            64, *contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2 &&
                              parsed.tool_calls[0].arguments_json == "{\"a\":1}" &&
                              parsed.tool_calls[1].arguments_json == "{}",
                          "json string and missing arguments");
    }
    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"name\":\"g\",\"arguments\":null}</tool_call>", 64, *contract);
        failures += check(parsed.is_tool_call_response &&
                              parsed.tool_calls[0].arguments_json == "{}",
                          "json null arguments were not normalized to an empty object");
    }
    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"name\":\"g\",\"arguments\":{\"b\":\"h\\u00e9llo\"}}</tool_call>",
            64, *contract);
        failures += check(parsed.is_tool_call_response &&
                              parsed.tool_calls[0].arguments_json.find("héllo") !=
                                  std::string::npos,
                          "json unicode arguments");
    }
    for (const auto& [text, reason] : std::vector<std::pair<std::string,
                                                            ninfer::ToolCallParseFallbackReason>>{
             {"<tool_call>{\"name\":\"nope\",\"arguments\":{}}</tool_call>",
              ninfer::ToolCallParseFallbackReason::UndeclaredTool},
             {"<tool_call>{\"name\":\"f\",\"arguments\":{}}",
              ninfer::ToolCallParseFallbackReason::MalformedStructure},
             {"<tool_call>[1,2]</tool_call>",
              ninfer::ToolCallParseFallbackReason::MalformedStructure},
             {"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call> trailing",
              ninfer::ToolCallParseFallbackReason::TrailingContent},
             {"<tool_call>{\"name\":\"\",\"arguments\":{}}</tool_call>",
              ninfer::ToolCallParseFallbackReason::InvalidToolName}}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *contract);
        failures += check(!parsed.is_tool_call_response && parsed.content == text &&
                              parsed.diagnostics.fallback_reason == reason,
                          "json fallback reason for: " + text);
    }
    {
        const std::string text =
            "prefix<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":1}}</tool_call>\n"
            "<tool_call>{\"name\":\"g\",\"arguments\":{\"b\":\"x\"}}</tool_call>";
        std::string reference;
        bool stable = true;
        for (std::size_t split = 0; split <= text.size(); ++split) {
            fi::ToolCallOutputDecoder decoder(contract, 64);
            std::string visible = decoder.feed(text.substr(0, split));
            visible += decoder.feed(text.substr(split));
            const auto terminal = decoder.finish();
            std::string signature = visible + "|" + terminal.content;
            for (const auto& call : terminal.tool_calls) {
                signature += "|" + call.name + ":" + call.arguments_json;
            }
            if (split == 0) {
                reference = signature;
            } else if (signature != reference) {
                stable = false;
            }
        }
        failures += check(stable, "json decoder is chunk-independent");
    }
    return failures;
}

int test_json_tool_calls_openai_wrapper() {
    const auto contract =
        json_contract({tool_definition("f", Json{{"a", Json{{"type", "integer"}}}}),
                       tool_definition("g", Json{{"b", Json{{"type", "string"}}}})});
    int failures = 0;

    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}</tool_call>",
            64, *contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls[0].name == "f" &&
                              parsed.tool_calls[0].arguments_json == "{\"a\":1}",
                          "wrapped json call was not normalized");
    }
    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}"
            "</tool_call>",
            64, *contract);
        failures += check(parsed.is_tool_call_response &&
                              parsed.tool_calls[0].arguments_json == "{\"a\":1}",
                          "wrapped json string arguments were not unwrapped");
    }
    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"function\":{\"name\":\"g\"}}</tool_call>", 64, *contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls[0].name == "g" &&
                              parsed.tool_calls[0].arguments_json == "{}",
                          "wrapped json call without arguments");
    }
    {
        const auto parsed = fi::parse_qwen_tool_call_output(
            "<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":2},\"function\":"
            "{\"name\":\"g\",\"arguments\":{\"b\":\"x\"}}}</tool_call>",
            64, *contract);
        failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                              parsed.tool_calls[0].name == "f" &&
                              parsed.tool_calls[0].arguments_json == "{\"a\":2}",
                          "native json shape did not win over the wrapper");
    }
    for (const auto& [text, reason] : std::vector<std::pair<std::string,
                                                            ninfer::ToolCallParseFallbackReason>>{
             {"<tool_call>{\"function\":\"nope\"}</tool_call>",
              ninfer::ToolCallParseFallbackReason::MalformedStructure},
             {"<tool_call>{\"function\":{\"name\":\"nope\",\"arguments\":{}}}</tool_call>",
              ninfer::ToolCallParseFallbackReason::UndeclaredTool},
             {"<tool_call>{\"function\":{\"name\":\"\",\"arguments\":{}}}</tool_call>",
              ninfer::ToolCallParseFallbackReason::InvalidToolName},
             {"<tool_call>{\"function\":{\"arguments\":{}}}</tool_call>",
              ninfer::ToolCallParseFallbackReason::InvalidToolName},
             {"<tool_call>{\"function\":{\"function\":{\"name\":\"f\"}}}</tool_call>",
              ninfer::ToolCallParseFallbackReason::InvalidToolName}}) {
        const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *contract);
        failures += check(!parsed.is_tool_call_response && parsed.content == text &&
                              parsed.diagnostics.fallback_reason == reason,
                          "wrapped json fallback reason for: " + text);
    }
    {
        const std::string text =
            "prefix<tool_call>{\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}"
            "</tool_call>\n"
            "<tool_call>{\"name\":\"g\",\"arguments\":{\"b\":\"x\"}}</tool_call>";
        std::string reference;
        bool stable = true;
        for (std::size_t split = 0; split <= text.size(); ++split) {
            fi::ToolCallOutputDecoder decoder(contract, 64);
            std::string visible = decoder.feed(text.substr(0, split));
            visible += decoder.feed(text.substr(split));
            const auto terminal = decoder.finish();
            std::string signature = visible + "|" + terminal.content;
            for (const auto& call : terminal.tool_calls) {
                signature += "|" + call.name + ":" + call.arguments_json;
            }
            if (split == 0) {
                reference = signature;
            } else if (signature != reference) {
                stable = false;
            }
        }
        failures += check(stable, "wrapped json decoder is chunk-independent");
    }
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish();
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish();

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    auto malformed_terminal = malformed.finish();
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == original && malformed_terminal.diagnostics.marker_seen &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call lost raw bytes or fallback diagnostics");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_basic_legacy_parsing();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_embedded_parameter_delimiters_are_preserved();
    failures += test_xml_candidate_anchor();
    failures += test_xml_parameter_close_disambiguation();
    failures += test_xml_parameter_close_budget();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_strict_structure_and_active_tool_set();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_conflicting_duplicate_tool_contracts_use_legacy_normalization();
    failures += test_all_or_nothing_structural_commit();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    failures += test_json_tool_calls();
    failures += test_json_tool_calls_openai_wrapper();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
