#pragma once

#include <ninfer/types.h>

#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Tool preamble and instructions. Byte-for-byte copies of the pinned upstream template
// (froggeric/Qwen-Fixed-Chat-Templates @ 855bffc49448e299789730ff92c9b8d834d6cc14); the
// oracle fixtures and tools/oracle_froggeric_v22_5/check_prompts.py pin them mechanically.
inline constexpr std::string_view kToolsHeader =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
inline constexpr std::string_view kXmlInstructionsThinking =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<think>\nBrief explanation of tool call\n</think>\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- You can use the <think></think> block to plan your next tool call OR to synthesize "
    "data and formulate your final response to the user.\n"
    "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> "
    "block.\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
    "thinking, with NO conversational text before it.\n"
    "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with "
    "NO spaces or indentation before them.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user "
    "without any tool call.\n</IMPORTANT>";
inline constexpr std::string_view kXmlInstructionsOff =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with "
    "NO conversational text before it.\n"
    "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with "
    "NO spaces or indentation before them.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user "
    "without any tool call.\n</IMPORTANT>";
inline constexpr std::string_view kJsonInstructionsThinking =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<think>\nBrief explanation of tool call\n</think>\n<tool_call>\n"
    "{\"name\": \"example_function_name\", \"arguments\": {\"example_parameter_1\": "
    "\"value_1\", \"example_parameter_2\": \"This is the value for the second parameter\"}}"
    "\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- You can use the <think></think> block to plan your next tool call OR to synthesize "
    "data and formulate your final response to the user.\n"
    "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> "
    "block.\n"
    "- Function calls MUST follow the specified format: a single JSON object with \"name\" "
    "and \"arguments\" keys inside <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after "
    "thinking, with NO conversational text before it.\n"
    "- The <tool_call> tag MUST be at the very beginning of a new line, with NO spaces or "
    "indentation before it.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user "
    "without any tool call.\n</IMPORTANT>";
inline constexpr std::string_view kJsonInstructionsOff =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n{\"name\": \"example_function_name\", \"arguments\": "
    "{\"example_parameter_1\": \"value_1\", \"example_parameter_2\": \"This is the value "
    "for the second parameter\"}}\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: a single JSON object with \"name\" "
    "and \"arguments\" keys inside <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with "
    "NO conversational text before it.\n"
    "- The <tool_call> tag MUST be at the very beginning of a new line, with NO spaces or "
    "indentation before it.\n"
    "- To call multiple functions, output a separate, completely closed "
    "<tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n"
    "- If you have all necessary data, provide your final answer directly to the user "
    "without any tool call.\n</IMPORTANT>";

inline constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

inline constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

std::string_view reasoning_instructions_for(bool thinking, ReasoningEffort effort) noexcept {
    if (thinking && effort == ReasoningEffort::Low) { return kLowReasoningInstructions; }
    if (thinking && effort == ReasoningEffort::XHigh) { return kXHighReasoningInstructions; }
    return "";
}


} // namespace ninfer::targets::qwen3_6::frontend_internal
