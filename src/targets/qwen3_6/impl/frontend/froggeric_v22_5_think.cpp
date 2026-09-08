#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_think.h"

#include "targets/qwen3_6/impl/frontend/froggeric_v22_5_python.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {

ThinkExtraction extract_think(const std::string& explicit_reasoning, const std::string& body) {
    ThinkExtraction out;
    if (!explicit_reasoning.empty()) {
        std::string_view lead_end;
        if (body.rfind("<think>", 0) == 0 && body.find("</think>") != std::string::npos) {
            lead_end = "</think>";
        } else if (body.rfind("<thinking>", 0) == 0 &&
                   body.find("</thinking>") != std::string::npos) {
            lead_end = "</thinking>";
        } else if (body.rfind("</think>", 0) == 0) {
            lead_end = "</think>";
        } else if (body.rfind("</thinking>", 0) == 0) {
            lead_end = "</thinking>";
        }
        if (!lead_end.empty()) {
            // jinja: content.split(_lead_end)[-1].lstrip('\n')
            out.body_begin = body.rfind(lead_end) + lead_end.size();
            while (out.body_begin < body.size() && body[out.body_begin] == '\n') {
                ++out.body_begin;
            }
        }
        const auto [r_begin, r_end] = py_trim_bounds(explicit_reasoning);
        out.explicit_reasoning = explicit_reasoning.substr(r_begin, r_end - r_begin);
        return out;
    }
    std::string_view think_end;
    if (body.rfind("</think>", 0) == 0) {
        think_end = "</think>";
    } else if (body.rfind("</thinking>", 0) == 0) {
        think_end = "</thinking>";
    } else if (body.find("\n</think>") != std::string::npos) {
        think_end = "\n</think>";
    } else if (body.find("\n</thinking>") != std::string::npos) {
        think_end = "\n</thinking>";
    } else if (body.find("\n</ think>") != std::string::npos) {
        think_end = "\n</ think>";
    } else if (body.find("\n</think >") != std::string::npos) {
        think_end = "\n</think >";
    } else if (body.rfind("<think>", 0) == 0 && body.find("</think>") != std::string::npos) {
        think_end = "</think>";
    } else if (body.rfind("<thinking>", 0) == 0 && body.find("</thinking>") != std::string::npos) {
        think_end = "</thinking>";
    }
    if (!think_end.empty()) {
        const std::string think_start =
            think_end.find("thinking") != std::string_view::npos ? "<thinking>" : "<think>";
        // jinja: reasoning = content.split(_think_end)[0].rstrip('\n') ...
        std::size_t before_end = body.find(think_end);
        while (before_end > 0 && body[before_end - 1] == '\n') { --before_end; }
        std::size_t r_begin = 0;
        const std::size_t open =
            std::string_view(body).substr(0, before_end).rfind(think_start);
        if (open != std::string_view::npos) {
            r_begin = open + think_start.size();
            while (r_begin < before_end && body[r_begin] == '\n') { ++r_begin; }
        }
        const auto [trim_begin, trim_end] =
            py_trim_bounds(std::string_view(body).substr(r_begin, before_end - r_begin));
        out.reasoning_begin = r_begin + trim_begin;
        out.reasoning_end   = r_begin + trim_end;
        // jinja: content = content.split(_think_end)[-1].lstrip('\n')
        out.body_begin = body.rfind(think_end) + think_end.size();
        while (out.body_begin < body.size() && body[out.body_begin] == '\n') {
            ++out.body_begin;
        }
    }
    return out;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
