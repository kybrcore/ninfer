#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline ChatStyle parse_chat_style(std::string_view value) {
    if (value == "artifact") { return ChatStyle::Artifact; }
    if (value == "froggeric-v22.5") { return ChatStyle::FroggericV225; }
    throw std::invalid_argument("invalid chat style: " + std::string(value) +
                                " (valid values: artifact, froggeric-v22.5)");
}

[[nodiscard]] inline const char* chat_style_name(ChatStyle style) noexcept {
    switch (style) {
    case ChatStyle::Artifact:
        return "artifact";
    case ChatStyle::FroggericV225:
        return "froggeric-v22.5";
    }
    return "unknown";
}

} // namespace ninfer::product
