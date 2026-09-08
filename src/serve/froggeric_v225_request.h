#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <optional>

namespace ninfer::serve {

// Decodes the froggeric-v22.5 chat_template_kwargs of an OpenAI request. The style-independent
// base keys stay in the caller (its value/conflict rules are unchanged); this entry point owns
// the kwargs whitelist, the v22.5 value validation, and the preserve alias conflict.
//
//   - `accept_enable_thinking` lists enable_thinking as a style-independent key (Chat
//     Completions) in addition to preserve_thinking (both protocols).
//   - Null values are neutral for every key, including unknown ones (pre-existing behavior).
//   - v22.5 keys are rejected unless the engine style is froggeric-v22.5.
//   - `merged_preserve_thinking` is the value after the caller merged the top-level and kwargs
//     forms; the renderer prefers preserve_reasoning, so an explicit conflict is a 400.
[[nodiscard]] ninfer::FroggericV225Options
decode_froggeric_v225_kwargs(ninfer::ChatStyle style, const RequestJson& kwargs,
                             bool accept_enable_thinking,
                             const std::optional<bool>& merged_preserve_thinking);

// froggeric-v22.5 effort aliases: minimal -> low, high/max -> xhigh. Returns nullopt when the
// style is not froggeric-v22.5 or the requested effort is not one of those aliases.
[[nodiscard]] std::optional<ninfer::ReasoningEffort>
froggeric_v225_effort_alias(ninfer::ChatStyle style, RequestedReasoningEffort requested);

} // namespace ninfer::serve
