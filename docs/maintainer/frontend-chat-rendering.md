# NInfer Chat Template Rendering

This document is the maintainer reference for the prompt-rendering layer: the two chat renderers,
the shared provenance contract between them, the rendering invariants the Engine relies on, and the
parity verification that pins the compiled renderer to its upstream template.

Upstream anchor for the compiled renderer:
[`froggeric/Qwen-Fixed-Chat-Templates@855bffc49448e299789730ff92c9b8d834d6cc14`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/tree/855bffc49448e299789730ff92c9b8d834d6cc14),
template self-reported version `qwen3.8-froggeric-v22.5`, `chat_template.jinja` SHA-256
`e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`.

## 1. Renderer selection and lifetime

`--chat-style artifact|froggeric-v22.5` selects the prompt renderer at Engine construction and the
value is frozen for the process; HTTP requests cannot switch it. `artifact` (the default and zero
value of `ninfer::ChatStyle`) keeps the semantics compiled for the template embedded in the loaded
`.ninfer`. `froggeric-v22.5` replaces rendering with compiled semantics while **still running the
artifact validation first**: `compile_chat_template(resources, style)` always resolves the embedded
template (tokenizer-config consistency and registered digest) before choosing the override, so the
flag cannot start an artifact whose template is inconsistent or unknown. An invalid value fails
during option parsing, before any model load, and lists the complete valid set.

`PromptCapabilities` follows the selected style: the artifact renderer reports the effort surface
implied by its embedded template; `froggeric-v22.5` reports `low`, `medium`, and `xhigh` with a
`medium` default. The resolved style is published in the startup log line and in the
`server_start` JSONL record (`engine.chat_style`).

Request-level template options are parsed by the serve layer and carried on `PromptOptions`
(`preserve_reasoning`, `auto_disable_thinking_with_tools`, `tool_call_format`,
`max_tool_arg_chars`, `max_tool_response_chars`). The artifact renderer deliberately ignores them;
the serve layer only accepts them under `froggeric-v22.5`.

## 2. Shared render-fragment contract

`src/targets/qwen3_6/impl/frontend/render_fragment.h` is the single provenance contract used by
both the artifact renderer (`chat_template.cpp`) and the compiled renderer
(`froggeric_v22_5/renderer.cpp`):

```cpp
struct RenderedFragment {
    std::string text;
    std::vector<ByteSpan> literal_spans;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
};
```

`RenderBuilder` appends template text (`append_template`), client content (`append_literal`),
media placeholders (`append_media_placeholder`), and whole fragments (`append`). The rules below are
contract, not convention:

- A media placeholder span covers **only the pad token** (`<|image_pad|>` / `<|video_pad|>`), never
  the surrounding `<|vision_start|>` / `<|vision_end|>`. The Processor replaces exactly that byte
  range with expanded vision embeddings.
- `literal_spans` cover client-provided content only; template text and media placeholders are not
  literal. Adjacent literal spans merge; overlapping spans are a logic error.
- `slice_fragment` may clip literal spans at the slice boundary but must never cut a media
  placeholder; callers that slice rendered content (think extraction, tool-response truncation)
  rely on this to keep placeholder metadata aligned with the emitted bytes.
- The Processor path requires `rendered.media_placeholders.size()` to equal its collected media
  items. A renderer branch that emits placeholder text without publishing the metadata produces a
  request-time `chat media count does not match rendered placeholders` failure.

Both renderers must publish literal/media provenance, message boundaries, cache boundaries, the
rewrite checkpoint, and rewrite execution boundaries from the same builder so the Processor,
context cache, and rewrite machinery see one contract.

## 3. Compiled froggeric v22.5 module layout

| File | Responsibility |
|---|---|
| `froggeric_v22_5/prompts.h` | Pinned tool and reasoning instruction texts |
| `froggeric_v22_5/python.{h,cpp}` | Python string semantics and `|tojson` parity |
| `froggeric_v22_5/tags.{h,cpp}` | Inline-tag pre-scan and concatenated-render stripping with an offset map |
| `froggeric_v22_5/think.{h,cpp}` | Assistant think-block extraction |
| `froggeric_v22_5/renderer.cpp` | Orchestration, per-role handlers, metadata resolution |
| `froggeric_v22_5/CMakeLists.txt` | Own source list; the parent adds one `add_subdirectory` line |
| `render_fragment.{h,cpp}` | Shared provenance contract (section 2) |

Two rules keep this layer honest:

1. **Instruction text lives only in `froggeric_v22_5/prompts.h`.** `tools/oracle_froggeric_v22_5/oracle.py
   check` re-renders the pinned template for the four tool-instruction combinations
   (XML/JSON × thinking on/off) and the low/xhigh reasoning instructions, and compares the exact
   bytes with those constants. Hand-copied instruction text is a defect even if the goldens happen
   to pass.
2. **Python semantics live only in `froggeric_v22_5/python.{h,cpp}`.** New semantics need a direct
   unit test in `tests/targets/qwen3_6/test_froggeric_v22_5_helpers.cpp` before they are used by
   the renderer; the end-to-end fixtures alone are not enough to localize a helper regression.

## 4. Rendering invariants

- **Preserve default.** `froggeric-v22.5` defaults `preserve_thinking` to true; an explicit
  `preserve_reasoning` takes precedence, and the serve layer rejects conflicting explicit values
  with `conflicting_template_option`.
- **Generation thinking state.** `RenderedChat::generation_starts_in_thinking` is computed from the
  rendered result (request `enable_thinking`, `auto_disable_thinking_with_tools`, effort `none`,
  and inline control tags), and `PreparedPromptData::starts_in_reasoning` reads it. The output
  session must not start in reasoning-split mode when a tag or option closed thinking before the
  generation suffix.
- **Multi-step tool pre-scan.** The template's `multi_step_tool` rule renders user content without
  vision counting or `add_vision_id`; text parts do not append media placeholders. A user message
  whose literal text is exactly `<tool_response>...</tool_response>` counts as a tool query. When
  every user turn is a tool query, `last_query_index` is `0` (or the last body index when the body
  exceeds 50 messages).
- **Inline tags.** The pre-scan applies one if/elif chain per text item in message order (the
  template's priority, not text position). Stripping happens on the **concatenated** rendered
  content, one tag at a time in template order, with a trim after each removal; a source-offset map
  rebases literal spans, media placeholders, and part boundaries so provenance always matches the
  emitted bytes. Tags split across two text parts are therefore stripped exactly like the template.
- **Boundaries.** Message boundaries are recorded after each serialized message (the merged leading
  block publishes the boundary after the whole block); tool boundaries after each tool definition;
  the rewrite checkpoint is `ResponseReplay` at the generation suffix under `preserve_thinking` and
  `TurnClosure` at the first assistant turn that closes an unpreserved turn otherwise; rewrite
  execution boundaries mark the assistant opener, the think opener, and the think close.

## 5. Parity and verification

The compiled renderer is pinned by four independent layers:

1. **Byte parity.** `tests/fixtures/frontend/froggeric_v22_5/` holds 113 oracle inputs and their
   expected outputs generated by the pinned Jinja environment (Python 3.11 + jinja2 3.1.6,
   `StrictUndefined`, `keep_trailing_newline`, `lstrip_blocks`, `trim_blocks`). The C++ test renders
   every input and compares byte-for-byte; `oracle.py check` re-renders the pretty and oneline
   templates and asserts both agree with the committed goldens.
2. **Mechanical prompt check.** The same `oracle.py check` run diffs the C++ instruction constants
   against the pinned template (section 3).
3. **Direct helper tests.** `ninfer_qwen3_6_froggeric_v22_5_helpers_test` covers Python string
   semantics, the tojson escape table, the tag-stripper offset map, think extraction, and the
   pinned constants.
4. **Property tests.** `ninfer_qwen3_6_froggeric_v22_5_properties_test` generates 500 deterministic
   cases (seed `0x5eed1234`, `NINFER_FROGGERIC_PROP_CASES/SEED` override) over roles, multipart and
   media content, reasoning shapes, inline tags, tool calls/results and the v22.5 request options,
   and asserts the design invariants: determinism, boundary/provenance sanity, generation thinking
   state, `preserve_thinking` prefix extension, and rewrite-checkpoint stability across every
   published prefix. Deterministic cases pin the exact branch divergence point. Documented
   exceptions (leading merge, an open consecutive-tool batch, an inline tag in a later
   system/user message) are skipped explicitly, not silently weakened.

Verification commands:

```bash
# Local oracle (pinned venv; see tools/oracle_froggeric_v22_5/README.md)
python3 -m venv /tmp/ov && /tmp/ov/bin/pip install -r tools/oracle_froggeric_v22_5/requirements.txt
/tmp/ov/bin/python tools/oracle_froggeric_v22_5/oracle.py check

# Host tests (isolated build on the GPU host; never installs over production)
ctest --test-dir build --output-on-failure \
  -R "froggeric|qwen3_6_frontend|tool_call_parser|openai_schema|openai_responses|anthropic_schema"
```

The artifact path is protected by `ninfer_qwen3_6_frontend_test`, which must pass unchanged; the
artifact renderer shares the fragment contract but not the v22.5 semantics.

## 6. Known boundaries

- Python numeric serialization is emulated for the shapes tool schemas actually use. Integers beyond
  `uint64` lose precision (the JSON parser stores them as doubles) and some float spellings differ
  from Python's `repr`; the user-facing list is in `docs/serving.md`.
- NInfer's wire model carries tool arguments as a JSON string. An object string is normalized
  through the template's mapping branch (sorted keys, Python tojson spacing); a non-object XML
  argument string is rejected because the wire contract requires an object.
- Unknown chat roles cannot be represented by the closed `ChatRole` enum and are rejected.
- XML tool-call output parsing is terminal and all-or-nothing: when the model's tool region cannot
  be structured, the complete region is returned as ordinary assistant content with no `tool_calls`
  and `finish_reason=stop`, and `request_done.result.tool_call_parse` records the fallback reason.
  XML-like text inside an argument value is the common trigger; `tool_call_format: "json"` avoids
  the XML ambiguity for such payloads.
- `max_tool_response_chars` that would truncate or drop a media placeholder is rejected because the
  Processor expands exactly the placeholder byte range.

## Troubleshooting

| Symptom | Diagnosis |
|---|---|
| `FIXTURE DRIFT` from the parity test | `chat_template.jinja` or `chat_template_oneline.txt` bytes changed; re-fetch the pinned revision, never edit fixtures in place |
| `frozen oracle requires jinja2 3.1.6` / Python 3.11 | the oracle venv is not the pinned one; recreate it from `tools/oracle_froggeric_v22_5/requirements.txt` |
| `FAIL kXmlInstructions… differs from the pinned template` | instruction text in `froggeric_v22_5/prompts.h` drifted from the template; fix the constant, do not update the checker |
| `chat media count does not match rendered placeholders` | a renderer branch emitted placeholder text without publishing `media_placeholders`; check every path that appends content |
| `rendered fragment slice intersects a media placeholder` | a slice (think extraction, tool-response truncation) cut the pad token; adjust the boundary or reject the option |
| A tag or think marker renders differently from the oracle | run `ninfer_qwen3_6_froggeric_v22_5_helpers_test` first to localize the helper, then `oracle.py check` |
| `--chat-style` rejected before model load | the value is outside `artifact, froggeric-v22.5`; the error lists the full set |
