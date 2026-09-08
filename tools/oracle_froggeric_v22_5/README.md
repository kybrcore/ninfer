# Froggeric v22.5 oracle tooling

Offline oracle for the compiled `froggeric-v22.5` chat renderer. Everything here reads local files
only; the renderer itself never loads the template at runtime.

## Pinned environment

- Python 3.11 (tested 3.11.15) and `jinja2==3.1.6` (`requirements.txt`)
- Upstream commit and file SHA-256 values: `../../tests/fixtures/frontend/froggeric_v22_5/PROVENANCE.md`
- The oracle refuses to run on any other Python/Jinja version.

```bash
python3 -m venv /tmp/ov
/tmp/ov/bin/pip install -r tools/oracle_froggeric_v22_5/requirements.txt
```

## What `oracle.py check` verifies

1. **Fixture integrity** — both template files match the SHA-256 values recorded in
   `PROVENANCE.md`.
2. **Golden parity** — every input under `tests/fixtures/frontend/froggeric_v22_5/inputs/` is
   rendered with both the pretty and the oneline template, the two are asserted byte-identical, and
   the result is compared with the committed golden. `expect_error` inputs are checked for the
   expected raise.
3. **Prompt-constant parity** — the tool instructions (XML/JSON × thinking on/off) and the low/xhigh
   reasoning instructions are re-rendered from the pinned template and compared byte-for-byte with
   the constants in `src/targets/qwen3_6/impl/frontend/froggeric_v22_5_prompts.h`. This is why the
   renderer must never hand-copy instruction text.

```bash
/tmp/ov/bin/python tools/oracle_froggeric_v22_5/oracle.py check     # verify
/tmp/ov/bin/python tools/oracle_froggeric_v22_5/oracle.py generate  # rewrite goldens after a deliberate input change
```

`make_inputs.py` deterministically regenerates every input from the matrix in this directory; run it
before `generate` when adding cases so inputs and goldens stay reproducible.

## C++ side

The C++ parity test (`ninfer_qwen3_6_froggeric_v22_5_test`) reads the same inputs and goldens and
compares the compiled renderer byte-for-byte. It also runs `tests/targets/qwen3_6/test_froggeric_v22_5_helpers.cpp`
companions through `ninfer_qwen3_6_froggeric_v22_5_helpers_test` for the Python string/tojson,
tag-stripper, and think-extraction helpers.

Architecture and troubleshooting: `docs/maintainer/frontend-chat-rendering.md`.
