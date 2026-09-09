# froggeric v22.5 fixture provenance

- Upstream: `froggeric/Qwen-Fixed-Chat-Templates`, fixed commit
  [`855bffc49448e299789730ff92c9b8d834d6cc14`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/tree/855bffc49448e299789730ff92c9b8d834d6cc14)
- Template self-reported version: `qwen3.8-froggeric-v22.5` (first line of `chat_template.jinja`)
- `chat_template.jinja` — 28,234 bytes, SHA-256
  `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`
- `chat_template_oneline.txt` — 22,391 bytes, SHA-256
  `eecae0e068e60f9c8665f0085b589d3e1c41508d359776c62018512c40b5879b`

Files are raw bytes fetched from the fixed revision URL
(`https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/<commit>/<file>`);
no Markdown/HTML conversion. A SHA-256 self-check test (see
`tests/targets/qwen3_6/test_froggeric_v22_5.cpp`) fails the build if either file drifts.

## License / provenance

The upstream repository carries no standalone LICENSE file; its README front matter
declares `license: apache-2.0` and states "Apache-2.0, inherited from Qwen" (Qwen model
licenses for the 3.x line are Apache-2.0). NInfer is Apache-2.0, so the fixtures may be
redistributed under the repository license with this provenance note. Adoption mode:
test fixtures and oracle inputs only — the templates are never loaded at runtime by the
NInfer binary (the v22.5 semantics are compiled in).

## Oracle environment (pinned)

- Python 3.11 (tested: 3.11.15)
- jinja2 3.1.6
- Environment: `StrictUndefined`, `keep_trailing_newline=True`, `lstrip_blocks=True`,
  `trim_blocks=True`, global `raise_exception` raising a plain `Exception`
- Offline: the generator and checker only read local files.
