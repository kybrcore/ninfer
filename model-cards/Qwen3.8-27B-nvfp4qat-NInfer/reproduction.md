# Qwen3.8-27B nvfp4qat: maintained recipe and reproduction

`feat/qwen3.8-nvfp4qat` preserves this profile lineage as a scoped conversion
patch over `feat/qwen3.8-profile-base`. Full and QAT are
siblings; neither depends on the other, DFlash execution, Windows, HQ, 1M context,
or kernel-performance changes. The shared prerequisite owns only vocabulary/optional
allocation and physical parent grouping. There is no new runtime identity or registry.

## Represented profile

QAT imports all 256 Text attention/GDN/MLP NVFP4 parents and their packed words,
E4M3FN block scales, FP32 weight divisors and activation input divisors from QUASAR.
All NVFP4 Text uses are AllowA4. The 48 paired GDN control parents are decoded
from the QAT source and rounded once to BF16, not taken from the BF16 teacher.
Vocabulary endpoints use Q8 group-32/FP16 scales; other direct tensors and optional
components come from the BF16 base. No local NVFP4 encoder or calibration JSON is
needed. This is neither the upstream mixed-FP8 source profile nor the fuller local
requantization profile, and is not an independently trained QAT checkpoint.

Vision/MTP use upstream optional allocation from the BF16 base. An optional DFlash2
component uses W8 allocation. NVFP4 DFlash2 components require separate conversion
and runtime support; the base recipe does not reproduce that component encoding.
Text-only and Text/MTP artifacts use the existing upstream runtime routes.

## Explicit local inputs

Use local snapshots, not glob/latest discovery or automatic downloads:

| Input | Historical snapshot | Role |
| --- | --- | --- |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | BF16 base |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | `d8e6fbfa3e3a78899b440222b827430045a05b44` | packed profile source |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | optional BF16 W8 companion |

These revisions identify historical inputs, not byte-identical v3 output requirements.
QAT imports its stored divisors; it requires no Full calibration file.

## Reproduce and run

From this checkout root, set `PYTHON` to an existing managed Python 3.11 interpreter
with repository conversion dependencies, `BASE` and `QUANTIZED` to the explicit
local snapshots above, `OUT` to a new `.ninfer` output path, and `ENGINE` to the
built NInfer executable.
No dependency installation or model download is performed by these commands.
On Windows, apply separately maintained Windows Python IO/build support first:
bare upstream imports use POSIX `os.sysconf`/positional IO. This platform prerequisite
is not included in either profile patch. Build commands remain in the platform branch.

```bash
"$PYTHON" -m tools.convert --help
"$PYTHON" -m tools.convert --model "$BASE" --source quantized="$QUANTIZED" --recipe tools/convert/recipes/qwen3_8_27b_nvfp4qat.py --components text,mtp --out "$OUT"
"$ENGINE" "$OUT" --prompt "Write a haiku about a GPU." --greedy --no-thinking --kv-dtype int8 --max-context 8192 --max-new 128
"$ENGINE" "$OUT" --prompt "Write a haiku about a GPU." --greedy --no-thinking --kv-dtype int8 --max-context 8192 --max-new 128 --spec mtp --draft-tokens 3
"$PYTHON" -m tests.convert.test_nvfp4qat_profile
```

Use `--components text` for the smallest plain-decode artifact, or add `vision` for
Vision. For an upstream-supported W8 companion add `dflash2` to the components and
`--source dflash2="$DFLASH2"`; select `--spec dflash2 --draft-tokens 7` at runtime.
Do not add the optional NVFP4 override without separately integrated runtime support.
Do not enable proposal-head options unless conversion included a proposal head.

## Provenance and verification boundary

The [original model card](https://github.com/cometkim/ninfer/blob/91f8a447e3d53f764c30116d7df53c5b4e213511/model-cards/Qwen3.8-27B-nvfp4qat-NInfer/README.md)
and [original artifact authority](https://github.com/cometkim/ninfer/blob/91f8a447e3d53f764c30116d7df53c5b4e213511/docs/maintainer/qwen3.8-27b-artifact.md)
preserve original calibration, release and evaluation provenance.
Old v2 commands, hashes and benchmark/quality tables describe historical shipped
artifacts, not newly converted v3 artifacts or this patch.

Executable CPU tests prepare real 64-layer logical shapes without large weights,
check allocation/policies and exact packed-word/divisor imports and BF16 decode.
They verify 256 NVFP4 parents, 48 BF16 control groups and Q8 vocabulary.
These CPU tests do not establish full-checkpoint conversion, GPU inference quality
or performance qualification.

## Prepare a v3 release

Choose a new output path; do not overwrite the published v2 artifact. Conversion
emits a v3 `.ninfer` file and its report. The explicit `--components` selection
is part of the release contract: optional weights are not added implicitly.

For all optional components, extend the conversion command above with
`--components text,mtp,vision,dflash2 --source dflash2="$DFLASH2"`. Set `DFLASH2`
to the local BF16 companion checkpoint. This selects W8 companion weights, not
the historical release's NVFP4 module. Add `--proposal` only when including an
optimized proposal head in the release; enable its runtime option only then.

Before publishing, record the new file's size and SHA-256 and retain the conversion
report. Check plain Text and MTP with the commands above; check Vision and DFlash2
on the same artifact if those components are included. Existing v2 benchmark and
quality tables do not qualify the new component allocation or container. Update
the model card with measured release facts rather than reusing the v2 hash.
