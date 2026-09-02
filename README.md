# NInfer + YaRN — Qwen3.8-27B at up to ~600K context on one RTX 5090

Qwen3.8-27B running past its native 262,144-token window on a single RTX 5090 (32 GB), using
[Neroued/ninfer](https://github.com/Neroued/ninfer)'s **NVFP4 KV cache** for the memory side and an
**optional YaRN context extension** (this repository's contribution) for the positional side.

**Validated: 590,053-token prompt, 5/5 needle retrieval, factor-4 YaRN, upstream NVFP4 KV.**
The practical configuration on this card is 500K; ~600K runs but sits at the VRAM edge with a large
throughput drop. Details in [Results](#results) and [Caveats](#caveats).

Upstream's own README is preserved as [`README-upstream.md`](README-upstream.md).

---

## Provenance — read this first

- **Upstream project:** [Neroued/ninfer](https://github.com/Neroued/ninfer), Apache-2.0. This
  repository is a branch of upstream `master` at `6e2786c5`.
- **NVFP4 / K8V4 KV cache support is upstream's work**, landed in Neroued/ninfer (commit `4ac73c4`
  and follow-ups). Nothing in the KV cache format, the NVFP4 kernels or the quantized weights was
  written or modified here.
- **This repository adds** the optional YaRN positional extension (`--rope-yarn-factor`,
  `--rope-original-max-position`), the context-ceiling change that lets `--max-context` exceed the
  native window when YaRN is enabled, and two attention-side changes required to *operate* beyond
  the native window (a visible-keys ceiling and a fixed-size page staging array). Three commits,
  ~280 lines, all under `splickz`.
- **Model weights are unchanged.** The artifact is upstream's
  [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
  (`qwen3_8_27b_nvfp4.ninfer`, sha256 `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`),
  itself derived from `Qwen/Qwen3.8-27B` via `unsloth/Qwen3.8-27B-NVFP4`. Nothing was retrained,
  fine-tuned or re-quantized.
- The KV *memory* question and the *positional* question are different problems. A compressed KV
  cache decides how many tokens fit in VRAM; YaRN decides how positions past the trained window are
  encoded. Upstream's NVFP4 KV solves the first; this repository only addresses the second.

Licensing: Apache-2.0, unchanged from upstream ([`LICENSE`](LICENSE)). Model weights carry their own
Apache-2.0 license from Qwen.

---

## What YaRN does here, precisely

- **Qwen3.8's native context remains 262,144 tokens.** `Variant::maximum_context` is untouched. With
  the default `--rope-yarn-factor 1` the engine is bit-for-bit upstream behaviour: same inverse
  frequency table, same attention scale, same context ceiling, byte-identical greedy output.
- **600K is not native context.** YaRN rescales the rotary frequencies (NTK-by-parts) so positions
  beyond 262,144 are representable, and applies the usual attention-scale correction. The model was
  not trained at that length; quality past the native window is *not* guaranteed (see caveats).
- With `--rope-yarn-factor F` the engine accepts `--max-context` up to `262144 × F`. Factor 4 was
  used for every long-context number below; the flag is capped at 4 in this release because that is
  the attention visible-keys ceiling that was validated.

---

## Results

One RTX 5090 (32,607 MiB), SM120, WSL2 (Ubuntu 22.04), CUDA 13.1, driver 610.47.
Long-context table measured at code commit `5f9b5d2c`; the release code commit `727c1897` (tag `v0.1.0-yarn-nvfp4-sm120`) adds only the factor-4 range check and was re-verified by the regression suite and a fresh-clone smoke test. Weights 19.0 GiB resident in every run.

Settings for every row: CUDA graphs **on**, speculative decoding **off**, prefix reuse **off**,
`--rope-yarn-factor 4 --rope-original-max-position 262144`, single request. Retrieval is 5
codewords planted at 10/25/50/75/90% depth with the question at the *end* of the prompt; each
codeword scored individually; `finish_reason=stop` required (a truncated answer is not a pass).
Prompt tokens are the engine's own `usage.prompt_tokens`. Prefill tok/s = prompt tokens ÷
time-to-first-token from a streamed response; decode tok/s over the remaining tokens.

| KV mode | `--max-context` | prompt tokens | retrieval | prefill tok/s | decode tok/s | KV runtime | peak VRAM |
|---|---|---|---|---|---|---|---|
| nvfp4 | 300,000 | 270,055 | 5/5 | 2,237 | 53.3 | 5.47 GiB | 26,809 MiB |
| nvfp4 | 300,000 | 290,105 | 5/5 | 2,130 | 51.6 | 5.47 GiB | 26,809 MiB |
| nvfp4 | 400,000 | 389,954 | 5/5 | 1,662 | 49.2 | 7.18 GiB | 28,511 MiB |
| nvfp4 | 500,000 | 489,803 | 5/5 | 1,357 | 48.6 | 8.90 GiB | 30,474 MiB |
| k8v4  | 400,000 | 389,954 | 5/5 | 1,820 | 49.7 | 9.90 GiB | 31,507 MiB |
| nvfp4 | 600,000 | 590,053 | 5/5 |   655 | 33.1 | 10.6 GiB | 32,045 MiB |

"KV runtime" is the engine's own reservation from its `capacity | KV …` startup line, not inferred
from `nvidia-smi`.

**Reservation frontier** (engine planner, same settings): nvfp4 at 600,000 fits with 407–500 MiB
free after startup; nvfp4 at 650,000 and k8v4 at 500,000 are refused by the planner
(`requires 12,320,728,576 / 13,203,857,920 bytes, 12,056,834,048 available`). Those refusals were
left in place — nothing bypasses the allocator.

**Memory control:** at `--max-context 262144` factor 1 and factor 4 reserve the same KV (4.82 GiB)
with the same free memory after startup (6.12 GiB). Enabling YaRN and raising the visible-keys
ceiling add no memory.

**Regression (32K context, greedy):** factor 1 with the flag is byte-identical to no flag for
`nvfp4` and `k8v4`; factor 4 at 32K answers correctly; factor 1 refuses `--max-context 300000`,
factor 4 accepts it. The `ninfer_yarn_test` unit test checks that factor 1 reproduces the engine's
shipped frequency table to < 1e-6.

---

## Usage

Build exactly as upstream (requirements: 64-bit Linux, RTX 5090, CUDA Toolkit 13.1+, CMake 3.28+,
C++20 compiler, Ninja, `pkg-config`, FFmpeg dev libraries, `libcurl`; see `README-upstream.md`):

```bash
git clone https://github.com/splickz/ninfer-yarn-nvfp4.git
cd ninfer-yarn-nvfp4
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Download upstream's artifact (unchanged):

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
sha256sum models/qwen3_8_27b_nvfp4.ninfer   # bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32
```

### Flags added by this repository

| Flag | Default | Meaning |
|---|---|---|
| `--rope-yarn-factor F` | `1` | YaRN scale factor, range `[1, 4]`. `1` disables YaRN entirely (upstream behaviour). `F > 1` rescales RoPE frequencies NTK-by-parts, applies the `0.1·ln(F) + 1` attention-scale correction, and raises the accepted `--max-context` ceiling to `262144 × F`. |
| `--rope-original-max-position N` | `262144` | The window the model was trained at, used as the YaRN reference length. Explicit rather than inferred; leave it at the default for Qwen3.8. |

### Native context (factor 1) — identical to upstream

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 5800 --kv-dtype nvfp4 --max-context 262144
```

### Practical long context — 500K, factor 4, NVFP4 KV

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 5800 --kv-dtype nvfp4 \
  --rope-yarn-factor 4 --rope-original-max-position 262144 \
  --max-context 500000
```

Reserves KV 8.90 GiB, leaves ~2.2 GiB free after startup on a 32 GB card; 489,803-token prompt
prefilled at ~1,360 tok/s, decoded at ~49 tok/s (single stream, no speculation).

### Edge — 600K, factor 4, NVFP4 KV

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 5800 --kv-dtype nvfp4 \
  --rope-yarn-factor 4 --rope-original-max-position 262144 \
  --max-context 600000
```

Starts with ~400–500 MiB free and peaks at 32,045 of 32,607 MiB during a 590K request. Retrieval
was 5/5, but prefill fell to ~655 tok/s and decode to ~33 tok/s. Treat it as a demonstration of the
ceiling, not a working configuration. Anything else on the GPU (a desktop compositor, a second
model) will make it fail to start or fail mid-request.

The validation runs additionally passed `--no-thinking --no-prefix-reuse` so that each tier was a
cold prefill with no cache reuse and no reasoning tokens inflating decode counts. Neither flag is
required for normal use.

### Reproducing the retrieval numbers

[`bench/ctxbench.py`](bench/ctxbench.py) is the harness that produced the table. It builds the
needle prompt, streams the response, and reports engine-counted prompt tokens, TTFT-derived prefill
rate, decode rate, peak VRAM, `finish_reason`, and which of the five codewords came back.

```bash
# server started as in the 500K example above, on port 5800
python3 bench/ctxbench.py 5800 nvfp4-f4 270000 390000 490000
```

Tiers are target prompt sizes; the engine's count will land within a few hundred tokens of each.
A tier must leave room for the answer under `--max-context` (700 output tokens are requested).

---

## Caveats

- **Retrieval is not proof of quality.** 5/5 codewords says the positions are addressable and the
  cache is intact. It says nothing about reasoning, summarisation or instruction-following quality at
  4× the trained window. YaRN past ~2× typically costs perplexity; measure your own task.
- **Context beyond native can degrade the model.** This is a runtime positional trick, not a
  600K-trained checkpoint.
- **600K is at the VRAM edge and roughly halves throughput** relative to 500K (prefill 1,357 → 655
  tok/s, decode 48.6 → 33.1). The cause was not isolated — candidates are memory pressure at ~98%
  VRAM under WSL2 and the split-K clamp doing more work per split at that key count. 500K is the
  number to plan around on a 32 GB card.
- **MTP / DFlash speculative decoding has not been validated with this port.** The draft head shares
  the rope tables so it should follow, but every number here is single-stream, no speculation.
- **`k8v4` uses ~40% more KV memory per token than `nvfp4`** (9.90 vs 7.18 GiB at 400K) and does not
  fit 500K on this card.
- **Multi-concurrency, vision inputs, and the `bf16` / `int8` decode paths past 262,144 were not
  exercised.** The page-staging fix was applied to all four decode kernels but only `nvfp4` and
  `k8v4` were run past the native window.
- **Experimental.** Three commits on top of a moving upstream; expect to rebase.

---

## Implementation notes

All changes are on top of upstream `6e2786c5`; `git log origin/master..HEAD` shows them.

- **Native context stays 262,144.** `Variant::maximum_context` is unchanged. `validate_target_options`
  (`src/targets/qwen3_6/impl/runtime/layouts_impl.h`) computes the accepted ceiling as
  `maximum_context × max(F, 1)` and additionally refuses any ceiling above the attention
  visible-keys constant, so a factor the kernels cannot serve is rejected at the engine level, not
  just by the CLI.
- **YaRN NTK-by-parts** (`src/ops/kernel/yarn.h`). For each rotary dimension pair the inverse
  frequency is blended between the original value (high-frequency dims, extrapolated unchanged) and
  the value divided by `F` (low-frequency dims, interpolated), with a linear ramp between the two
  boundary dimensions derived from `beta_fast = 32` and `beta_slow = 1` rotations across the
  original window. Constants match the YaRN paper and Qwen's published long-context configs.
- **Attention-scale correction.** `mscale = 0.1·ln(F) + 1` is folded into the sin/cos tables via the
  existing `kTextRopeAttentionScale` path, so no attention kernel changes for it.
- **Installation.** The scaled table and scale are written into the existing rope constant symbols
  after `Engine` construction (`rope_install_text_inv_frequency` / `rope_install_text_attention_scale`,
  `src/ops/launcher/rope.cu`). Factor 1 installs nothing. A cleaner upstream integration would thread
  the factor into the target's rope table construction instead of writing the symbols afterwards.
- **Visible-keys ceiling.** `kCausalAttentionMaximumVisibleKeys` (`include/ninfer/ops/softmax_attention.h`)
  raised from 262,144 to 1,048,576 (4× native). Split counts remain clamped to `SmallTMaximumSplits`,
  so the decode grid's Y dimension is bounded exactly as before; the memory control above shows no
  reservation change.
- **Fixed `PageIds = 64` staging removed.** The four small-T decode kernels
  (`src/ops/softmax_attention/dense/causal_cache/small_t_{bf16,i8,nvfp4,k8v4}.cuh`) staged page ids
  into a 64-entry shared array `physical_pages_s` before the main loop. Past roughly 500K tokens a
  split can span more pages than that, which is an out-of-bounds write. The kernels now index
  `block_table[...]` directly, which removes both the bound and the staging loop.
- **Tests.** `tests/test_yarn.cpp` (`ninfer_yarn_test`, built with `-DBUILD_TESTING=ON`) checks that
  factor 1 reproduces upstream's shipped `kTextRopeInvFrequency` table to < 1e-6 relative error with
  scale 1.0, and that factors 2 and 4 leave dims 0–13 untouched, divide the lowest frequency by
  exactly `F`, and produce the expected scale.

---

## Related

- Upstream: [Neroued/ninfer](https://github.com/Neroued/ninfer). A proposal to bring the YaRN option
  upstream is filed as an issue there; no PR has been opened.
- Weights: [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).
- Reproduction page / model card for this configuration:
  [splickz/qwen3.8-27b-yarn-nvfp4-sm120](https://huggingface.co/splickz/qwen3.8-27b-yarn-nvfp4-sm120).
- Earlier, separate work by the same author on a 4-bit E8-lattice KV cache for SM120:
  [splickz/ninfer-rk4v4-e8](https://github.com/splickz/ninfer-rk4v4-e8). That is a KV-compression
  approach and is unrelated to the positional extension here.
