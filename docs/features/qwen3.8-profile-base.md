# Shared Qwen3.8 profile allocation

This page describes the shared helper that the sibling full and QAT recipes once
used: it required a 64-layer dense model, assigned Q8 vocabulary endpoints, used
upstream optional-component allocation and grouped related Text projections. Its
implementation, `qwen3_8_profile.py`, was removed together with
`qwen3_8_27b_nvfp4qat.py` in `61773f73`, after the self-contained
[qwen3_8_27b_quasar.py](../../tools/convert/recipes/qwen3_8_27b_quasar.py) proved
byte-equivalent (same assignment order, same formats, byte-identical output outside
the random metadata header); both originals remain in git history.

It was not a standalone conversion recipe and selected no full/QAT weight
encoding; owning recipes called it under the [conversion contract](../weight-conversion.md).
It introduced no runtime identity, artifact registry or DFlash execution route.
