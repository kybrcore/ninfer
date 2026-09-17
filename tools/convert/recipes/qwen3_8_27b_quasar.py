"""QUASAR line conversion: import QAT-trained W4A4 text, decode GDN controls, Q8 vocabulary.

Self-contained replacement for the cometkim nvfp4qat recipe pair
(qwen3_8_27b_nvfp4qat.py + qwen3_8_profile.py), semantically equivalent:
same assignment order, same formats, byte-equivalent artifacts.

Format decisions, one layer each:
- text/layers projections  nvfp4 W4A4 imported bit-exact from the QUASAR-QAT
                          checkpoint (packed words + stored divisors, AllowA4)
- GDN a/b projections     bf16 values from the same checkpoint (recurrence
                          control path: tiny matrices, error compounds over
                          every step - kept full precision by design)
- token_embedding /
  text/output_head        q8_g32 grouped_absmax computed locally from the BF16
                          base (the QAT checkpoint ships no packed vocabulary;
                          same byte cost as FP8, groupwise scales track the
                          embedding outlier structure better)
- vision / mtp / dflash2  upstream _optional allocation, unchanged
"""
from tools.convert.methods import cast_direct, grouped_absmax, import_encoded
from tools.convert.official_recipes import Q8, _optional


def configure(model, recipe, sources):
    # 64-layer Qwen3.8-27B dense only: this line's grouping and QAT import
    # geometry are trained for exactly this architecture.
    if model.config.get('num_hidden_layers') != 64 or 'num_experts' in model.config:
        raise ValueError('this recipe requires the 64-layer Qwen3.8-27B dense model')

    # Optional components first (vision/mtp/dflash2 formats + dflash KV share),
    # exactly like the official recipes.
    _optional(model, recipe)

    # Vocabulary: locally computed int8-groupwise from the BF16 base.
    for name in ('text/token_embedding', 'text/output_head'):
        recipe.assign(name, format=Q8, method=grouped_absmax)

    # Physical parent grouping (packing efficiency only, no precision effect):
    # full-attention layers group q/k/gate/v; GDN layers group q/k/v/z and
    # a/b separately; every layer groups mlp gate+up.
    for layer in range(64):
        prefix = f'text/layers/{layer}/'
        if prefix + 'attention/query' in model.parameters:
            recipe.group(tuple(prefix + 'attention/' + role
                               for role in ('query', 'key', 'gate', 'value')))
        else:
            recipe.group(tuple(prefix + 'gdn/' + role
                               for role in ('query', 'key', 'value', 'z')))
            recipe.group((prefix + 'gdn/a_projection', prefix + 'gdn/b_projection'))
        recipe.group((prefix + 'mlp/gate', prefix + 'mlp/up'))

    # Text trunk: import the QAT-trained weights verbatim. The 496 W4A4
    # projections arrive bit-exact (packed codes + stored divisors); the GDN
    # control gates are decoded from the same checkpoint and stored as BF16.
    quantized = sources['quantized']
    for name, parameter in model.parameters.items():
        if not name.startswith('text/layers/') or not parameter.projection:
            continue
        _, _, _, family, role = name.split('/')
        source = model.source(name, quantized, 'nvfp4')
        if family == 'gdn' and role in ('a_projection', 'b_projection'):
            recipe.assign(name, format='bf16', method=cast_direct, source=source)
        else:
            recipe.assign(name, format='nvfp4', method=import_encoded,
                          source=source, activation_policy='AllowA4')
