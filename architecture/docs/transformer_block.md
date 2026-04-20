# Decoder Layer - Gemma 4 E2B

> One block of the 35-layer stack. It wires `GemmaAttention` and
> `GemmaFFN` together with **five** RMSNorms, two residual adds, the
> per-layer-embedding (PLE) gate, and a final scalar multiply. Stack
> 35 of these with a final norm and the LM head and you have the
> language model.

---

## 1. The block in one diagram

```
in (B, S, 1536)
   │
   ├──► input_layernorm ──► self_attn ──► post_attention_layernorm ──► (+) ──► hidden
   │                                                                     ▲
   └─────────────────────────────────────────────────────────────────────┘  residual

   ├──► pre_feedforward_layernorm ──► mlp (GeGLU) ──► post_feedforward_layernorm ──► (+) ──► hidden
   │                                                                                 ▲
   └─────────────────────────────────────────────────────────────────────────────────┘  residual

   ├──► per_layer_input_gate (1536→256) ──► gelu_tanh ──► ⊙ per_layer_input
   │                                                       │
   │                                              per_layer_projection (256→1536)
   │                                                       │
   │                                              post_per_layer_input_norm
   │                                                       │
   └────────────────────────────────────────────────► (+) ──► hidden
                                                            ▲
                                                            └  residual

   hidden *= layer_scalar   (a (1,) buffer in the shard — NOT 1.0)
   │
   ▼
out (B, S, 1536)
```

Three sub-blocks, each with its own residual add. The PLE block only
fires because `hidden_size_per_layer_input = 256` is set in config; if
it weren't, the layer would just be sandwich-norm attn + sandwich-norm
ffn.

## 2. Sandwich norm

Most modern decoders use *pre-norm only*: `x + sublayer(norm(x))`.
Gemma puts a norm on **both sides** of each sublayer:

```python
x = x + post_norm(sublayer(pre_norm(x)))
```

The post-norm sits *inside* the residual branch. Without it the
residual stream's variance can drift across 35 layers. The post-norm
clamps the sublayer's contribution to unit RMS before it adds back.
Cheap insurance - two more 1536-dim gain vectors per sublayer.

That's where four of the five norms come from: input/post-attn for the
attention sublayer, pre-ffn/post-ffn for the FFN. The fifth (post-PLE)
is the same trick on the PLE branch.

## 3. The PLE block

Per-layer embeddings inject a *token-specific, layer-specific* signal
of width 256 into every layer. The signal is built **once at the model
level** (`GemmaPerLayerInputs`, see §4 - lives in
[`architecture/ple.py`](../ple.py)) and the i-th 256-dim slice goes
into layer i.

Inside the block:

```python
residual = hidden
h = per_layer_input_gate(hidden)        # 1536 → 256
h = gelu_pytorch_tanh(h)
h = h * per_layer_input                 # elementwise; both (B, S, 256)
h = per_layer_projection(h)             # 256 → 1536
h = post_per_layer_input_norm(h)
hidden = residual + h
```

### How the per-layer signal flows in

PLE is built once at the top of the forward pass and *fed into* every
layer. The flow at run time:

```
input_ids (B, S)
   │
   ├──► GemmaEmbedding ─────────────────────► inputs_embeds (B, S, 1536)
   │                                                │
   └────────┐  ┌────────────────────────────────────┘
            ▼  ▼
       GemmaPerLayerInputs  (model-level — runs ONCE)
            │
            ▼
     per_layer_inputs (B, S, 35, 256)         ← shape: one 256-dim row per (token, layer)
            │
   ┌────────┼─────────────────── slice along dim 2 ──────────────────┐
   │        │                                                         │
   ▼        ▼                                                         ▼
 [:,:,0,:]  [:,:,1,:]                                              [:,:,34,:]
   │        │                                                         │
   ▼        ▼                                                         ▼
DecoderL0  DecoderL1   ...                                       DecoderL34
   │        │                                                         │
   └─► hidden ─► hidden ─►  ...  ─► hidden ─► final norm ─► LM head ─►
```

So PLE is **not** mixed into the residual stream at the top. Each
layer receives its own 256-dim slice through a side-channel (the
`per_layer_input` arg of `forward`), gates it through the 1536→256
projection, multiplies elementwise, projects back to 1536, then adds.
The main residual stream never has to "carry" the PLE signal across
layers - each layer reads its slice fresh.

Why it exists: the residual stream is a 1536-dim shared bus that every
layer reads from and writes to. PLE gives each layer a *private*,
low-rank channel to receive token-conditioned information that doesn't
have to compete for bandwidth with the main stream. The memory cost
is the model-level table (262 144 × 8960 ≈ 2.3 G params), but at
inference each token uses only one row - cheap.

## 4. Building `per_layer_input` (model level)

Two parallel paths combine into the (B, S, 35, 256) tensor:

**Path A - token-id lookup.** A 262 144-vocab embedding table of width
`35 × 256 = 8960`. Per token, scaled by `√256 = 16`:

```python
ple = embed_tokens_per_layer(input_ids) * sqrt(256)   # (B, S, 8960)
ple = ple.view(B, S, 35, 256)
```

**Path B - residual projection.** The freshly-embedded (and
√hidden-scaled) `inputs_embeds` is projected to 8960, scaled by
`1/√1536`, reshaped, and RMS-normed with a per-256-dim learned gain:

```python
proj = per_layer_model_projection(inputs_embeds) * (1/sqrt(1536))
proj = proj.view(B, S, 35, 256)
proj = per_layer_projection_norm(proj)               # gain shape (256,)
```

**Combine.** Average them in a unit-RMS sense:

```python
per_layer_inputs = (proj + ple) * (1/sqrt(2))
```

That tensor is sliced once per layer in the decoder loop:
`per_layer_inputs[:, :, i, :]` → fed into layer i's PLE block.

## 5. `layer_scalar` - not 1.0

The very last line of `forward` is:

```python
hidden = hidden * self.layer_scalar
```

`layer_scalar` is a `(1,)` buffer loaded from the shard. **Do not skip
loading it** - it is not initialized to 1, and the values vary across
layers:

| layer | `layer_scalar` |
|-------|----------------|
| 0     | 0.01782…       |
| 34    | 0.16699…       |

It dampens the per-layer contribution into the residual stream - small
early, growing later. Forgetting to load it produces drift on the
order of **50×** in the output, which is loud enough to spot but a
costly hour to track down.

## 6. Shapes for E2B

Per-layer (one of 35):

| weight                                         | shape         |
|------------------------------------------------|---------------|
| `input_layernorm.weight`                       | `(1536,)`     |
| `post_attention_layernorm.weight`              | `(1536,)`     |
| `pre_feedforward_layernorm.weight`             | `(1536,)`     |
| `post_feedforward_layernorm.weight`            | `(1536,)`     |
| `per_layer_input_gate.weight`                  | `(256, 1536)` |
| `per_layer_projection.weight`                  | `(1536, 256)` |
| `post_per_layer_input_norm.weight`             | `(1536,)`     |
| `layer_scalar`                                 | `(1,)`        |
| + nested `self_attn.*` and `mlp.*` keys (see attention.md, ffn.md) |

Model-level PLE:

| weight                                                   | shape             |
|----------------------------------------------------------|-------------------|
| `model.language_model.embed_tokens_per_layer.weight`     | `(262144, 8960)`  |
| `model.language_model.per_layer_model_projection.weight` | `(8960, 1536)`    |
| `model.language_model.per_layer_projection_norm.weight`  | `(256,)`          |

## 7. The modules

[`architecture/ple.py`](../ple.py) - model-level (runs once):

```python
class GemmaPerLayerInputs(nn.Module):
    @classmethod
    def from_safetensors(cls, shard_path, ..., state_dict=None): ...
    def forward(self, input_ids, inputs_embeds):
        # → (B, S, num_layers, 256)
```

[`architecture/transformer_block.py`](../transformer_block.py) -
per-layer:

```python
class GemmaDecoderLayer(nn.Module):
    @classmethod
    def from_safetensors(cls, shard_path, layer_idx, layer_type,
                         impl="eager", state_dict=None):
        # layer_type ∈ {"sliding_attention", "full_attention"} — picks
        # head_dim/window. intermediate_size auto-detected from gate_proj.
    def forward(self, hidden, per_layer_input, cos, sin, attention_mask=None,
                cached_kv=None, return_kv=False):
        # per_layer_input is per_layer_inputs[:, :, layer_idx, :] — one slice
        # → (B, S, 1536)
```

`GemmaDecoderLayer.from_safetensors` does *all* the per-layer loading:
attention nested keys, mlp keys, four norms, PLE gate/projection/norm,
and `layer_scalar`. One classmethod, one layer.

## 8. The approver

For each layer type, build cos/sin and a causal mask, then compare
ours against HF's `model.model.language_model.layers[L]`:

```python
import torch
from architecture.transformer_block import GemmaDecoderLayer
from architecture.attention         import causal_mask
from architecture.rope              import rope_local, rope_global

torch.manual_seed(0)
B, S = 1, 8
hidden = torch.randn(B, S, 1536, dtype=torch.bfloat16)
pli    = torch.randn(B, S,  256, dtype=torch.bfloat16)   # one layer's slice
position_ids = torch.arange(S)[None]
mask = causal_mask(S, hidden.device, torch.float32)

# ── sliding layer 0 ───────────────────────────
ours_l = GemmaDecoderLayer.from_safetensors(WEIGHTS, layer_idx=0,
                                            layer_type="sliding_attention")
cos_l, sin_l = rope_local()(torch.zeros(1, S, 256, dtype=torch.bfloat16), position_ids)
out_l = ours_l(hidden, pli, cos_l, sin_l, attention_mask=mask)

hf_l = model.model.language_model.layers[0]
hf_out_l = hf_l(hidden_states=hidden.clone(), per_layer_input=pli,
                shared_kv_states={}, position_embeddings=(cos_l, sin_l),
                attention_mask=mask, position_ids=position_ids)
assert torch.equal(out_l, hf_out_l)

# ── full layer 4 ──────────────────────────────
ours_g = GemmaDecoderLayer.from_safetensors(WEIGHTS, layer_idx=4,
                                            layer_type="full_attention")
cos_g, sin_g = rope_global()(torch.zeros(1, S, 512, dtype=torch.bfloat16), position_ids)
out_g = ours_g(hidden, pli, cos_g, sin_g, attention_mask=mask)

hf_g = model.model.language_model.layers[4]
hf_out_g = hf_g(hidden_states=hidden.clone(), per_layer_input=pli,
                shared_kv_states={}, position_embeddings=(cos_g, sin_g),
                attention_mask=mask, position_ids=position_ids)
assert torch.equal(out_g, hf_out_g)
```

We tested layer 4 (not 15+) on purpose: the last 20 layers are
KV-shared, which is a *stack-level* concern. A single layer in
isolation always projects fresh K/V, so testing a kv-shared layer in
isolation is fine - but the parity assertion only tells you about K/V
*routing* once the stack is wired up. Layer 4 is well below the
sharing point.

## 9. The HF eager-mode trap

HF's default `attn_implementation="sdpa"` will give you 1-2 bf16 ulps
of drift even when our attention's `impl` matches. Always load HF
with:

```python
hf = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
)
```

If parity fails on this layer test and *didn't* fail on the standalone
attention test, `layer_scalar` is the next place to look - the 50×
scale-down makes most other bugs look like sign-flips.

## 10. KV sharing for layers 15-34

The last 20 layers don't compute their own K/V. They reuse K/V from
the **last non-shared layer of the same attention type**:

```
non-shared (0..14)                       shared (15..34)
  …
  layer 13 (sliding) ──┐                  layer 15 (sliding) ◀──┐
  layer 14 (full)   ──┼┐                  layer 16 (sliding) ◀──┤  reuses
                      ││                  layer 17 (sliding) ◀──┤  layer 13's
                      ││                  layer 18 (sliding) ◀──┤  K, V
                      ││                  layer 19 (full)    ◀──┐
                      ││                  layer 20 (sliding) ◀──┤
                      ││                  ...                   │
                      └┴───────────────►  layer 19, 24, 29, 34  reuse layer 14
                                          layers 15..33 sliding reuse layer 13
```

**Why "stack-level"**: a single shared layer can't compute its own K/V
- it needs K/V that some *earlier* layer in the same forward pass
produced. The broker has to be the stack.

The routing rule (extracted from `config.text_config`):

```python
first_kv = num_hidden_layers - num_kv_shared_layers   # 15
prev     = layer_types[:first_kv]                     # types of layers 0..14

# For shared layer i (i >= first_kv), donor index = last index in `prev`
# whose type matches layer_types[i]:
src = len(prev) - 1 - prev[::-1].index(layer_types[i])
```

For E2B that resolves to: every shared `sliding_attention` layer
reuses layer 13; every shared `full_attention` layer reuses layer 14.

**API on our side**: `GemmaAttention.forward` and
`GemmaDecoderLayer.forward` both take two optional kwargs:

| arg          | meaning                                                                                |
|--------------|----------------------------------------------------------------------------------------|
| `cached_kv`  | `(k, v)` to use *instead of* projecting; skips k_proj/k_norm/RoPE-K/v_proj/v_norm    |
| `return_kv`  | also return `(out, k, v)` so the stack can stash them                                  |

K and V are cached **pre-GQA-expand** (shape `(B, H_kv, S, head_dim)`)
- matching HF's `shared_kv_states` storage. K is post-norm and
post-RoPE; V is post-norm.

A subtle point: the shard *does* still ship `k_proj`/`v_proj`/`k_norm`
weights for layers 15-34, even though HF never uses them. They're
dead weight at inference. Loading via `from_safetensors` doesn't fail;
we just never call those projections when `cached_kv` is provided.

The stack-level loop (see the "FULL-STACK cumulative through layer N"
cell in [`load_pytorch_model.ipynb`](../../load_pytorch_model.ipynb)):

```python
shared_kv = {}
for i in range(N + 1):
    src    = kv_source(i)                          # 13, 14, or None
    cached = shared_kv[src] if src is not None else None
    if is_donor(i):                                # i ∈ {13, 14}
        hidden, k, v = layer(hidden, pli[:,:,i,:], cos, sin, mask,
                              cached_kv=cached, return_kv=True)
        shared_kv[i] = (k, v)
    else:
        hidden = layer(hidden, pli[:,:,i,:], cos, sin, mask, cached_kv=cached)
```

## 11. Where it goes next

With KV-share routing in place, the cumulative-N cell is bit-equal
against HF for any N in [0, 34]. Remaining for a full text model:

1. Wrap the loop in a `GemmaTextModel` module so callers don't manage
   `shared_kv` or `from_safetensors` per layer.
2. Final `GemmaRMSNorm` on `hidden`.
3. LM head - tied `nn.Linear` onto the embedding matrix transpose;
   logits parity against `hf(input_ids).logits`.
