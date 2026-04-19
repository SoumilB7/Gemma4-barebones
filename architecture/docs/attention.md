# Self-Attention — Gemma 4 E2B

> The densest block in the model. One residual-stream vector per token goes
> in; a refined vector per token comes out. Inside, every token gets to
> *look at* every earlier token (or just the last 512 of them) through the
> usual `softmax(Q · Kᵀ) · V` lens — but with a handful of E2B-specific
> twists that you have to get exactly right or nothing downstream matches.

---

## 1. The block in one diagram

```
hidden (B, S, 1536)
   │
   ├──► q_proj (1536 → H_q · D) ─► reshape ─► q_norm ─► RoPE ─┐
   │                                                          │
   ├──► k_proj (1536 → H_kv · D) ─► reshape ─► k_norm ─► RoPE ─┼─► softmax(Q · Kᵀ + mask) · V ─► reshape ─► o_proj ─► out
   │                                                          │
   └──► v_proj (1536 → H_kv · D) ─► reshape ─► v_norm ─────────┘
```

The reshape splits the projected tensor across heads:
`(B, S, H · D)` → `(B, S, H, D)` → `(B, H, S, D)` for the attention matmul.

## 2. The E2B numbers

E2B has two interleaved layer types. Same kernel, different shape constants:

| layer type        | head_dim | Q heads | KV heads | GQA ratio | sliding window | RoPE        |
|-------------------|----------|---------|----------|-----------|----------------|-------------|
| local  (sliding)  | 256      | 8       | 1        | 8 : 1     | 512            | full, θ=10k |
| global (full)     | 512      | 8       | 1        | 8 : 1     | —              | p-RoPE 0.25, θ=1M |

So both layer types use **8 : 1 GQA**, contradicting the overview's earlier
"2 : 1 local" line — the safetensors shapes are the source of truth:

```
local  q_proj  (2048, 1536)   = 8 × 256
local  k_proj  ( 256, 1536)   = 1 × 256        ← 8:1
global q_proj  (4096, 1536)   = 8 × 512
global k_proj  ( 512, 1536)   = 1 × 512        ← 8:1
```

GQA fan-out is just `K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)`
— no parameters, just a memory broadcast before the dot product.

## 3. Three details that bite

### 3a. `q_norm`, `k_norm`, **weightless** `v_norm`

```python
q_norm = GemmaRMSNorm(head_dim, with_scale=True)   # scaled
k_norm = GemmaRMSNorm(head_dim, with_scale=True)   # scaled
v_norm = GemmaRMSNorm(head_dim, with_scale=False)  # weightless!
```

Q and K get normalized *and* rescaled by a learned per-dim gain. V gets
normalized but its gain is fixed at 1. That's why the safetensors shard
has `q_norm.weight` and `k_norm.weight` keys but **no** `v_norm.weight` —
we don't load anything for v_norm.

### 3b. Softmax scaling = 1.0 (not `1/√d`)

The classical attention `softmax(Q · Kᵀ / √d)` divides by `√head_dim` so
the dot products don't explode. Gemma drops that scale:

```python
attn = q @ k.transpose(-1, -2)        # no /√d
attn = attn + mask
attn = softmax(attn, dim=-1, dtype=fp32).to(q.dtype)
```

Why it's safe: `q_norm` and `k_norm` already RMS-normalize Q and K to unit
length per head. The dot product magnitude is bounded by construction —
adding `1/√d` on top would shrink logits twice and flatten the softmax.

### 3c. Apply RoPE *after* the norms, *before* the transpose

```python
q = q_proj(hidden).view(B, S, H, D)
q = q_norm(q)
q = apply_rope(q, cos, sin, unsqueeze_dim=2)   # cos, sin: (B, S, D)
q = q.transpose(1, 2)                          # → (B, H, S, D)
```

`unsqueeze_dim=2` matches the `(B, S, H, D)` layout: cos/sin become
`(B, S, 1, D)` and broadcast over heads. If you transpose first and use
`unsqueeze_dim=1`, the math is the same — but in this codebase we keep
`(B, S, H, D)` until after RoPE so the broadcast is the obvious one.

## 4. The mask

```python
def causal_mask(S, device, dtype, window=None):
    i = torch.arange(S, device=device)
    allowed = i[:, None] >= i[None, :]              # lower triangle
    if window is not None:
        allowed &= (i[:, None] - i[None, :]) < window
    mask = torch.zeros(S, S, dtype=dtype).masked_fill(~allowed, -inf)
    return mask[None, None]                         # (1, 1, S, S)
```

The mask is *additive*: `attn = q @ kᵀ + mask`, so `0` means "attend"
and `-inf` means "don't". Sliding adds one extra constraint
(`i - j < window`). For S ≤ window, sliding equals plain causal — handy,
because our parity test runs at S = 8 and the local/global mask shapes
collapse to the same thing.

## 5. KV sharing (deferred to the stack)

E2B has `num_kv_shared_layers = 20`. For the **last 20 layers** (indices
15-34 out of 35), K and V are reused from the most recent **earlier
same-type** layer. The shared layers don't have their own `k_proj`,
`v_proj`, `k_norm`, or `v_norm` weights at all.

This is a **stack-level** optimization, not a per-layer one. Our
`GemmaAttention` always projects fresh K/V; the decoder stack will be
responsible for caching K/V from the source layer and threading it into
the dependent layers. For now, our parity test uses layers 0 (local) and
4 (global) — both well before the sharing point.

## 6. The module

[`architecture/attention.py`](../attention.py):

```python
class GemmaAttention(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim,
                 sliding_window=None, rms_norm_eps=1e-6, dtype=None): ...
    @classmethod
    def from_safetensors(cls, shard_path, layer_idx, layer_type, ...): ...
    def forward(self, hidden, cos, sin, attention_mask=None): ...

# convenience
attn_local()    # head_dim=256, sliding_window=512
attn_global()   # head_dim=512, no window
```

`from_safetensors(shard, layer_idx, layer_type)` is the load path — pass
either `"sliding_attention"` or `"full_attention"` and it picks
`head_dim` and `sliding_window` for you.

## 7. The approver

For each layer type, build cos/sin and a causal mask, then compare ours
against HF's `model.model.language_model.layers[L].self_attn`:

```python
import torch
from architecture.attention import GemmaAttention, causal_mask
from architecture.rope      import rope_local, rope_global

torch.manual_seed(0)
B, S = 1, 8
hidden = torch.randn(B, S, 1536, dtype=torch.bfloat16)
position_ids = torch.arange(S)[None]
mask = causal_mask(S, hidden.device, torch.float32)

# ── local layer 0 ─────────────────────────────
ours_l = GemmaAttention.from_safetensors(WEIGHTS / "model.safetensors",
                                          layer_idx=0, layer_type="sliding_attention")
cos_l, sin_l = rope_local()(torch.zeros(1, S, 256, dtype=torch.bfloat16), position_ids)
out_l = ours_l(hidden, cos_l, sin_l, attention_mask=mask)

hf_l = model.model.language_model.layers[0].self_attn
hf_out_l, _ = hf_l(hidden, position_embeddings=(cos_l, sin_l),
                   attention_mask=mask, shared_kv_states={})

assert torch.equal(out_l, hf_out_l)

# ── global layer 4 ────────────────────────────
ours_g = GemmaAttention.from_safetensors(..., layer_idx=4, layer_type="full_attention")
cos_g, sin_g = rope_global()(torch.zeros(1, S, 512, dtype=torch.bfloat16), position_ids)
out_g = ours_g(hidden, cos_g, sin_g, attention_mask=mask)

hf_g = model.model.language_model.layers[4].self_attn
hf_out_g, _ = hf_g(hidden, position_embeddings=(cos_g, sin_g),
                   attention_mask=mask, shared_kv_states={})

assert torch.equal(out_g, hf_out_g)
```

If parity fails, narrow the search by intercepting after each step: `q_proj`
output → after `q_norm` → after RoPE → softmax weights → V-weighted output
→ post-`o_proj`. The most common drift sources are mask shape/dtype, the
softmax fp32 cast, and forgetting that `scaling=1.0`.

### One trap: HF's SDPA backend

`AutoModelForCausalLM.from_pretrained(MODEL_ID)` defaults to
`attn_implementation="sdpa"` on most builds. SDPA fuses the
`softmax(Q·Kᵀ) · V` pipeline into one kernel whose summation order
differs from a hand-written eager loop. Numerically equivalent in fp32,
but in **bf16** the two paths drift by 1-3 ulps per token (any token
that attends to >1 position). The first token always matches because
its softmax has only one entry — `softmax([0, -∞, …]) = [1, 0, …]` —
so the output is just `V[0]`, no summation, no precision loss.

Both kernels are correct; they just round in different bits. We expose
the choice with an `impl=` flag so you can match either:

```python
GemmaAttention(..., impl="eager")  # default — visible math, matches HF eager
GemmaAttention(..., impl="sdpa")   # F.scaled_dot_product_attention, matches HF sdpa
```

So the parity matrix is:

| ours        | HF                              | bit-equal? |
|-------------|---------------------------------|------------|
| `impl="eager"` | `attn_implementation="eager"` | yes |
| `impl="sdpa"`  | `attn_implementation="sdpa"`  | yes |
| `impl="eager"` | `attn_implementation="sdpa"`  | no, 1-2 bf16 ulps |
| `impl="sdpa"`  | `attn_implementation="eager"` | no, 1-2 bf16 ulps |

Default is `eager` because the math is visible — you can see `Q · Kᵀ`,
the mask add, the softmax, and the multiply by V as separate lines, which
is the whole point of building from scratch. For matching a default-
loaded HF model (no `attn_implementation` arg), pass `impl="sdpa"`.

## 8. Where it goes next

Attention is the load-bearing piece. With it pinned down, the remaining
per-layer block is just **GeGLU FFN** (`down(gelu(gate(x)) * up(x))`),
two RMSNorms, and the residual adds. After that we stack 35 layers, wire
the per-layer-embedding (PLE) gate, and we have a full forward pass.

Next up: **GeGLU FFN** — the simplest of the remaining bricks.
