# RoPE — Gemma 4 E2B

> Position is injected into Q and K (not into the residual stream) by
> *rotating* pairs of dimensions by an angle proportional to the token's
> position. Two variants live in this model — one for local layers, one for
> global — and they differ only in their inverse-frequency table.

---

## 1. Why rotary positions

Vanilla transformers added a learned or sinusoidal position vector to the
token embedding. That bakes position into the residual stream, which is
overkill — the only place position actually needs to be visible is inside
attention's `Q · Kᵀ` dot product.

RoPE solves this by **rotating** Q and K (just before the dot product) so
that the dot product `Qᵢ · Kⱼᵀ` becomes a function of `i - j`. Position
information is encoded *geometrically* — no extra parameters, no addition
to the residual, and the model gets a relative-position bias for free.

The trick: pair up the head's dimensions and rotate each pair by an angle
proportional to `position × 1/θ^(2k/D)`, where θ is a base wavelength and k
indexes the pair.

```
   pair (a, b) at position p, frequency f
       ┌                  ┐   ┌ a ┐
       │  cos(pf)  -sin(pf)│ · │ b │
       │  sin(pf)   cos(pf)│   └   ┘
       └                  ┘
```

Different pairs spin at different rates — small-index pairs rotate fast
(short-wavelength position bits), large-index pairs rotate slowly
(long-wavelength).

## 2. Two variants in Gemma 4 E2B

The two attention variants in this model use *different* RoPE configs:

| layer type        | head_dim | θ          | partial_rotary_factor |
|-------------------|----------|------------|-----------------------|
| local  (sliding)  | 256      | 10 000     | 1.0   (all dims rotate)         |
| global (full)     | 512      | 1 000 000  | 0.25  (only first 25% rotate)   |

Two things stand out:

- **Different head_dim per layer type.** Local heads are 256-dim, global
  heads are 512-dim. Q/K/V projections in attention will be sized
  differently per layer.
- **Proportional RoPE on global layers (a.k.a. p-RoPE).** Only the first
  25 % of the head's dimensions get rotated; the remaining 75 % stay
  position-invariant. This gives the global layer stable, content-only
  signals for long-range reasoning. Combined with θ = 1 000 000, the rotated
  dims also have very long wavelengths — no aliasing across long contexts.

## 3. How "only 25 % of dims rotate" is implemented

A neat trick: keep the cos/sin tensor full-width, but **set the inverse
frequency to zero for the non-rotated dims**:

```python
rope_angles = int(p * head_dim // 2)               # dims that rotate
nope_angles = head_dim // 2 - rope_angles          # dims that don't

inv_freq = cat([rotated_freqs, zeros(nope_angles)])
```

For the zero-frequency dims, the angle is always 0, so `cos = 1`, `sin = 0`,
and `apply_rope` becomes `x * 1 + rotate_half(x) * 0 = x` — identity. No
branching needed in the apply function; it works on full-width tensors
regardless of the partial factor.

## 4. The GPT-NeoX layout

The original RoPE paper paired dims by `(2i, 2i+1)`. HuggingFace adopted
the GPT-NeoX layout instead: pair `(i, i + head_dim/2)`. Mathematically
equivalent, but the tensor ops are simpler:

```python
def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def apply_rope(x, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)
```

Critically, the `cos`/`sin` produced by RoPE forward are also in this
layout (concatenated `[freqs, freqs]`, not interleaved). If you mix
conventions, every dot product silently drifts.

## 5. The module

[`architecture/rope.py`](../rope.py):

```python
class GemmaRoPE(nn.Module):
    def __init__(self, head_dim, theta, partial_rotary_factor=1.0): ...
    def forward(self, x, position_ids):
        # returns (cos, sin), each (B, S, head_dim), at x.dtype

# convenience
rope_local()    # head_dim=256, theta=10_000, full rotation
rope_global()   # head_dim=512, theta=1_000_000, p=0.25
```

`forward(x, position_ids)` only uses `x` for its dtype and device — it does
not actually transform `x`. The cos/sin are then passed to `apply_rope` at
the Q and K sites inside attention.

One implementation detail copied from HF: the `inv_freq @ positions`
multiply runs with `torch.autocast(enabled=False)` to force fp32. At long
positions, `cos(p · f)` is precision-sensitive — bf16 would lose enough
bits to drift the dot products downstream.

## 6. The approver

Three things to check against HF's `Gemma4TextRotaryEmbedding`:

```python
hf_rope = model.model.language_model.rotary_emb        # built by HF
ids     = torch.arange(8)[None]                        # positions [0..7]
x       = torch.zeros(1, 8, 512, dtype=torch.bfloat16) # dtype/device carrier

# global (full_attention) ────────────────────────────
cos_hf, sin_hf = hf_rope(x, ids, layer_type="full_attention")
cos_us, sin_us = rope_global()(x, ids)
assert torch.equal(cos_hf, cos_us)
assert torch.equal(sin_hf, sin_us)

# local (sliding_attention) ──────────────────────────
x256 = torch.zeros(1, 8, 256, dtype=torch.bfloat16)
cos_hf, sin_hf = hf_rope(x256, ids, layer_type="sliding_attention")
cos_us, sin_us = rope_local()(x256, ids)
assert torch.equal(cos_hf, cos_us)
assert torch.equal(sin_hf, sin_us)
```

If `cos`/`sin` are bit-equal, applying them inside attention will
necessarily match too — `apply_rope` is just elementwise multiplies and a
concat.

## 7. Where it goes next

RoPE has no weights to load and produces fully deterministic tables once
the config numbers are set. With it pinned down, **attention** becomes
testable: feed the same `(hidden, position_ids)` to ours and HF, and
compare Q/K (post-RoPE) directly.

Next up: **attention** — the densest block in the model.
