# Feed-Forward Network (GeGLU) — Gemma 4 E2B

> The second sub-block of every decoder layer. Takes a residual-stream
> vector, runs it through a gated MLP, writes it back. No heads, no
> positions, no masks — just three matmuls and a multiply.

---

## 1. The block

```
x ──► gate_proj (1536 → inter) ─► GELU(tanh) ─┐
                                              ▼
                                              ⊙
                                              ▲
x ──► up_proj   (1536 → inter) ────────────────┘
                                              │
                                              ▼
                                   down_proj (inter → 1536) ──► y
```

In code, one line:

```python
y = down_proj(gelu(gate_proj(x), approximate="tanh") * up_proj(x))
```

No biases. No dropout in inference. The only non-linearity is the GELU.

## 2. Why *gated*

A plain MLP is `down(act(up(x)))`. A gated MLP multiplies two parallel
projections: `down(act(gate(x)) * up(x))`. The `act(gate(x))` term can
*mask* individual dimensions of `up(x)` — push them toward zero or let
them through. That multiplicative interaction gives the block much more
expressive power per parameter than a plain MLP, which is why every
modern decoder (Llama, Mistral, Gemma) uses it.

Historical names:
- **GLU**: `sigmoid(gate(x)) * up(x)` — the original (Dauphin et al.).
- **SwiGLU**: `silu(gate(x)) * up(x)` — Llama / Mistral.
- **GeGLU**: `gelu(gate(x)) * up(x)` — Gemma. What we have here.

## 3. The one activation detail that bites

HF uses `gelu_pytorch_tanh`:

```python
F.gelu(x, approximate="tanh")
```

Not the default `approximate="none"` (exact erf-based GELU). The two
are close numerically but **not bit-equal**. The JAX reference that
Gemma was trained with uses the tanh approximation, and HF matched it
for weight parity. So must we.

## 4. Width varies across the stack

E2B has two FFN widths:

| layers     | intermediate_size | notes                                 |
|------------|-------------------|---------------------------------------|
| 0-14       | 6 144             | standard                              |
| 15-34      | 12 288 (= 2×)     | kv-shared layers, double-wide MLP     |

The last 20 layers reuse K/V from earlier same-type layers (the
`num_kv_shared_layers=20` setting). Those layers save compute on
`k_proj`/`v_proj` — the saved budget is spent on a 2× wider FFN
instead. Same FLOPs overall, different allocation.

Our `from_safetensors` reads `gate_proj.weight.shape[0]` and picks the
width automatically; one class handles both.

## 5. Shapes for E2B

| weight              | standard (L<15) | double-wide (L≥15) |
|---------------------|-----------------|--------------------|
| `gate_proj.weight`  | `(6144, 1536)`  | `(12288, 1536)`    |
| `up_proj.weight`    | `(6144, 1536)`  | `(12288, 1536)`    |
| `down_proj.weight`  | `(1536, 6144)`  | `(1536, 12288)`    |

No biases, so no `*.bias` keys.

## 6. The module

[`architecture/ffn.py`](../ffn.py):

```python
class GemmaFFN(nn.Module):
    def __init__(self, hidden_size=1536, intermediate_size=6144, dtype=None): ...
    @classmethod
    def from_safetensors(cls, shard_path, layer_idx, hidden_size=1536): ...
    def forward(self, x):
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
```

## 7. The approver

Run one standard-width layer (0) and one double-wide layer (15)
through ours and HF's, expect bit-equal output.

```python
import torch
from architecture.ffn import GemmaFFN

torch.manual_seed(0)
x = torch.randn(1, 8, 1536, dtype=torch.bfloat16)

# standard width (layer 0)
ours_0 = GemmaFFN.from_safetensors(WEIGHTS / "model.safetensors", layer_idx=0)
hf_0   = model.model.language_model.layers[0].mlp
assert torch.equal(ours_0(x), hf_0(x))

# double-wide (layer 15, kv-shared)
ours_15 = GemmaFFN.from_safetensors(WEIGHTS / "model.safetensors", layer_idx=15)
hf_15   = model.model.language_model.layers[15].mlp
assert torch.equal(ours_15(x), hf_15(x))
```

If parity fails, the usual culprit is the GELU variant
(`approximate="tanh"` vs `"none"`). The matmul order is unambiguous
and doesn't drift.

## 8. What about the MoE second route?

HF's `Gemma4TextDecoderLayer` has a conditional second FFN path:

```python
if self.enable_moe_block:
    moe_out  = self.experts(x, *self.router(x))
    hidden   = dense_out + moe_out
else:
    hidden   = dense_out
```

Driven by `enable_moe_block` in the config. For **E2B** this is `False`
and the checkpoint has no `router.*` or `experts.*` weights at all —
zero MoE keys in the shard. The second route is compile-time gated off
for this size; `GemmaFFN` alone is the full FFN.

The MoE route is reserved for larger Gemma 4 variants (E4B / flagship).
If we ever target those, we'd add a sibling `GemmaMoE` module and the
`if` in the decoder-layer glue — but for E2B it would be dead code we
have no weights to populate, so we leave it out.

## 9. Where it goes next

With attention and FFN both parity-checked, a full decoder layer is
just `attn + ffn` sandwiched in the right norms and residual adds:

```python
x = x + attn(input_layernorm(x))      # but actually: both pre- and
                                      # post-attn norms, see below
x = x + ffn (pre_feedforward_layernorm(x))
```

E2B actually has **four** norms per layer (input, post-attention,
pre-feedforward, post-feedforward) plus PLE gating on top. That's
the decoder-layer doc.

Next up: the **decoder layer** — wiring attention + FFN with the four
norms and the PLE gate.
