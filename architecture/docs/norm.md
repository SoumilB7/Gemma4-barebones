# RMSNorm - Gemma 4 E2B

> The only normalization the model uses. Shows up at least four times
> per decoder layer, plus inside attention (q/k/v norms), inside the
> PLE block, on multimodal projector inputs, and on the final hidden
> state before the LM head. If our RMSNorm is wrong by 1 ulp, *every*
> downstream parity check is poisoned.

---

## 1. What RMSNorm is

LayerNorm subtracts the mean, divides by the standard deviation, then
scales and shifts. RMSNorm drops the mean subtraction and the bias:

```
LayerNorm(x) = ((x - mean(x)) / std(x)) * weight + bias
RMSNorm (x) = ( x             / rms(x)) * weight
                                          └─── optional in Gemma
```

Where `rms(x) = sqrt(mean(x²) + eps)`.

Why it works as well as LayerNorm in practice: for transformer hidden
states the mean-centering doesn't add much. Dropping it saves one pass
over the data and one parameter (the bias). RMSNorm is now the default
in Llama, Mistral, Gemma - basically every modern decoder.

## 2. Two flavours, one kernel

Gemma uses RMSNorm in two configurations:

```python
GemmaRMSNorm(dim, eps=1e-6, with_scale=True)    # learned per-dim gain  (most uses)
GemmaRMSNorm(dim, eps=1e-6, with_scale=False)   # no learned weight      (a few uses)
```

The weightless variant is just `x / rms(x)` - pure normalization, no
parameters. Gemma uses it on:

- the **multimodal projector input** (image/audio features before they
  enter the residual stream - the bug we hit in `embedding.md`),
- the **value head** in attention (`v_norm`),
- the **router input** in the MoE block (only used in larger Gemma 4
  variants - E2B has no MoE weights at all).

Everywhere else (input/post-attn/pre-FFN/post-FFN norms, q_norm,
k_norm, the final norm, the post-PLE norm) uses the scaled variant.

## 3. Two parity-critical implementation details

Two choices in HF's `Gemma4RMSNorm` are easy to miss and both cause
silent drift if you skip them:

### 3a. fp32 math, even when the input is bf16

```python
def forward(self, x):
    orig_dtype = x.dtype
    y = self._norm(x.float())            # ← fp32
    if self.with_scale:
        y = y * self.weight.float()
    return y.to(orig_dtype)              # cast back at the end
```

`x.pow(2).mean(-1)` in bf16 simply loses too many bits for the
per-token RMS to be accurate. The cast-up + cast-back trick is
universal in modern transformers - cheap insurance against precision
loss on a per-token reduction.

### 3b. `pow(-0.5)`, not `rsqrt`

HF computes the inverse square root as:

```python
return x * torch.pow(mean_squared, -0.5)
```

instead of the more obvious `x * torch.rsqrt(mean_squared)`. They are
mathematically the same, but the JIT compilers behind PyTorch and JAX
produce *byte-identical* outputs only with `pow(-0.5)`. We match it so
weight-by-weight parity survives.

## 4. The module

[`architecture/RMSnorm.py`](../RMSnorm.py):

```python
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True, dtype=None): ...

    @classmethod
    def from_safetensors(cls, shard_path, weight_key, eps=1e-6, dtype=None):
        """Load a scaled RMSNorm whose gain lives at `weight_key` in the shard."""

    def forward(self, x):
        # fp32 math, optional learned gain, cast back
```

Two ways to construct it:

```python
# fresh, weight initialised to ones (acts as identity until trained)
norm = GemmaRMSNorm(dim=1536)

# loaded from the model shard
norm = GemmaRMSNorm.from_safetensors(
    "model_weights/model.safetensors",
    "model.language_model.norm.weight",     # the final norm before LM head
)
```

## 5. The approver

The notebook test picks a norm whose weight is easy to grab - the
**final norm** at `model.language_model.norm`. Pump a random
`(B, S, 1536)` tensor through both ours and HF's, expect bit-equal
output:

```python
import torch
from architecture.RMSnorm import GemmaRMSNorm

torch.manual_seed(0)
x = torch.randn(1, 8, 1536, dtype=torch.bfloat16)

ours   = GemmaRMSNorm.from_safetensors(
    "model_weights/model.safetensors",
    "model.language_model.norm.weight",
)
theirs = model.model.language_model.norm

assert torch.equal(ours(x), theirs(x))      # bit-exact
```

If this fails, the suspects in priority order are: wrong eps, missing
fp32 cast, `rsqrt` instead of `pow(-0.5)`, wrong weight key.

## 6. Where it goes next

Once we trust RMSNorm, every later block (attention, FFN, PLE gate)
becomes testable in isolation: pull the right weight out of the shard,
build the block on top of `GemmaRMSNorm`, parity-check against HF.

Next up: **RoPE** - the only other thing attention needs before we can
build the attention block itself.
