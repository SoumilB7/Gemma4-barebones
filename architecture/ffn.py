"""
Gemma 4 E2B · Feed-Forward Network (GeGLU)
──────────────────────────────────────────
The second sub-block of every decoder layer. Two parallel projections up,
one of them gets a GELU, they multiply elementwise, one projection back
down:

    y = down_proj(  gelu(gate_proj(x))  *  up_proj(x)  )
                    └── "gate" ──┘        └ "up" ┘

The elementwise multiply is what makes it *gated* — `gate_proj(x)` can
suppress or amplify each dimension of `up_proj(x)`. A plain MLP lacks
that multiplicative interaction.

## Width varies across the stack

For E2B, `intermediate_size = 6144` for the first 15 layers (0-14) and
**12 288** (= 6144 × 2) for the last 20 layers (15-34). The wider MLP in
the kv-shared layers is a deliberate trade: those layers save compute on
`k_proj`/`v_proj` (reused from earlier layers), and spend that budget on
a fatter FFN instead. Same FLOPs, different allocation.

## Activation

`gelu_pytorch_tanh` — the tanh-approximated GELU, bit-equivalent to
`F.gelu(x, approximate="tanh")`. HF uses this specific variant to stay
byte-identical with the JAX reference; `approximate="none"` (the exact
erf-based GELU) would drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class GemmaFFN(nn.Module):
    """
    GeGLU FFN. No biases.

    Args:
        hidden_size:        residual stream width (1536)
        intermediate_size:  FFN inner width (6144 standard, 12288 for
                            kv-shared layers)
    """

    def __init__(self, hidden_size=1536, intermediate_size=6144, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    @classmethod
    def from_safetensors(cls, shard_path, layer_idx, hidden_size=1536):
        """
        Load one decoder layer's FFN from `model.safetensors`. The
        intermediate size is read from the gate_proj shape, so this
        works for both standard (6144) and double-wide (12288) layers.
        """
        sd = load_file(str(shard_path))
        prefix = f"model.language_model.layers.{layer_idx}.mlp."

        gate_w = sd[prefix + "gate_proj.weight"]
        up_w   = sd[prefix + "up_proj.weight"]
        down_w = sd[prefix + "down_proj.weight"]

        intermediate_size = gate_w.shape[0]            # (inter, hidden)
        m = cls(hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=gate_w.dtype)

        m.gate_proj.weight.data.copy_(gate_w)
        m.up_proj  .weight.data.copy_(up_w)
        m.down_proj.weight.data.copy_(down_w)
        return m

    def forward(self, x):
        # tanh-approx GELU to match HF's `gelu_pytorch_tanh` byte-for-byte.
        gated = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        return self.down_proj(gated)
