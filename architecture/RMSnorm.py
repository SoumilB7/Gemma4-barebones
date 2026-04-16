"""
Gemma 4 E2B · RMSNorm
─────────────────────
Root-Mean-Square normalization. The one norm used throughout the model:

    y = x / rms(x)              (weightless  — with_scale=False)
    y = (x / rms(x)) * weight   (scaled      — with_scale=True)

Two details that matter for bit-parity with HuggingFace:

- Math runs in **fp32**, then casts back to the input dtype.
- Uses `pow(mean_sq, -0.5)` rather than `rsqrt` — HF chose this to stay
  byte-identical with their JAX reference. We match it.

The weightless variant already showed up inline in embedding.py (the
pre-projection norm on image/audio features). This module generalizes it:
same kernel, optional learned gain.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True, dtype=None):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    @classmethod
    def from_safetensors(cls, shard_path, weight_key, eps=1e-6, dtype=None):
        """Load a scaled RMSNorm whose gain lives at `weight_key` in the shard."""
        w = load_file(str(shard_path))[weight_key]
        m = cls(w.shape[0], eps=eps, with_scale=True, dtype=dtype or w.dtype)
        m.weight.data.copy_(w)
        return m

    def _norm(self, x):
        mean_sq = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_sq, -0.5)

    def forward(self, x):
        orig_dtype = x.dtype
        y = self._norm(x.float())
        if self.with_scale:
            y = y * self.weight.float()
        return y.to(orig_dtype)
