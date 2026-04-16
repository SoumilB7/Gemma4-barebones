"""
Gemma 4 E2B · Rotary Position Embeddings (RoPE)
───────────────────────────────────────────────
Position is injected into Q and K (not into the residual stream) by **rotating**
pairs of dimensions by an angle proportional to the token's position.

Two variants live in this model, differing only in their inv_freq table:

    local  layers (sliding_attention) → standard RoPE
        head_dim = 256, theta = 10_000, all dims rotate
    global layers (full_attention)    → proportional RoPE  (a.k.a. p-RoPE)
        head_dim = 512, theta = 1_000_000, only first 25 % of dims rotate

p-RoPE keeps the bottom 75 % of dims position-invariant — useful for long
contexts where global attention needs stable, content-only signals.

We use the GPT-NeoX layout that HuggingFace adopted: dims are paired by
(i, i + head_dim/2), not (2i, 2i+1). `apply_rope` and `rotate_half` follow
that convention so our outputs line up tensor-for-tensor with HF's
`apply_rotary_pos_emb`.
"""

import torch
import torch.nn as nn


def rotate_half(x):
    """Pair dims (i, i+H/2) and rotate them: (a, b) → (-b, a)."""
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x, cos, sin, unsqueeze_dim=2):
    """
    Apply RoPE to a tensor of shape (..., head_dim).
    `cos`, `sin` come in as (B, S, head_dim) and are unsqueezed to broadcast
    over the head axis. `unsqueeze_dim=2` matches x of shape (B, S, H, D).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class GemmaRoPE(nn.Module):
    """
    Precomputes inv_freq once and emits (cos, sin) on demand.

    Args:
        head_dim:               size of one attention head (e.g. 256 local, 512 global)
        theta:                  base wavelength (10_000 local, 1_000_000 global)
        partial_rotary_factor:  fraction of dims to rotate (1.0 standard, 0.25 p-RoPE)
    """

    def __init__(self, head_dim, theta, partial_rotary_factor=1.0):
        super().__init__()
        self.head_dim = head_dim
        rope_angles = int(partial_rotary_factor * head_dim // 2)

        # Frequencies for the rotated portion only. Zero-pad the rest so those
        # dims get angle=0 → cos=1, sin=0 → identity (no rotation).
        rotated = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim))
        nope = head_dim // 2 - rope_angles
        if nope > 0:
            inv_freq = torch.cat([rotated, torch.zeros(nope, dtype=torch.float32)])
        else:
            inv_freq = rotated
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Returns (cos, sin), each of shape (B, S, head_dim), at the dtype of x.
        `x` is just used for dtype/device; only its dtype/device matter.
        """
        # (B, head_dim/2, 1)  @  (B, 1, S)  →  (B, head_dim/2, S)  →  (B, S, head_dim/2)
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()

        # Force fp32 even under autocast — sin/cos at long positions are precision-sensitive.
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ positions).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ──────────────────────────────────────────────────────────────────────
#  Convenience constructors for the two layer types in Gemma 4 E2B
# ──────────────────────────────────────────────────────────────────────

def rope_local(head_dim=256, theta=10_000.0):
    return GemmaRoPE(head_dim=head_dim, theta=theta, partial_rotary_factor=1.0)


def rope_global(head_dim=512, theta=1_000_000.0, partial_rotary_factor=0.25):
    return GemmaRoPE(head_dim=head_dim, theta=theta, partial_rotary_factor=partial_rotary_factor)
