"""
Gemma 4 E2B · Self-Attention
────────────────────────────
The densest block in the model. Per layer:

    hidden ─► q_proj  ─► q_norm ─► RoPE ─┐
    hidden ─► k_proj  ─► k_norm ─► RoPE ─┼─► softmax(Q·Kᵀ) · V ─► o_proj ─► out
    hidden ─► v_proj  ─► v_norm ─────────┘

Two layer types share this kernel and only differ in shape constants:

    local  (sliding_attention)  head_dim=256, sliding_window=512
    global (full_attention)     head_dim=512, sliding_window=None

E2B uses 8:1 GQA in BOTH layer types — 8 query heads share 1 KV head.

Three details that bite if you miss them:

- `softmax` scaling is **1.0**, not 1/√d. The q_norm normalizes Q to unit-RMS
  per head, which replaces the usual scaling. Multiplying by 1/√d on top would
  shrink logits twice.
- `v_norm` is **weightless** RMSNorm (with_scale=False) — V is normalized but
  not rescaled. That's why the shard has no `v_norm.weight` key.
- KV sharing for the last `num_kv_shared_layers` layers is a *stack-level*
  concern, not a layer-level one. This module always projects QKV; the stack
  decides whether to reuse a previous layer's K/V.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from architecture.RMSnorm import GemmaRMSNorm
from architecture.rope import apply_rope


def causal_mask(seq_len, device, dtype, window=None):
    """
    Build a (1, 1, S, S) additive mask: 0 where attendable, -inf elsewhere.
    `window=None` is plain causal. `window=W` is sliding causal — a query
    at position i can only attend to positions in [i-W+1, i].
    """
    i = torch.arange(seq_len, device=device)
    allowed = i[:, None] >= i[None, :]          # lower triangle inc. diag
    if window is not None:
        allowed &= (i[:, None] - i[None, :]) < window
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[None, None]                     # (1, 1, S, S)


class GemmaAttention(nn.Module):
    """
    A single self-attention block (one decoder layer's worth).

    Args:
        hidden_size:     residual stream width (1536 for E2B)
        num_q_heads:     number of query heads (8 for E2B)
        num_kv_heads:    number of K/V heads (1 for E2B → 8:1 GQA)
        head_dim:        per-head dim (256 local, 512 global)
        sliding_window:  None for global, 512 for local (only used by callers
                         that build masks; the module itself is window-agnostic)
        rms_norm_eps:    matches the rest of the model (1e-6)
    """

    def __init__(
        self,
        hidden_size=1536,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=256,
        sliding_window=None,
        rms_norm_eps=1e-6,
        impl="eager",
        dtype=None,
    ):
        super().__init__()
        assert impl in ("eager", "sdpa"), f"impl must be 'eager' or 'sdpa', got {impl!r}"
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.impl = impl                                 # "eager" or "sdpa"
        self.kv_groups = num_q_heads // num_kv_heads     # GQA fan-out

        self.q_proj = nn.Linear(hidden_size, num_q_heads  * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_q_heads  * head_dim, hidden_size, bias=False, dtype=dtype)

        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps, with_scale=True,  dtype=dtype)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps, with_scale=True,  dtype=dtype)
        self.v_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps, with_scale=False, dtype=dtype)

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_safetensors(cls, shard_path, layer_idx, layer_type,
                         hidden_size=1536, num_q_heads=8, num_kv_heads=1,
                         local_head_dim=256, global_head_dim=512,
                         sliding_window=512, rms_norm_eps=1e-6, impl="eager"):
        """
        Load one decoder layer's attention from `model.safetensors`.

        `layer_type` is "sliding_attention" (local) or "full_attention" (global).
        Picks head_dim and sliding_window automatically.
        """
        is_global = layer_type == "full_attention"
        head_dim  = global_head_dim if is_global else local_head_dim
        window    = None            if is_global else sliding_window

        sd = load_file(str(shard_path))
        prefix = f"model.language_model.layers.{layer_idx}.self_attn."

        # Use weight dtype (bf16) for the module
        dtype = sd[prefix + "q_proj.weight"].dtype
        m = cls(hidden_size=hidden_size, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, head_dim=head_dim,
                sliding_window=window, rms_norm_eps=rms_norm_eps,
                impl=impl, dtype=dtype)

        m.q_proj.weight.data.copy_(sd[prefix + "q_proj.weight"])
        m.k_proj.weight.data.copy_(sd[prefix + "k_proj.weight"])
        m.v_proj.weight.data.copy_(sd[prefix + "v_proj.weight"])
        m.o_proj.weight.data.copy_(sd[prefix + "o_proj.weight"])
        m.q_norm.weight.data.copy_(sd[prefix + "q_norm.weight"])
        m.k_norm.weight.data.copy_(sd[prefix + "k_norm.weight"])
        # v_norm is weightless — nothing to load
        return m

    # ──────────────────────────────────────────────────────────────────
    def forward(self, hidden, cos, sin, attention_mask=None):
        """
        hidden          : (B, S, hidden_size)         bf16
        cos, sin        : (B, S, head_dim)            from GemmaRoPE
        attention_mask  : (B|1, 1, S, S) additive mask of {0, -inf}, or None

        Returns: (B, S, hidden_size)
        """
        B, S, _ = hidden.shape

        q = self.q_proj(hidden).view(B, S, self.num_q_heads,  self.head_dim)
        q = self.q_norm(q)
        q = apply_rope(q, cos, sin, unsqueeze_dim=2)
        q = q.transpose(1, 2)                                # (B, H_q,  S, D)

        k = self.k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = apply_rope(k, cos, sin, unsqueeze_dim=2)
        k = k.transpose(1, 2)                                # (B, H_kv, S, D)

        v = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_norm(v)
        v = v.transpose(1, 2)                                # (B, H_kv, S, D)

        # GQA: expand K/V heads to match Q heads.
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        if self.impl == "sdpa":
            # PyTorch's fused kernel. Bit-equal to a `from_pretrained(...)`
            # HF model (which defaults to attn_implementation="sdpa").
            # SDPA's mask must match q's dtype.
            mask = attention_mask.to(q.dtype) if attention_mask is not None else None
            out  = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            # Eager: visible math, bit-equal to HF loaded with
            # attn_implementation="eager". Drifts 1-2 bf16 ulps from sdpa.
            # Scaling=1.0 because q_norm already normalized Q.
            attn = q @ k.transpose(-1, -2)                   # (B, H_q, S, S)
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out  = attn @ v                                  # (B, H_q, S, D)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


# ──────────────────────────────────────────────────────────────────────
#  Convenience constructors for the two layer types in Gemma 4 E2B
# ──────────────────────────────────────────────────────────────────────

def attn_local(hidden_size=1536, num_q_heads=8, num_kv_heads=1,
               head_dim=256, sliding_window=512, rms_norm_eps=1e-6,
               impl="eager", dtype=None):
    return GemmaAttention(hidden_size, num_q_heads, num_kv_heads, head_dim,
                          sliding_window=sliding_window,
                          rms_norm_eps=rms_norm_eps, impl=impl, dtype=dtype)


def attn_global(hidden_size=1536, num_q_heads=8, num_kv_heads=1,
                head_dim=512, rms_norm_eps=1e-6, impl="eager", dtype=None):
    return GemmaAttention(hidden_size, num_q_heads, num_kv_heads, head_dim,
                          sliding_window=None,
                          rms_norm_eps=rms_norm_eps, impl=impl, dtype=dtype)
