"""
Gemma 4 E2B · Decoder Layer
───────────────────────────
One block of the 35-layer stack. Wraps `GemmaAttention` + `GemmaFFN` with
five RMSNorms, residual adds, the per-layer-embedding (PLE) gate, and a
`layer_scalar` post-multiply.

The model-level PLE machinery that builds the `(B, S, 35, 256)` tensor of
per-layer signals lives in `architecture/ple.py` — one slice from that
tensor (`per_layer_inputs[:, :, i, :]`) is what the i-th layer's PLE block
consumes via the `per_layer_input` argument of `forward`.

Sandwich-norm pattern (pre AND post norm around both attn and ffn):

    residual = x
    x = input_layernorm(x)
    x = self_attn(x, ...)
    x = post_attention_layernorm(x)        ← post-norm INSIDE the block
    x = residual + x

    residual = x
    x = pre_feedforward_layernorm(x)
    x = mlp(x)                             ← GemmaFFN (GeGLU)
    x = post_feedforward_layernorm(x)
    x = residual + x

PLE block (only because `hidden_size_per_layer_input=256` in config):

    residual = x
    x = per_layer_input_gate(x)            ← Linear 1536 → 256
    x = gelu_pytorch_tanh(x)
    x = x * per_layer_input                ← per-token signal of width 256
    x = per_layer_projection(x)            ← Linear 256 → 1536
    x = post_per_layer_input_norm(x)
    x = residual + x

    x *= layer_scalar                      ← scalar buffer, NOT 1.0
                                             (layer 0 = 0.01782, layer 34 = 0.16699)

The `per_layer_input` slice is computed once at the model level by
`GemmaPerLayerInputs`, then the i-th slice is fed into the i-th decoder
layer. KV sharing for the last 20 layers is still a stack-level concern;
this module always projects fresh QKV through its own `self_attn`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from architecture.RMSnorm import GemmaRMSNorm
from architecture.attention import GemmaAttention
from architecture.ffn import GemmaFFN
from architecture.ple import GemmaPerLayerInputs   # re-export: PLE lives in ple.py now

__all__ = ["GemmaDecoderLayer", "GemmaPerLayerInputs"]


# ──────────────────────────────────────────────────────────────────────
#  Decoder layer
# ──────────────────────────────────────────────────────────────────────

class GemmaDecoderLayer(nn.Module):
    """
    One Gemma 4 E2B decoder layer. Owns:

        self_attn  : GemmaAttention   (local or global flavour)
        mlp        : GemmaFFN         (standard 6144 or double-wide 12288)
        five RMSNorms (input, post-attn, pre-ffn, post-ffn, post-PLE)
        per_layer_input_gate    : Linear 1536 → 256
        per_layer_projection    : Linear 256  → 1536
        layer_scalar            : (1,)  — buffer, loaded from shard
    """

    def __init__(self,
                 hidden_size=1536,
                 hidden_size_per_layer_input=256,
                 num_q_heads=8,
                 num_kv_heads=1,
                 head_dim=256,
                 sliding_window=None,
                 intermediate_size=6144,
                 rms_norm_eps=1e-6,
                 impl="eager",
                 dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = hidden_size_per_layer_input

        self.self_attn = GemmaAttention(
            hidden_size=hidden_size, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, sliding_window=sliding_window,
            rms_norm_eps=rms_norm_eps, impl=impl, dtype=dtype,
        )
        self.mlp = GemmaFFN(hidden_size=hidden_size, intermediate_size=intermediate_size, dtype=dtype)

        self.input_layernorm           = GemmaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm  = GemmaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

        self.per_layer_input_gate    = nn.Linear(hidden_size, hidden_size_per_layer_input, bias=False, dtype=dtype)
        self.per_layer_projection    = nn.Linear(hidden_size_per_layer_input, hidden_size, bias=False, dtype=dtype)
        self.post_per_layer_input_norm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

        self.register_buffer("layer_scalar", torch.ones(1, dtype=dtype))

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_safetensors(cls, shard_path, layer_idx, layer_type,
                         hidden_size=1536, hidden_size_per_layer_input=256,
                         num_q_heads=8, num_kv_heads=1,
                         local_head_dim=256, global_head_dim=512,
                         sliding_window=512, rms_norm_eps=1e-6, impl="eager"):
        """
        Load layer `layer_idx`. `layer_type` is "sliding_attention" or
        "full_attention". Picks head_dim/window automatically and reads
        intermediate_size from the gate_proj shape (handles double-wide).
        """
        is_global = layer_type == "full_attention"
        head_dim  = global_head_dim if is_global else local_head_dim
        window    = None            if is_global else sliding_window

        sd = load_file(str(shard_path))
        prefix = f"model.language_model.layers.{layer_idx}."
        dtype  = sd[prefix + "input_layernorm.weight"].dtype
        intermediate_size = sd[prefix + "mlp.gate_proj.weight"].shape[0]

        m = cls(hidden_size=hidden_size,
                hidden_size_per_layer_input=hidden_size_per_layer_input,
                num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, sliding_window=window,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps, impl=impl, dtype=dtype)

        # Attention sub-block.
        ap = prefix + "self_attn."
        m.self_attn.q_proj.weight.data.copy_(sd[ap + "q_proj.weight"])
        m.self_attn.k_proj.weight.data.copy_(sd[ap + "k_proj.weight"])
        m.self_attn.v_proj.weight.data.copy_(sd[ap + "v_proj.weight"])
        m.self_attn.o_proj.weight.data.copy_(sd[ap + "o_proj.weight"])
        m.self_attn.q_norm.weight.data.copy_(sd[ap + "q_norm.weight"])
        m.self_attn.k_norm.weight.data.copy_(sd[ap + "k_norm.weight"])
        # v_norm is weightless — nothing to load.

        # FFN sub-block.
        mp = prefix + "mlp."
        m.mlp.gate_proj.weight.data.copy_(sd[mp + "gate_proj.weight"])
        m.mlp.up_proj  .weight.data.copy_(sd[mp + "up_proj.weight"])
        m.mlp.down_proj.weight.data.copy_(sd[mp + "down_proj.weight"])

        # Layer-level norms + PLE block + scalar.
        m.input_layernorm           .weight.data.copy_(sd[prefix + "input_layernorm.weight"])
        m.post_attention_layernorm  .weight.data.copy_(sd[prefix + "post_attention_layernorm.weight"])
        m.pre_feedforward_layernorm .weight.data.copy_(sd[prefix + "pre_feedforward_layernorm.weight"])
        m.post_feedforward_layernorm.weight.data.copy_(sd[prefix + "post_feedforward_layernorm.weight"])
        m.per_layer_input_gate      .weight.data.copy_(sd[prefix + "per_layer_input_gate.weight"])
        m.per_layer_projection      .weight.data.copy_(sd[prefix + "per_layer_projection.weight"])
        m.post_per_layer_input_norm .weight.data.copy_(sd[prefix + "post_per_layer_input_norm.weight"])
        m.layer_scalar.data.copy_(sd[prefix + "layer_scalar"])
        return m

    # ──────────────────────────────────────────────────────────────────
    def forward(self, hidden_states, per_layer_input, cos, sin, attention_mask=None):
        """
        hidden_states   : (B, S, hidden_size)
        per_layer_input : (B, S, hidden_size_per_layer_input)   one layer's slice
        cos, sin        : (B, S, head_dim)                      from GemmaRoPE
        attention_mask  : (1, 1, S, S) additive mask, or None

        Returns: (B, S, hidden_size)
        """
        # 1. Attention block (sandwich norm + residual).
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cos, sin, attention_mask=attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        # 2. FFN block (sandwich norm + residual).
        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        hidden_states = residual + h

        # 3. PLE block: gate residual to 256-dim, multiply by per-token signal,
        #               project back to 1536, norm, residual add.
        residual = hidden_states
        h = self.per_layer_input_gate(hidden_states)
        h = F.gelu(h, approximate="tanh")
        h = h * per_layer_input
        h = self.per_layer_projection(h)
        h = self.post_per_layer_input_norm(h)
        hidden_states = residual + h

        # 4. Per-layer scalar (NOT 1.0; it's a learned buffer in the shard).
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states
