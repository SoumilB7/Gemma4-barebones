"""
┌─────────────────────────────────────────────────────────┐
│  Gemma 4 · Mixture of Experts · Barebones PyTorch       │
│─────────────────────────────────────────────────────────│
│  Model:  26B-A4B (128 experts, top-8, 1 shared)        │
│  Style:  Load HuggingFace weights directly              │
│  License: Apache 2.0                                    │
└─────────────────────────────────────────────────────────┘

How it all fits together:

    ┌─────────────── One Decoder Layer's FFN ───────────────┐
    │                                                       │
    │   input ──┬──→ DenseMLP ──────────→ dense_out ──┐     │
    │           │                                     + = output
    │           └──→ MoEBlock ──────────→ moe_out ────┘     │
    │                  │                                    │
    │                  ├─ Router → pick top-8 of 128        │
    │                  ├─ Run 8 ExpertFFNs, weighted sum    │
    │                  └─ Run SharedExpert (always on)      │
    │                                                       │
    └───────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class MoEConfig:
    """All the numbers that define the MoE architecture."""

    hidden_size:          int = 3072    # model dimension (d_model)
    intermediate_size:    int = 12288   # dense MLP width
    moe_intermediate_size:int = 1024    # each expert's FFN width (much smaller)
    num_experts:          int = 128     # total expert pool
    top_k:                int = 8       # experts activated per token
    rms_norm_eps:       float = 1e-6


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Building Blocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RMSNorm(nn.Module):
    """Normalize by root-mean-square. Simpler & cheaper than LayerNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class GeGLU_FFN(nn.Module):
    """
    Gated FFN used everywhere in Gemma 4.

        gate = GELU(gate_proj(x))
        out  = down_proj(gate × up_proj(x))

    Shapes:
        gate_proj : [hidden, intermediate]
        up_proj   : [hidden, intermediate]
        down_proj : [intermediate, hidden]
    """

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Router
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Router(nn.Module):
    """
    Picks which experts handle each token.

        logits = linear(x)            # [tokens, 128]
        top_k  = logits.topk(8)       # pick 8 highest
        weights = softmax(top_k)      # normalize

    One weight matrix: [hidden_size, num_experts] → [3072, 128]
    """

    def __init__(self, hidden, num_experts, top_k):
        super().__init__()
        self.gate  = nn.Linear(hidden, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(values, dim=-1)
        return indices, weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MoE Block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MoEBlock(nn.Module):
    """
    Sparse Mixture of Experts.

    Components:
        router         → decides which 8 experts each token uses
        experts[0..127] → 128 small GeGLU FFNs (only 8 run per token)
        shared_expert  → 1 GeGLU FFN that runs on EVERY token

    Output = weighted_sum(8 selected experts) + shared_expert
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k       = cfg.top_k

        self.router = Router(cfg.hidden_size, cfg.num_experts, cfg.top_k)

        self.experts = nn.ModuleList([
            GeGLU_FFN(cfg.hidden_size, cfg.moe_intermediate_size)
            for _ in range(cfg.num_experts)
        ])

        self.shared_expert = GeGLU_FFN(cfg.hidden_size, cfg.moe_intermediate_size)

    def forward(self, x):
        """x: [batch, seq, hidden] → same shape out."""
        B, S, H = x.shape
        flat = x.view(-1, H)                                   # [B*S, H]

        # Route
        indices, weights = self.router(flat)                    # [B*S, 8], [B*S, 8]

        # Run selected experts
        out = torch.zeros_like(flat)                            # [B*S, H]

        for i in range(self.num_experts):
            mask = (indices == i)                                # [B*S, 8]
            token_mask = mask.any(dim=-1)                       # [B*S]
            if not token_mask.any():
                continue

            tokens = flat[token_mask]                            # [n, H]
            result = self.experts[i](tokens)                    # [n, H]

            w = (weights * mask.float()).sum(dim=-1)[token_mask] # [n]
            out[token_mask] += result * w.unsqueeze(-1)

        # Shared expert (always on)
        out = out + self.shared_expert(flat)

        return out.view(B, S, H)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Layer FFN  (Dense + MoE, summed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Gemma4LayerFFN(nn.Module):
    """
    Gemma 4's unusual design: keep the dense MLP AND add MoE on top.

        output = DenseMLP(x) + MoEBlock(x)

    Most other MoE models (DeepSeek, Qwen) replace the MLP.
    Gemma sums them — trades efficiency for simpler architecture.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.dense = GeGLU_FFN(cfg.hidden_size, cfg.intermediate_size)
        self.moe   = MoEBlock(cfg)

    def forward(self, x):
        return self.dense(x) + self.moe(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Weight Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_weights(layer_ffn: Gemma4LayerFFN, state_dict: dict, layer: int):
    """
    Load HuggingFace weights into our modules.

    Usage:
        hf = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-26B-A4B-it")
        sd = hf.state_dict()

        cfg = MoEConfig()
        ffn = Gemma4LayerFFN(cfg)
        load_weights(ffn, sd, layer=0)
    """
    p = f"model.layers.{layer}"

    # Dense MLP
    layer_ffn.dense.gate_proj.weight.data = state_dict[f"{p}.mlp.gate_proj.weight"]
    layer_ffn.dense.up_proj.weight.data   = state_dict[f"{p}.mlp.up_proj.weight"]
    layer_ffn.dense.down_proj.weight.data = state_dict[f"{p}.mlp.down_proj.weight"]

    # Router
    layer_ffn.moe.router.gate.weight.data = state_dict[f"{p}.block_sparse_moe.router.gate.weight"]

    # 128 experts
    for e in range(layer_ffn.moe.num_experts):
        ep = f"{p}.block_sparse_moe.experts.{e}"
        layer_ffn.moe.experts[e].gate_proj.weight.data = state_dict[f"{ep}.gate_proj.weight"]
        layer_ffn.moe.experts[e].up_proj.weight.data   = state_dict[f"{ep}.up_proj.weight"]
        layer_ffn.moe.experts[e].down_proj.weight.data = state_dict[f"{ep}.down_proj.weight"]

    # Shared expert
    sp = f"{p}.block_sparse_moe.shared_expert"
    layer_ffn.moe.shared_expert.gate_proj.weight.data = state_dict[f"{sp}.gate_proj.weight"]
    layer_ffn.moe.shared_expert.up_proj.weight.data   = state_dict[f"{sp}.up_proj.weight"]
    layer_ffn.moe.shared_expert.down_proj.weight.data = state_dict[f"{sp}.down_proj.weight"]

    print(f"  ✓ layer {layer} MoE weights loaded")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":

    # Tiny config so it runs anywhere
    cfg = MoEConfig(hidden_size=64, intermediate_size=256,
                    moe_intermediate_size=32, num_experts=8, top_k=2)

    ffn = Gemma4LayerFFN(cfg)

    x = torch.randn(2, 4, 64)
    y = ffn(x)

    print(f"in:  {x.shape}")
    print(f"out: {y.shape}")
    print(f"ok:  {x.shape == y.shape}")

    # What the real 26B numbers look like
    r = MoEConfig()
    exp = 3 * r.hidden_size * r.moe_intermediate_size
    print(f"\n── Real 26B-A4B per layer ──")
    print(f"dense MLP:    {3 * r.hidden_size * r.intermediate_size:>12,} params")
    print(f"router:       {r.hidden_size * r.num_experts:>12,} params")
    print(f"128 experts:  {exp * 128:>12,} params")
    print(f"shared:       {exp:>12,} params")
    print(f"active/token: {3*r.hidden_size*r.intermediate_size + r.hidden_size*r.num_experts + exp*9:>12,} params")