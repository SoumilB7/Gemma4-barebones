"""
Gemma 4 E2B · Per-Layer Inputs (PLE)
────────────────────────────────────
Model-level machinery that builds the `(B, S, num_layers, 256)` tensor of
per-layer signals. The i-th 256-dim slice is consumed by the i-th decoder
layer's PLE block (see `transformer_block.py`).

Two parallel paths combine:

    Path A — token-id lookup
        embed_tokens_per_layer(input_ids) * √256
        reshape (B, S, num_layers, 256)

    Path B — residual projection
        per_layer_model_projection(inputs_embeds) * 1/√1536
        reshape (B, S, num_layers, 256)
        per_layer_projection_norm   (gain shape (256,))

    Combine
        (A + B) * 2**-0.5

`inputs_embeds` here is the *already √hidden-scaled* output of `GemmaEmbedding`
— matching that scaling exactly is critical for bit-parity (one bf16 ulp
of drift here propagates through every layer).
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from architecture.RMSnorm import GemmaRMSNorm


class GemmaPerLayerInputs(nn.Module):
    """
    Builds the `(B, S, num_layers, hidden_size_per_layer_input)` tensor that
    feeds the PLE block in every decoder layer.
    """

    EMBED_KEY = "model.language_model.embed_tokens_per_layer.weight"
    PROJ_KEY  = "model.language_model.per_layer_model_projection.weight"
    NORM_KEY  = "model.language_model.per_layer_projection_norm.weight"

    def __init__(self,
                 vocab_size_per_layer_input=262144,
                 hidden_size=1536,
                 hidden_size_per_layer_input=256,
                 num_hidden_layers=35,
                 rms_norm_eps=1e-6,
                 dtype=None):
        super().__init__()
        self.num_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input

        out_dim = num_hidden_layers * hidden_size_per_layer_input          # 8960
        self.embed_tokens_per_layer    = nn.Embedding(vocab_size_per_layer_input, out_dim, dtype=dtype)
        self.per_layer_model_projection = nn.Linear(hidden_size, out_dim, bias=False, dtype=dtype)
        self.per_layer_projection_norm = GemmaRMSNorm(hidden_size_per_layer_input, eps=rms_norm_eps,
                                                      with_scale=True, dtype=dtype)

        # Scales (Python floats; multiplied at compute time).
        self.embed_scale = hidden_size_per_layer_input ** 0.5              # √256
        self.proj_scale  = hidden_size ** -0.5                             # 1/√1536
        self.combine_scale = 2.0 ** -0.5                                    # 1/√2

    @classmethod
    def from_safetensors(cls, shard_path,
                         vocab_size_per_layer_input=262144,
                         hidden_size=1536,
                         hidden_size_per_layer_input=256,
                         num_hidden_layers=35,
                         rms_norm_eps=1e-6,
                         state_dict=None):
        sd = state_dict if state_dict is not None else load_file(str(shard_path))
        dtype = sd[cls.EMBED_KEY].dtype
        m = cls(vocab_size_per_layer_input=vocab_size_per_layer_input,
                hidden_size=hidden_size,
                hidden_size_per_layer_input=hidden_size_per_layer_input,
                num_hidden_layers=num_hidden_layers,
                rms_norm_eps=rms_norm_eps,
                dtype=dtype)
        m.embed_tokens_per_layer    .weight.data.copy_(sd[cls.EMBED_KEY])
        m.per_layer_model_projection.weight.data.copy_(sd[cls.PROJ_KEY])
        m.per_layer_projection_norm .weight.data.copy_(sd[cls.NORM_KEY])
        return m

    def forward(self, input_ids, inputs_embeds):
        """
        input_ids     : (B, S)               long
        inputs_embeds : (B, S, hidden_size)  bf16  (already √hidden-scaled)

        Returns: (B, S, num_layers, hidden_size_per_layer_input)
        """
        B, S = input_ids.shape

        # Path A: per-token lookup, scaled by √D_per_layer.
        ple = self.embed_tokens_per_layer(input_ids)                         # (B, S, 8960)
        ple = ple * torch.tensor(self.embed_scale, dtype=ple.dtype)
        ple = ple.view(B, S, self.num_layers, self.hidden_size_per_layer_input)

        # Path B: project residual stream, scale, reshape, norm.
        proj = self.per_layer_model_projection(inputs_embeds) * self.proj_scale
        proj = proj.view(B, S, self.num_layers, self.hidden_size_per_layer_input)
        proj = self.per_layer_projection_norm(proj)

        return (proj + ple) * self.combine_scale
