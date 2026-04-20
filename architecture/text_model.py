"""
Gemma 4 E2B · Text Model  (end-to-end)
──────────────────────────────────────
Wraps the pieces we built into a causal language model:

    input_ids  ─►  embed  ─►  PLE precompute  ─►  35 decoder layers
                                                        │
                                       KV-share routing (layers 15-34 reuse
                                       layer 13/14's K/V)
                                                        │
                              ─►  final RMSNorm  ─►  tied LM head  ─►  softcap
                                                        │
                                                     logits (B, S, vocab)

Two per-token gotchas inherited from earlier pieces:

- `inputs_embeds` is already √hidden-scaled by `GemmaEmbedding`, which is what
  `GemmaPerLayerInputs` expects. Don't rescale here.
- Final logits go through `softcap = 30 * tanh(logits / 30)` to match HF —
  `final_logit_softcapping = 30.0` in `config.text_config`.

The LM head is tied to the embedding matrix: we don't own a separate Linear.
Just `logits = hidden @ embed_tokens.weight.T`.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from architecture.RMSnorm import GemmaRMSNorm
from architecture.embedding import GemmaEmbedding
from architecture.ple import GemmaPerLayerInputs
from architecture.transformer_block import GemmaDecoderLayer
from architecture.attention import causal_mask
from architecture.rope import rope_local, rope_global


class GemmaTextModel(nn.Module):
    """
    The full text decoder stack. Call `from_safetensors(shard_path)` to build.

    Forward signature: `forward(input_ids) -> logits` of shape (B, S, V).
    """

    FINAL_NORM_KEY = "model.language_model.norm.weight"

    def __init__(self,
                 vocab_size=262144,
                 hidden_size=1536,
                 hidden_size_per_layer_input=256,
                 num_hidden_layers=35,
                 num_kv_shared_layers=20,
                 num_q_heads=8, num_kv_heads=1,
                 local_head_dim=256, global_head_dim=512,
                 sliding_window=512,
                 rms_norm_eps=1e-6,
                 final_logit_softcapping=30.0,
                 layer_types=None,
                 impl="eager",
                 dtype=None):
        super().__init__()
        assert layer_types is not None and len(layer_types) == num_hidden_layers

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping

        # KV-share routing: the last `num_kv_shared_layers` layers reuse K/V
        # from the last non-shared layer of the same type.
        self.first_kv_shared_idx = num_hidden_layers - num_kv_shared_layers
        prev = layer_types[:self.first_kv_shared_idx]
        self._prev_types = prev

        # Embedding + PLE precompute (model-level).
        self.embed_tokens    = GemmaEmbedding(vocab_size, hidden_size, dtype=dtype)
        self.per_layer_inputs = GemmaPerLayerInputs(
            vocab_size_per_layer_input=vocab_size,
            hidden_size=hidden_size,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
        )

        # 35 decoder layers.
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(
                hidden_size=hidden_size,
                hidden_size_per_layer_input=hidden_size_per_layer_input,
                num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                head_dim=(global_head_dim if layer_types[i] == "full_attention" else local_head_dim),
                sliding_window=(None if layer_types[i] == "full_attention" else sliding_window),
                # intermediate_size will be overwritten by from_safetensors' actual loader;
                # for plain __init__ we default to 6144 (layers 0-14 size).
                intermediate_size=6144,
                rms_norm_eps=rms_norm_eps, impl=impl, dtype=dtype,
            ) for i in range(num_hidden_layers)
        ])

        # Final norm before LM head.
        self.norm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps, with_scale=True, dtype=dtype)

        # RoPE tables, one per layer type. These cache inv_freq only; we call
        # them per forward to emit (cos, sin) at the current sequence length.
        self.rope_local  = rope_local(head_dim=local_head_dim)
        self.rope_global = rope_global(head_dim=global_head_dim)

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_safetensors(cls, shard_path, model_id=None, impl="eager"):
        """
        Build the full text model by loading `shard_path` ONCE and copying
        weights into every submodule. `model_id` (optional) is used to pull
        exact config values from HF — defaults are Gemma 4 E2B.
        """
        if model_id is not None:
            from transformers import AutoConfig
            tc = AutoConfig.from_pretrained(model_id).text_config
            kw = dict(
                vocab_size=tc.vocab_size,
                hidden_size=tc.hidden_size,
                hidden_size_per_layer_input=tc.hidden_size_per_layer_input,
                num_hidden_layers=tc.num_hidden_layers,
                num_kv_shared_layers=tc.num_kv_shared_layers,
                num_q_heads=tc.num_attention_heads,
                num_kv_heads=tc.num_key_value_heads,
                local_head_dim=tc.head_dim,
                global_head_dim=tc.head_dim * 2,           # E2B: 256 local, 512 global
                sliding_window=tc.sliding_window,
                rms_norm_eps=tc.rms_norm_eps,
                final_logit_softcapping=tc.final_logit_softcapping,
                layer_types=list(tc.layer_types),
            )
        else:
            # Hard-coded E2B defaults (match the config values above).
            kw = dict(
                layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 7,
            )

        sd = load_file(str(shard_path))
        dtype = sd[cls.FINAL_NORM_KEY].dtype

        m = cls(impl=impl, dtype=dtype, **kw)

        # Reuse component loaders by passing the pre-loaded state dict.
        m.embed_tokens     = GemmaEmbedding.from_safetensors(shard_path, state_dict=sd)
        m.per_layer_inputs = GemmaPerLayerInputs.from_safetensors(shard_path, state_dict=sd)

        for i, lt in enumerate(m.layer_types):
            m.layers[i] = GemmaDecoderLayer.from_safetensors(
                shard_path, layer_idx=i, layer_type=lt, impl=impl, state_dict=sd,
            )

        m.norm.weight.data.copy_(sd[cls.FINAL_NORM_KEY])
        return m

    # ──────────────────────────────────────────────────────────────────
    def _kv_source(self, i):
        """For shared layer i, return donor idx. None if non-shared."""
        if i < self.first_kv_shared_idx:
            return None
        return (len(self._prev_types) - 1
                - self._prev_types[::-1].index(self.layer_types[i]))

    def _is_donor(self, i):
        """True if i is the LAST non-shared layer of its type."""
        if i >= self.first_kv_shared_idx:
            return False
        return i == (len(self._prev_types) - 1
                     - self._prev_types[::-1].index(self.layer_types[i]))

    # ──────────────────────────────────────────────────────────────────
    def forward(self, input_ids, position_ids=None, return_hidden=False):
        """
        input_ids   : (B, S) long
        position_ids: (B, S) long, defaults to arange(S)
        return_hidden: if True, also return the pre-LM-head hidden state.

        Returns: logits (B, S, V), or (logits, hidden) if return_hidden=True.
        """
        B, S = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(S, dtype=torch.long, device=device)[None].expand(B, -1)

        # 1. Embed + PLE precompute (model-level).
        inputs_embeds    = self.embed_tokens(input_ids)                            # (B, S, H)
        per_layer_inputs = self.per_layer_inputs(input_ids, inputs_embeds)         # (B, S, L, 256)

        # 2. RoPE tables per layer type. `dummy` is just for dtype/device.
        dummy_l = inputs_embeds.new_zeros(B, S, self.rope_local.head_dim)
        dummy_g = inputs_embeds.new_zeros(B, S, self.rope_global.head_dim)
        cos_l, sin_l = self.rope_local(dummy_l, position_ids)
        cos_g, sin_g = self.rope_global(dummy_g, position_ids)

        # 3. Masks. For sliding layers with S > window, positions outside the
        #    window are blocked. For S <= window, sliding_mask == full_mask.
        mask_full = causal_mask(S, device, torch.float32, window=None)
        mask_loc  = causal_mask(S, device, torch.float32, window=self.sliding_window)

        # 4. Run layers with KV-share routing.
        hidden    = inputs_embeds
        shared_kv = {}                                                             # {donor_idx: (k, v)}
        for i, layer in enumerate(self.layers):
            is_global = self.layer_types[i] == "full_attention"
            cos, sin  = (cos_g, sin_g) if is_global else (cos_l, sin_l)
            mask      = mask_full if is_global else mask_loc
            pli_i     = per_layer_inputs[:, :, i, :]

            src    = self._kv_source(i)
            cached = shared_kv[src] if src is not None else None

            if self._is_donor(i):
                hidden, k, v = layer(hidden, pli_i, cos, sin, attention_mask=mask,
                                      cached_kv=cached, return_kv=True)
                shared_kv[i] = (k, v)
            else:
                hidden = layer(hidden, pli_i, cos, sin, attention_mask=mask,
                                cached_kv=cached)

        # 5. Final norm.
        hidden = self.norm(hidden)

        # 6. Tied LM head + softcap. `embed_tokens.weight` has shape (V, H).
        logits = hidden @ self.embed_tokens.embed.weight.t()                       # (B, S, V)
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap

        if return_hidden:
            return logits, hidden
        return logits
