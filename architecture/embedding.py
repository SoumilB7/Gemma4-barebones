"""
Gemma 4 E2B · Embeddings
────────────────────────
Three modules that produce vectors in the 1536-dim residual stream:

    GemmaEmbedding         text id        → 1536    (scaled by √hidden)
    GemmaVisionProjector   vision feat 768→ 1536    (last-mile of image path)
    GemmaAudioProjector    audio  feat 1536→1536    (last-mile of audio path)

The two projectors are bias-free linear maps. They expect features produced
by the vision/audio towers. Until we've built those towers in this repo,
`.embed_file()` borrows the HuggingFace tower to get features, then runs
them through *our* projector weights.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


# ──────────────────────────────────────────────────────────────────────
#  Text embedding
# ──────────────────────────────────────────────────────────────────────

class GemmaEmbedding(nn.Module):
    """Token-id → residual vector, scaled by √hidden (Gemma convention)."""

    WEIGHT_KEY = "model.language_model.embed_tokens.weight"

    def __init__(self, vocab_size, hidden_size, dtype=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, dtype=dtype)

    @classmethod
    def from_safetensors(cls, shard_path, key=None, state_dict=None):
        k = key or cls.WEIGHT_KEY
        w = state_dict[k] if state_dict is not None else _load_weight(shard_path, k)
        vocab, hidden = w.shape
        m = cls(vocab, hidden, dtype=w.dtype)
        m.embed.weight.data.copy_(w)
        return m

    def forward(self, ids):
        hidden = self.embed.weight.shape[1]
        # Round √hidden into the weight's dtype BEFORE multiplying — HF stores
        # the scale as a bf16 buffer, so the scalar is 39.25 (not 39.1918…).
        # Skipping the cast diverges by ~1 bf16 ulp.
        scale = torch.tensor(hidden ** 0.5, dtype=self.embed.weight.dtype)
        return self.embed(ids) * scale


# ──────────────────────────────────────────────────────────────────────
#  Vision / audio projectors  (last mile into residual stream)
# ──────────────────────────────────────────────────────────────────────

class _Projector(nn.Module):
    """Bias-free Linear(in_dim → out_dim). Subclasses set WEIGHT_KEY + MEDIA."""

    WEIGHT_KEY: str
    MEDIA:      str           # "image" | "audio"
    TOWER_ATTR: str           # "vision_tower" | "audio_tower"
    INPUT_KEYS: dict           # {tower_arg_name: processor_output_key}

    EPS = 1e-6

    def __init__(self, in_dim, out_dim, dtype=None):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False, dtype=dtype)

    def _rms_norm(self, x):
        # Weightless RMSNorm: HF's Gemma4RMSNorm(with_scale=False) computed in fp32.
        orig_dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.EPS)
        return x32.to(orig_dtype)

    @classmethod
    def from_safetensors(cls, shard_path, key=None):
        w = _load_weight(shard_path, key or cls.WEIGHT_KEY)
        out_dim, in_dim = w.shape
        m = cls(in_dim, out_dim, dtype=w.dtype)
        m.proj.weight.data.copy_(w)
        return m

    def forward(self, feats):
        return self.proj(self._rms_norm(feats))

    def embed_file(self, path, model_id):
        """File → soft tokens. Borrows HF's tower until we build our own."""
        proc, hf = _hf(model_id)
        msg    = [{"role": "user", "content": [{"type": self.MEDIA, "path": str(path)}]}]
        inputs = proc.apply_chat_template(
            msg, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        tower    = getattr(hf.model, self.TOWER_ATTR)
        tower_kw = {arg: inputs[src] for arg, src in self.INPUT_KEYS.items() if src in inputs}
        with torch.no_grad():
            feats = tower(**tower_kw).last_hidden_state
        return self(feats)


class GemmaVisionProjector(_Projector):
    WEIGHT_KEY = "model.embed_vision.embedding_projection.weight"
    MEDIA      = "image"
    TOWER_ATTR = "vision_tower"
    INPUT_KEYS = {"pixel_values": "pixel_values", "pixel_position_ids": "image_position_ids"}


class GemmaAudioProjector(_Projector):
    WEIGHT_KEY = "model.embed_audio.embedding_projection.weight"
    MEDIA      = "audio"
    TOWER_ATTR = "audio_tower"
    INPUT_KEYS = {"input_features": "input_features", "input_features_mask": "input_features_mask"}


# ──────────────────────────────────────────────────────────────────────
#  Internals
# ──────────────────────────────────────────────────────────────────────

def _load_weight(shard_path, key):
    return load_file(str(shard_path))[key]


_HF_CACHE = {}

def _hf(model_id):
    """Lazy-load (processor, model) once per model_id. Only the towers get used."""
    if model_id not in _HF_CACHE:
        from transformers import AutoProcessor, AutoModelForCausalLM
        proc  = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.eval()
        _HF_CACHE[model_id] = (proc, model)
    return _HF_CACHE[model_id]
