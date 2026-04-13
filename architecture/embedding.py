"""
Gemma 4 E2B · Main Embedding
────────────────────────────
id → 1536-dim vector. Thin wrapper around nn.Embedding that loads the
`model.embed_tokens.weight` tensor directly from a safetensors shard.

(PLE — the per-layer 256-dim embeddings — lives in its own module.)
"""

import torch.nn as nn
from safetensors.torch import load_file


class GemmaEmbedding(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    @classmethod
    def from_safetensors(cls, shard_path, key="model.embed_tokens.weight"):
        w = load_file(str(shard_path))[key]
        vocab, hidden = w.shape
        m = cls(vocab, hidden)
        m.embed.weight.data.copy_(w)
        return m

    def forward(self, ids):
        return self.embed(ids)
