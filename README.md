# Understanding Gemma 4 E2B

Taking Gemma 4 E2B apart to understand it — not just use it. Two parallel tracks: rebuilding the architecture from scratch in PyTorch (verified tensor-by-tensor against HuggingFace), and running experiments on the parts that feel interesting or underexplained.

---

## What is Gemma 4 E2B?

A 2-billion-effective-parameter decoder-only transformer from Google. The **E** stands for *Effective*: part of the weights live in flash memory and stream in layer by layer, so the model has more capacity than its VRAM footprint suggests. Beyond that it's a modern transformer — 35 layers, 1536-dim residual stream, 262k vocabulary — with a few specific design choices worth understanding deeply.

---

## How it works

> Full docs in [architecture/docs/](architecture/docs/). This is the intuition.

**The residual stream** is the backbone. Each token enters as a 1536-dim vector. Every sublayer across all 35 layers just reads from it and writes a small update back. Nothing creates new memory, nothing discards anything. The final vector — the sum of 35 rounds of updates — gets projected to a next-token distribution. When we ask "where does information flow," we're asking what each layer writes, and where.
→ [transformer_block.md](architecture/docs/transformer_block.md)

**4:1 sliding-to-full attention.** 28 of 35 layers use sliding-window attention (window = 512 tokens). Every 5th layer (4, 9, 14, 19, 24, 29, 34) uses full attention. The bet: most of what you need to predict the next word is nearby; full attention every 5 layers is enough for long-range integration. This also means BOS gets evicted from the sliding window for any query past position 512 — which is exactly what the experiment below explores.
→ [attention.md](architecture/docs/attention.md)

**8:1 grouped-query attention (GQA).** All 8 query heads share a single key head and value head. The KV projection is `(1536 → 1 × 256)` instead of `(1536 → 8 × 256)`. This makes the KV cache 8× smaller — critical at long context — at a modest quality cost because query projections are still independent.
→ [attention.md](architecture/docs/attention.md)

**KV sharing across layers 15–34.** Starting at layer 15, K and V aren't recomputed. Every sliding layer from 15–34 reuses layer 13's K/V; every full layer reuses layer 14's. Each layer still computes its own Q, so attention patterns differ — but the "facts" being looked up are frozen. The freed compute goes into a 2× wider FFN in those layers (12,288 vs 6,144 units).
→ [attention.md](architecture/docs/attention.md) · [text_model.md](architecture/docs/text_model.md)

**Per-layer embeddings (PLE).** At every layer, the model looks up an additional 256-dim vector keyed to the token id, gates it, and adds it into the residual stream. These live in flash, not VRAM — that's the "E." The effect: token identity gets re-injected at every layer rather than having to survive 35 rounds of updates from a single upfront embedding.
→ [embedding.md](architecture/docs/embedding.md)

**RoPE and p-RoPE.** Position is encoded by rotating Q and K just before the dot product. Sliding layers use standard RoPE (all 256 head dims rotate, θ = 10k). Full layers use partial RoPE — only the first 25% of 512 head dims rotate, the rest carry position-free content. Combined with θ = 1M, this gives full-attention layers clean long-range signals without aliasing.
→ [rope.md](architecture/docs/rope.md)

---

## Experiment — attention sinks in sliding layers

**Question**: When BOS is evicted from the sliding window (any query at position ≥ 512), where does the attention go — and does losing it matter?

**The prior** (Cancedda 2024): BOS is a low-norm no-op. Models dump attention there when they have nothing better to attend to, and it barely affects the residual stream. Losing it should be free.

**What we found**:

BOS in full-attention layers produces **16.76 units of exact residual shift per unit of attention mass**. A true no-op would be ~1×. It's a real signal channel.

When evicted, nothing replaces it. The window-boundary token (edge) is **134× weaker** in exact residual impact in sliding layers vs full layers. No edge sink forms.

The mass disperses into middle tokens (58.5%) and recent tokens (36.4%) — but these aren't doing the same thing. The middle group **inverts its character**:

| | full-attention | sliding-attention |
|---|---|---|
| what middle tokens are | 68% content words | 64% structural tokens (`\n`, `.`, `-`, format markers) |
| mean exact residual | 14.83 | 3.22 |

In full-attention layers, BOS handles structural grounding, freeing middle to route to content. Without BOS, middle absorbs the structural role — and pays a **4.6× residual penalty**. The `\n` newline token appears as a top contributor in every one of the 5 test prompts across 5 task types and languages.

→ [experiments/AttentionSinks/result.md](experiments/AttentionSinks/result.md) — full results  
→ [experiments/AttentionSinks/RUNBOOK.md](experiments/AttentionSinks/RUNBOOK.md) — how to reproduce

---

## Where to start

| goal | link |
|---|---|
| Understand the full model | [architecture/docs/overview.md](architecture/docs/overview.md) |
| Understand attention in depth | [architecture/docs/attention.md](architecture/docs/attention.md) |
| Read the attention sink findings | [experiments/AttentionSinks/result.md](experiments/AttentionSinks/result.md) |
| Run the experiment | [experiments/AttentionSinks/RUNBOOK.md](experiments/AttentionSinks/RUNBOOK.md) |
