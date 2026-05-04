# Understanding Gemma 4 E2B

This is me taking Gemma 4 E2B apart — building it from scratch in PyTorch, verifying every tensor against HuggingFace, and then running experiments on the things that feel surprising or underexplained in the literature.

The two tracks run in parallel: **architecture/** is the build, **experiments/** is the exploration. Both are meant to be readable.

---

## The model in one paragraph

Gemma 4 E2B is a 2-billion-effective-parameter decoder-only transformer from Google. The "E" stands for Effective: the model stores extra per-layer embeddings in flash memory and streams them in layer by layer, so the on-device VRAM footprint stays small while the model still gets a richer signal at every layer. Beyond that it's a fairly standard modern transformer — 35 layers, 1536-dim residual stream, SentencePiece tokenizer with 262k vocabulary — but with a few specific design choices that are worth understanding deeply: a 4:1 sliding-to-full attention interleave, 8:1 grouped-query attention throughout, KV reuse across the last 20 layers, and partial RoPE on full-attention layers. Each of these has a real reason behind it.

---

## Architecture: how it actually works

> Full docs in [architecture/docs/](architecture/docs/). What follows is the intuition.

### The residual stream is the whole game

Every token enters the model as a 1536-dimensional vector. That vector is the **residual stream** — the model's working memory for that token. For 35 layers, nothing creates new memory. Every sublayer (attention, FFN, PLE) reads from this stream and **adds a small update back to it**. At the end, the stream for the last token is projected to a vocabulary distribution and you get the next word.

This matters because it means attention and FFN aren't "doing computation" in isolation — they're writing notes into a shared notebook that every layer can read. When we say "the residual stream at layer 17," we mean the sum of everything that has been written into that token's vector up to that point.

→ [transformer_block.md](architecture/docs/transformer_block.md)

---

### Sliding vs full attention: why 4:1

At every layer, each token asks: *which other tokens should I look at?* The answer determines attention. Full attention says look at everything. Sliding-window attention says only look at the last 512 tokens.

Gemma 4 E2B does 4 sliding layers for every 1 full layer. The pattern repeats: `S S S S F S S S S F ...` all the way to layer 34. The full-attention layers are at positions 4, 9, 14, 19, 24, 29, 34.

**Why?** Most of what a token needs to predict the next word is nearby — the last sentence or two. Sliding attention captures this cheaply. But occasionally a token needs to reach back further — across a document boundary, back to the original question in a long conversation. The full-attention layers do that expensive cross-range lookup every 5 layers. 80% cheap, 20% powerful.

The practical consequence: if your prompt is 600 tokens long, a query at position 600 *cannot see position 1* in a sliding layer. BOS is gone. We ran an experiment on exactly this — see [experiments/AttentionSinks/result.md](experiments/AttentionSinks/result.md).

→ [attention.md](architecture/docs/attention.md)

---

### Grouped-query attention: 8 brains, 1 memory

Standard attention gives every query head its own key and value heads — 8 Q heads means 8 K heads and 8 V heads. Grouped-query attention (GQA) collapses the K and V side: all 8 query heads share a single K head and a single V head.

Concretely: the key projection goes from `(1536 → 8 × 256)` in standard attention to `(1536 → 1 × 256)` in GQA. The single K is broadcast to all 8 Q heads before the dot product. Same for V.

**Why?** The KV cache grows with context length. At 8:1 GQA, the KV cache is 8× smaller than standard attention for the same number of query heads. At long context lengths this is the difference between fitting in memory and not.

**What it trades away**: each Q head gets less differentiated memory — they all read from the same K and V. In practice the quality difference is small because the Q projections are independent and can still specialize.

→ [attention.md](architecture/docs/attention.md)

---

### KV sharing across layers: same eyes, different brain

Starting at layer 15, Gemma stops computing new K and V projections. Instead:

- Every sliding layer from 15–34 reuses **layer 13's** K and V
- Every full layer from 15–34 reuses **layer 14's** K and V

The Q projections are still computed fresh at each layer, so the *attention pattern* is different — each layer is asking a different question. But the "memory" being read (K = what's at each position, V = what to retrieve) is frozen to whatever layer 13/14 computed.

This saves half the attention compute for 20 out of 35 layers. The budget freed up is spent on a **2× wider FFN** in those layers (12,288 units vs 6,144 in the first 15). Same total FLOPs, different allocation: less on key/value projection, more on feedforward computation.

Think of it as: early layers figure out *what's at each position*, late layers reuse that answer while spending more energy on *what to do with it*.

→ [attention.md](architecture/docs/attention.md) · [text_model.md](architecture/docs/text_model.md)

---

### Per-layer embeddings: a USB drive at every layer

Every classical transformer does one embedding lookup at the start: token id → 1536-dim vector. Done. The rest of the network never sees the token id again — only the vector.

E2B adds a second embedding table that is indexed at **every single layer**. For each token, at each of 35 layers, the model looks up a 256-dim vector from this table, gates it, and adds it into the residual stream. These per-layer embeddings (PLE) live in flash memory and are streamed in rather than sitting in VRAM.

**Why?** The residual stream carries information across 35 layers, and a lot of what needs to flow through is tied to the token identity itself — its part of speech, its semantic class, its typical contexts. With standard embeddings that information is injected once and then has to survive 35 layers of attention and FFN updates without getting washed out. PLE lets the model reinforce identity-specific signals at every layer without paying for them in VRAM.

→ [embedding.md](architecture/docs/embedding.md)

---

### RoPE and p-RoPE: position as rotation

Position information in Gemma is injected not into the residual stream, but directly into the Q and K vectors inside attention. The mechanism: take pairs of dimensions in Q (and K), treat each pair as a 2D vector, and rotate it by an angle proportional to the token's position. The rotation angle differs per pair — some pairs spin fast (sensitive to nearby positions), some spin slow (sensitive to long-range separation). The dot product `Q · Kᵀ` then naturally encodes relative distance.

Sliding layers use standard RoPE with θ = 10,000 — all 256 dimensions of each head rotate.

Full-attention layers use **p-RoPE with p = 0.25 and θ = 1,000,000**: only the first 25% of the 512 head dimensions rotate. The other 75% are position-invariant — they carry pure content signal, no position. This is intentional: full-attention layers need to reason across very long distances, and rotating fewer dimensions with a much larger base prevents aliasing at long context lengths while keeping content-rich signals stable.

→ [rope.md](architecture/docs/rope.md)

---

### One token's full journey

1. Text → SentencePiece tokenizer → integer ids (vocab size 262,144)
2. Id → **1536-dim vector** via embedding table (scaled by √1536)
3. Vector enters the residual stream
4. For each of 35 layers:
   - **RMSNorm** → **attention** (sliding or full, 8:1 GQA) → **RMSNorm** → add back
   - **RMSNorm** → **GeGLU FFN** → **RMSNorm** → add back
   - Gate a **256-dim PLE** slice → project back to 1536 → **RMSNorm** → add back
   - Multiply by a learned `layer_scalar` (grows from ~0.018 at L0 to ~0.167 at L34)
   - If layer ≥ 15: K and V came from layer 13 or 14, not from this layer's projections
5. Final **RMSNorm**
6. Project 1536 → 262,144 logits via the embedding table (weight-tied with step 2)
7. **Softcap**: `30 · tanh(logits / 30)` — squashes extreme logits toward ±30
8. Softmax → next token

→ [overview.md](architecture/docs/overview.md) · [text_model.md](architecture/docs/text_model.md)

---

## Experiments

### Attention Sinks in sliding layers

**The question**: When BOS is evicted from the sliding window (which happens for all queries at position ≥ 512), where does the attention mass go — and does it matter for the residual stream?

**Why it's interesting**: The standard assumption (Cancedda 2024) is that BOS is a "low-V no-op" — a trash bin that absorbs excess attention without affecting the output because its value vector has near-zero norm. If true, losing BOS in sliding layers would be essentially free. We checked.

**What we found** (5 prompts × 35 layers, all strata):

- BOS produces **16.76× its attention mass** in exact residual-stream shift in full-attention layers. Not a no-op.
- When evicted, **no edge sink forms**. The window-boundary token is 134× weaker than the full-layer edge on exact residual.
- The mass lands in the **middle group** (58.5%) and **recent group** (36.4%). But these groups aren't doing the same thing.
- **Middle group completely inverts**: in full-attention layers it's 68% content words; in sliding layers it flips to 64% structural tokens (newlines, punctuation, format markers).
- **Deep layers (L25–34) are 68.6% structural** in their residual routing, vs 38% at L0–4. The model progressively routes to document scaffolding rather than semantic content as it goes deeper.
- **`\n` appears in all 5 prompts** as a top middle-group contributor — the most cross-prompt-consistent structural anchor.

The core finding: losing BOS doesn't just remove BOS's direct contribution. It triggers a role reassignment in the middle group — from content retrieval to structural grounding — with a **4.6× penalty** in exact residual impact.

→ Full results: [experiments/AttentionSinks/result.md](experiments/AttentionSinks/result.md)  
→ Runbook (how to reproduce): [experiments/AttentionSinks/RUNBOOK.md](experiments/AttentionSinks/RUNBOOK.md)

---

## Repo layout

```
architecture/
  docs/
    overview.md          ← start here if you're new
    attention.md         ← sliding vs full, GQA, KV-sharing, RoPE
    transformer_block.md ← sandwich norm, PLE gate, layer_scalar
    embedding.md         ← main embedding + PLE
    rope.md              ← rotary positions, p-RoPE
    ffn.md               ← GeGLU feedforward
    norm.md              ← RMSNorm details
    text_model.md        ← full stack, KV-share routing, generation
    tokenization.md      ← SentencePiece, 262k vocab

experiments/
  AttentionSinks/
    result.md            ← findings
    RUNBOOK.md           ← how to run
    experiment.ipynb     ← data collection
    analyze.ipynb        ← plots and tables
    token_residuals.ipynb← which tokens carry the residuals

load_hftf_model.ipynb    ← load HuggingFace weights for comparison
load_pytorch_model.ipynb ← load our implementation
```
