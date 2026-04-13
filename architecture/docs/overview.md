# Gemma 4 E2B — Architecture Deep Dive

> A first-principles walkthrough. If you've never opened a transformer before, start here.
> We'll rebuild this model from scratch in PyTorch. This doc is the map.
>
> _Numbers below are taken from Maarten Grootendorst's visual guide to Gemma 4
> (newsletter.maartengrootendorst.com). We'll double-check each one against
> `config.json` as we implement._

---

## 1. What is Gemma 4 E2B?

**Gemma 4** is Google's open-weights LLM family. It ships in several sizes:

| variant       | type             | total params | active / token |
|---------------|------------------|--------------|----------------|
| **E2B**       | dense + PLE      | ~5B on disk  | **2B** effective |
| E4B           | dense + PLE      | larger       | 4B effective   |
| 31B           | dense            | 31B          | 31B            |
| 26B-A4B       | **MoE**          | 26B          | 4B             |

**"E" = Effective.** E2B stores a chunk of its weights (per-layer embeddings, PLE)
in flash memory — they get *streamed in* per layer instead of being counted as
"active" compute. So the model on disk is bigger than 2B, but the math per token
feels like a 2B dense model. That's the trick.

> **Heads up**: the file [gemma4_moe.py](../../gemma4_moe.py) in this repo describes
> the **26B-A4B MoE** variant, *not* E2B. E2B itself is **not MoE**. We'll come back
> to MoE later when we want it; the main build target is dense E2B.

### The headline numbers (E2B)

| property                    | value          |
|-----------------------------|----------------|
| effective params / token    | ~2 B           |
| vocabulary                  | **262,144**    |
| main embedding dim          | **1,536**      |
| per-layer embedding (PLE)   | **256**        |
| sliding window              | **512** tokens |
| attention interleave        | **4 local : 1 global** |
| global attention GQA        | 8 query heads share 1 KV head |
| local attention GQA         | 2 query heads share 1 KV head |
| p-RoPE (global only)        | p = 0.25       |
| K = V optimization          | global layers only |
| vision encoder              | 150 M params, SigLIP-style, 2D RoPE |
| vision resolution budget    | 70 / 140 / 280 / 560 / 1120 soft tokens |
| audio encoder               | Conformer-based |

---

## 2. The 10,000-foot view

A decoder-only transformer. Tokens go in, next-token distribution comes out.
Everything in between is a stack of **identical decoder layers**.

```
text  ──►  Tokenizer (text → ids)
image ──►  Vision Tower (pixels → image tokens)
audio ──►  Audio Encoder (waveform → audio tokens)
               │
               ▼
       Embedding table (ids → 1536-dim vectors)
               │
      ┌────────┴─────────┐
      │ + PLE (256-dim)  │  ← streamed from flash per layer,
      │   via gating     │    gated into the residual stream
      └────────┬─────────┘
               ▼
       Decoder Layer × N
         ├── RMSNorm
         ├── Self-Attention   (RoPE, KV cache, GQA)
         │     · 4 layers local (window=512)
         │     · 1 layer global (p-RoPE 0.25, K=V)
         ├── RMSNorm
         └── FFN (GeGLU)
               │
               ▼
       Final RMSNorm
       LM Head (tied to embedding) → logits over 262,144 vocab
```

Two things to internalize:

1. **The layer is repeated.** Understand one layer = understand the whole model.
2. **Residual stream.** Every sub-block (attn, FFN) reads from and *adds back to*
   a single vector per token. That vector is the model's working memory.

---

## 3. What's unusual about Gemma 4 E2B

### 3a. Per-Layer Embeddings (PLE)

Classical transformer: one embedding lookup at the input, done.
Gemma 4: *every layer also gets its own extra embedding* (256-dim) for each token,
**stored in flash** (not VRAM), **streamed in layer by layer**, **gated** into the
residual stream.

```
token id ──► main embedding (1536)
          │
          ▼
       layer 0 ─── + PLE_0(id)  via gate
          │
          ▼
       layer 1 ─── + PLE_1(id)  via gate
          │
          ▼
        ...
```

Why? It lets the model carry ID-specific information *all the way through* without
paying for it in active compute. It trades VRAM for flash bandwidth.

### 3b. 4 : 1 local-to-global attention

Out of every 5 layers, 4 use a cheap **sliding-window** attention (window = 512),
and 1 uses full **global** attention.

```
layer:  L  L  L  L  G   L  L  L  L  G   ...
        └── local ──┘   └── local ──┘
        window=512      window=512
```

Most information is local (nearby tokens). Every 5th layer does the heavy lifting
of cross-document reasoning. Huge speedup, small quality cost.

### 3c. Group-Query Attention (GQA), asymmetric

- **Global** layers: **8 Q heads share 1 KV head** (aggressive sharing → small KV cache)
- **Local** layers:  **2 Q heads share 1 KV head** (less sharing; local KV is already cheap)

### 3d. K = V on global layers

On the global-attention layers, K and V *are the same tensor*. You only project once.
Cuts KV cache in half and roughly halves the projection compute on those layers.

### 3e. p-RoPE on global layers

Normal RoPE rotates every pair of Q/K dimensions by position.
**p-RoPE with p=0.25** only rotates the first 25% of dimensions — the rest carry
position-invariant content. Gives long-context stability.

### 3f. Multimodal: vision + audio

- **Vision**: 150M SigLIP-style encoder, 2D RoPE, variable aspect ratios.
  Images become a *variable number* of soft tokens (budgets: 70, 140, 280, 560, 1120).
  After a projector, image tokens live in the same 1536-dim residual stream.
- **Audio**: Conformer-based encoder. Same story — audio tokens are injected into
  the stream alongside text.

The decoder can't tell text / image / audio tokens apart once they're in the
residual stream. Unified sequence.

---

## 4. The pieces you should know by name

| piece              | what it does                                             |
|--------------------|----------------------------------------------------------|
| **Tokenizer**      | text ↔ ids, SentencePiece, 262k vocab                   |
| **Embedding**      | id → 1536-dim vector                                     |
| **PLE**            | per-layer extra embedding (256-dim), gated, from flash   |
| **RMSNorm**        | normalize by root-mean-square                            |
| **RoPE / p-RoPE**  | rotary positions inside Q/K                              |
| **Sliding Attn**   | local window=512 attention (4 of every 5 layers)         |
| **Global Attn**    | full attention, GQA 8:1, K=V, p-RoPE (1 of every 5)      |
| **GeGLU FFN**      | `down(gelu(gate(x)) * up(x))`                            |
| **Vision Tower**   | SigLIP-style ViT, 150M, 2D RoPE, variable resolution     |
| **Audio Tower**    | Conformer encoder                                        |
| **Projector**      | maps vision/audio dim → 1536                             |
| **LM Head**        | hidden → 262,144 logits (tied to embedding)              |

Each gets its own doc as we build it.

---

## 5. How one token travels through the model

1. Text is chopped into subword ids by the SentencePiece tokenizer (262k vocab).
2. Id → **1536-dim vector** via the embedding table.
3. Vector enters the residual stream.
4. For each decoder layer:
   - Add the layer's **PLE** (streamed from flash, gated).
   - Copy → RMSNorm → **attention** (local or global) → add back.
   - Copy → RMSNorm → **GeGLU FFN** → add back.
5. Final RMSNorm.
6. **LM head** projects 1536 → 262,144 logits.
7. Softmax → sample → next token.

Every fancy thing (PLE, sliding window, p-RoPE, vision, audio, MoE in the bigger
variant) is a variation on one of these steps.

---

## 6. Our build plan

Order matches the information flow:

1. **Tokenization** ← next
2. Embedding + residual stream
3. Per-Layer Embeddings (PLE) + gating
4. RMSNorm
5. RoPE (and p-RoPE for global layers)
6. Attention: GQA, KV cache, sliding window, K=V optimization
7. GeGLU FFN
8. Full decoder layer (local variant, global variant)
9. Stacking layers + final norm + LM head
10. Vision tower + projector
11. Audio tower + projector
12. Weight loading from HuggingFace
13. Sampling / generation loop

For each step:

- Implement the module in `architecture/`.
- Write the intuition doc in `architecture/docs/`.
- Compare outputs tensor-by-tensor against the HuggingFace model in
  [load_model.ipynb](../../load_model.ipynb) to catch drift early.

Next up: **tokenization** — how raw strings (and images, and audio) become the
integer ids / soft tokens the rest of the model operates on.
