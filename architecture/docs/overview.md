# Gemma 4 E2B - Architecture Deep Dive

> A first-principles walkthrough. If you've never opened a transformer
> before, start here. We'll rebuild this model from scratch in PyTorch.
> This doc is the map.
>
> _Numbers below come from Maarten Grootendorst's visual guide to
> Gemma 4 (newsletter.maartengrootendorst.com), but every one is
> double-checked against the actual `config.json` and the safetensors
> shapes as we implement. Where the visual guide and the real config
> disagree, we go with the config - and call it out._

---

## 1. What is Gemma 4 E2B?

**Gemma 4 E2B** is an open-weights, multimodal, decoder-only language
model from Google. The **"E" stands for Effective**: the model stores
a chunk of its weights (the per-layer embeddings, "PLE") in **flash
memory** and streams them in layer by layer instead of keeping them in
VRAM. So the math per token feels like a **2 B dense model**, even
though the file on disk is bigger than that.

That flash-backed trick is the single most important thing to remember
about E2B. Everything else is standard-ish transformer machinery,
tuned for speed.

### The headline numbers

| property                      | value                  |
|-------------------------------|------------------------|
| effective params / token      | ~2 B                   |
| vocabulary                    | **262,144**            |
| main embedding dim            | **1,536**              |
| per-layer embedding (PLE)     | **256**                |
| number of decoder layers      | **35**                 |
| sliding window                | **512** tokens         |
| attention interleave          | **4 sliding : 1 full** |
| GQA ratio (both layer types)  | 8 query heads : 1 KV head |
| KV sharing                    | last 20 layers reuse earlier same-type K/V |
| RoPE θ (local / global)       | 10 000 / 1 000 000     |
| p-RoPE (global only)          | p = 0.25 (only first 25 % of dims rotate) |
| FFN width (L0-14 / L15-34)    | 6 144 / 12 288         |
| final logit softcap           | `30 · tanh(logits / 30)` |
| vision encoder                | 150 M, SigLIP-style ViT, 16×16 patches, 2D positions |
| vision token budgets          | 70 / 140 / 280 / 560 / 1120 |
| audio encoder                 | Conformer-based        |

A note on a number you'll see in older Gemma write-ups: "**2 : 1 GQA
on local layers**". That's a Gemma 3 fact. In E2B, both layer types
are **8 : 1** - the safetensors `q_proj`/`k_proj` shapes are the
ground truth and they confirm 8 : 1 across the board.

---

## 2. The 10,000-foot view

Decoder-only transformer. Tokens go in, next-token distribution comes
out. Everything in between is a stack of **identical decoder layers**.

```
   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │      TEXT       │     │     IMAGE       │     │     AUDIO       │
   │  "hello world"  │     │   pixels (HWC)  │     │   waveform      │
   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
            │                       │                       │
            ▼                       ▼                       ▼
     ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
     │  Tokenizer  │         │ Vision Tower│         │Audio Encoder│
     │ SP · 262 k  │         │ SigLIP 150M │         │  Conformer  │
     └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
            │ ids                   │ soft tokens           │ soft tokens
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                        ╔═══════════════════════╗
                        ║   Embedding Table     ║
                        ║   id → vec[1536]      ║
                        ╚═══════════╤═══════════╝
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   RESIDUAL STREAM     │  ← working memory, per token
                        └───────────┬───────────┘
                                    │
   ┌────────────────────────────────┼────────────────────────────────┐
   │ DECODER LAYER × 35                                              │
   │                                 │                               │
   │    PLE[ℓ](id) ──(gate)─► (+) ◄──┤      ← streamed from flash    │
   │                                 │                               │
   │                          RMSNorm│                               │
   │                                 ▼                               │
   │                   ┌──────────────────────────┐                  │
   │                   │   Self-Attention (GQA)   │                  │
   │                   │   · sliding (w=512) × 4  │                  │
   │                   │   · full (p-RoPE) × 1    │                  │
   │                   └─────────────┬────────────┘                  │
   │                                 │ (+) residual                  │
   │                          RMSNorm│                               │
   │                                 ▼                               │
   │                   ┌──────────────────────────┐                  │
   │                   │      FFN  (GeGLU)        │                  │
   │                   └─────────────┬────────────┘                  │
   │                                 │ (+) residual                  │
   └─────────────────────────────────┼───────────────────────────────┘
                                     ▼
                             ┌───────────────┐
                             │ Final RMSNorm │
                             └───────┬───────┘
                                     ▼
                        ╔═══════════════════════╗
                        ║   LM Head  (tied)     ║
                        ║   1536 → 262,144      ║
                        ╚═══════════╤═══════════╝
                                    ▼
                          softcap → next-token logits
```

Two things to internalize:

1. **The layer is repeated.** Understand one layer = understand the
   whole model.
2. **Residual stream.** Every sub-block (attention, FFN, PLE) reads
   from and *adds back to* a single vector per token. That vector is
   the model's working memory.

---

## 3. What's unusual about E2B

### 3a. Per-Layer Embeddings (PLE) - the "E" in E2B

Classical transformer: one embedding lookup at the input, done.
E2B: *every layer also gets its own extra embedding* (256-dim) for
each token, **stored in flash** (not VRAM), **streamed in layer by
layer**, and **gated** into the residual stream:

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

This lets the model carry id-specific information *all the way through*
the stack without paying for it in active compute. Trades VRAM for
flash bandwidth.

### 3b. 4 : 1 sliding-to-full attention

Out of every 5 layers, 4 use a cheap **sliding-window** attention
(window = 512), and 1 uses **full** attention.

```
layer:  S  S  S  S  F   S  S  S  S  F   ...
        └─ sliding ─┘   └─ sliding ─┘
        window=512      window=512
```

Most information is local (nearby tokens). Every 5th layer does the
heavy lifting of cross-document reasoning. Big speedup, small quality
cost.

### 3c. Group-Query Attention (GQA), uniform 8 : 1

Both layer types use 8 query heads sharing 1 KV head. That makes the
KV cache per layer small in both flavours, which keeps long-context
memory low.

### 3d. KV sharing across layers (the actual K/V optimization)

For the **last 20 layers** (indices 15-34 out of 35), K and V are
**not recomputed** - they are reused from earlier same-type layers:

- every shared `sliding` layer reuses **layer 13's** K/V,
- every shared `full` layer reuses **layer 14's** K/V.

(13 and 14 are the *last* sliding and full layers respectively before
the sharing region begins.) The shared layers still compute their own
Q, so attention output still differs - but they save half the
projection compute and skip the K/V cache write.

The saved budget gets spent on a **2× wider FFN** in those same layers
(12 288 vs 6 144). Same total FLOPs, different allocation.

> Older Gemma 3 write-ups describe a "K = V on global layers"
> optimization (use the same tensor for K and V on full-attention
> layers). E2B does *not* do that. The `k_proj` and `v_proj` weights
> are independent. The K/V optimization in E2B is the layer-15+
> sharing rule above.

### 3e. p-RoPE on full-attention layers

Normal RoPE rotates every pair of Q/K dimensions by position.
**p-RoPE with p = 0.25** only rotates the first 25 % of dimensions -
the rest carry position-invariant content. Combined with a much larger
RoPE base (θ = 1 000 000 vs 10 000 on local), this gives the full-
attention layer stable, content-dominated signals at long contexts
without aliasing.

### 3f. Sandwich norm and per-layer scalar

Every sublayer (attention, FFN, PLE) sits between **two** RMSNorms -
one before, one after - inside its residual branch. And the very last
line of each layer multiplies `hidden` by a learned `layer_scalar`
(varies from ~0.018 at layer 0 to ~0.167 at layer 34). Both are
silent-failure traps if you skip them.

### 3g. Final logit softcap

Right before the next-token softmax, the logits go through
`30 · tanh(logits / 30)` - squashing extreme values toward ±30.
Skipping this makes every probability slightly wrong.

### 3h. Multimodal: vision + audio

- **Vision**: 150 M SigLIP-style encoder, 16×16 patches, separable 2D
  positional embeddings, variable aspect ratios. Images become a
  *variable number* of soft tokens (budgets: 70, 140, 280, 560, 1120).
  After a projector, image tokens live in the same 1536-dim residual
  stream as text.
- **Audio**: Conformer-based encoder. Same story - audio tokens are
  injected into the stream alongside text.

The decoder can't tell text/image/audio tokens apart once they're in
the residual stream. One unified sequence.

---

## 4. The pieces you should know by name

| piece              | what it does                                             |
|--------------------|----------------------------------------------------------|
| **Tokenizer**      | text ↔ ids, SentencePiece, 262 k vocab                   |
| **Embedding**      | id → 1536-dim vector (× √hidden)                         |
| **PLE**            | per-layer extra embedding (256-dim), gated, from flash   |
| **RMSNorm**        | normalize by root-mean-square (fp32 math, optional gain) |
| **RoPE / p-RoPE**  | rotary positions inside Q/K                              |
| **Sliding Attn**   | local window = 512 attention (4 of every 5 layers)       |
| **Full Attn**      | full attention with p-RoPE (1 of every 5)                |
| **GeGLU FFN**      | `down(gelu(gate(x)) * up(x))`                            |
| **KV-share**       | layers 15-34 reuse K/V from layer 13/14                  |
| **Vision Tower**   | SigLIP-style ViT, 150 M, 2D positions, variable resolution |
| **Audio Tower**    | Conformer encoder                                        |
| **Projector**      | maps vision/audio dim → 1536, with weightless RMSNorm    |
| **LM Head**        | hidden → 262 144 logits (tied to embedding)              |
| **Softcap**        | `30 · tanh(logits / 30)` before softmax                  |

Each gets its own doc as we build it.

---

## 5. How one token travels through the model

1. Text is chopped into subword ids by the SentencePiece tokenizer
   (262 k vocab).
2. Id → **1536-dim vector** via the embedding table (× √1536, rounded
   to bf16).
3. Vector enters the residual stream.
4. PLE precompute runs once at the model level, producing a
   `(B, S, 35, 256)` tensor.
5. For each of 35 decoder layers:
   - Sandwich-norm + **attention** (sliding or full) → residual add.
   - Sandwich-norm + **GeGLU FFN** → residual add.
   - **PLE block**: gate the layer's 256-dim slice, multiply, project
     back, norm → residual add.
   - Multiply by `layer_scalar`.
   - For layers 15-34, K/V are read from the donor's cache rather
     than recomputed.
6. Final RMSNorm.
7. **Tied LM head**: `hidden @ embed_tokens.weight.T` projects
   1536 → 262 144 logits.
8. Softcap.
9. Softmax → sample → next token.

Every fancy thing (PLE, sliding window, p-RoPE, KV-share, vision,
audio) is a variation on one of these steps.

---

## 6. Our build plan

The order matches the information flow. For each step we:

- implement the module in `architecture/`,
- write the intuition doc in `architecture/docs/`,
- compare outputs tensor-by-tensor against the HuggingFace model in
  the parity notebooks ([load_hftf_model.ipynb](../../load_hftf_model.ipynb)
  / [load_pytorch_model.ipynb](../../load_pytorch_model.ipynb)) to
  catch drift early.

1. **Tokenization** ✓
2. Embedding + residual stream ✓
3. RMSNorm ✓
4. Per-Layer Embeddings (PLE) precompute ✓
5. RoPE (and p-RoPE for full layers) ✓
6. Attention: GQA, sliding window, KV-sharing API ✓
7. GeGLU FFN ✓
8. Full decoder layer (sandwich norms + PLE block + layer_scalar) ✓
9. Stack 35 layers + KV-share routing + final norm + tied LM head + softcap ✓
10. Vision tower + projector (projector ✓; tower borrowed)
11. Audio tower + projector  (projector ✓; tower borrowed)
12. KV cache for incremental generation
13. Sampling / generation loop ✓ (greedy)

The full text-only model is bit-equal against HF eager. See
[`text_model.md`](text_model.md) for the end-to-end story.
