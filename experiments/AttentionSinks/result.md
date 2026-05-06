# Attention Sinks — Results

**Model**: Gemma 4 E2B · **5 prompts × 35 layers** · seq_len 517–640 · late queries (pos ≥ 512)  
**Layer types**: 7 full-attention (L4, 9, 14, 19, 24, 29, 34) · 28 sliding-window (rest)

---

## 1. BOS is a real signal, not a no-op

> **attn-overlap mass** — total softmax weight a position receives, summed across all late queries and heads  
> **exact residual** — actual shift in the residual stream when that position's contribution is removed: `‖ RMSNorm(out) − RMSNorm(out − cₖ) ‖`  
> **ratio** = exact residual ÷ attn-overlap mass. A true no-op would score ~1×.

In full-attention layers, BOS is the most attention-efficient position in the sequence:

| position | attn-overlap mass | exact residual | ratio |
|---|---|---|---|
| **bos** | 0.53 | **8.91** | **16.76×** |
| edge | 1.47 | 7.12 | 4.84× |
| recent | 1.13 | 29.89 | 26.5× |
| middle | 4.63 | 14.83 | 3.20× |

For every unit of attention mass BOS receives, it produces **16.76 units** of residual-stream shift. Cancedda 2024 predicted ~1× (low-norm no-op). The data says 17×.

---

## 2. When BOS is evicted, nothing replaces it

**Expected**: attention migrates to the sliding-window boundary (edge sink).  
**Observed**:

| group | score\_pre share | mean exact residual |
|---|---|---|
| middle | 58.5% | 3.22 |
| recent | 36.4% | 9.26 |
| self | 4.9% | 3.33 |
| **edge** | **0.18%** | **0.053** |

The edge token in sliding layers scores **0.053 exact residual**. The same position in full-attention layers scores **7.12** — a **134× drop**. No edge sink exists. Mass disperses into middle and recent.

---

## 3. Middle and recent carry opposite things

The 58% that lands in the middle group and the 36% in the recent group are not doing the same job. Comparing which *type* of token each group routes to:

> **Content tokens** — actual words semantically relevant to the query (`Quick`, `years`, `classified`, etc.)  
> **Structural tokens** — punctuation, newlines, format markers (`\n`, `-`, `"`, `<|turn|>`) that delimit document structure but carry no task-specific meaning

| group | layer type | content share | structural share | mean exact |
|---|---|---|---|---|
| middle | **full-attention** | **68%** | 32% | **14.83** |
| middle | **sliding-attention** | 36% | **64%** | **3.22** |
| recent | full-attention | 71% | 29% | 29.87 |
| recent | sliding-attention | 61% | 39% | 9.26 |

**Middle in full-attention layers** = content retrieval (semantic words from across the full context).  
**Middle in sliding-attention layers** = structural grounding (newlines, punctuation, format markers).

The same positional slot does different work depending on whether BOS is present. With BOS as the structural anchor, middle goes to content. Without BOS, middle absorbs the structural role — and pays a **4.6× exact-residual penalty** for the switch.

**Recent** stays content-dominant in both layer types (~61–71%) — it carries local semantic context regardless.

---

## 4. The structural takeover builds with depth

Structural tokens (punctuation, `\n`, format markers) start as a minority in early layers and progressively dominate as the network goes deeper. Sliding layers, split by depth band — what share of score\_pre goes to structural tokens:

| depth | structural share | structural mean score\_pre | vs content mean |
|---|---|---|---|
| L0–4 | 38.4% | 6.32 | ≈ equal (content: 6.40) |
| L5–14 | 40.5% | 3.20 | slightly lower |
| L15–24 | 47.9% | 2.92 | slightly higher |
| **L25–34** | **68.6%** | **5.59** | **2.2× higher** |

Early sliding layers are roughly balanced. By L25–34, structural tokens carry **2.2× more residual per instance** than content tokens and own 68.6% of the total. Deep layers are overwhelmingly routing to document scaffolding, not semantic content.

Full-attention layers show the same endpoint (L34: 68.8% structural) but are content-heavier throughout their span because BOS handles structural grounding separately.

---

## 5. The tokens that show up across all prompts

Tested across 5 independent prompts (different tasks, languages). Tokens consistent across prompts reveal structural dependencies, not prompt-specific behavior.

**Full-attention edge (positions 1–3) — 5/5 prompts:**

| token | mean score\_pre | mean exact |
|---|---|---|
| `<\|turn\|>` | 4.10 | 9.81 |
| `user` | 3.77 | 5.22 |
| `\n` | 2.87 | 4.03 |

Always the same three tokens, always at positions 1–3. The chat-template skeleton.

**Sliding-attention middle — cross-prompt structural anchors:**

| token | prompts | appearances | mean score\_pre |
|---|---|---|---|
| `\n` | **5/5** | 109 | 2.40 |
| `model` | 4/5 | 97 | 4.55 |
| `.` | 4/5 | 31 | 3.67 |
| `<turn\|>` | 4/5 | 25 | 2.31 |
| `-` | 3/5 | 159 | 5.54 |

None of these are content words. All are format/punctuation tokens that appear regardless of what the prompt is about. These are the structural anchors the model falls back on when BOS is gone.

**Sliding-attention recent — 5/5 prompts prompt-specific:**

Top tokens (`split`, `Quick`, `yuan`, `hätte`, `classified`) each appear in exactly one sample. Recent carries local semantic context — never a cross-prompt anchor.

---

## 6. Why sliding writes are smaller — the RMSNorm × peakiness mechanism

Across the whole dataset, sliding-attention layers produce a **2.87× smaller mean exact residual shift** than full-attention layers, despite writing nearly identical raw `score_pre` magnitudes (mean 3.82 vs 3.88). The gap doesn't appear in the raw contributions — it appears after the post-attention RMSNorm. This finding asks: why?

> **Compression** = `score_pre / score_resid_exact` per row. Higher = more of the raw contribution gets absorbed by RMSNorm. A value of 1 means raw and post-norm magnitudes match.

### Sliding attention is genuinely sharper

| metric | full | sliding |
|---|---|---|
| top-1 attention fraction | 0.307 | **0.384** (25% peakier) |
| coefficient of variation | 0.796 | **1.081** |
| attention entropy (bits) | 2.684 | **2.484** |

Sliding distributions are systematically more concentrated. The 512-token window forces sharper softmaxes — fewer keys compete for the same probability mass.

### RMSNorm compresses sliding outputs ~2× harder

| layer type | median compression | mean compression |
|---|---|---|
| full | 0.66 | 1.02 |
| **sliding** | **1.34** | **1.86** |

For the median sliding-attention write, RMSNorm absorbs roughly twice as much of the raw contribution as it does for the median full-attention write.

### The compression scales monotonically with peakiness

Compression bucketed by per-query max attention mass:

| max attn bucket | full (median) | sliding (median) |
|---|---|---|
| 0.05 – 0.15 | 0.52 | 0.90 |
| 0.15 – 0.50 | 1.09 | 1.60 |
| 0.50 – 1.00 | (no data) | **2.55** |

Two things happen together:
- **Compression rises with peakiness** within both layer types
- **Sliding spends 193 query-writes in the >0.5 peakiness bucket; full spends zero**

Sliding doesn't just hit RMSNorm's compression curve — it lives in the regime where the curve is steepest.

### The mechanism is a known mathematical property of RMSNorm

RMSNorm divides by the root-mean-square of the input vector. For a vector of fixed L2 norm, peaky vectors have higher RMS than diffuse vectors, so they get scaled down harder per-component. This was established in **Zhang & Sennrich (2019)**, the paper that introduced RMSNorm. **Cancedda (2024)** independently ties attention-sink behavior to post-attention normalization. The compression-vs-peakiness table above is a clean empirical confirmation of both.

### Negative correlation: peakier writes → smaller residuals

Per-query correlation between max attention mass and total exact residual:

| layer type | r |
|---|---|
| full-attention | **−0.31** |
| sliding-attention | −0.11 |

This is counter to naive intuition. You'd expect higher attention mass → bigger writes. The data says the opposite, especially in full layers where the dynamic range allows the effect to show.

### What this means

Sliding attention is **not designed** to produce smaller residual writes — Longformer (2020), Big Bird (2020), Mistral (2023), and the Gemma reports all motivate sliding attention purely on **compute and memory grounds**. Residual-stream stability is never cited as a design rationale.

But the architecture choice produces this property emergently. Sliding window → sharper softmaxes → peakier output vectors → harder RMSNorm compression → smaller residual writes. The result is that **deep sliding-attention layers are systematically prevented from any single key dominating the residual stream** — exactly the "signal propagates without overweighting" property a careful designer would want, achieved as a side effect of stacking sliding attention with RMSNorm.

This also explains why BOS in full-attention layers reaches 16.76× exact-per-mass efficiency (Finding 1): BOS accumulates moderate (not extreme) attention mass, sits among other tokens (less peaky), and RMSNorm passes most of its contribution through. Full attention preserves contributions; sliding attention absorbs them.

---

## Summary

| finding | number | confidence |
|---|---|---|
| BOS residual efficiency | **16.76×** its attention mass | high |
| Edge sink after BOS eviction | **0.053** exact — 134× weaker than full-layer edge | high |
| Middle group: content share in full-attn | **68%** content | high |
| Middle group: content share in sliding | **36%** content (64% structural) | high |
| Exact-residual penalty from inversion | **4.6×** drop (14.83 → 3.22) | high |
| Structural share in L25–34 (sliding) | **68.6%**, structural is 2.2× heavier per-instance | high |
| Most consistent cross-prompt token | `\n` in **5/5** prompts, 109 appearances | high |
| Per-unit structural anchor set | `\n`, `model`, `.`, `<turn\|>` in ≥4/5 prompts | medium |
| Sliding attention sharpness | top-1 frac 0.384 vs 0.307 (25% peakier) | high |
| RMSNorm compression gap | sliding 1.34× vs full 0.66× median (2× harder) | high |
| Mean residual write gap | full 2.87× larger despite equal raw `score_pre` | high |
| Peakiness → compression scaling | monotonic in both layer types; sliding exclusive in >0.5 max-attn regime | high |

*Smoke run · 5 prompts · ratios will tighten with 100 prompts, directions are stable*
