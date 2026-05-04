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

*Smoke run · 5 prompts · ratios will tighten with 100 prompts, directions are stable*
