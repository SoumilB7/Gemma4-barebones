# Text Model — Gemma 4 E2B

> The end of the road. `GemmaTextModel` wires every piece we've built
> (embedding, PLE precompute, 35 decoder layers with KV-share routing,
> final RMSNorm, tied LM head, softcap) into one `nn.Module` whose
> `forward(input_ids)` returns logits ready to sample from.

---

## 1. The pipeline in one diagram

```
input_ids  (B, S)
   │
   ▼
embed_tokens                         GemmaEmbedding  (× √1536, rounded to bf16)
   │  inputs_embeds  (B, S, 1536)
   ▼
per_layer_inputs                     GemmaPerLayerInputs — runs ONCE
   │                                   → (B, S, 35, 256)
   ▼
┌────────── layer loop, i = 0..34 ──────────┐
│  mask  = causal_mask(S, window=None)      │   full attention layers
│  mask' = causal_mask(S, window=512)       │   sliding attention layers
│  cos/sin = rope_local or rope_global      │
│  pli_i  = per_layer_inputs[:, :, i, :]    │
│                                            │
│  if i ≤ 14 (non-shared):                   │
│      hidden, k, v = layer(hidden, pli_i,   │
│                            cos, sin, mask) │
│      if i is a donor:  stash (k, v)        │
│                                            │
│  if i ≥ 15 (shared):                       │
│      cached_kv = shared_kv[donor_of(i)]    │
│      hidden    = layer(hidden, pli_i,      │
│                        cos, sin, mask,     │
│                        cached_kv=cached_kv)│
└────────────────────────────────────────────┘
   │  hidden  (B, S, 1536)
   ▼
norm                                 GemmaRMSNorm
   │
   ▼
logits = hidden @ embed_tokens.weight.T       ← tied LM head, NO separate Linear
   │
   ▼
logits = 30 * tanh(logits / 30)               ← final_logit_softcapping
   │
   ▼
logits  (B, S, vocab=262_144)
```

Three facts to keep in mind:

1. **No separate LM head.** `tie_word_embeddings=True`, so the projection
   back to vocab is just a matmul against the embedding matrix. HF's
   `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` — same
   storage. We don't own a `Linear`; we transpose and multiply.
2. **Softcap is not optional.** Before the softmax, logits are squashed
   through `30 · tanh(x / 30)`. Skip it and every log-prob shifts.
3. **Masks differ per layer type.** Sliding layers use a windowed causal
   mask; full layers use plain causal. For `S ≤ 512` these coincide; past
   that, sliding layers start blocking old positions.

---

## 2. Why the shard loads exactly once

Each component (embedding, PLE, 35 decoder layers, final norm) used to
call `load_file(shard_path)` inside its own `from_safetensors`. At the
stack level that's **37 passes** over the same 3 GB file — wasteful and
slow.

`GemmaTextModel.from_safetensors` loads the shard into `sd` once and
passes the dict down via the new `state_dict=sd` kwarg on every
component loader. The per-component loaders stay usable standalone
(the parity cells still call them one at a time), but the aggregate
path no longer re-reads.

---

## 3. KV-share routing lives at the stack level

`GemmaAttention` and `GemmaDecoderLayer` have no idea which layers share
K/V. They expose two kwargs:

- `cached_kv=(k, v)` → skip k_proj/k_norm/RoPE-on-K + v_proj/v_norm,
  reuse these instead.
- `return_kv=True` → also return this layer's (pre-GQA-expand) `(k, v)`
  so the caller can stash them.

`GemmaTextModel.forward` decides:

```python
src    = self._kv_source(i)                   # donor idx or None
cached = shared_kv[src] if src is not None else None

if self._is_donor(i):                          # layer 13 or 14
    hidden, k, v = layer(..., cached_kv=cached, return_kv=True)
    shared_kv[i] = (k, v)
else:
    hidden = layer(..., cached_kv=cached)
```

Routing rule for E2B's specific interleave:

    shared sliding (15,16,17,18, 20,21,22,23, 25,...,33)  ←  layer 13
    shared full    (19, 24, 29, 34)                       ←  layer 14

We compute it from `config.layer_types` rather than hard-coding it, so
a future config change doesn't silently mis-route.

---

## 4. Forward shape + dtype reference

| tensor                       | shape                  | dtype    |
|------------------------------|------------------------|----------|
| `input_ids`                  | (B, S)                 | long     |
| `inputs_embeds`              | (B, S, 1536)           | bf16     |
| `per_layer_inputs`           | (B, S, 35, 256)        | bf16     |
| `cos_l, sin_l`               | (B, S, 256)            | bf16     |
| `cos_g, sin_g`               | (B, S, 512)            | bf16     |
| `mask_full, mask_loc`        | (1, 1, S, S)           | fp32     |
| `hidden` (each layer out)    | (B, S, 1536)           | bf16     |
| `logits`                     | (B, S, 262_144)        | bf16     |

The masks are fp32 because the additive `-inf` must survive the add
before softmax; the softmax itself is done in fp32 (inside
`GemmaAttention`) and cast back.

---

## 5. Generating our first outputs

The final notebook cell does greedy decoding:

```python
for step in range(max_new):
    logits  = ours(input_ids)               # (1, S, vocab)
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_id], dim=1)
    if next_id.item() in eos_ids:           # {1, 106}
        break
```

No KV cache across generation steps yet — each token re-prefills the
whole prompt. That's O(S²) in tokens and plenty for a sanity check; a
real inference loop is the next milestone.

Prompt formatting still routes through `processor.apply_chat_template`
because the chat tokens (`<start_of_turn>user`, `<end_of_turn>`, etc.)
are easier to let HF template than to re-derive. The *model* running
generation is ours.

---

## 6. What parity guarantees

The logits-parity cell asserts `torch.equal(ours_logits, hf_logits)`
against HF loaded with `attn_implementation="eager"`. If that holds:

- embedding is right (including the √hidden scale, rounded to bf16),
- PLE precompute is right (both paths, norms, scale factors),
- every decoder layer is right (all 5 norms, attn, ffn, PLE block,
  per-layer scalar),
- KV-share routing is right (sharing math is not a free pass — wrong
  donor selection diverges instantly),
- final RMSNorm + tied LM head + softcap is right.

Anything that goes wrong downstream (in generation, in KV caching, in
batched inference) isn't a model bug. That narrows the search space.

---

## 7. What comes next

- **KV cache for generation** — currently re-prefilling. Need an
  incremental path that caches K/V per layer across steps *and* respects
  the existing layer-15+ shared-K/V routing.
- **Sliding window > 512 tokens** — the mask code is correct; we just
  haven't actually tested a prompt long enough for it to matter.
- **Vision / audio fusion into the residual stream** — the towers and
  projectors exist; the soft tokens need to be inserted at the right
  positions in `input_ids`-space before the text stack runs.
