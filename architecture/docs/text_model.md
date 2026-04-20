# Text Model - Gemma 4 E2B

> The end of the road. `GemmaTextModel` is the wrapper that holds every piece we've built so far - the embedding, the input computation of each layer,
> all 35 decoder layers (with their KV-share routing), the final RMSNorm,
> the tied LM head, and the softcap - and exposes one method:
> `forward(input_ids) -> logits`. Logits are the raw scores over the vocabulary.

---

## 1. The pipeline in one diagram

```
input_ids  (B, S)
   │
   ▼
embed_tokens                         GemmaEmbedding  (multiplies by √1536, then rounds to bf16)
   │  inputs_embeds  (B, S, 1536)
   ▼
per_layer_inputs                     GemmaPerLayerInputs - runs ONCE for the whole stack
   │                                   → (B, S, 35, 256)
   ▼
┌────────── layer loop, i = 0..34 ──────────┐
│  mask  = causal_mask(S, window=None)      │   used by full-attention layers
│  mask' = causal_mask(S, window=512)       │   used by sliding-attention layers
│  cos/sin = rope_local or rope_global      │
│  pli_i  = per_layer_inputs[:, :, i, :]    │
│                                            │
│  if i ≤ 14 (computes its own K and V):     │
│      hidden, k, v = layer(hidden, pli_i,   │
│                            cos, sin, mask) │
│      if i is a donor:  stash (k, v)        │
│                                            │
│  if i ≥ 15 (reuses someone else's K, V):    │
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
logits = hidden @ embed_tokens.weight.T       ← tied LM head - no separate Linear
   │
   ▼
logits = 30 * tanh(logits / 30)               ← final_logit_softcapping
   │
   ▼
logits  (B, S, vocab=262_144)
```

Three things to keep in mind:

1. **There is no separate output layer.** Gemma sets
   `tie_word_embeddings=True`, which means the matrix that maps token IDs
   into vectors at the bottom of the model is the *same* matrix used to
   map vectors back into vocabulary scores at the top. You can prove this
   with `hf.lm_head.weight.data_ptr() == hf.embed_tokens.weight.data_ptr()`
   - both names point at the same chunk of memory. So instead of building
   a `Linear` for the LM head, we just transpose the embedding matrix and
   matmul against it.
2. **The softcap is part of the model, not optional smoothing.** Before
   the next-token softmax, every logit goes through `30 · tanh(x / 30)`,
   which squashes very large or very small values toward ±30. If you
   skip it, every probability the model assigns is wrong by a small but
   real amount.
3. **Sliding and full attention layers use different masks.** A sliding
   layer can only see the most recent 512 tokens; a full layer can see
   everything before it. When the sequence is shorter than 512 the two
   masks happen to be identical, so you can't catch this bug on short
   prompts - it only shows up once you cross the window.

---

## 2. Why the shard loads exactly once

The model weights live in one big file (`model.safetensors`, around 3 GB).
Earlier in the build, every component had its own `from_safetensors`
that called `safetensors.load_file(shard_path)` on its own. That was
fine when we were testing pieces in isolation. At the stack level it
became a problem: the embedding loads it, the PLE loads it, then *each
of the 35 decoder layers* loads it. That's 37 reads of the same 3 GB
file just to build the model once.

`GemmaTextModel.from_safetensors` reads the file once into a dict
called `sd` and hands that dict down to every component via a new
`state_dict=sd` keyword. The component loaders still work standalone
(the parity cells still call them one at a time, without `state_dict`),
but when the full model assembles itself, the dict is passed in and no
extra disk reads happen.

---

## 3. KV-share routing lives at the stack level

A quick recap on what KV-sharing means: when an attention layer runs,
it normally projects the input three times - into queries (Q), keys
(K), and values (V). The Q, K, V are then used together to compute
attention. Gemma 4 saves work by having later layers **reuse** the K
and V that an earlier layer already computed (they still compute their
own Q). This cuts both compute and memory in half for the shared
layers.

`GemmaAttention` and `GemmaDecoderLayer` themselves don't know which
layers share with which. They just expose two simple knobs:

- `cached_kv=(k, v)` → "skip your own k_proj/k_norm/RoPE-on-K and your
  v_proj/v_norm; use these K and V instead."
- `return_kv=True` → "after you compute your K and V, hand them back
  to me too - I want to stash them for whoever is going to share them."

The stack-level forward is the one that knows the routing rule:

```python
src    = self._kv_source(i)                   # who does layer i borrow from? None if it computes its own.
cached = shared_kv[src] if src is not None else None

if self._is_donor(i):                          # only true for layer 13 and layer 14
    hidden, k, v = layer(..., cached_kv=cached, return_kv=True)
    shared_kv[i] = (k, v)
else:
    hidden = layer(..., cached_kv=cached)
```

The actual routing rule for E2B's specific layer pattern works out to:

    shared sliding (15,16,17,18, 20,21,22,23, 25,...,33)  ←  borrow from layer 13
    shared full    (19, 24, 29, 34)                       ←  borrow from layer 14

Why 13 and 14? Because they're the *last* sliding and full layers
respectively before the sharing region begins at layer 15. Each
shared layer borrows from the most recent layer of the same attention
type that actually did its own K/V projection.

We compute this rule from `config.layer_types` instead of hard-coding
indices, so if Google ever retunes the layer pattern (more sliding
layers, a different interleave, etc.) the code adapts instead of
silently routing to the wrong donor.

---

## 4. Forward shape + dtype reference

Helpful when you're stepping through with a debugger and want to know
what to expect at each stage.

| tensor                       | shape                  | dtype    |
|------------------------------|------------------------|----------|
| `input_ids`                  | (B, S)                 | long     |
| `inputs_embeds`              | (B, S, 1536)           | bf16     |
| `per_layer_inputs`           | (B, S, 35, 256)        | bf16     |
| `cos_l, sin_l`               | (B, S, 256)            | bf16     |
| `cos_g, sin_g`               | (B, S, 512)            | bf16     |
| `mask_full, mask_loc`        | (1, 1, S, S)           | fp32     |
| `hidden` (each layer's out)  | (B, S, 1536)           | bf16     |
| `logits`                     | (B, S, 262_144)        | bf16     |

The masks stay in fp32 because they contain `-inf` in the blocked
positions. If you cast `-inf` down to bf16 it survives, but the
addition that mixes the mask into attention scores is more reliable in
fp32. The softmax inside `GemmaAttention` is also computed in fp32 and
cast back at the end - that's a standard "be careful at the edges"
move.

---

## 5. Generating our first outputs

The last cell of the notebook does plain greedy decoding. Greedy means:
at each step, pick the single token with the highest score and append
it. No sampling, no beam search, no temperature. Boring but
deterministic - perfect for a sanity check.

```python
for step in range(max_new):
    logits  = ours(input_ids)               # (1, S, vocab)
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_id], dim=1)
    if next_id.item() in eos_ids:           # {1, 106}
        break
```

There's no KV cache *across* generation steps yet. That means every
time we want one new token, we rerun the whole stack on the entire
prompt-so-far. This is `O(S²)` in tokens - fine for a 64-token sanity
output, terrible for real inference. Building an incremental version
is on the next-steps list.

We still let HF's processor build the prompt, because Gemma's chat
format uses special tokens like `<start_of_turn>user` and
`<end_of_turn>` that are tedious to assemble by hand. Once we have the
list of token IDs, the model running generation is entirely ours.

---

## 6. What parity guarantees

The logits-parity cell asserts `torch.equal(ours_logits, hf_logits)` -
every single number in the (B, S, 262_144) output tensor is exactly
the same as HuggingFace's, down to the last bit. We compare against HF
loaded with `attn_implementation="eager"` so the math kernels match
step-for-step (the default SDPA kernel is numerically equivalent but
sums things in a different order, which drifts in bf16).

If that check passes, here's everything that's silently certified
correct:

- the embedding (including the √1536 scale, rounded to bf16),
- the PLE precompute (both paths, the norms, all the scale factors),
- every decoder layer (all 5 RMSNorms, attention, FFN, the PLE block,
  the per-layer scalar),
- the KV-share routing (one wrong donor index would diverge instantly
  - the math doesn't forgive a bad cache),
- the final RMSNorm,
- the tied LM head matmul,
- the softcap.

That's the whole text model. Anything that goes wrong from here -
generation looking weird, batched inference being off, a long prompt
producing garbage - is no longer a model bug. It's a glue bug. That's
a much smaller search space.

---

## 7. What comes next

- **KV cache for generation.** Right now we re-prefill the entire
  prompt every step. We need an incremental forward that caches K and
  V *across steps*, while still respecting the layers-15+ routing
  (those layers don't add to the cache; they read from the donor's
  cache).
- **Sliding window past 512 tokens.** The mask code already supports
  it; we just haven't tested with a prompt long enough to exercise it.
- **Vision and audio fusion into the residual stream.** The towers and
  projectors already exist. The remaining work is splicing the soft
  tokens into the right positions in `input_ids`-space *before* the
  text stack runs.
