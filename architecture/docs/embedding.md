# Embedding — Gemma 4 E2B

> Step 2 in the pipeline. Integer ids and raw multimodal features in → 1536-dim
> residual-stream vectors out. This is where text, images, and audio all become
> the *same kind of thing*.

---

## 1. What "embedding" means here

The decoder only knows how to push vectors of shape `(seq_len, 1536)` through
attention + FFN stacks. Everything upstream exists to produce that tensor.

Gemma 4 E2B has three separate on-ramps into the residual stream, one per modality:

```
text ids  ──► GemmaEmbedding      ─┐
image     ──► vision tower         ─┼─► (seq_len, 1536)  ──► decoder
audio     ──► audio  tower         ─┘
```

All three land in the **same 1536-dim space**. From the decoder's point of view
there is no difference between a text token and an image soft token — they're
just vectors at positions in the sequence.

## 2. The text path — `GemmaEmbedding`

Plain `nn.Embedding(262_144, 1536)` lookup, plus one twist.

```python
vec = embed_tokens(id) * sqrt(hidden_size)    # sqrt(1536) ≈ 39.2
```

That √hidden scaling is a Gemma convention baked into HuggingFace's
`ScaledWordEmbedding`. Without it our embeddings are ~39× smaller than HF's and
the residual stream enters the first decoder layer at the wrong magnitude.

The weight lives in the safetensors shard under:

```
model.language_model.embed_tokens.weight      # (262_144, 1536)  bf16
```

Note the `language_model.` prefix — this is a multimodal model, so the text
embedder is nested one level deeper than a text-only Gemma.

## 3. The image & audio paths — `_Projector`

Images and audio don't have a vocabulary. Instead, a **tower** (a ViT for
images, a Conformer-style stack for audio) turns the raw signal into a sequence
of feature vectors. Then a **projector** — the last mile — maps those features
into the 1536-dim residual stream.

```
image pixels ──► vision_tower ──► (N_img, 768)  ──► projector ──► (N_img, 1536)
audio waveform ──► audio_tower ──► (N_aud, 1536) ──► projector ──► (N_aud, 1536)
```

The projector is not just a `Linear`. HF's `Gemma4MultimodalEmbedder` does two
things in sequence:

```python
x = rms_norm_no_scale(features)       # weightless RMSNorm, eps=1e-6, fp32
y = linear(x)                         # bias-free Linear(in_dim → 1536)
```

The RMSNorm has **no learned gain** (`with_scale=False`). It just divides by
the per-token RMS of the features — bringing image features (huge magnitudes
coming out of the ViT) and audio features (already on a sane scale) onto a
common footing before projection. Skipping this norm was exactly the bug we
hit: our image soft tokens came out ~2290× too large, audio ~8× too large,
directions correct, scales way off. The normalization is what makes both
modalities land in a range the residual stream expects.

Weights:

```
model.embed_vision.embedding_projection.weight   # (1536, 768)    bf16
model.embed_audio .embedding_projection.weight   # (1536, 1536)   bf16
```

No bias, no norm weight — just the linear map.

### A peek inside the vision tower — ViT-style patches

Gemma's vision tower is a classical ViT, with the patch embedder split across
the processor/model boundary:

- **Patch size = 16.** Each patch is `3 × 16 × 16 = 768` raw values — that's
  where the `768` dim comes from.
- The **processor** does the patchification (split the image into non-overlapping
  16×16 tiles, flatten each to a 768-vector) *before* the tensor enters the
  model. That's why `pixel_values` arrives as `(1, 2520, 768)` — already
  pre-tokenized patches.
- The model-side `input_proj` is a `Linear(768 → hidden)`.

This is mathematically identical to the textbook ViT `Conv2d(3, hidden,
kernel=16, stride=16)` patch embedder — a stride=kernel conv over an image is
exactly a per-patch flatten followed by a linear projection. HF just factored
the two halves across the processor/model boundary.

One Gemma twist: positions aren't a single learned vector per patch index.
They use **separable 2D positional embeddings** — one table for x, one for y —
summed per patch. This handles pan-and-scan tiling and variable-size inputs
more gracefully than a flat 1D position table.

## 4. Why we borrow the towers (for now)

The vision and audio towers together are hundreds of MB of custom architecture
(patch-embed, Conformer blocks, positional schemes, …). Rebuilding them before
the decoder is a detour.

So `GemmaVisionProjector.embed_file(...)` and `GemmaAudioProjector.embed_file(...)`
lazily load HuggingFace's towers via a cached `(processor, model)` pair, run
the file through the tower to get features, and then push those features
through **our own** `_rms_norm` + `proj`. When we later replace the towers
with our own, the projector API stays the same — only the feature source
changes.

This is the hybrid pattern we'll reuse: borrow parts we haven't built yet,
but always make the piece under study ours.

## 5. The approver: both notebooks, side by side

- `load_hftf_model.ipynb` embeds the same inputs with HF's full model.
- `load_pytorch_model.ipynb` embeds them with our three modules.

For text, we compare `our_emb(ids)` vs `hf.get_input_embeddings()(ids)` — they
should match to bf16 precision.

For image and audio, the check is subtler: HF's `fused = hidden_states[0]`
contains text + image + audio soft tokens interleaved. We partition it using
`mm_token_type_ids` (0 = text, 1 = image, 3 = audio) and compare the
image/audio slices against our projector outputs:

```python
tt   = inputs["mm_token_type_ids"][0]
img  = fused[0, tt == 1]       # (N_img, 1536)  — HF's image soft tokens
aud  = fused[0, tt == 3]       # (N_aud, 1536)  — HF's audio soft tokens
```

If the first `[:6]` of these matches our `img_embeds` and `aud_embeds`, the
projector (weights + norm) is correct.

## 6. What to watch for

| symptom                                              | likely cause                              |
|------------------------------------------------------|-------------------------------------------|
| text embeddings ~39× smaller than HF                 | missing `* sqrt(hidden)` scale            |
| image/audio directions right, magnitudes way off     | missing weightless RMSNorm pre-projection |
| `KeyError` on `model.embed_tokens.weight`            | forgot the `language_model.` prefix       |
| `IndexError: too many indices for 2D tensor`         | tower output has no batch dim — use `flatten(0,-2)` before indexing |

That's embedding. Next: **per-layer embeddings (PLE)** — Gemma's 256-dim
sidecar stream that gets gated into every decoder layer.
