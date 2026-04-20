# Embedding - Gemma 4 E2B

> Step 2 in the pipeline. Integer ids and raw multimodal features go in,
> 1536-dim residual-stream vectors come out. This is where text, images,
> and audio all become *the same kind of thing* - vectors the decoder
> stack can chew on.

---

## 1. What "embedding" means here

The decoder only knows how to push tensors of shape `(seq_len, 1536)`
through attention and FFN stacks. Everything upstream of it exists to
produce that tensor.

Gemma 4 E2B has three on-ramps into the residual stream, one per
modality:

```
text ids  ──► GemmaEmbedding      ─┐
image     ──► vision tower         ─┼─► (seq_len, 1536)  ──► decoder
audio     ──► audio  tower         ─┘
```

All three land in **the same 1536-dim space**. From the decoder's point
of view there is no difference between a text token and an image soft
token - they are just vectors at positions in the sequence.

## 2. The text path - `GemmaEmbedding`

Plain `nn.Embedding(262_144, 1536)` lookup, plus one twist:

```python
vec = embed_tokens(id) * sqrt(hidden_size)    # sqrt(1536) ≈ 39.2
```

That `* √hidden` scaling is a Gemma convention (HuggingFace bakes it
into a class called `ScaledWordEmbedding`). Without it our embeddings
land in the residual stream ~39× smaller than HF's, and every
downstream parity check fails immediately.

A small but biting detail: HF stores the scale as a **bf16** scalar,
which rounds `sqrt(1536) = 39.1918…` to `39.25`. If you multiply in
fp32 first, you get a different number. So we cast the scalar into the
weight's dtype *before* multiplying:

```python
scale = torch.tensor(hidden ** 0.5, dtype=self.embed.weight.dtype)
return self.embed(ids) * scale
```

Skip the cast and you drift by ~1 bf16 ulp per token - small, but
silently wrong.

The weight lives in the safetensors shard at:

```
model.language_model.embed_tokens.weight      # (262_144, 1536)  bf16
```

The `language_model.` prefix is there because this is a multimodal
model - the text embedder is nested one level deeper than it would be
in a text-only Gemma. Easy to forget when writing key strings by hand.

## 3. The image and audio paths - `_Projector`

Images and audio don't have a vocabulary. Instead, a **tower** (a ViT
for images, a Conformer for audio) turns the raw signal into a sequence
of feature vectors. Then a **projector** - the last mile - maps those
features into the 1536-dim residual stream:

```
image pixels   ──► vision_tower ──► (N_img, 768)   ──► projector ──► (N_img, 1536)
audio waveform ──► audio_tower  ──► (N_aud, 1536)  ──► projector ──► (N_aud, 1536)
```

The projector is *not* just a `Linear`. HF's `Gemma4MultimodalEmbedder`
does two things in sequence:

```python
x = rms_norm_no_scale(features)       # weightless RMSNorm, eps=1e-6, fp32
y = linear(x)                         # bias-free Linear(in_dim → 1536)
```

The RMSNorm has **no learned gain** (`with_scale=False`). It just
divides by the per-token RMS of the features. This brings image
features (huge magnitudes coming out of the ViT) and audio features
(already on a sane scale) onto a common footing before the linear
projection. Skipping this norm was exactly the bug we hit during
implementation: image soft tokens came out ~2290× too large, audio ~8×
too large, directions correct but scales wildly off.

Weights:

```
model.embed_vision.embedding_projection.weight   # (1536, 768)    bf16
model.embed_audio .embedding_projection.weight   # (1536, 1536)   bf16
```

No bias, no norm weight - just the linear map. (The norm is parameter-
free, so there's nothing to load for it.)

### A peek inside the vision tower - ViT-style patches

Gemma's vision tower is a classical Vision Transformer, with the patch
embedder split across the processor/model boundary:

- **Patch size = 16.** Each patch is `3 × 16 × 16 = 768` raw values -
  that's where the `768` input dim comes from.
- The **processor** does the patchification (split the image into
  non-overlapping 16×16 tiles, flatten each to a 768-vector) *before*
  the tensor enters the model. So `pixel_values` arrives as
  `(1, 2520, 768)` - already pre-tokenized patches.
- The model-side `input_proj` is a `Linear(768 → hidden)`.

This is mathematically identical to the textbook ViT recipe of
`Conv2d(3, hidden, kernel=16, stride=16)` - a stride=kernel conv over
an image *is* a per-patch flatten followed by a linear projection. HF
just split the two halves across the processor/model boundary.

One Gemma twist on positions: instead of one learned vector per patch
index, it uses **separable 2D positional embeddings** - one table for
x, one for y - summed per patch. This handles pan-and-scan tiling and
variable-size inputs more gracefully than a flat 1D table would.

## 4. Why we borrow the towers (for now)

The vision and audio towers together are hundreds of MB of custom
architecture - patch embedders, Conformer blocks, positional schemes,
the works. Rebuilding them before the decoder is a detour from the main
goal.

So `GemmaVisionProjector.embed_file(...)` and
`GemmaAudioProjector.embed_file(...)` lazily load HuggingFace's towers
through a cached `(processor, model)` pair, run the file through the
tower to get features, and then push those features through **our own**
`_rms_norm` + `proj`. When we eventually replace the towers with our
own implementations, the projector API stays the same - only the
feature source changes.

This is the hybrid pattern we'll keep reusing: borrow parts we haven't
built yet, but always make the piece *under study* ours.

## 5. The approver: both notebooks, side by side

- `load_hftf_model.ipynb` embeds the same inputs with HF's full model.
- `load_pytorch_model.ipynb` embeds them with our three modules.

For text, we compare `our_emb(ids)` vs `hf.get_input_embeddings()(ids)`.
They should match to bf16 precision.

For image and audio the check is subtler. HF's `fused = hidden_states[0]`
contains text + image + audio soft tokens interleaved. We partition it
using `mm_token_type_ids` (0 = text, 1 = image, 3 = audio) and compare
the image/audio slices against our projector outputs:

```python
tt   = inputs["mm_token_type_ids"][0]
img  = fused[0, tt == 1]       # (N_img, 1536)  — HF's image soft tokens
aud  = fused[0, tt == 3]       # (N_aud, 1536)  — HF's audio soft tokens
```

If the first `[:6]` of these matches our `img_embeds` and `aud_embeds`,
the projector (weights + norm) is correct.

## 6. What to watch for

| symptom                                             | likely cause                              |
|-----------------------------------------------------|-------------------------------------------|
| text embeddings ~39× smaller than HF                | missing `* sqrt(hidden)` scale            |
| text embeddings drift by 1 bf16 ulp                 | scale not cast to bf16 before multiply    |
| image/audio directions right, magnitudes way off    | missing weightless RMSNorm pre-projection |
| `KeyError` on `model.embed_tokens.weight`           | forgot the `language_model.` prefix       |
| `IndexError: too many indices for 2D tensor`        | tower output has no batch dim - flatten before indexing |

That's embedding. Next: **per-layer embeddings (PLE)** - Gemma's
256-dim sidecar stream that gets gated into every decoder layer.
