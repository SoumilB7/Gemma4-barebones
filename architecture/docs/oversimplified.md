# Gemma 4 E2B — The Oversimplified Version

> One or two lines per step. No nuance, no caveats. Just what each brick actually does.

---

## 1. Tokenizer
String → list of integers. SentencePiece BPE over a 262 k vocab. `"hello"` becomes a few ints the model can index with.

## 2. Embedding
Each integer id indexes one row of a `(262144, 1536)` table. Out comes a 1536-dim vector per token — the residual stream is born.

## 3. Vision tower + projector
Image → flattened 16×16 patches → SigLIP ViT → Linear into 1536. You now have *N* "image soft tokens" that sit in the residual stream next to text tokens.

## 4. Audio tower + projector
Waveform → mel spectrogram → Conformer → Linear into 1536. Same idea: audio becomes soft tokens in the same stream. The decoder can't tell them apart from text.

## 5. PLE (per-layer embeddings)
Every layer gets an extra 256-dim lookup per token, streamed from flash, gated, and added back to the stream. Free "id memory" without keeping the weights in VRAM.

## 6. RMSNorm
Divide each token vector by its own root-mean-square, then multiply by a learned per-dim gain. Keeps activations on a unit sphere so the next matmul doesn't explode.

## 7. RoPE
Before attention, rotate pairs of Q/K dims by an angle proportional to the token's position. Dot products `Q·K` then depend on *relative* position — no learned position vectors needed.

## 8. Attention (GQA)
Project stream → Q, K, V. Rotate Q/K with RoPE, softmax `Q·Kᵀ`, weighted sum of V. Q heads outnumber KV heads (8:1 global, 2:1 local) to shrink the KV cache. Add the result back to the stream.

## 9. Sliding vs global
Local layers only attend to the last 512 tokens (cheap, 4 of every 5). Global layers attend to everything, use p-RoPE (only 25 % of dims rotate) and reuse K as V. 1 of every 5.

## 10. GeGLU FFN
Two parallel projections up to 8192-ish, GELU one of them, multiply elementwise, project back to 1536. A gated MLP — the "thinking" happens here. Add back to stream.

## 11. Decoder layer
`stream += attn(norm(stream))` then `stream += ffn(norm(stream))`. Plus PLE gating. That's one layer. Repeat 35 times.

## 12. Final norm + LM head
One last RMSNorm, then multiply by the embedding table transposed (tied weights): 1536 → 262 144 logits. Softmax → sample → next token id.

## 13. Loop
Append the new id, run forward again (or just feed it through with a KV cache). Stop on EOS or a length limit.
