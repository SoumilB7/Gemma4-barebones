# Tokenization — Gemma 4 E2B

> Step 1 in the pipeline. Strings in → integer ids out. Nothing else in the
> architecture runs until this does.

---

## 1. What tokenization is

Models don't read strings; they read integer ids. A **tokenizer** is a deterministic,
reversible function:

```
 text ──► [id, id, id, ...]        (encode)
 [id, id, id, ...] ──► text         (decode)
```

The choice of tokenizer decides the **vocabulary** — the fixed set of pieces the
model will ever see. Every other weight in the model (embedding table, LM head)
is shaped by it.

## 2. Gemma 4 E2B's tokenizer

- **Algorithm**: SentencePiece (BPE, byte-fallback).
- **Vocabulary size**: **262,144** — large. Room for many languages, code, and
  dedicated special tokens for tools, chat roles, vision, and audio.
- **Reversible**: every byte a user types can always be encoded; `decode(encode(s)) == s`.
- **Shared across modalities**: image and audio "tokens" live in the same id space
  as text via dedicated special tokens (`<start_of_image>`, `<image_soft_token>`,
  `<end_of_image>`, and audio analogues). Images don't produce normal text tokens —
  they get *placeholder* ids that the model replaces with soft embeddings from the
  vision tower.

## 3. Why we don't reimplement it

The tokenizer file (`tokenizer.json`, ~32 MB) is part of the release. It encodes
merges and pieces Google trained. Any reimplementation would be a port with a risk
of drift — and if our ids differ by even one, every downstream check fails.

**So we wrap it.** The Python class in [tokenization.py](../tokenization.py) is a
thin shell around HuggingFace's `AutoProcessor`, giving us a consistent
PyTorch-friendly API (`encode` / `decode` / `apply_chat_template` returning
`torch.LongTensor`s) that fits with the rest of the architecture.

## 4. What the wrapper gives us

```python
tok = GemmaTokenizer.from_pretrained("google/gemma-4-E2B-it")

tok.vocab_size        # 262_144
tok.special           # SpecialTokens(bos=..., eos=..., pad=..., boi=..., eoi=..., image=...)

ids = tok.encode("hello world")           # torch.LongTensor [1, T]
tok.decode(ids[0])                        # "hello world"
tok.pretty(ids)                           # [(id, "▁hello"), (id, "▁world"), ...]

tok.apply_chat_template(messages)         # dict of tensors, multimodal-aware
```

Two design choices:

- **Always returns `torch.LongTensor`**. No numpy, no lists at API edges. Keeps
  the rest of the pipeline typed.
- **Specials collected eagerly** into a dataclass so the vision-tower code can
  ask "what's the `<image_soft_token>` id?" without poking into HF internals.

## 5. The approver: load_model.ipynb

The notebook is our ground truth. Every time we run an experiment we encode the
same string both ways and assert the ids match:

```python
# in the notebook
from architecture.tokenization import GemmaTokenizer

ours  = GemmaTokenizer(processor)          # reuses the notebook's processor
their = processor                          # HF directly

s = "Explain MoE in transformers in 3 sentences."

a = ours.encode(s)[0]
b = their(s, return_tensors="pt")["input_ids"][0]

assert torch.equal(a, b), "tokenizer drift!"
```

If this ever fires, we stop and fix it before moving on. That's the approver
pattern we'll use through the whole rebuild.

## 6. What to look at when you open ids

Handy things to print once you have token ids:

| question                       | how                                      |
|--------------------------------|------------------------------------------|
| How many pieces is my string?  | `ids.shape[-1]`                          |
| What are the literal pieces?   | `tok.pretty(ids)`                        |
| Which ids are special?         | `tok.special`                            |
| Are image tokens where I expect? | scan for `tok.special.boi` / `eoi`    |

That's enough to move on. Next: the **embedding table** — how each of these ids
becomes a 1536-dim vector and enters the residual stream.
