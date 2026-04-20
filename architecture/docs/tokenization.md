# Tokenization - Gemma 4 E2B

> Step 1 in the pipeline. Strings go in, lists of integers come out.
> Nothing else in the model can run until this does.

---

## 1. What tokenization is

Models can't read strings - they only know how to look up rows of a
table by integer index. A **tokenizer** is the deterministic, reversible
function that bridges those two worlds:

```
 text ──► [id, id, id, ...]    (encode)
 [id, id, id, ...] ──► text    (decode)
```

The tokenizer also *defines the vocabulary* - the fixed set of pieces
the model will ever see. Every other weight downstream (the embedding
table, the LM head) is shaped by it. Change the tokenizer and you've
changed the model.

## 2. Gemma 4 E2B's tokenizer

- **Algorithm:** SentencePiece, BPE-style, with byte-level fallback for
  any character it doesn't know a piece for.
- **Vocabulary size:** **262,144** - large. Big enough to cover many
  languages, code, and special tokens for chat roles, vision, and audio.
- **Reversible:** any sequence of bytes the user types can be encoded,
  and `decode(encode(s)) == s`.
- **Shared across modalities:** images and audio don't get text tokens.
  They get *placeholder* ids like `<start_of_image>`,
  `<image_soft_token>`, `<end_of_image>` (and audio analogues). The
  model will later swap those placeholders for soft embeddings produced
  by the vision and audio towers - but as far as the integer stream is
  concerned, they're just more ids.

## 3. Why we don't reimplement it

The shipped `tokenizer.json` (~32 MB) encodes the merges and pieces that
Google trained. Rebuilding that from scratch would be a port - and if
even one id came out different, every downstream parity check would
fail. So we just *load the file* with the lightweight `tokenizers`
library:

```python
from tokenizers import Tokenizer
self.tk = Tokenizer.from_file("model_weights/tokenizer.json")
```

No `transformers` dependency, no special-token registry to maintain -
the file already knows its own specials.

## 4. The wrapper

[`architecture/tokenization.py`](../tokenization.py) is intentionally a
thin shell:

```python
class GemmaTokenizer:
    def __init__(self, tokenizer_path):
        self.tk = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, text)  -> list[int]: ...
    def decode(self, ids)   -> str:       ...   # skips special tokens
    def pretty(self, ids)   -> list[tuple[int, str]]:
        # [(id, "▁hello"), (id, "▁world"), ...]  — handy for debugging
```

That's the whole API. Full chat-template formatting (with
`<start_of_turn>user` etc.) is left to HuggingFace's `AutoProcessor`
when we need it for generation, because reimplementing the chat-template
DSL adds risk for no real learning.

## 5. The approver

The check is: encode the same string with our wrapper and with HF's
processor and assert the id lists match.

```python
from architecture.tokenization import GemmaTokenizer

ours  = GemmaTokenizer(WEIGHTS / "tokenizer.json")
their = processor                          # HF's AutoProcessor

s = "Explain MoE in transformers in 3 sentences."

a = ours.encode(s)
b = their(s, return_tensors="pt")["input_ids"][0].tolist()

assert a == b, "tokenizer drift!"
```

If this ever fires, we stop and fix it before moving on. That's the
parity-check pattern we'll use through the whole rebuild: every new
component is judged against the corresponding HF output.

## 6. Useful things to print once you have ids

| question                          | how                                  |
|-----------------------------------|--------------------------------------|
| How many pieces is my string?     | `len(ids)`                           |
| What are the literal pieces?      | `tok.pretty(ids)`                    |
| Which ids are specials?           | inspect `tok.tk.get_vocab()` or just the chat template output |
| Are image tokens where I expect?  | scan for the `<image_soft_token>` id in `tok.pretty(ids)`     |

That's enough to move on. Next: the **embedding table** - how each id
turns into a 1536-dim vector and enters the residual stream.
