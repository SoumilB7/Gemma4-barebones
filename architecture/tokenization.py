"""
Gemma 4 E2B · Tokenization
──────────────────────────
Loads Google's shipped tokenizer.json directly via the `tokenizers` library.
No HF `transformers` dependency here — just the raw SentencePiece/BPE file
we saved into `model_weights/`.
"""

from tokenizers import Tokenizer


class GemmaTokenizer:

    def __init__(self, tokenizer_path):
        self.tk = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, text) -> list[int]:
        return self.tk.encode(text).ids

    def decode(self, ids) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=True)

    def pretty(self, ids) -> list[tuple[int, str]]:
        return [(i, self.tk.id_to_token(i)) for i in ids]
