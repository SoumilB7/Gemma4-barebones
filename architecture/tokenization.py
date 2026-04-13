"""
Gemma 4 E2B · Tokenization
──────────────────────────
Thin PyTorch-style wrapper around the HF processor. We don't reimplement
SentencePiece — we just expose the parts we need with a clean API that
returns torch tensors.
"""

from transformers import AutoProcessor


class GemmaTokenizer:

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.vocab_size = self.tokenizer.vocab_size

    @classmethod
    def from_pretrained(cls, model_id="google/gemma-4-E2B-it"):
        return cls(AutoProcessor.from_pretrained(model_id))

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt")["input_ids"]

    def decode(self, ids):
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    def apply_chat_template(self, messages):
        return self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
        )

    def pretty(self, ids):
        flat = ids.flatten().tolist()
        pieces = self.tokenizer.convert_ids_to_tokens(flat)
        return list(zip(flat, pieces))


if __name__ == "__main__":
    tok = GemmaTokenizer.from_pretrained()
    ids = tok.encode("Explain MoE in transformers in 3 sentences.")
    print(f"vocab_size = {tok.vocab_size:,}")
    print(f"ids.shape  = {tuple(ids.shape)}")
    for tid, piece in tok.pretty(ids)[:12]:
        print(f"  {tid:>6}  {piece!r}")
    print(f"roundtrip  = {tok.decode(ids[0])!r}")
