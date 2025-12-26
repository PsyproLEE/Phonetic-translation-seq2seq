from collections import Counter

class CharVocab:
    def __init__(self, texts, specials=["<pad>", "<sos>", "<eos>"]):
        counter = Counter(
            ch for text in texts if isinstance(text, str) for ch in text
        )
        self.itos = specials + sorted(counter)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def encode(self, text):
        if not isinstance(text, str):
            text = ""
        return (
            [self.stoi["<sos>"]]
            + [self.stoi.get(c, self.stoi["<pad>"]) for c in text]
            + [self.stoi["<eos>"]]
        )

    def decode(self, ids):
        return ''.join(
            self.itos[i]
            for i in ids
            if self.itos[i] not in ("<sos>", "<eos>", "<pad>")
        )

    def __len__(self):
        return len(self.itos)
