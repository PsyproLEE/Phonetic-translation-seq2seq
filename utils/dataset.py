import torch
from torch.utils.data import Dataset

class PronunciationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return (
            torch.tensor(self.src_vocab.encode(src)),
            torch.tensor(self.tgt_vocab.encode(tgt))
        )
