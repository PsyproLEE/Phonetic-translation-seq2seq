import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathlib import Path

from utils.vocab import CharVocab
from utils.dataset import PronunciationDataset
from utils.config import load_config
from seq2seq import Encoder, Attention, AttentionDecoder, Seq2Seq

cfg = load_config("configs/seq2seq.yaml")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = Path(cfg["logging"]["save_dir"])
MODEL_PATH = SAVE_DIR / "model_best.pt"
SRC_VOCAB_PATH = SAVE_DIR / "src_vocab.pkl"
TGT_VOCAB_PATH = SAVE_DIR / "tgt_vocab.pkl"
TEST_PATH = cfg["data"]["test_path"]

with open(SRC_VOCAB_PATH, "rb") as f:
    src_vocab = pickle.load(f)
with open(TGT_VOCAB_PATH, "rb") as f:
    tgt_vocab = pickle.load(f)

test_df = pd.read_csv(TEST_PATH, encoding="utf-8")
test_df.dropna(subset=["input", "target"], inplace=True)
test_pairs = list(zip(test_df["input"], test_df["target"]))

test_dataset = PronunciationDataset(test_pairs, src_vocab, tgt_vocab)

def collate_batch(batch):
    src, tgt = zip(*batch)
    return (
        torch.nn.utils.rnn.pad_sequence(src).to(DEVICE),
        torch.nn.utils.rnn.pad_sequence(tgt).to(DEVICE),
    )

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

enc = Encoder(len(src_vocab), cfg["model"]["emb_dim"], cfg["model"]["hid_dim"])
attn = Attention(cfg["model"]["hid_dim"])
dec = AttentionDecoder(len(tgt_vocab), cfg["model"]["emb_dim"], cfg["model"]["hid_dim"], attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

smoother = SmoothingFunction()
total_bleu = 0

with torch.no_grad():
    for src, tgt in tqdm(test_loader):
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        pred = tgt_vocab.decode(output.argmax(-1)[:, 0].cpu().numpy())
        ref = tgt_vocab.decode(tgt[:, 0].cpu().numpy())
        total_bleu += sentence_bleu([list(ref)], list(pred), smoothing_function=smoother.method1)

print(f"Final Test BLEU: {total_bleu / len(test_loader):.4f}")
