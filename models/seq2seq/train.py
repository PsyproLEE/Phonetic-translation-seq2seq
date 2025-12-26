import os
import torch
import pickle
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.vocab import CharVocab
from utils.dataset import PronunciationDataset
from utils.config import load_config
from seq2seq import Encoder, Attention, AttentionDecoder, Seq2Seq

# ---------- Config ----------
cfg = load_config("configs/seq2seq.yaml")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = cfg["train"]["batch_size"]
EMB_DIM = cfg["model"]["emb_dim"]
HID_DIM = cfg["model"]["hid_dim"]
EPOCHS = cfg["train"]["epochs"]
LEARNING_RATE = cfg["train"]["learning_rate"]
TEACHER_FORCING_RATIO = cfg["train"]["teacher_forcing_ratio"]
PATIENCE = cfg["train"]["patience"]

TRAIN_PATH = cfg["data"]["train_path"]
VAL_PATH = cfg["data"]["val_path"]

SAVE_DIR = Path(cfg["logging"]["save_dir"])
LOG_DIR = Path(cfg["logging"]["log_dir"])
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVE_DIR / "model_best.pt"
SRC_VOCAB_PATH = SAVE_DIR / "src_vocab.pkl"
TGT_VOCAB_PATH = SAVE_DIR / "tgt_vocab.pkl"

writer = SummaryWriter(log_dir=LOG_DIR)
smoother = SmoothingFunction()

# ---------- Early Stopping ----------
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ---------- Data ----------
train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8")
val_df = pd.read_csv(VAL_PATH, encoding="utf-8")

train_df.dropna(subset=["input", "target"], inplace=True)
val_df.dropna(subset=["input", "target"], inplace=True)

train_pairs = list(zip(train_df["input"], train_df["target"]))
val_pairs = list(zip(val_df["input"], val_df["target"]))

# ---------- Vocab ----------
src_vocab = CharVocab([p[0] for p in train_pairs])
tgt_vocab = CharVocab([p[1] for p in train_pairs])

with open(SRC_VOCAB_PATH, "wb") as f:
    pickle.dump(src_vocab, f)
with open(TGT_VOCAB_PATH, "wb") as f:
    pickle.dump(tgt_vocab, f)

# ---------- Dataset ----------
train_dataset = PronunciationDataset(train_pairs, src_vocab, tgt_vocab)
val_dataset = PronunciationDataset(val_pairs, src_vocab, tgt_vocab)

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_pad.to(DEVICE), tgt_pad.to(DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

# ---------- Model ----------
enc = Encoder(len(src_vocab), EMB_DIM, HID_DIM)
attn = Attention(HID_DIM)
dec = AttentionDecoder(len(tgt_vocab), EMB_DIM, HID_DIM, attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
early_stopper = EarlyStopping(patience=PATIENCE)

# ---------- Training ----------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt[1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch + 1)

    # ---------- Validation ----------
    model.eval()
    total_bleu = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            pred = tgt_vocab.decode(output.argmax(-1)[:, 0].cpu().numpy())
            ref = tgt_vocab.decode(tgt[:, 0].cpu().numpy())
            total_bleu += sentence_bleu([list(ref)], list(pred), smoothing_function=smoother.method1)

    avg_bleu = total_bleu / len(val_loader)
    writer.add_scalar("BLEU/val", avg_bleu, epoch + 1)

    if early_stopper(avg_bleu):
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if early_stopper.early_stop:
        break

writer.close()
