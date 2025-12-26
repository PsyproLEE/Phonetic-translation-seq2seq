import torch
import pickle
from pathlib import Path

from utils.vocab import CharVocab
from utils.config import load_config
from seq2seq import Encoder, Attention, AttentionDecoder, Seq2Seq

cfg = load_config("configs/seq2seq.yaml")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = Path(cfg["logging"]["save_dir"])
MODEL_PATH = SAVE_DIR / "model_best.pt"
SRC_VOCAB_PATH = SAVE_DIR / "src_vocab.pkl"
TGT_VOCAB_PATH = SAVE_DIR / "tgt_vocab.pkl"
MAX_LEN = cfg["inference"]["max_len"]

with open(SRC_VOCAB_PATH, "rb") as f:
    src_vocab = pickle.load(f)
with open(TGT_VOCAB_PATH, "rb") as f:
    tgt_vocab = pickle.load(f)

enc = Encoder(len(src_vocab), cfg["model"]["emb_dim"], cfg["model"]["hid_dim"])
attn = Attention(cfg["model"]["hid_dim"])
dec = AttentionDecoder(len(tgt_vocab), cfg["model"]["emb_dim"], cfg["model"]["hid_dim"], attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def infer(text):
    tokens = src_vocab.encode(text)
    src = torch.tensor(tokens).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        input_token = torch.tensor([tgt_vocab.stoi["<sos>"]], device=DEVICE)
        result = []

        for _ in range(MAX_LEN):
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)
            if top1.item() == tgt_vocab.stoi["<eos>"]:
                break
            result.append(top1.item())
            input_token = top1

    return tgt_vocab.decode(result)

if __name__ == "__main__":
    while True:
        text = input("발음 입력 (quit 종료): ").strip()
        if text == "quit":
            break
        print("→", infer(text))
