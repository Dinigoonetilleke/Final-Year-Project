import re
import torch
import pandas as pd
from pathlib import Path
from collections import Counter
from torch import nn
from sklearn.model_selection import train_test_split

# Paths
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "final_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "charcnn.pt"

MAX_LEN = 160  # must match training


def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_char_vocab(texts):
    # IMPORTANT: same logic as training script (Counter + most_common order)
    chars = Counter()
    for t in texts:
        chars.update(list(clean_text(t)))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in chars.most_common():
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_chars(text: str, vocab):
    text = clean_text(text)
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in text][:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    return ids


class CharCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        emb_dim = 64
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.Conv1d(emb_dim, 128, kernel_size=7, padding=3),
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        e = self.emb(x)          # (B, T, D)
        e = e.transpose(1, 2)    # (B, D, T)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(e))         # (B, C, T)
            h = torch.max(h, dim=2).values  # (B, C)
            pooled.append(h)
        z = torch.cat(pooled, dim=1)
        z = self.dropout(z)
        return self.fc(z)


def main():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(str)

    # IMPORTANT: same split as training (random_state=42, stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # IMPORTANT: vocab built from TRAIN ONLY (same as training)
    char_vocab = build_char_vocab(X_train)

    labels = sorted(df["label"].unique())
    id2label = {i: l for i, l in enumerate(labels)}

    device = "cpu"
    model = CharCNN(vocab_size=len(char_vocab), num_classes=len(labels)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sentence = input("Enter a sentence: ").strip()
    x = torch.tensor([encode_chars(sentence, char_vocab)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        pred_id = logits.argmax(dim=1).item()

    print("Prediction:", id2label[pred_id])


if __name__ == "__main__":
    main()
