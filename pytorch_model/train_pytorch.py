import re
import math
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Paths
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_dataset.csv"
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUT_DIR / "textclf.pt"
VOCAB_PATH = OUT_DIR / "vocab.json"
LABELS_PATH = OUT_DIR / "labels.json"

# Hyperparameters
MAX_LEN = 40
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
EMB_DIM = 128
HID_DIM = 128


def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(text, vocab):
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokenize(text)]
    ids = ids[:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.label2id[self.labels[idx]], dtype=torch.long)
        return x, y


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=EMB_DIM,
            hidden_size=HID_DIM,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HID_DIM * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())
    return all_labels, all_preds


def main():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    vocab = build_vocab(X_train)
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    VOCAB_PATH.write_text(json.dumps(vocab))
    LABELS_PATH.write_text(json.dumps(labels))

    train_ds = TextDataset(X_train, y_train, vocab, label2id)
    test_ds = TextDataset(X_test, y_test, vocab, label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMClassifier(len(vocab), len(labels)).to(device)

    # Class weights
    counts = df["label"].value_counts().to_dict()
    weights = torch.tensor(
        [1.0 / math.sqrt(counts[l]) for l in labels],
        dtype=torch.float
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0
    best_state = None

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        y_true, y_pred = evaluate(model, test_loader, device)

        y_true_lbl = [id2label[i] for i in y_true]
        y_pred_lbl = [id2label[i] for i in y_pred]

        report = classification_report(y_true_lbl, y_pred_lbl, zero_division=0, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"\nEpoch {epoch+1}/{EPOCHS}  Loss={loss:.4f}  Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_PATH)

    print("\n=== FINAL TEST REPORT (PyTorch) ===")
    y_true, y_pred = evaluate(model, test_loader, device)
    y_true_lbl = [id2label[i] for i in y_true]
    y_pred_lbl = [id2label[i] for i in y_pred]
    print(classification_report(y_true_lbl, y_pred_lbl, zero_division=0))

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
