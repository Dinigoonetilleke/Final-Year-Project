import re
import math
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "final_dataset_v2.csv"
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUT_DIR / "charcnn.pt"

MAX_LEN = 160          # characters per sentence (trim/pad)
BATCH_SIZE = 64
EPOCHS = 12
LR = 2e-3


def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_char_vocab(texts):
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


class CharDataset(Dataset):
    def __init__(self, texts, labels, char_vocab, label2id):
        self.texts = list(texts)
        self.labels = list(labels)
        self.char_vocab = char_vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode_chars(self.texts[idx], self.char_vocab), dtype=torch.long)
        y = torch.tensor(self.label2id[self.labels[idx]], dtype=torch.long)
        return x, y


class CharCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        emb_dim = 64
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # multiple kernel sizes capture different character patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1),
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.Conv1d(emb_dim, 128, kernel_size=7, padding=3),
        ])

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * len(self.convs), num_classes)

    def forward(self, x):
        # x: (B, T)
        e = self.emb(x)              # (B, T, D)
        e = e.transpose(1, 2)        # (B, D, T) for Conv1d

        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(e))          # (B, C, T)
            h = torch.max(h, dim=2).values   # global max pool -> (B, C)
            pooled.append(h)

        z = torch.cat(pooled, dim=1)         # (B, C*kernels)
        z = self.dropout(z)
        return self.fc(z)


def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(y.tolist())
    return y_true, y_pred


def main():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    char_vocab = build_char_vocab(X_train)

    train_ds = CharDataset(X_train, y_train, char_vocab, label2id)
    test_ds = CharDataset(X_test, y_test, char_vocab, label2id)

    # Weighted sampler to balance classes in training batches
    train_counts = pd.Series(y_train).value_counts().to_dict()
    sample_weights = [1.0 / train_counts[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharCNN(vocab_size=len(char_vocab), num_classes=len(labels)).to(device)

    # Also keep class weights (extra help)
    full_counts = df["label"].value_counts().to_dict()
    weights = torch.tensor([1.0 / math.sqrt(full_counts[l]) for l in labels], dtype=torch.float)
    weights = (weights / weights.mean()).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optim, loss_fn, device)
        yt, yp = eval_epoch(model, test_loader, device)

        yt_lbl = [id2label[i] for i in yt]
        yp_lbl = [id2label[i] for i in yp]

        report = classification_report(yt_lbl, yp_lbl, zero_division=0, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"\nEpoch {epoch}/{EPOCHS}  loss={loss:.4f}  macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_PATH)

    print("\n=== FINAL TEST REPORT (PyTorch CharCNN) ===")
    yt, yp = eval_epoch(model, test_loader, device)
    yt_lbl = [id2label[i] for i in yt]
    yp_lbl = [id2label[i] for i in yp]
    print(classification_report(yt_lbl, yp_lbl, zero_division=0))

    print(f"\n Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
