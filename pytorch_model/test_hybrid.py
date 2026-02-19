import sys
import re
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split

# Add project root to path so "ai_model" can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai_model.punctuation_override import is_punctuation_issue
from ai_model.spelling_override import spelling_suspects
from ai_model.grammar_override2 import (
    is_subject_verb_agreement_error,
    is_did_not_past_error,
    is_sva_basic_error,
    is_simple_past_marker_error,
)

from ai_model.structure_override import is_sentence_fragment
from ai_model.article_override import article_issue


BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "final_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "charcnn.pt"

MAX_LEN = 160
NO_ERROR_THRESHOLD = 0.80  # tune 0.65–0.85


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
            h = torch.relu(conv(e))
            h = torch.max(h, dim=2).values
            pooled.append(h)
        z = torch.cat(pooled, dim=1)
        z = self.dropout(z)
        return self.fc(z)


def main():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(str)

    # Recreate training split so vocab matches saved model
    X_train, _, y_train, _ = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    char_vocab = build_char_vocab(X_train)

    labels = sorted(df["label"].unique())
    id2label = {i: l for i, l in enumerate(labels)}

    device = "cpu"
    model = CharCNN(len(char_vocab), len(labels)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    while True:
        sentence = input("\nEnter a sentence (or type exit): ").strip()

        if sentence.lower() == "exit":
            break

        # 1 Spelling override
        suspects = spelling_suspects(sentence)
        if suspects:
            print(f"Prediction: Spelling (override)  suspects={suspects}")
            continue

        # 2 Punctuation override
        if is_punctuation_issue(sentence):
            print("Prediction: Punctuation (override)")
            continue

        # 3 Grammar overrides
        if (
            is_did_not_past_error(sentence)
            or is_subject_verb_agreement_error(sentence)
            or is_sva_basic_error(sentence)
            or is_simple_past_marker_error(sentence)
        ):
            print("Prediction: Grammar (override)")
            continue


        # 4 Sentence fragment override
        if is_sentence_fragment(sentence):
            print("Prediction: Sentence_Structure (fragment override)")
            continue

        # 5 Article override
        if article_issue(sentence):
            print("Prediction: Preposition_Article (override)")
            continue


        # 6 CharCNN prediction + No_Error threshold
        x = torch.tensor([encode_chars(sentence, char_vocab)], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_id = torch.max(probs, dim=0)

        conf = float(conf.item())
        pred_label = id2label[int(pred_id.item())]

        if conf < NO_ERROR_THRESHOLD:
            print(f"Prediction: No_Error (low confidence={conf:.2f})")
        else:
            print(f"Prediction: {pred_label} (confidence={conf:.2f})")


if __name__ == "__main__":
    main()
