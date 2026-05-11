import sys
import re
import json
from pathlib import Path

import torch
from torch import nn

# Add project root to path so "ai_model" can be imported
BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from ai_model.punctuation_override import is_punctuation_issue
from ai_model.spelling_override import spelling_suspects
from ai_model.article_override import article_issue
from ai_model.structure_override import is_sentence_fragment
from ai_model.grammar_override2 import (
    is_subject_verb_agreement_error,
    is_did_not_past_error,
    is_sva_basic_error,
    is_simple_past_marker_error,
)

# Paths (all inside pytorch_model/)
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUT_DIR / "charcnn.pt"
VOCAB_PATH = OUT_DIR / "vocab.json"      # must be the CHAR vocab used in training
LABELS_PATH = OUT_DIR / "labels.json"

MAX_LEN = 160
NO_ERROR_THRESHOLD = 0.60  # since you now have No_Error in dataset, 0.55–0.70 is typical


def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def encode_chars(text: str, vocab: dict) -> list[int]:
    text = clean_text(text)
    unk = vocab.get("<UNK>", 1)
    pad = vocab.get("<PAD>", 0)

    ids = [vocab.get(ch, unk) for ch in text][:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids += [pad] * (MAX_LEN - len(ids))
    return ids


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
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


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 1) Load saved vocab + labels (CRITICAL to match model)
    char_vocab = load_json(VOCAB_PATH)
    labels = load_json(LABELS_PATH)

    # labels.json should be a LIST in the exact order used during training
    if not isinstance(labels, list):
        raise ValueError("labels.json must be a JSON list like ['Grammar', 'No_Error', ...]")

    id2label = {i: l for i, l in enumerate(labels)}

    # 2) Load model with matching sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharCNN(vocab_size=len(char_vocab), num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Hybrid tester ready.")
    print(f"- Device: {device}")
    print(f"- Classes: {labels}")
    print(f"- No_Error threshold: {NO_ERROR_THRESHOLD}\n")

    while True:
        sentence = input("Enter a sentence (or type exit): ").strip()
        if sentence.lower() == "exit":
            break

        # 1) Spelling override
        suspects = spelling_suspects(sentence)
        if suspects:
            print(f"Prediction: Spelling (override)  suspects={suspects}\n")
            continue

        # 2) Punctuation override
        if is_punctuation_issue(sentence):
            print("Prediction: Punctuation (override)\n")
            continue

        # 3) Article override
        if article_issue(sentence):
            print("Prediction: Preposition_Article (override)\n")
            continue

        # 4) Grammar overrides
        if (
            is_did_not_past_error(sentence)
            or is_subject_verb_agreement_error(sentence)
            or is_sva_basic_error(sentence)
            or is_simple_past_marker_error(sentence)
        ):
            print("Prediction: Grammar (override)\n")
            continue

        # 5) Sentence fragment override
        if is_sentence_fragment(sentence):
            print("Prediction: Sentence_Structure (fragment override)\n")
            continue

        # 6) CharCNN prediction + No_Error threshold
        x = torch.tensor([encode_chars(sentence, char_vocab)], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_id = torch.max(probs, dim=0)

        conf = float(conf.item())
        pred_label = id2label[int(pred_id.item())]

        # If model predicts No_Error confidently, trust it.
        # Otherwise use threshold to reduce false positives.
        if pred_label == "No_Error":
            print(f"Prediction: No_Error (model) (confidence={conf:.2f})\n")
        else:
            if conf < NO_ERROR_THRESHOLD:
                print(f"Prediction: No_Error (low confidence={conf:.2f})\n")
            else:
                print(f"Prediction: {pred_label} (confidence={conf:.2f})\n")


if __name__ == "__main__":
    main()