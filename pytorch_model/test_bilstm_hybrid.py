import sys
import re
import json
from pathlib import Path

import torch
from torch import nn

# Project root so ai_model imports work
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

OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUT_DIR / "textclf.pt"              # BiLSTM weights
VOCAB_PATH = OUT_DIR / "bilstm_vocab.json"       # word vocab
LABELS_PATH = OUT_DIR / "bilstm_labels.json"     # label list

MAX_LEN = 40  # must match training
NO_ERROR_THRESHOLD = 0.55  # optional: only used if model isn't confident


def tokenize(text: str):
    # Must match training tokenizer
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def encode(text: str, vocab: dict):
    pad = vocab.get("<PAD>", 0)
    unk = vocab.get("<UNK>", 1)

    ids = [vocab.get(w, unk) for w in tokenize(text)]
    ids = ids[:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids += [pad] * (MAX_LEN - len(ids))
    return ids


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, emb_dim=128, hid_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hid_dim * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Load vocab + labels used during training
    vocab = load_json(VOCAB_PATH)
    labels = load_json(LABELS_PATH)

    if not isinstance(labels, list):
        raise ValueError("bilstm_labels.json must be a JSON list.")

    id2label = {i: l for i, l in enumerate(labels)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMClassifier(len(vocab), len(labels)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("BiLSTM Hybrid tester ready.")
    print(f"- Device: {device}")
    print(f"- Labels: {labels}\n")

    while True:
        sentence = input("Enter a sentence (or type exit): ").strip()
        if sentence.lower() == "exit":
            break

        # 1) Spelling override
        suspects = spelling_suspects(sentence)
        if suspects:
            print(f"Prediction: Spelling (override) suspects={suspects}\n")
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

        # 5) Structure override
        if is_sentence_fragment(sentence):
            print("Prediction: Sentence_Structure (fragment override)\n")
            continue

        # 6) BiLSTM prediction
        x = torch.tensor([encode(sentence, vocab)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_id = torch.max(probs, dim=0)

        conf = float(conf.item())
        pred_label = id2label[int(pred_id.item())]

        # If model is uncertain and it is not predicting No_Error, allow No_Error fallback
        if pred_label != "No_Error" and conf < NO_ERROR_THRESHOLD:
            print(f"Prediction: No_Error (low confidence={conf:.2f})\n")
        else:
            print(f"Prediction: {pred_label} (confidence={conf:.2f})\n")


if __name__ == "__main__":
    main()