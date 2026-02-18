import sys
from pathlib import Path

# Adding project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
import torch
import pandas as pd
from pathlib import Path
from collections import Counter
from torch import nn
from sklearn.model_selection import train_test_split

from ai_model.punctuation_override import is_punctuation_issue
from ai_model.spelling_override import spelling_suspects
from ai_model.grammar_override import is_did_not_past_error
from ai_model.structure_override import is_sentence_fragment



BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "final_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "charcnn.pt"
MAX_LEN = 160


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
        e = self.emb(x)
        e = e.transpose(1, 2)
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

    # recreate training split so vocab matches saved model
    X_train, _, y_train, _ = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    char_vocab = build_char_vocab(X_train)
    labels = sorted(df["label"].unique())
    id2label = {i: l for i, l in enumerate(labels)}

    device = "cpu"
    model = CharCNN(len(char_vocab), len(labels))
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

        # 3 Grammar override (did not + past tense)
        if is_did_not_past_error(sentence):
            print("Prediction: Grammar (override: did not + past verb)")
            continue
	
	# 4 Sentence fragment override
        if is_sentence_fragment(sentence):
            print("Prediction: Sentence_Structure (fragment override)")
            continue


        # 5 Otherwise CharCNN prediction
        x = torch.tensor([encode_chars(sentence, char_vocab)], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            pred_id = logits.argmax(dim=1).item()

        print("Prediction:", id2label[pred_id])



if __name__ == "__main__":
    main()
