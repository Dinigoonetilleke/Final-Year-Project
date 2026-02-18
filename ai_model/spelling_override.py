import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MISSPELL_PATH = BASE / "data" / "misspelled_dataset.csv"

_misspell_inputs = None
_correct_words = None

def _load_sets():
    global _misspell_inputs, _correct_words
    if _misspell_inputs is not None:
        return

    import pandas as pd
    df = pd.read_csv(MISSPELL_PATH)

    # Words that are known misspellings
    _misspell_inputs = set(df["input"].astype(str).str.lower().str.strip())

    # Words that are correct forms
    _correct_words = set(df["label"].astype(str).str.lower().str.strip())


def spelling_suspects(sentence: str):
    _load_sets()
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence.lower())

    suspects = []

    for w in words:
        # Only consider words longer than 2 characters
        if len(w) <= 2:
            continue

        # Word is considered misspelled ONLY IF:
        # 1) It appears in misspelled input column
        # 2) It does NOT appear as a correct word
        if w in _misspell_inputs and w not in _correct_words:
            suspects.append(w)

    return suspects
