import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MISSPELL_PATH = BASE / "data" / "misspelled_dataset.csv"

_misspell_inputs = None
_correct_words = None

COMMON_WORDS = {
    "the","is","are","was","were","am","a","an","to","of","in","on",
    "and","or","but","for","with","at","by","from","this","that",
    "you","he","she","they","we","it","his","her","their","my",
    "wow","bag","late","home","school"
}

def _load_sets():
    global _misspell_inputs, _correct_words
    if _misspell_inputs is not None:
        return

    import pandas as pd
    df = pd.read_csv(MISSPELL_PATH)

    _misspell_inputs = set(df["input"].astype(str).str.lower().str.strip())
    _correct_words = set(df["label"].astype(str).str.lower().str.strip())


def spelling_suspects(sentence: str):
    _load_sets()
    words = re.findall(r"[A-Za-z]+", sentence.lower())

    suspects = []

    for w in words:
        if len(w) <= 3:
            continue

        if w in COMMON_WORDS:
            continue

        if w in _misspell_inputs and w not in _correct_words:
            suspects.append(w)

    return suspects
