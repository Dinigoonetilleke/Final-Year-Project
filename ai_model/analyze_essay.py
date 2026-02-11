import joblib
import re
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "essay_error_model.joblib"
model = joblib.load(MODEL_PATH)

def split_sentences(text: str):
    # first split on line breaks (students often press enter instead of using full stop)
    parts = re.split(r"[\r\n]+", text.strip())
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # then split using punctuation
        sents = re.split(r"(?<=[.!?])\s+", part)
        sentences.extend([s.strip() for s in sents if s.strip()])
    return sentences


def analyze(essay: str):
    sentences = split_sentences(essay)
    grouped = {}

    for i, s in enumerate(sentences, 1):
        label = model.predict([s])[0]
        grouped.setdefault(label, []).append({"sentence_no": i, "sentence": s})

    return grouped

if __name__ == "__main__":
    essay = """He go to school everyday. Hello how are you
I went to the shop. I recieved the message. Where you are going?"""

    grouped = analyze(essay)

    print("=== STRUCTURED FEEDBACK ===")
    for label, items in grouped.items():
        print(f"\n{label.upper()} ({len(items)} issues):")
        for item in items:
            print(f"  - Sentence {item['sentence_no']}: {item['sentence']}")
