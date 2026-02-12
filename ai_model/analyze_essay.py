import joblib
import re
from pathlib import Path
from collections import OrderedDict
from wordfreq import zipf_frequency

MODEL_PATH = Path(__file__).resolve().parent / "essay_error_model.joblib"
model = joblib.load(MODEL_PATH)

LABEL_ORDER = [
    "Grammar",
    "Preposition_Article",
    "Punctuation",
    "Sentence_Structure",
    "Spelling",
    "Word_Usage",
]

LABEL_TIPS = {
    "Grammar": "Check verb tense, subject–verb agreement, auxiliaries, conditionals, etc.",
    "Preposition_Article": "Check articles (a/an/the), prepositions, pronouns, conjunction usage, modifiers.",
    "Punctuation": "Check commas/full stops, apostrophes, capitalization, abbreviations, ellipsis.",
    "Sentence_Structure": "Check word order, fragments, run-on sentences, clauses, parallelism.",
    "Spelling": "Check misspelled words.",
    "Word_Usage": "Check word choice, register, clichés, redundancy, ambiguity, idioms/metaphors.",
}

def split_sentences(text: str):
    parts = re.split(r"[\r\n]+", text.strip())
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        sents = re.split(r"(?<=[.!?])\s+", part)
        sentences.extend([s.strip() for s in sents if s.strip()])
    return sentences

def tokenize_words(sentence: str):
    # words with optional apostrophes (don't, it's, student's)
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)

def detect_spelling(sentence: str, threshold_zipf: float = 2.0):
    """
    Returns a list of suspicious words.
    zipf_frequency(word, 'en') is ~ 0 (rare) to ~ 7 (very common).
    Low score usually means rare/unknown word -> possible misspelling.
    """
    words = tokenize_words(sentence)
    suspicious = []

    for w in words:
        lw = w.lower()

        # ignore very short tokens
        if len(lw) <= 2:
            continue

        # ignore proper nouns (simple heuristic: starts with capital and not first word)
        # (we won't be too strict)
        # ignore words containing digits (none will due to regex)
        score = zipf_frequency(lw, "en")

        # If extremely rare, flag it
        if score < threshold_zipf:
            suspicious.append(w)

    return suspicious

def predict_label(sentence: str):
    # 1) spelling override (hybrid)
    suspicious = detect_spelling(sentence)

    # If we detect likely misspelling(s), label as Spelling
    if suspicious:
        return "Spelling", suspicious

    # 2) otherwise ML label
    return model.predict([sentence])[0], []

def analyze_essay(essay: str):
    sentences = split_sentences(essay)

    grouped = {label: [] for label in LABEL_ORDER}
    for i, s in enumerate(sentences, 1):
        label, suspicious = predict_label(s)
        if label not in grouped:
            grouped[label] = []
        grouped[label].append({
            "sentence_no": i,
            "sentence": s,
            "spelling_suspects": suspicious
        })

    grouped_non_empty = OrderedDict(
        (label, items) for label, items in grouped.items() if len(items) > 0
    )

    return {
        "sentence_count": len(sentences),
        "counts": {label: len(items) for label, items in grouped.items()},
        "tips": LABEL_TIPS,
        "grouped": grouped_non_empty,
    }

if __name__ == "__main__":
    essay = """She is a beutiful girl.
The boy wnet to school.
They are playng outside."""


    result = analyze_essay(essay)

    print("=== ESSAY ANALYSIS (HYBRID) ===")
    print(f"Total sentences detected: {result['sentence_count']}\n")

    print("=== COUNT SUMMARY ===")
    for label in LABEL_ORDER:
        print(f"{label:20s}: {result['counts'][label]}")
    print()

    print("=== DETAILED FEEDBACK ===")
    for label, items in result["grouped"].items():
        print(f"\n{label.upper()} ({len(items)}):")
        print(f"Tip: {LABEL_TIPS.get(label, '')}")
        for item in items:
            extra = ""
            if label == "Spelling" and item["spelling_suspects"]:
                extra = f"  [suspects: {', '.join(item['spelling_suspects'])}]"
            print(f"  - Sentence {item['sentence_no']}: {item['sentence']}{extra}")
