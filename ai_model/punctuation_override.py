import re

QUESTION_WORDS = r"\b(what|why|how|where|when|who)\b"

def is_punctuation_issue(sentence: str) -> bool:
    s = sentence.strip()
    low = s.lower()

    # Treat as punctuation if it's likely a question missing '?'
    if re.search(QUESTION_WORDS, low) and not s.endswith("?"):
        return True

    # If there is NO terminal punctuation (very common punctuation issue)
    if not re.search(r"[.!?]$", s):
        return True

    # Missing space after punctuation: "Hello,how"
    if re.search(r"[,.!?][A-Za-z]", s):
        return True

    return False
