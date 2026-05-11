import re

QUESTION_WORDS = r"\b(what|why|how|where|when|who)\b"

def is_punctuation_issue(sentence: str) -> bool:
    s = sentence.strip()
    low = s.lower()

    # Missing terminal punctuation completely
    if s and s[-1] not in ".!?":
        return True

    # Question word present but no '?'
    if re.search(QUESTION_WORDS, low) and not s.endswith("?"):
        return True

    return False
