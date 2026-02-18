import re

# Common wrong pattern: "did not + past tense"
PAST_VERBS = {
    "ate", "went", "saw", "did", "took", "gave", "made", "wrote", "bought",
    "came", "ran", "drank", "drove", "spoke", "caught", "thought", "felt",
    "found", "left", "read", "slept", "told", "understood"
}

def is_did_not_past_error(sentence: str) -> bool:
    s = sentence.lower().strip()
    # match: did not ate / didn't ate
    m = re.search(r"\b(did not|didn't)\s+([a-z]+)\b", s)
    if not m:
        return False
    verb = m.group(2)
    return verb in PAST_VERBS
