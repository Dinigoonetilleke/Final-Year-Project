import re

# Words that often start dependent clauses
FRAGMENT_STARTERS = {
    "because", "although", "though", "while",
    "when", "if", "since", "unless", "before", "after"
}

def is_sentence_fragment(sentence: str) -> bool:
    s = sentence.strip().lower()

    words = s.split()

    # If sentence starts with dependent clause word
    if words and words[0] in FRAGMENT_STARTERS:
        # And does not contain comma (which might indicate full clause)
        if "," not in s:
            return True

    return False
