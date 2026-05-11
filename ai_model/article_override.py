import re

VOWEL_SOUND_START = tuple("aeiou")

def article_issue(sentence: str) -> bool:
    s = sentence.lower()

    # a/an wrong (very common ESL error)
    # "a umbrella" (should be an umbrella)
    if re.search(r"\ba\s+[aeiou]", s):
        return True

    # "an university" (often should be a university - 'yoo' sound)
    if re.search(r"\ban\s+univ", s):
        return True

    # Missing article for singular count noun patterns like: "She is honest person"
    # simple heuristic: "is/was" + adjective + "person/man/woman/student/teacher" without a/an/the
    if re.search(r"\b(is|was)\s+\w+\s+(person|man|woman|student|teacher|engineer|doctor)\b", s):
        # if already has an article before noun, skip
        if not re.search(r"\b(a|an|the)\s+\w+\s+(person|man|woman|student|teacher|engineer|doctor)\b", s):
            return True

    return False
