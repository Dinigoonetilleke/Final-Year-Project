import re

def is_subject_verb_agreement_error(sentence: str) -> bool:
    s = sentence.lower().strip()

    # She/He/It don't -> should be doesn't
    if re.search(r"\b(she|he|it)\s+don['’]?t\b", s):
        return True

    # I/You/We/They doesn't -> should be don't
    if re.search(r"\b(i|you|we|they)\s+doesn['’]?t\b", s):
        return True

    # He/She/It have -> should be has
    if re.search(r"\b(she|he|it)\s+have\b", s):
        return True

    # They/We/You was -> should be were
    if re.search(r"\b(we|you|they)\s+was\b", s):
        return True

    # He/She/It were -> should be was
    if re.search(r"\b(she|he|it)\s+were\b", s):
        return True

    return False


# (Keep your old rule too: did not + past tense)
PAST_VERBS = {
    "ate","went","saw","did","took","gave","made","wrote","bought","came","ran",
    "drank","drove","spoke","caught","thought","felt","found","left","read",
    "slept","told","understood"
}

def is_did_not_past_error(sentence: str) -> bool:
    s = sentence.lower().strip()
    m = re.search(r"\b(did not|didn't)\s+([a-z]+)\b", s)
    if not m:
        return False
    return m.group(2) in PAST_VERBS

def is_sva_basic_error(sentence: str) -> bool:
    s = sentence.lower()

    # "students ... is"
    if re.search(r"\b(students|people|children|they|we)\b.*\bis\b", s):
        return True

    # "everyone have" / "everybody have"
    if re.search(r"\b(everyone|everybody|someone|somebody|no one|nobody)\s+have\b", s):
        return True

    # "they has"
    if re.search(r"\b(they|we|you)\s+has\b", s):
        return True

    # "he/she/it have"
    if re.search(r"\b(he|she|it)\s+have\b", s):
        return True

    # uncountable/subject singular treated plural: "mathematics are"
    if re.search(r"\bmathematics\s+are\b", s):
        return True

    return False


def is_simple_past_marker_error(sentence: str) -> bool:
    s = sentence.lower()

    # "yesterday/last/ago" but verb looks base (very rough but works for 'go yesterday')
    if re.search(r"\byesterday\b", s) and re.search(r"\b(go|eat|come|finish|buy)\b", s):
        return True

    return False

