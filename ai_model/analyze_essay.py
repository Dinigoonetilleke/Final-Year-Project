from __future__ import annotations

import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import joblib
from wordfreq import zipf_frequency

MODEL_PATH = Path(__file__).resolve().parent / 'essay_error_model.joblib'
model = joblib.load(MODEL_PATH)

LABEL_ORDER = [
    'Grammar',
    'Preposition_Article',
    'Punctuation',
    'Sentence_Structure',
    'Spelling',
    'Word_Usage',
]

LABEL_TIPS = {
    'Grammar': 'Check verb tense, subject–verb agreement, auxiliaries, and sentence agreement.',
    'Preposition_Article': 'Check articles, prepositions, pronouns, conjunctions, and modifiers.',
    'Punctuation': 'Check commas, full stops, apostrophes, capitalization, and sentence endings.',
    'Sentence_Structure': 'Check fragments, run-on sentences, word order, and clause balance.',
    'Spelling': 'Check misspelled words and typing mistakes.',
    'Word_Usage': 'Check word choice, repetition, ambiguity, and inappropriate wording.',
}

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'to', 'of', 'in', 'on', 'for', 'with', 'at', 'by', 'from', 'as', 'that', 'this', 'these', 'those',
    'it', 'its', 'their', 'there', 'they', 'them', 'he', 'she', 'his', 'her', 'we', 'our', 'you',
    'your', 'i', 'me', 'my', 'mine', 'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would',
    'can', 'could', 'should', 'may', 'might', 'must', 'not', 'so', 'because', 'than', 'then', 'very',
}


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'[\r\n]+', text.strip())
    sentences: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        sents = re.split(r'(?<=[.!?])\s+', part)
        sentences.extend([s.strip() for s in sents if s.strip()])

    return sentences


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def count_syllables(word: str) -> int:
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 1

    groups = re.findall(r'[aeiouy]+', word)
    count = len(groups)

    if word.endswith('e') and count > 1:
        count -= 1

    return max(1, count)


def detect_spelling(sentence: str, threshold_zipf: float = 2.0) -> list[str]:
    words = tokenize_words(sentence)
    suspicious: list[str] = []
    ignore_words = {'ai', 'nlp', 'bci', 'nsbm', 'chatgpt'}

    for word in words:
        lower = word.lower()

        if len(lower) <= 2:
            continue
        if lower in STOPWORDS or lower in ignore_words:
            continue

        if zipf_frequency(lower, 'en') < threshold_zipf:
            suspicious.append(word)

    return suspicious


def detect_grammar_rules(sentence: str) -> tuple[str, list[str], str] | None:
    s = sentence.strip()
    lower = s.lower()
    exceptions = {'news', 'mathematics', 'physics', 'economics'}

    # Avoid false positive: Reading books is...
    if re.match(r'^[A-Za-z]+ing\s+[A-Za-z]+s\s+is\b', s):
        return None

    match = re.search(r'^(these|those)?\s*([A-Za-z]+s)\s+is\b', lower)
    if match:
        subject = match.group(2)
        if subject not in exceptions:
            return 'Grammar', [], f"Subject–verb agreement error: '{subject} is' should be '{subject} are'."

    match = re.search(r'\b(they|we|you)\s+([a-z]+s)\b', lower)
    if match:
        subject, verb = match.group(1), match.group(2)
        if verb not in {'is', 'was', 'has', 'does'}:
            return 'Grammar', [], f"Subject–verb agreement error: '{subject} {verb}' should use the base verb form."

    match = re.search(
        r'\b(people|students|children|teachers|lecturers|animals|dogs|cats|books)\s+([a-z]+s)\b',
        lower
    )
    if match:
        subject, verb = match.group(1), match.group(2)
        if verb not in {'is', 'was', 'has', 'does'}:
            return 'Grammar', [], f"Subject–verb agreement error: '{subject} {verb}' should use the base verb form."

    match = re.search(
        r'\b(he|she|it)\s+(help|make|go|play|use|provide|create|affect|need|want|like|enjoy|show|bark|see)\b',
        lower
    )
    if match:
        subject, verb = match.group(1), match.group(2)
        return 'Grammar', [], f"Subject–verb agreement error: '{subject} {verb}' should usually use the singular verb form."

    match = re.search(r'\b(technology|education|school|learning|system|essay)\s+have\b', lower)
    if match:
        subject = match.group(1)
        return 'Grammar', [], f"Subject–verb agreement error: '{subject} have' should usually be '{subject} has'."

    match = re.search(r'\b(can|will|should|must|could|would|may|might)\s+([a-z]+s)\b', lower)
    if match:
        modal, verb = match.group(1), match.group(2)
        return 'Grammar', [], f"Verb form error: after '{modal}', use the base verb form, not '{verb}'."

    return None


def detect_article_rules(sentence: str) -> tuple[str, list[str], str] | None:
    lower = sentence.lower()

    if re.search(r'\ba\s+[aeiou]', lower):
        return 'Preposition_Article', [], "Article error: use 'an' before words beginning with a vowel sound."

    if re.search(r'\ban\s+[bcdfghjklmnpqrstvwxyz]', lower):
        return 'Preposition_Article', [], "Article error: use 'a' before words beginning with a consonant sound."

    return None


def detect_word_usage_rules(sentence: str) -> tuple[str, list[str], str] | None:
    lower = sentence.lower()

    rules = {
        'did a mistake': "Word usage error: use 'made a mistake' instead of 'did a mistake'.",
        'less mistakes': "Word usage error: use 'fewer mistakes' instead of 'less mistakes'.",
        'less people': "Word usage error: use 'fewer people' instead of 'less people'.",
        'informations': "Word usage error: 'information' is uncountable, not 'informations'.",
        'overusing of': "Word usage error: use 'overuse of' instead of 'overusing of'.",
        'many student': "Word usage error: use 'many students' instead of 'many student'.",
    }

    for phrase, reason in rules.items():
        if phrase in lower:
            return 'Word_Usage', [], reason

    return None


def detect_sentence_structure_rules(sentence: str) -> tuple[str, list[str], str] | None:
    lower = sentence.strip().lower()
    words = tokenize_words(sentence)

    if lower.startswith(('because ', 'when ', 'although ', 'while ', 'if ')) and len(words) < 8:
        return 'Sentence_Structure', [], "Sentence fragment: this dependent clause may need a main clause."

    if lower.startswith(('and ', 'but ', 'or ')) and len(words) < 8:
        return 'Sentence_Structure', [], "Sentence fragment: avoid using a short incomplete sentence beginning with a conjunction."

    # Run-on detection
    if len(words) > 18 and re.search(r'\bhowever\b|\bi enjoy\b|\btherefore\b|\bbut\b', lower):
        return 'Sentence_Structure', [], "Possible run-on sentence: consider splitting the sentence or adding proper punctuation."

    if len(words) > 45:
        return 'Sentence_Structure', [], "Sentence is very long and may be unclear. Consider splitting it."

    return None


def detect_punctuation_rules(sentence: str) -> list[tuple[str, list[str], str]]:
    issues = []
    stripped = sentence.strip()
    lower = stripped.lower()

    if stripped and not stripped.endswith(('.', '!', '?')):
        issues.append(('Punctuation', [], "Sentence may be missing proper ending punctuation."))

    if stripped and stripped[0].islower():
        issues.append(('Punctuation', [], "Capitalization issue: the sentence should begin with a capital letter."))

    if re.search(r'\benglish\b', lower):
        issues.append(('Punctuation', [], "Capitalization issue: the word 'English' should begin with a capital letter."))

    if re.search(r'\bmath science and english\b', lower):
        issues.append(('Punctuation', [], "Comma issue: use commas in a list, e.g., 'math, science, and English'."))

    if re.search(r'\bhowever\b', lower) and not re.search(r'[.;]\s*however,', lower):
        issues.append(('Punctuation', [], "Punctuation issue: use a semicolon or full stop before 'however' and a comma after it."))

    return issues


def predict_labels(sentence: str) -> list[tuple[str, list[str], str]]:
    issues: list[tuple[str, list[str], str]] = []

    for checker in [
        detect_grammar_rules,
        detect_article_rules,
        detect_word_usage_rules,
        detect_sentence_structure_rules,
    ]:
        result = checker(sentence)
        if result:
            issues.append(result)

    issues.extend(detect_punctuation_rules(sentence))

    suspicious = detect_spelling(sentence)
    if suspicious:
        issues.append(('Spelling', suspicious, f"Possible spelling mistake in: {', '.join(suspicious)}"))

    # ML fallback only if no rule-based issues were found
    if not issues:
        try:
            if hasattr(model, 'predict_proba'):
                prediction = model.predict([sentence])[0]
                probabilities = model.predict_proba([sentence])[0]
                confidence = max(probabilities)

                if prediction in LABEL_ORDER and confidence >= 0.85:
                    issues.append((prediction, [], f"Model detected a possible {prediction.replace('_', ' ').lower()} issue."))
            else:
                prediction = model.predict([sentence])[0]
                if prediction in {'Spelling', 'Word_Usage'}:
                    issues.append((prediction, [], f"Model detected a possible {prediction.replace('_', ' ').lower()} issue."))
        except Exception:
            pass

    if not issues:
        issues.append(('No_Error', [], 'No significant issues detected.'))

    return issues


def flesch_reading_ease(text: str, sentence_count: int, word_count: int) -> float:
    words = tokenize_words(text)
    syllables = sum(count_syllables(word) for word in words) or 1
    sentence_count = max(1, sentence_count)
    word_count = max(1, word_count)

    return round(
        206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count),
        2
    )


def readability_level(score: float) -> str:
    if score >= 80:
        return 'Easy'
    if score >= 60:
        return 'Standard'
    if score >= 40:
        return 'Fairly Difficult'
    return 'Difficult'


def detect_repeated_words(words: list[str]) -> list[dict[str, Any]]:
    filtered = [w.lower() for w in words if len(w) > 3 and w.lower() not in STOPWORDS]
    counts = Counter(filtered)

    return [
        {'word': word, 'count': count}
        for word, count in counts.most_common(5)
        if count >= 3
    ]


def infer_structure(paragraphs: list[str], sentence_count: int) -> dict[str, Any]:
    intro = len(paragraphs) >= 1 and len(tokenize_words(paragraphs[0])) >= 20
    conclusion_markers = ('in conclusion', 'to conclude', 'overall', 'therefore', 'thus', 'to sum up')

    conclusion = False
    if paragraphs:
        last_paragraph = paragraphs[-1].lower()
        conclusion = any(marker in last_paragraph for marker in conclusion_markers) or len(tokenize_words(paragraphs[-1])) >= 15

    body = len(paragraphs) >= 3 or sentence_count >= 6
    missing_parts = []

    if not intro:
        missing_parts.append('strong introduction')
    if not body:
        missing_parts.append('clear body development')
    if not conclusion:
        missing_parts.append('clear conclusion')

    return {
        'paragraphCount': len(paragraphs),
        'hasIntroduction': intro,
        'hasBodyDevelopment': body,
        'hasConclusion': conclusion,
        'missingParts': missing_parts,
    }


def essay_rating(error_total: int, sentence_count: int, paragraph_count: int) -> str:
    if sentence_count == 0:
        return 'Insufficient Content'

    density = error_total / max(1, sentence_count)

    if density == 0:
        return 'Good'
    if density <= 0.3:
        return 'Good'
    if density <= 0.7:
        return 'Satisfactory'
    if density <= 1.2:
        return 'Needs Improvement'
    return 'Weak'


def build_summary(
    counts: dict[str, int],
    structure: dict[str, Any],
    repeated: list[dict[str, Any]],
    rating: str,
) -> tuple[list[str], list[str], str]:

    strongest = sorted(counts.items(), key=lambda item: item[1])[:2]
    weakest = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:2]

    strengths = []
    for label, count in strongest:
        if count == 0:
            strengths.append(f'No major {label.replace("_", " ").lower()} issues were detected.')
        else:
            strengths.append(f'{label.replace("_", " ")} errors were comparatively lower than other categories.')

    improvements = []
    for label, count in weakest:
        if count > 0:
            improvements.append(f'Focus on improving {label.replace("_", " ").lower()} because {count} issue(s) were flagged.')

    if structure['missingParts'] and rating != 'Good':
        improvements.append('Improve essay organization by adding ' + ', '.join(structure['missingParts']) + '.')

    if repeated:
        repeated_words = ', '.join(item['word'] for item in repeated[:3])
        improvements.append(f'Reduce repeated vocabulary such as {repeated_words}.')

    overview = {
        'Good': 'The essay shows a good level of writing with no major issues detected.',
        'Satisfactory': 'The essay communicates its ideas, but there are noticeable writing issues to revise.',
        'Needs Improvement': 'The essay has a clear attempt at communication, but several writing problems reduce clarity.',
        'Weak': 'The essay needs major revision in language accuracy and organization.',
        'Insufficient Content': 'The submitted text is too short to evaluate meaningfully.',
    }[rating]

    return strengths[:3], improvements[:4], overview


def analyze_essay(essay: str) -> dict[str, Any]:
    essay = (essay or '').strip()

    paragraphs = split_paragraphs(essay)
    sentences = split_sentences(essay)
    words = tokenize_words(essay)

    grouped = {label: [] for label in LABEL_ORDER}

    for index, sentence in enumerate(sentences, 1):
        issues = predict_labels(sentence)

        for label, suspicious, reason in issues:
            if label == 'No_Error':
                continue

            if label not in grouped:
                grouped[label] = []

            grouped[label].append({
                'sentence_no': index,
                'sentence': sentence,
                'spelling_suspects': suspicious,
                'reason': reason,
                'suggestion': LABEL_TIPS.get(label, 'No suggestion needed.'),
            })

    grouped_non_empty = OrderedDict(
        (label, items)
        for label, items in grouped.items()
        if items
    )

    counts = {
        label: len(items)
        for label, items in grouped.items()
    }

    total_issues = sum(counts.values())
    structure = infer_structure(paragraphs, len(sentences))
    repeated = detect_repeated_words(words)

    readability_score = flesch_reading_ease(essay, len(sentences), len(words)) if essay else 0.0
    rating = essay_rating(total_issues, len(sentences), len(paragraphs))
    strengths, improvements, overview = build_summary(counts, structure, repeated, rating)

    return {
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'word_count': len(words),
        'average_sentence_length': round(len(words) / max(1, len(sentences)), 2),
        'counts': counts,
        'tips': LABEL_TIPS,
        'grouped': grouped_non_empty,
        'essay_metrics': {
            'totalIssues': total_issues,
            'readabilityScore': readability_score,
            'readabilityLevel': readability_level(readability_score),
            'lexicalDiversity': round(len({w.lower() for w in words}) / max(1, len(words)), 2),
            'repeatedWords': repeated,
        },
        'structure': structure,
        'overall_assessment': {
            'rating': rating,
            'overview': overview,
            'strengths': strengths,
            'improvements': improvements,
        },
    }


if __name__ == '__main__':
    sample = '''My favorite subjects are math science and english I enjoy learning new things and solving problems however sometimes it can be challenging'''
    from pprint import pprint
    pprint(analyze_essay(sample))