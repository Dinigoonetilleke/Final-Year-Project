import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "grammar_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Mapping original labels to Six Main categories
def map_label(label):
    grammar = [
        "Verb Tense Errors",
        "Subject-Verb Agreement",
        "Gerund and Participle Errors",
        "Infinitive Errors",
        "Mixed Conditionals",
        "Quantifier Errors",
        "Contractions Errors",
        "Incorrect Auxiliaries",
        "Agreement in Comparative and Superlative Forms"
    ]

    sentence_structure = [
        "Sentence Structure Errors",
        "Sentence Fragments",
        "Run-on Sentences",
        "Parallelism Errors",
        "Lack of Parallelism in Lists or Series",
        "Faulty Comparisons",
        "Relative Clause Errors"
    ]

    punctuation = [
        "Punctuation Errors",
        "Ellipsis Errors",
        "Capitalization Errors",
        "Abbreviation Errors"
    ]

    word_usage = [
        "Word Choice/Usage",
        "Clichés",
        "Redundancy/Repetition",
        "Tautology",
        "Mixed Metaphors/Idioms",
        "Inappropriate Register",
        "Slang, Jargon, and Colloquialisms",
        "Ambiguity"
    ]

    spelling = [
        "Spelling Mistakes"
    ]

    preposition_article = [
        "Preposition Usage",
        "Article Usage",
        "Pronoun Errors",
        "Modifiers Misplacement",
        "Conjunction Misuse"
    ]

    if label in grammar:
        return "Grammar"
    elif label in sentence_structure:
        return "Sentence_Structure"
    elif label in punctuation:
        return "Punctuation"
    elif label in word_usage:
        return "Word_Usage"
    elif label in spelling:
        return "Spelling"
    elif label in preposition_article:
        return "Preposition_Article"
    else:
        return None

# Mapping
df["major_label"] = df["Error Type"].apply(map_label)

# Remove the rows that did not map (if any)
df = df.dropna(subset=["major_label"])

# Keep only the needed columns
df = df[["Ungrammatical Statement", "major_label"]]
df.columns = ["text", "label"]

# Save the processed dataset
df.to_csv(OUTPUT_PATH, index=False)

print(" Processed dataset saved as processed_dataset.csv")
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nTotal samples:", len(df))
