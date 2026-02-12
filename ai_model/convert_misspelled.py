import pandas as pd
import random

# Load dataset
df = pd.read_csv("data/misspelled_dataset.csv")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit number of spelling samples
LIMIT = 1200
df = df.head(LIMIT)

templates = [
    "I saw {} yesterday.",
    "She wrote {} in her notebook.",
    "He typed {} in the message.",
    "This is {} example.",
    "They said {} in class.",
    "The word {} is wrong."
]

sentences = []

for _, row in df.iterrows():
    wrong_word = str(row["input"]).strip()
    sentence = random.choice(templates).format(wrong_word)
    sentences.append({
        "text": sentence,
        "label": "Spelling"
    })

new_df = pd.DataFrame(sentences)

new_df.to_csv("data/spelling_augmented.csv", index=False)

print("Created spelling_augmented.csv with", len(new_df), "rows")
