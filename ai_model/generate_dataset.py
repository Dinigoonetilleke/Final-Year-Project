import random
import csv
from pathlib import Path

out_path = Path(__file__).resolve().parents[1] / "data" / "labeled_sentences.csv"

grammar = [
    "He go to school everyday.",
    "She have a pen.",
    "They is playing football.",
    "I does my homework at night.",
    "She don't like coffee.",
    "He were late to class.",
]
punct = [
    "Hello how are you",
    "I went to the shop, and bought apples",
    "She said I am coming",
    "Its raining today",
    "We learned grammar today it was fun",
]
spelling = [
    "I cannt believe it.",
    "I recieved the message.",
    "Definately I will come.",
    "I seperated the papers.",
    "She is beautifull.",
]
structure = [
    "Where you are going?",
    "Because I was late.",
    "This essay is good it has many points and examples",
    "Went to school yesterday I.",
    "He angry very.",
]
no_error = [
    "I went to the shop.",
    "She is reading a book.",
    "We completed the assignment on time.",
    "They are playing football.",
    "This paragraph is clear and structured.",
]

rows = []
for _ in range(80):
    rows.append((random.choice(grammar), "grammar"))
for _ in range(70):
    rows.append((random.choice(punct), "punctuation"))
for _ in range(60):
    rows.append((random.choice(spelling), "spelling"))
for _ in range(60):
    rows.append((random.choice(structure), "sentence_structure"))
for _ in range(80):
    rows.append((random.choice(no_error), "no_error"))

random.shuffle(rows)

out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Generated dataset: {out_path} with {len(rows)} rows")
