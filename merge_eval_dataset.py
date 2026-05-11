import pandas as pd

base = pd.read_csv("data/final_dataset.csv")
grammar = pd.read_csv("data/eval_grammar_1000.csv")
no_error = pd.read_csv("data/eval_no_error_1000.csv")

df = pd.concat([base, grammar, no_error], ignore_index=True)

# Clean
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/final_dataset_v2.csv", index=False)

print("New dataset size:", len(df))
print(df["label"].value_counts())

