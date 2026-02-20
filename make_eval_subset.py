import pandas as pd

# Load dataset
df = pd.read_csv("data/eval.csv")

# Remove missing rows
df = df.dropna(subset=["input", "target"])

# Random 1000 samples
df = df.sample(1000, random_state=42)

# Create Grammar dataset (incorrect sentences)
grammar_df = pd.DataFrame({
    "text": df["input"].astype(str),
    "label": "Grammar"
})

# Create No_Error dataset (correct sentences)
no_error_df = pd.DataFrame({
    "text": df["target"].astype(str),
    "label": "No_Error"
})

# Save them
grammar_df.to_csv("data/eval_grammar_1000.csv", index=False)
no_error_df.to_csv("data/eval_no_error_1000.csv", index=False)

print("Saved 1000 Grammar + 1000 No_Error samples.")
