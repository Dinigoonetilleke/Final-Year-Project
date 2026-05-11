import pandas as pd

VALID_LABELS = [
    "Grammar",
    "Spelling",
    "Punctuation",
    "Sentence_Structure",
    "Word_Usage",
    "Preposition_Article",
    "No_Error"
]

df1 = pd.read_csv("final_dataset_v2.csv")[["text", "label"]]
df4 = pd.read_csv("labeled_sentences.csv")[["text", "label"]]
df5 = pd.read_csv("spelling_augmented.csv")[["text", "label"]]

df2 = pd.read_csv("misspelled_dataset.csv")
df2 = df2[["input", "label"]]
df2.columns = ["text", "label"]

df3 = pd.read_csv("grammar_dataset.csv")
df3 = df3[["Ungrammatical Statement", "Error Type"]]
df3.columns = ["text", "label"]

df = pd.concat([df1, df2, df3, df4, df5])

# Clean spaces
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Remove very short/noisy sentences
df = df[df["text"].str.len() > 5]

# Remove weak No_Error samples (IMPORTANT FIX)
df = df[~((df["label"] == "No_Error") & (df["text"].str.contains(" go ", case=False)))]
df = df[~((df["label"] == "No_Error") & (df["text"].str.contains(" dont ", case=False)))]
df = df[~((df["label"] == "No_Error") & (df["text"].str.contains(" is are ", case=False)))]

# Keep only valid labels
df = df[df["label"].isin(VALID_LABELS)]

# Remove empty and duplicate rows
df = df[df["text"] != ""]
df = df.drop_duplicates()

# Save
df.to_csv("master_dataset.csv", index=False)

# Print distribution
print("\nClean Dataset Distribution:\n")
print(df["label"].value_counts())

print("\nClean master_dataset.csv created successfully!")