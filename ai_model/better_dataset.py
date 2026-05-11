import pandas as pd
from pathlib import Path

# Base datasets
base = Path(__file__).resolve().parents[1] / "data" / "processed_dataset.csv"
spelling = Path(__file__).resolve().parent / "spelling_augmented.csv"
no_error = Path(__file__).resolve().parent / "clean_no_error_fixed.csv"

# Load datasets
df1 = pd.read_csv(base)
df2 = pd.read_csv(spelling)
df3 = pd.read_csv(no_error)
df3 = df3[["text", "label"]]

# Ensure correct format
df2 = df2[["text", "label"]]
df3 = df3[["text", "label"]] 

# Combine datasets
df = pd.concat([df1, df2, df3])  

# Clean dataset

# Keep all No_Error rows (important)
df_no_error = df[df["label"] == "No_Error"]

# Remove duplicates from other labels
df_other = df[df["label"] != "No_Error"]
df_other = df_other.drop_duplicates()

# Combine back
df = pd.concat([df_no_error, df_other])

# Remove very short text
df = df[df["text"].str.len() > 5]

# Save
df.to_csv("better_dataset.csv", index=False)

# Show distribution
print(df["label"].value_counts())
print("better_dataset created")