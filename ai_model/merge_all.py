import pandas as pd

main = pd.read_csv("data/processed_dataset.csv")
spell = pd.read_csv("data/spelling_augmented.csv")

combined = pd.concat([main, spell], ignore_index=True)
combined = combined.drop_duplicates(subset=["text"])

combined.to_csv("data/final_dataset.csv", index=False)

print("\nFinal dataset size:", len(combined))
print("\nLabel distribution:")
print(combined["label"].value_counts())

