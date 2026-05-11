import pandas as pd

sentences = [
    "Reading books improves vocabulary.",
    "Students should complete their homework on time.",
    "Technology helps people communicate easily.",
    "Teachers provide feedback to students.",
    "English is important for academic writing.",
    "A good essay has a clear introduction.",
    "Students can improve through regular practice.",
    "The teacher explained the lesson clearly.",
    "Many students enjoy reading short stories.",
    "Writing essays helps students organize ideas."
]

data = []

for i in range(30):
    for sentence in sentences:
        data.append({
            "text": sentence,
            "label": "No_Error"
        })

df = pd.DataFrame(data)
df.to_csv("clean_no_error_fixed.csv", index=False)

print("Created clean_no_error_fixed.csv")
print(df["label"].value_counts())