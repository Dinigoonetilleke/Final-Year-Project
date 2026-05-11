import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

DATASETS = {
    "old_processed": Path(__file__).resolve().parents[1] / "data" / "processed_dataset.csv",
    "master": Path(__file__).resolve().parent / "master_dataset.csv",
    "final_v2": Path(__file__).resolve().parent / "final_dataset_v2.csv",
}

features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, lowercase=True)),
    ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2, lowercase=True))
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=6000, C=4.0, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced")
}

for dataset_name, path in DATASETS.items():
    print("\n==============================")
    print("DATASET:", dataset_name)
    print("==============================")

    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    print("Rows:", len(df))
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    for model_name, clf in models.items():
        pipe = Pipeline([
            ("features", features),
            ("clf", clf)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        print("\nMODEL:", model_name)
        print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2))
        print(classification_report(y_test, pred, zero_division=0))