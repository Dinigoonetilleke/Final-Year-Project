import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_dataset.csv"
MODEL_OUT = Path(__file__).resolve().parent / "essay_error_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Combine word-level + character-level TF-IDF
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2
        ))
    ])

    model = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            max_iter=4000,
            n_jobs=None  # keep safe on Windows
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, pred, zero_division=0))

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    labels = sorted(df["label"].unique())
    print("Labels order:", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    joblib.dump(model, MODEL_OUT)
    print(f"\n Saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()
