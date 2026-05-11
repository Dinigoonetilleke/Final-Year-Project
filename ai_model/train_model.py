import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC


DATA_PATH = Path(__file__).resolve().parent / "better_dataset.csv"
MODEL_OUT = Path(__file__).resolve().parent / "essay_error_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    print("Using dataset:", DATA_PATH)
    print("Total rows:", len(df))
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            lowercase=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            lowercase=True
        ))
    ])

    model = ImbPipeline([
        ("features", features),


        ("clf", LinearSVC(class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, pred)
    print("\n=== Accuracy ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")

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
