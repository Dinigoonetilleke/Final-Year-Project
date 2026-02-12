import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed_dataset.csv"
MODEL_OUT = Path(__file__).resolve().parent / "essay_error_model.joblib"


def build_pipeline():
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(lowercase=True)),
        ("char_tfidf", TfidfVectorizer(lowercase=True)),
    ])

    pipe = ImbPipeline([
        ("features", features),
        ("sampler", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=6000))
    ])
    return pipe


def main():
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    pipe = build_pipeline()

    # Parameter grid (kept small enough to run on a laptop)
    param_grid = {
        # word features
        "features__word_tfidf__analyzer": ["word"],
        "features__word_tfidf__ngram_range": [(1, 1), (1, 2)],
        "features__word_tfidf__min_df": [1, 2],

        # char features (best for spelling/punctuation)
        "features__char_tfidf__analyzer": ["char"],
        "features__char_tfidf__ngram_range": [(2, 5), (2, 6), (3, 6)],
        "features__char_tfidf__min_df": [1, 2],

        # sampler choices
        "sampler": [
            SMOTE(random_state=42),
            RandomOverSampler(random_state=42),
        ],

        # classifier
        "clf__C": [1.0, 2.0, 4.0, 8.0],
        "clf__class_weight": [None, "balanced"],
        "clf__solver": ["liblinear"],  # stable for multinomial-ish setups
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",   # good when classes are imbalanced
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\n Best Params:")
    print(grid.best_params_)
    print("\n Best CV f1_macro:", grid.best_score_)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)

    print("\n=== Test Classification Report ===")
    print(classification_report(y_test, pred, zero_division=0))

    print("\n=== Test Confusion Matrix (rows=true, cols=pred) ===")
    labels = sorted(df["label"].unique())
    print("Labels order:", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    joblib.dump(best_model, MODEL_OUT)
    print(f"\n Saved BEST model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
