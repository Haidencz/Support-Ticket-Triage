from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed_tickets.csv")
MODELS_PATH = Path("models")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_model():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_model(model_name, model, test_text, test_labels):
    predictions = model.predict(test_text)

    accuracy = accuracy_score(
        test_labels,
        predictions
    )

    macro_f1 = f1_score(
        test_labels,
        predictions,
        average="macro"
    )

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            digits=3,
            zero_division=0
        )
    )


if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Processed dataset not found. "
        "Run src/prepare_data.py first."
    )

tickets = pd.read_csv(DATA_PATH)

print(f"Loaded {len(tickets)} processed tickets.")

# We use one shared split so both models are evaluated on the same tickets.
train_tickets, test_tickets = train_test_split(
    tickets,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=tickets["queue"]
)

train_text = train_tickets["text"]
test_text = test_tickets["text"]

print(f"Training tickets: {len(train_tickets)}")
print(f"Test tickets: {len(test_tickets)}")

print("\nTraining routing model...")

queue_model = create_model()

queue_model.fit(
    train_text,
    train_tickets["queue"]
)

evaluate_model(
    "Routing model",
    queue_model,
    test_text,
    test_tickets["queue"]
)

print("\nTraining priority model...")

priority_model = create_model()

priority_model.fit(
    train_text,
    train_tickets["priority"]
)

evaluate_model(
    "Priority model",
    priority_model,
    test_text,
    test_tickets["priority"]
)

MODELS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    queue_model,
    MODELS_PATH / "queue_model.joblib"
)

joblib.dump(
    priority_model,
    MODELS_PATH / "priority_model.joblib"
)

print("\nModels saved:")
print(MODELS_PATH / "queue_model.joblib")
print(MODELS_PATH / "priority_model.joblib")