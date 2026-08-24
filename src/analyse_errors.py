from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed_tickets.csv")
QUEUE_MODEL_PATH = Path("models/queue_model.joblib")
PRIORITY_MODEL_PATH = Path("models/priority_model.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def analyse_model(model_name, model, test_tickets, target_column):
    test_text = test_tickets["text"]
    actual_labels = test_tickets[target_column].to_numpy()

    predicted_labels = model.predict(test_text)
    probabilities = model.predict_proba(test_text)

    class_names = model.named_steps["classifier"].classes_

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=class_names
    )

    confusions = []

    for actual_index, actual_class in enumerate(class_names):
        for predicted_index, predicted_class in enumerate(class_names):
            if actual_class == predicted_class:
                continue

            count = matrix[actual_index, predicted_index]

            if count > 0:
                confusions.append(
                    (count, actual_class, predicted_class)
                )

    confusions.sort(reverse=True)

    print(f"\n{model_name}")
    print("=" * len(model_name))

    print("\nMost common confusions:")

    for count, actual_class, predicted_class in confusions[:10]:
        print(
            f"Actual: {actual_class:<35} "
            f"Predicted: {predicted_class:<35} "
            f"Count: {count}"
        )

    incorrect_positions = np.where(
        predicted_labels != actual_labels
    )[0]

    incorrect_confidences = probabilities[
        incorrect_positions
    ].max(axis=1)

    ranked_positions = incorrect_positions[
        np.argsort(incorrect_confidences)[::-1]
    ]

    print("\nHighest-confidence incorrect predictions:")

    for position in ranked_positions[:5]:
        ticket = test_tickets.iloc[position]
        confidence = probabilities[position].max()

        preview = (
            ticket["text"]
            .replace("\n", " ")
            .replace("\\n", " ")
        )

        print("\n" + "-" * 80)
        print(f"Actual: {ticket[target_column]}")
        print(f"Predicted: {predicted_labels[position]}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Text: {preview[:400]}")


required_paths = [
    DATA_PATH,
    QUEUE_MODEL_PATH,
    PRIORITY_MODEL_PATH,
]

for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {required_path}"
        )

tickets = pd.read_csv(DATA_PATH)

_, test_tickets = train_test_split(
    tickets,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=tickets["queue"]
)

test_tickets = test_tickets.reset_index(drop=True)

queue_model = joblib.load(QUEUE_MODEL_PATH)
priority_model = joblib.load(PRIORITY_MODEL_PATH)

analyse_model(
    model_name="Routing model error analysis",
    model=queue_model,
    test_tickets=test_tickets,
    target_column="queue"
)

analyse_model(
    model_name="Priority model error analysis",
    model=priority_model,
    test_tickets=test_tickets,
    target_column="priority"
)