from functools import lru_cache
from pathlib import Path
from pprint import pprint

import joblib


QUEUE_MODEL_PATH = Path("models/queue_model.joblib")
PRIORITY_MODEL_PATH = Path("models/priority_model.joblib")
REVIEW_THRESHOLD = 0.55


@lru_cache(maxsize=1)
def load_models():
    if not QUEUE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Queue model not found: {QUEUE_MODEL_PATH}"
        )

    if not PRIORITY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Priority model not found: {PRIORITY_MODEL_PATH}"
        )

    queue_model = joblib.load(QUEUE_MODEL_PATH)
    priority_model = joblib.load(PRIORITY_MODEL_PATH)

    return queue_model, priority_model


def combine_ticket_text(subject, body):
    subject = subject.strip()
    body = body.strip()

    text = f"{subject}\n\n{body}".strip()

    if not text:
        raise ValueError(
            "A ticket must contain a subject or description."
        )

    return text


def get_ranked_predictions(model, text):
    probabilities = model.predict_proba([text])[0]
    class_names = model.named_steps["classifier"].classes_

    predictions = [
        {
            "label": class_name,
            "confidence": float(probability),
        }
        for class_name, probability in zip(
            class_names,
            probabilities
        )
    ]

    return sorted(
        predictions,
        key=lambda prediction: prediction["confidence"],
        reverse=True
    )


def predict_ticket(subject, body):
    queue_model, priority_model = load_models()

    text = combine_ticket_text(
        subject,
        body
    )

    queue_predictions = get_ranked_predictions(
        queue_model,
        text
    )

    priority_predictions = get_ranked_predictions(
        priority_model,
        text
    )

    predicted_queue = queue_predictions[0]
    predicted_priority = priority_predictions[0]

    requires_review = (
        predicted_queue["confidence"] < REVIEW_THRESHOLD
        or predicted_priority["confidence"] < REVIEW_THRESHOLD
    )

    return {
        "predicted_queue": predicted_queue,
        "alternative_queues": queue_predictions[1:3],
        "predicted_priority": predicted_priority,
        "requires_review": requires_review,
    }


if __name__ == "__main__":
    example_prediction = predict_ticket(
        subject="Company VPN unavailable",
        body=(
            "Our entire remote team has lost access to the company VPN. "
            "Restarting devices and resetting passwords did not resolve "
            "the problem, and employees cannot access critical systems."
        )
    )

    pprint(example_prediction)