from pathlib import Path

import joblib

from preprocessing import clean_text

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "ticket_classifier.pkl"


def load_model(path=MODEL_PATH):
    if not path.exists():
        from train import train_model

        train_model()
    return joblib.load(path)


def predict_ticket(description):
    cleaned_text = clean_text(description)
    if not cleaned_text:
        raise ValueError("Please enter a ticket description.")

    model = load_model()
    prediction = model.predict([cleaned_text])[0]
    confidence = max(model.predict_proba([cleaned_text])[0]) * 100
    return prediction, confidence


if __name__ == "__main__":
    print("=" * 50)
    print("Customer Support Classifier")
    print("=" * 50)

    description = input("Enter Ticket Description: ").strip()

    if not description:
        print("Please enter a ticket description.")
    else:
        category, confidence = predict_ticket(description)
        print(f"\nPredicted Category: {category}")
        print(f"Confidence: {confidence:.2f}%")
