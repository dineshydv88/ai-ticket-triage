import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


INPUT_FILE = "data/tickets.csv"

CATEGORY_MODEL_FILE = "models/category_model.pkl"
URGENCY_MODEL_FILE = "models/urgency_model.pkl"
VECTORIZER_FILE = "models/vectorizer.pkl"


def main():

    print("Loading dataset and trained models...")

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------
    # Load trained models
    # --------------------------------------------------

    category_model = joblib.load(
        CATEGORY_MODEL_FILE
    )

    urgency_model = joblib.load(
        URGENCY_MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    texts = df["text"].fillna("")

    category = df["category"]

    urgency = df["urgency"]

    # --------------------------------------------------
    # Recreate the same test split
    # --------------------------------------------------

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=category
    )

    X_test_text = texts.loc[test_idx]

    y_test_category = category.loc[test_idx]

    y_test_urgency = urgency.loc[test_idx]

    # --------------------------------------------------
    # Transform test text using saved TF-IDF
    # --------------------------------------------------

    X_test = vectorizer.transform(
        X_test_text
    )

    # ==================================================
    # CATEGORY EVALUATION
    # ==================================================

    print("\n========================================")
    print("CATEGORY MODEL EVALUATION")
    print("========================================")

    category_predictions = category_model.predict(
        X_test
    )

    category_accuracy = accuracy_score(
        y_test_category,
        category_predictions
    )

    print(
        f"\nAccuracy: {category_accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_category,
            category_predictions,
            labels=category_model.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    category_cm = confusion_matrix(
        y_test_category,
        category_predictions,
        labels=category_model.classes_
    )

    print(category_cm)

    # ==================================================
    # URGENCY EVALUATION
    # ==================================================

    print("\n========================================")
    print("URGENCY MODEL EVALUATION")
    print("========================================")

    urgency_predictions = urgency_model.predict(
        X_test
    )

    urgency_accuracy = accuracy_score(
        y_test_urgency,
        urgency_predictions
    )

    print(
        f"\nAccuracy: {urgency_accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_urgency,
            urgency_predictions,
            labels=urgency_model.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    urgency_cm = confusion_matrix(
        y_test_urgency,
        urgency_predictions,
        labels=urgency_model.classes_
    )

    print(urgency_cm)

    # ==================================================
    # SUMMARY
    # ==================================================

    print("\n========================================")
    print("EVALUATION SUMMARY")
    print("========================================")

    print(
        f"Category Accuracy: {category_accuracy:.4f}"
    )

    print(
        f"Urgency Accuracy:  {urgency_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
    