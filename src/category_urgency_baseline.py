import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


INPUT_FILE = "data/tickets.csv"


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    category = df[["category"]]
    urgency = df["urgency"]

    # --------------------------------------------------
    # Same split used throughout the project
    # --------------------------------------------------

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=df["category"]
    )

    X_train_category = category.loc[train_idx]
    X_test_category = category.loc[test_idx]

    y_train = urgency.loc[train_idx]
    y_test = urgency.loc[test_idx]

    # --------------------------------------------------
    # Convert category into numerical features
    # --------------------------------------------------

    encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    X_train = encoder.fit_transform(
        X_train_category
    )

    X_test = encoder.transform(
        X_test_category
    )

    # --------------------------------------------------
    # Train urgency model using category only
    # --------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========================================")
    print("CATEGORY → URGENCY BASELINE")
    print("========================================")

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
    