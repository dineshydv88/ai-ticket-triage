import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import joblib


INPUT_FILE = "data/tickets.csv"

CATEGORY_MODEL_FILE = "models/category_model.pkl"
URGENCY_MODEL_FILE = "models/urgency_model.pkl"
VECTORIZER_FILE = "models/vectorizer.pkl"


def main():

    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    # ==================================================
    # 1. Prepare data
    # ==================================================

    texts = df["text"].fillna("")

    category = df["category"]

    urgency = df["urgency"]

    # ==================================================
    # 2. Train/Test Split
    # ==================================================

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=category
    )

    X_train_text = texts.loc[train_idx]
    X_test_text = texts.loc[test_idx]

    y_train_category = category.loc[train_idx]
    y_test_category = category.loc[test_idx]

    y_train_urgency = urgency.loc[train_idx]
    y_test_urgency = urgency.loc[test_idx]

    # ==================================================
    # 3. TF-IDF
    # ==================================================

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)

    X_test = vectorizer.transform(X_test_text)

    print("Training matrix:", X_train.shape)
    print("Testing matrix:", X_test.shape)

    # ==================================================
    # 4. Train Category Model
    # ==================================================

    print("\nTraining category model...")

    category_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    category_model.fit(
        X_train,
        y_train_category
    )

    print("Category model trained successfully.")

    # ==================================================
    # 5. Train Urgency Model
    # ==================================================

    print("\nTraining urgency model...")

    urgency_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    urgency_model.fit(
        X_train,
        y_train_urgency
    )

    print("Urgency model trained successfully.")

    # ==================================================
    # 6. Save models
    # ==================================================

    print("\nSaving models...")

    joblib.dump(
        category_model,
        CATEGORY_MODEL_FILE
    )

    joblib.dump(
        urgency_model,
        URGENCY_MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    print("Saved:")
    print(CATEGORY_MODEL_FILE)
    print(URGENCY_MODEL_FILE)
    print(VECTORIZER_FILE)

    # ==================================================
    # 7. Basic verification
    # ==================================================

    print("\n--- MODEL INFORMATION ---")

    print(
        "Category classes:",
        category_model.classes_
    )

    print(
        "Urgency classes:",
        urgency_model.classes_
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
    