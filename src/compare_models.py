import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


INPUT_FILE = "data/tickets.csv"


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    texts = df["text"].fillna("")
    category = df["category"]
    urgency = df["urgency"]

    # --------------------------------------------------
    # Same train/test split used previously
    # --------------------------------------------------

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

    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

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
    # CATEGORY MODELS
    # ==================================================

    print("\n========================================")
    print("CATEGORY MODEL COMPARISON")
    print("========================================")

    # Logistic Regression
    print("\nTraining Logistic Regression...")

    logistic_category = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    logistic_category.fit(
        X_train,
        y_train_category
    )

    logistic_category_pred = logistic_category.predict(
        X_test
    )

    logistic_category_accuracy = accuracy_score(
        y_test_category,
        logistic_category_pred
    )

    logistic_category_f1 = f1_score(
        y_test_category,
        logistic_category_pred,
        average="macro"
    )

    # Linear SVM
    print("Training Linear SVM...")

    svm_category = LinearSVC(
        class_weight="balanced",
        random_state=42
    )

    svm_category.fit(
        X_train,
        y_train_category
    )

    svm_category_pred = svm_category.predict(
        X_test
    )

    svm_category_accuracy = accuracy_score(
        y_test_category,
        svm_category_pred
    )

    svm_category_f1 = f1_score(
        y_test_category,
        svm_category_pred,
        average="macro"
    )

    print("\nCategory Results:")
    print(
        f"Logistic Regression | "
        f"Accuracy: {logistic_category_accuracy:.4f} | "
        f"Macro F1: {logistic_category_f1:.4f}"
    )

    print(
        f"Linear SVM          | "
        f"Accuracy: {svm_category_accuracy:.4f} | "
        f"Macro F1: {svm_category_f1:.4f}"
    )

    # ==================================================
    # URGENCY MODELS
    # ==================================================

    print("\n========================================")
    print("URGENCY MODEL COMPARISON")
    print("========================================")

    # Logistic Regression
    print("\nTraining Logistic Regression...")

    logistic_urgency = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    logistic_urgency.fit(
        X_train,
        y_train_urgency
    )

    logistic_urgency_pred = logistic_urgency.predict(
        X_test
    )

    logistic_urgency_accuracy = accuracy_score(
        y_test_urgency,
        logistic_urgency_pred
    )

    logistic_urgency_f1 = f1_score(
        y_test_urgency,
        logistic_urgency_pred,
        average="macro"
    )

    # Linear SVM
    print("Training Linear SVM...")

    svm_urgency = LinearSVC(
        class_weight="balanced",
        random_state=42
    )

    svm_urgency.fit(
        X_train,
        y_train_urgency
    )

    svm_urgency_pred = svm_urgency.predict(
        X_test
    )

    svm_urgency_accuracy = accuracy_score(
        y_test_urgency,
        svm_urgency_pred
    )

    svm_urgency_f1 = f1_score(
        y_test_urgency,
        svm_urgency_pred,
        average="macro"
    )

    print("\nUrgency Results:")
    print(
        f"Logistic Regression | "
        f"Accuracy: {logistic_urgency_accuracy:.4f} | "
        f"Macro F1: {logistic_urgency_f1:.4f}"
    )

    print(
        f"Linear SVM          | "
        f"Accuracy: {svm_urgency_accuracy:.4f} | "
        f"Macro F1: {svm_urgency_f1:.4f}"
    )

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
    