import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/tickets.csv"


def main():
    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # 1. Prepare inputs and labels
    # --------------------------------------------------

    texts = df["text"].fillna("")

    category = df["category"]
    urgency = df["urgency"]

    # --------------------------------------------------
    # 2. Split BEFORE fitting TF-IDF
    # --------------------------------------------------

    # Split row indices once so that category and urgency
    # use the same training and testing tickets.

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=category
    )

    X_train_text = texts.loc[train_idx]
    X_test_text = texts.loc[test_idx]

    y_train_cat = category.loc[train_idx]
    y_test_cat = category.loc[test_idx]

    y_train_urg = urgency.loc[train_idx]
    y_test_urg = urgency.loc[test_idx]

    # --------------------------------------------------
    # 3. Create TF-IDF vectorizer
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    # Learn vocabulary ONLY from training tickets.
    X_train = vectorizer.fit_transform(X_train_text)

    # Apply the SAME vocabulary to testing tickets.
    X_test = vectorizer.transform(X_test_text)

    # --------------------------------------------------
    # 4. Display TF-IDF results
    # --------------------------------------------------

    print("\n--- TF-IDF RESULTS ---")

    print("Training matrix:", X_train.shape)
    print("Testing matrix:", X_test.shape)

    print(
        "Vocabulary size:",
        len(vectorizer.get_feature_names_out())
    )

    # --------------------------------------------------
    # 5. Display category distribution
    # --------------------------------------------------

    print("\n--- CATEGORY SPLIT ---")

    print("\nTraining:")
    print(y_train_cat.value_counts())

    print("\nTesting:")
    print(y_test_cat.value_counts())

    # --------------------------------------------------
    # 6. Display urgency distribution
    # --------------------------------------------------

    print("\n--- URGENCY SPLIT ---")

    print("\nTraining:")
    print(y_train_urg.value_counts())

    print("\nTesting:")
    print(y_test_urg.value_counts())

    # --------------------------------------------------
    # 7. Verify no overlapping tickets by row index
    # --------------------------------------------------

    overlap = set(train_idx).intersection(set(test_idx))

    print("\n--- DATA LEAKAGE CHECK ---")
    print("Overlapping row indices:", len(overlap))


if __name__ == "__main__":
    main()
    

