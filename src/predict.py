import joblib

from routing import get_support_team
from review_logger import log_human_review


CATEGORY_MODEL_FILE = "models/category_model.pkl"
URGENCY_MODEL_FILE = "models/urgency_model.pkl"
VECTORIZER_FILE = "models/vectorizer.pkl"

CONFIDENCE_THRESHOLD = 0.60


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


def predict_ticket(ticket_text):
    """
    Predict category, urgency, confidence,
    support team, and human-review status.
    """

    # --------------------------------------------------
    # 1. Convert ticket text to TF-IDF
    # --------------------------------------------------

    X = vectorizer.transform(
        [ticket_text]
    )

    # --------------------------------------------------
    # 2. Category prediction
    # --------------------------------------------------

    category_prediction = category_model.predict(X)[0]

    category_probabilities = (
        category_model.predict_proba(X)[0]
    )

    category_confidence = (
        category_probabilities.max()
    )

    # --------------------------------------------------
    # 3. Urgency prediction
    # --------------------------------------------------

    urgency_prediction = urgency_model.predict(X)[0]

    urgency_probabilities = (
        urgency_model.predict_proba(X)[0]
    )

    urgency_confidence = (
        urgency_probabilities.max()
    )

    # --------------------------------------------------
    # 4. Human-review decision
    # --------------------------------------------------

    human_review = (
        category_confidence < CONFIDENCE_THRESHOLD
        or urgency_confidence < CONFIDENCE_THRESHOLD
    )

    # --------------------------------------------------
    # 5. Routing decision
    # --------------------------------------------------

    if human_review:
        support_team = "Human Review"
    else:
        support_team = get_support_team(
            category_prediction
        )

    # --------------------------------------------------
    # 6. Complete result
    # --------------------------------------------------

    result = {
        "category": category_prediction,

        "category_confidence": round(
            float(category_confidence),
            4
        ),

        "urgency": urgency_prediction,

        "urgency_confidence": round(
            float(urgency_confidence),
            4
        ),

        "support_team": support_team,

        "human_review": human_review
    }

    # --------------------------------------------------
    # 7. Log low-confidence tickets
    # --------------------------------------------------

    if human_review:

        log_human_review(
            ticket_text,
            result
        )

    return result


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

if __name__ == "__main__":

    test_ticket = (
        "I cannot access my account and "
        "need help logging in."
    )

    result = predict_ticket(
        test_ticket
    )

    print("\n--- COMPLETE TICKET RESULT ---")

    print(
        "Category:",
        result["category"]
    )

    print(
        "Category Confidence:",
        result["category_confidence"]
    )

    print(
        "Urgency:",
        result["urgency"]
    )

    print(
        "Urgency Confidence:",
        result["urgency_confidence"]
    )

    print(
        "Support Team:",
        result["support_team"]
    )

    print(
        "Human Review Required:",
        result["human_review"]
    )
    


