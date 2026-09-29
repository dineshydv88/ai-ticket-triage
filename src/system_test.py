from predict import predict_ticket


# --------------------------------------------------
# System Test Tickets
# --------------------------------------------------

test_tickets = [
    {
        "name": "Billing Test",
        "text": (
            "I was charged twice for my subscription "
            "and there is an incorrect invoice."
        )
    },
    {
        "name": "Account Test",
        "text": (
            "I cannot access my account and need help "
            "resetting my account password."
        )
    },
    {
        "name": "Product Test",
        "text": (
            "I would like information about the product "
            "features and specifications."
        )
    },
    {
        "name": "Technical Test",
        "text": (
            "The application keeps crashing whenever "
            "I try to upload a file."
        )
    },
    {
        "name": "Other Test",
        "text": (
            "Hello, I have a general question and need "
            "some assistance."
        )
    }
]


# --------------------------------------------------
# Run Tests
# --------------------------------------------------

print("\n")
print("=" * 70)
print("AI CUSTOMER SUPPORT TICKET TRIAGE")
print("SYSTEM TEST")
print("=" * 70)


for number, ticket in enumerate(test_tickets, start=1):

    result = predict_ticket(ticket["text"])

    print("\n" + "-" * 70)
    print(f"TEST {number}: {ticket['name']}")
    print("-" * 70)

    print("Ticket:")
    print(ticket["text"])

    print("\nPrediction:")
    print("Category:", result["category"])
    print(
        "Category Confidence:",
        result["category_confidence"]
    )

    print("Urgency:", result["urgency"])
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


print("\n")
print("=" * 70)
print("SYSTEM TEST COMPLETE")
print("=" * 70)
