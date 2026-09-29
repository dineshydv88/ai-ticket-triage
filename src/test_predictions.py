from predict import predict_ticket


test_tickets = [
    "I cannot access my account and need help logging in.",

    "The application keeps crashing whenever I try to upload a file.",

    "I would like more information about your products and services.",

    "There is a problem with my invoice and I think I was charged incorrectly.",

    "Hello, I have a question and need some assistance."
]


print("\n========================================")
print("MULTIPLE TICKET PREDICTION TEST")
print("========================================")


for number, ticket in enumerate(test_tickets, start=1):

    result = predict_ticket(ticket)

    print(f"\n--- TICKET {number} ---")

    print("Text:", ticket)

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
    
