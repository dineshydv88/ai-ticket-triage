import os
import csv

LOG_FILE = "logs/human_review.csv"


# --------------------------------------------------
# Reset Human Review Log
# --------------------------------------------------

def reset_review_log():

    os.makedirs("logs", exist_ok=True)

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "ticket_text",
            "category",
            "category_confidence",
            "urgency",
            "urgency_confidence",
            "support_team",
            "review_status"
        ])

    print(
        "Human-review log has been reset successfully."
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    reset_review_log()
    