import os
import csv
from datetime import datetime

LOG_FILE = "logs/human_review.csv"


def log_human_review(ticket_text, prediction_result):
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
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

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            ticket_text,
            prediction_result["category"],
            prediction_result["category_confidence"],
            prediction_result["urgency"],
            prediction_result["urgency_confidence"],
            prediction_result["support_team"],
            "Pending Review"
        ])

    print(
        f"Human-review ticket logged to: {LOG_FILE}"
    )


def update_review_status(row_index, new_status):
    """
    Update the review status of a ticket
    in the human-review CSV.
    """

    if not os.path.exists(LOG_FILE):
        return False

    rows = []

    with open(
        LOG_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.reader(file)
        rows = list(reader)

    # Header + data rows
    if row_index < 0 or row_index >= len(rows) - 1:
        return False

    # +1 because row 0 is the CSV header
    csv_row_index = row_index + 1

    # review_status is the last column
    rows[csv_row_index][-1] = new_status

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerows(rows)

    return True


if __name__ == "__main__":
    print("Review logger module ready.")
    



