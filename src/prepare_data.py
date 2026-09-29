import re
import pandas as pd


INPUT_FILE = "data/customer_support_tickets.csv"
OUTPUT_FILE = "data/tickets.csv"


def clean_text(text):
    """Clean basic whitespace from text."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_ticket_text(row):
    """Combine subject and body into one text field."""
    subject = "" if pd.isna(row["subject"]) else str(row["subject"])
    body = "" if pd.isna(row["body"]) else str(row["body"])

    subject = clean_text(subject)
    body = clean_text(body)

    if subject:
        return f"{subject}. {body}"

    return body


def assign_category(row):
    """
    Create the project's five category labels.

    Categories:
    Billing, Account, Product, Technical, Other

    Account is assigned only when the actual ticket text
    contains strong account-management evidence.
    """

    queue = str(row["queue"]).lower()

    # Collect all available tags
    tag_columns = [c for c in row.index if c.startswith("tag_")]

    tags = " ".join(
        str(row[c])
        for c in tag_columns
        if not pd.isna(row[c])
    ).lower()

    subject = "" if pd.isna(row["subject"]) else str(row["subject"])
    body = "" if pd.isna(row["body"]) else str(row["body"])

    text = clean_text(f"{subject} {body}").lower()

    # ==================================================
    # 1. BILLING
    # ==================================================

    if "billing and payments" in queue:
        return "Billing"

    if re.search(r"\bbilling\b", tags):
        return "Billing"

    billing_pattern = (
        r"\binvoice\b|"
        r"\binvoices\b|"
        r"\bbilling issue\b|"
        r"\bbilling problem\b|"
        r"\bpayment issue\b|"
        r"\bpayment problem\b|"
        r"\bpayment failed\b|"
        r"\bcharged incorrectly\b|"
        r"\bbilling statement\b|"
        r"\bbilling details\b"
    )

    if re.search(billing_pattern, text):
        return "Billing"

    # ==================================================
    # 2. ACCOUNT
    # ==================================================

    # IMPORTANT:
    # Do NOT use the Account tag by itself.
    # We require strong account-management language
    # in the actual ticket text.

    account_pattern = (
        r"\baccount management\b|"
        r"\baccount administration\b|"
        r"\baccount management portal\b|"
        r"\baccount administration portal\b|"
        r"\baccount access\b|"
        r"\baccount settings\b|"
        r"\baccount login\b|"
        r"\baccount password\b|"
        r"\baccount creation\b|"
        r"\bcreate an account\b|"
        r"\bcreating an account\b|"
        r"\baccess my account\b|"
        r"\bmanage my account\b|"
        r"\bunable to access (my|the) account\b"
    )

    if re.search(account_pattern, text):
        return "Account"

    # ==================================================
    # 3. PRODUCT
    # ==================================================

    if "product support" in queue:
        return "Product"

    if re.search(r"\bproduct\b", tags):
        return "Product"

    product_pattern = (
        r"\bproduct feature\b|"
        r"\bproduct features\b|"
        r"\bproduct compatibility\b|"
        r"\bproduct specifications\b|"
        r"\bproduct integration\b|"
        r"\bproduct functionality\b|"
        r"\bproduct capabilities\b"
    )

    if re.search(product_pattern, text):
        return "Product"

    # ==================================================
    # 4. TECHNICAL
    # ==================================================

    technical_queues = [
        "technical support",
        "it support",
        "service outages and maintenance"
    ]

    if queue in technical_queues:
        return "Technical"

    technical_pattern = (
        r"\btechnical issue\b|"
        r"\btechnical problem\b|"
        r"\bnetwork outage\b|"
        r"\bconnectivity issue\b|"
        r"\bconnection problem\b|"
        r"\bserver error\b|"
        r"\bsystem outage\b|"
        r"\bsoftware crash\b|"
        r"\bapplication crash\b|"
        r"\bvpn issue\b|"
        r"\bnetwork problem\b|"
        r"\bsystem failure\b"
    )

    if re.search(technical_pattern, text):
        return "Technical"

    # ==================================================
    # 5. OTHER
    # ==================================================

    return "Other"


def main():
    print("Loading source dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original dataset: {df.shape}")

    # Create combined ticket text
    df["text"] = df.apply(build_ticket_text, axis=1)

    # Create category label
    df["category"] = df.apply(assign_category, axis=1)

    # Use original priority as urgency
    df["urgency"] = df["priority"].str.lower().str.strip()

    # Keep only the columns required by the ML pipeline
    tickets = df[["text", "category", "urgency"]].copy()

    # Remove empty text
    tickets = tickets[tickets["text"].str.len() > 0]

    # Keep only valid urgency labels
    valid_urgency = {"low", "medium", "high"}
    tickets = tickets[tickets["urgency"].isin(valid_urgency)]

    # Save cleaned dataset
    tickets.to_csv(OUTPUT_FILE, index=False)

    print("\n--- CLEANED DATASET ---")
    print("Shape:", tickets.shape)

    print("\n--- CATEGORY DISTRIBUTION ---")
    print(tickets["category"].value_counts())

    print("\n--- URGENCY DISTRIBUTION ---")
    print(tickets["urgency"].value_counts())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    
