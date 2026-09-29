ROUTING_MAP = {
    "Billing": "Billing Support",
    "Account": "Account Support",
    "Product": "Product Support",
    "Technical": "Technical Support",
    "Other": "General Support"
}


def get_support_team(category):
    """
    Return the appropriate support team
    for a predicted ticket category.
    """

    return ROUTING_MAP.get(
        category,
        "General Support"
    )


if __name__ == "__main__":

    print("--- ROUTING TEST ---")

    test_categories = [
        "Billing",
        "Account",
        "Product",
        "Technical",
        "Other"
    ]

    for category in test_categories:

        team = get_support_team(category)

        print(
            f"{category} -> {team}"
        )
        