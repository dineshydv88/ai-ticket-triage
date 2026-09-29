import pandas as pd


INPUT_FILE = "data/tickets.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    print("Dataset loaded.")
    print("Shape:", df.shape)

    # --------------------------------------------------
    # Overall urgency distribution
    # --------------------------------------------------

    print("\n--- URGENCY DISTRIBUTION ---")
    print(df["urgency"].value_counts())

    # --------------------------------------------------
    # Average text length by urgency
    # --------------------------------------------------

    df["text_length"] = df["text"].str.len()

    print("\n--- AVERAGE TEXT LENGTH ---")

    print(
        df.groupby("urgency")["text_length"]
        .mean()
        .round(2)
    )

    # --------------------------------------------------
    # Sample tickets from each urgency class
    # --------------------------------------------------

    for urgency_level in ["low", "medium", "high"]:

        print("\n========================================")
        print(f"{urgency_level.upper()} TICKETS")
        print("========================================")

        samples = (
            df[df["urgency"] == urgency_level]
            .sample(
                n=min(5, len(df[df["urgency"] == urgency_level])),
                random_state=42
            )
        )

        for i, text in enumerate(samples["text"], start=1):

            print(f"\n{i}. {text[:500]}")

    # --------------------------------------------------
    # Urgency vs category
    # --------------------------------------------------

    print("\n========================================")
    print("URGENCY BY CATEGORY")
    print("========================================")

    cross_tab = pd.crosstab(
        df["category"],
        df["urgency"],
        normalize="index"
    ) * 100

    print(cross_tab.round(2))


if __name__ == "__main__":
    main()
    