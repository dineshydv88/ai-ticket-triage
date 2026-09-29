import sys
import os

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Allow Python to find files inside src/
# --------------------------------------------------

SRC_PATH = os.path.join(
    os.path.dirname(__file__),
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


from predict import predict_ticket
from review_logger import update_review_status


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LOG_FILE = "logs/human_review.csv"
CONFIDENCE_THRESHOLD = 0.60


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎫 AI Customer Support Ticket Triage")

st.write(
    "Enter a customer support ticket below and the AI system "
    "will classify, prioritize, and route it."
)


# --------------------------------------------------
# Ticket Input
# --------------------------------------------------

st.subheader("Customer Support Ticket")

ticket_text = st.text_area(
    "Enter the customer's issue:",
    placeholder=(
        "Example: I was charged twice for my subscription "
        "and need help correcting the invoice."
    ),
    height=180
)


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

predict_button = st.button(
    "Analyze Ticket",
    type="primary"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    if not ticket_text.strip():

        st.warning(
            "Please enter a support ticket first."
        )

    else:

        result = predict_ticket(ticket_text)

        st.success(
            "Ticket analyzed successfully!"
        )

        # --------------------------------------------------
        # Prediction Results
        # --------------------------------------------------

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Category",
                result["category"]
            )

        with col2:
            st.metric(
                "Urgency",
                result["urgency"].upper()
            )

        with col3:
            st.metric(
                "Support Team",
                result["support_team"]
            )

        # --------------------------------------------------
        # Model Confidence
        # --------------------------------------------------

        st.subheader("Model Confidence")

        confidence_col1, confidence_col2 = st.columns(2)

        with confidence_col1:

            st.write("Category Confidence")

            st.progress(
                result["category_confidence"]
            )

            st.write(
                f"{result['category_confidence'] * 100:.2f}%"
            )

        with confidence_col2:

            st.write("Urgency Confidence")

            st.progress(
                result["urgency_confidence"]
            )

            st.write(
                f"{result['urgency_confidence'] * 100:.2f}%"
            )

        # --------------------------------------------------
        # Routing Decision
        # --------------------------------------------------

        st.subheader("Routing Decision")

        st.write(
            f"Confidence threshold: "
            f"**{CONFIDENCE_THRESHOLD * 100:.0f}%**"
        )

        if result["human_review"]:

            st.error(
                "⚠️ HUMAN REVIEW REQUIRED"
            )

            st.write(
                "At least one model has confidence below "
                "the 60% threshold. The ticket has not been "
                "automatically routed."
            )

        else:

            st.success(
                "✅ AUTOMATIC ROUTING APPROVED"
            )

            st.write(
                f"The ticket can be automatically routed "
                f"to **{result['support_team']}**."
            )

        # --------------------------------------------------
        # Urgency Status
        # --------------------------------------------------

        st.subheader("Urgency Status")

        urgency = result["urgency"].lower()

        if urgency == "high":

            st.error(
                "🔴 HIGH URGENCY"
            )

        elif urgency == "medium":

            st.warning(
                "🟡 MEDIUM URGENCY"
            )

        else:

            st.success(
                "🟢 LOW URGENCY"
            )


# ==================================================
# HUMAN REVIEW QUEUE
# ==================================================

st.divider()

st.subheader("Human Review Queue")


# --------------------------------------------------
# Check whether review log exists
# --------------------------------------------------

if os.path.exists(LOG_FILE):

    review_df = pd.read_csv(LOG_FILE)

    # --------------------------------------------------
    # Make sure review_status column exists
    # --------------------------------------------------

    if "review_status" not in review_df.columns:

        st.error(
            "The human-review log does not contain "
            "a review_status column."
        )

    else:

        # --------------------------------------------------
        # Get pending tickets
        # --------------------------------------------------

        pending_df = review_df[
            review_df["review_status"] == "Pending Review"
        ].copy()

        st.write(
            f"Pending review tickets: "
            f"**{len(pending_df)}**"
        )

        # --------------------------------------------------
        # Display pending tickets
        # --------------------------------------------------

        if not pending_df.empty:

            st.dataframe(
                pending_df,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------
            # Select ticket for review
            # --------------------------------------------------

            st.subheader("Review a Ticket")

            ticket_indices = pending_df.index.tolist()

            selected_ticket = st.selectbox(
                "Select a ticket to review:",
                ticket_indices,
                format_func=lambda index: (
                    f"Ticket {index + 1}: "
                    f"{str(pending_df.loc[index, 'ticket_text'])[:80]}"
                )
            )

            selected_row = pending_df.loc[
                selected_ticket
            ]

            # --------------------------------------------------
            # Selected Ticket
            # --------------------------------------------------

            st.write("### Ticket Details")

            st.write(
                selected_row["ticket_text"]
            )

            detail_col1, detail_col2, detail_col3 = st.columns(3)

            with detail_col1:

                st.write(
                    f"**Category:** "
                    f"{selected_row['category']}"
                )

            with detail_col2:

                st.write(
                    f"**Urgency:** "
                    f"{str(selected_row['urgency']).upper()}"
                )

            with detail_col3:

                st.write(
                    f"**Category Confidence:** "
                    f"{float(selected_row['category_confidence']) * 100:.2f}%"
                )

            st.write(
                f"**Urgency Confidence:** "
                f"{float(selected_row['urgency_confidence']) * 100:.2f}%"
            )

            st.write(
                f"**Current Status:** "
                f"{selected_row['review_status']}"
            )

            # --------------------------------------------------
            # Mark as Reviewed
            # --------------------------------------------------

            if st.button(
                "✅ Mark as Reviewed",
                type="primary"
            ):

                success = update_review_status(
                    selected_ticket,
                    "Reviewed"
                )

                if success:

                    st.success(
                        "Ticket marked as Reviewed."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Could not update the review status."
                    )

        else:

            st.success(
                "✅ No tickets are currently waiting "
                "for human review."
            )

else:

    st.info(
        "No human-review tickets have been logged yet."
    )
    
