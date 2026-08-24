import pandas as pd
import streamlit as st

from src.database import (
    get_ticket,
    get_tickets,
    initialise_database,
    review_ticket,
    save_ticket,
)
from src.predict import predict_ticket


SUPPORT_QUEUES = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]

PRIORITIES = [
    "low",
    "medium",
    "high",
]


st.set_page_config(
    page_title="Support Ticket Triage",
    page_icon="🎫",
    layout="wide",
)

initialise_database()

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 0.75rem;
            padding: 1rem;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.75rem;
            padding: 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_prediction(ticket):
    queue_confidence = ticket["queue_confidence"]
    priority_confidence = ticket["priority_confidence"]

    queue_column, priority_column, status_column = st.columns(3)

    with queue_column:
        st.metric(
            "Predicted queue",
            ticket["predicted_queue"],
            f"{queue_confidence:.1%} confidence",
        )

    with priority_column:
        st.metric(
            "Predicted priority",
            ticket["predicted_priority"].title(),
            f"{priority_confidence:.1%} confidence",
        )

    with status_column:
        if ticket["requires_review"]:
            status_label = "Human review"
        else:
            status_label = ticket["status"].replace(
                "_",
                " ",
            ).title()

        st.metric(
            "Routing status",
            status_label,
        )

    st.write("Queue confidence")
    st.progress(queue_confidence)

    st.write("Priority confidence")
    st.progress(priority_confidence)

    if ticket["requires_review"]:
        st.warning(
            "This prediction requires human review because at least "
            "one confidence score is below the 55% threshold."
        )
    else:
        st.success(
            "The confidence threshold was met and the ticket can be "
            "automatically routed."
        )

    alternatives = ticket["alternative_queues"]

    if alternatives:
        alternative_data = pd.DataFrame(
            [
                {
                    "Alternative queue": prediction["label"],
                    "Confidence": prediction["confidence"],
                }
                for prediction in alternatives
            ]
        )

        st.subheader("Alternative queues")

        st.dataframe(
            alternative_data,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
            },
            hide_index=True,
            width="stretch",
        )


def submission_page():
    st.title("Support Ticket Triage")
    st.write(
        "Submit a customer-support ticket to predict its routing "
        "queue and priority."
    )

    with st.form("ticket_submission_form"):
        subject = st.text_input(
            "Subject",
            placeholder="For example: Company VPN unavailable",
        )

        body = st.text_area(
            "Ticket description",
            placeholder=(
                "Describe the customer's issue, its impact and any "
                "troubleshooting already attempted."
            ),
            height=220,
        )

        submitted = st.form_submit_button(
            "Analyse and save ticket",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            prediction = predict_ticket(
                subject,
                body,
            )

            ticket_id = save_ticket(
                subject,
                body,
                prediction,
            )

            st.session_state["latest_ticket_id"] = ticket_id

        except ValueError as error:
            st.error(str(error))

    latest_ticket_id = st.session_state.get(
        "latest_ticket_id"
    )

    if latest_ticket_id is not None:
        ticket = get_ticket(latest_ticket_id)

        if ticket is not None:
            st.divider()
            st.success(
                f"Ticket #{ticket['id']} was analysed and saved."
            )

            display_prediction(ticket)


def review_page():
    st.title("Human Review Queue")
    st.write(
        "Review low-confidence predictions before tickets are routed."
    )

    pending_tickets = get_tickets(
        limit=500,
        review_only=True,
    )

    st.metric(
        "Tickets awaiting review",
        len(pending_tickets),
    )

    if not pending_tickets:
        st.success("There are currently no tickets awaiting review.")
        return

    ticket_options = {
        ticket["id"]: ticket
        for ticket in pending_tickets
    }

    selected_ticket_id = st.selectbox(
        "Select a ticket",
        options=list(ticket_options.keys()),
        format_func=lambda ticket_id: (
            f"#{ticket_id} — "
            f"{ticket_options[ticket_id]['subject'] or 'No subject'}"
        ),
    )

    selected_ticket = ticket_options[selected_ticket_id]

    st.subheader(f"Ticket #{selected_ticket['id']}")

    st.markdown("**Subject**")
    st.write(selected_ticket["subject"] or "No subject provided")

    st.markdown("**Description**")
    st.write(selected_ticket["body"])

    st.divider()

    display_prediction(selected_ticket)

    predicted_queue = selected_ticket["predicted_queue"]
    predicted_priority = selected_ticket["predicted_priority"]

    queue_index = (
        SUPPORT_QUEUES.index(predicted_queue)
        if predicted_queue in SUPPORT_QUEUES
        else 0
    )

    priority_index = (
        PRIORITIES.index(predicted_priority)
        if predicted_priority in PRIORITIES
        else 1
    )

    st.subheader("Review decision")

    with st.form(
        f"review_form_{selected_ticket_id}"
    ):
        final_queue = st.selectbox(
            "Final queue",
            options=SUPPORT_QUEUES,
            index=queue_index,
        )

        final_priority = st.selectbox(
            "Final priority",
            options=PRIORITIES,
            index=priority_index,
        )

        reviewer_notes = st.text_area(
            "Reviewer notes",
            placeholder=(
                "Explain why the prediction was approved or changed."
            ),
        )

        review_submitted = st.form_submit_button(
            "Complete review",
            type="primary",
            width="stretch",
        )

    if review_submitted:
        try:
            review_ticket(
                ticket_id=selected_ticket_id,
                final_queue=final_queue,
                final_priority=final_priority,
                reviewer_notes=reviewer_notes,
            )

            if (
                st.session_state.get("latest_ticket_id")
                == selected_ticket_id
            ):
                del st.session_state["latest_ticket_id"]

            st.success(
                f"Ticket #{selected_ticket_id} was reviewed."
            )

            st.rerun()

        except ValueError as error:
            st.error(str(error))


def analytics_page():
    st.title("Ticket Analytics")
    st.write(
        "Monitor submitted tickets, routing decisions and review status."
    )

    tickets = get_tickets(limit=500)

    if not tickets:
        st.info(
            "Submit some tickets before viewing the analytics dashboard."
        )
        return

    ticket_data = pd.DataFrame(tickets)

    total_tickets = len(ticket_data)
    awaiting_review = int(
        (ticket_data["status"] == "needs_review").sum()
    )
    reviewed_tickets = int(
        (ticket_data["status"] == "reviewed").sum()
    )
    auto_routed = int(
        (ticket_data["status"] == "auto_routed").sum()
    )

    total_column, review_column, reviewed_column, auto_column = (
        st.columns(4)
    )

    total_column.metric(
        "Total tickets",
        total_tickets,
    )

    review_column.metric(
        "Awaiting review",
        awaiting_review,
    )

    reviewed_column.metric(
        "Human reviewed",
        reviewed_tickets,
    )

    auto_column.metric(
        "Automatically routed",
        auto_routed,
    )

    ticket_data["routing_queue"] = ticket_data[
        "final_queue"
    ].fillna(ticket_data["predicted_queue"])

    ticket_data["routing_priority"] = ticket_data[
        "final_priority"
    ].fillna(ticket_data["predicted_priority"])

    queue_counts = (
        ticket_data["routing_queue"]
        .value_counts()
        .sort_values(ascending=True)
    )

    priority_counts = (
        ticket_data["routing_priority"]
        .value_counts()
        .reindex(PRIORITIES, fill_value=0)
    )

    chart_column_one, chart_column_two = st.columns(2)

    with chart_column_one:
        st.subheader("Tickets by queue")
        st.bar_chart(
            queue_counts,
            horizontal=True,
        )

    with chart_column_two:
        st.subheader("Tickets by priority")
        st.bar_chart(priority_counts)

    st.subheader("Recent tickets")

    recent_tickets = ticket_data[
        [
            "id",
            "subject",
            "predicted_queue",
            "queue_confidence",
            "predicted_priority",
            "priority_confidence",
            "status",
            "created_at",
        ]
    ].copy()

    recent_tickets.columns = [
        "ID",
        "Subject",
        "Predicted queue",
        "Queue confidence",
        "Predicted priority",
        "Priority confidence",
        "Status",
        "Created",
    ]

    st.dataframe(
        recent_tickets,
        column_config={
            "Queue confidence": st.column_config.ProgressColumn(
                "Queue confidence",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "Priority confidence": (
                st.column_config.ProgressColumn(
                    "Priority confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
            ),
        },
        hide_index=True,
        width="stretch",
    )


with st.sidebar:
    st.title("Ticket Triage")

    selected_page = st.radio(
        "Navigation",
        options=[
            "Submit ticket",
            "Review queue",
            "Analytics",
        ],
    )

    st.divider()

    st.caption(
        "Predictions below 55% confidence are sent for human review."
    )

    st.caption(
        "This application provides decision support and should not "
        "replace human judgement."
    )


if selected_page == "Submit ticket":
    submission_page()
elif selected_page == "Review queue":
    review_page()
else:
    analytics_page()