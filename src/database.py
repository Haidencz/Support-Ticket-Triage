import json
import sqlite3
from pathlib import Path


DATABASE_PATH = Path("data/tickets.db")


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialise_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,

                predicted_queue TEXT NOT NULL,
                queue_confidence REAL NOT NULL,
                alternative_queues TEXT NOT NULL,

                predicted_priority TEXT NOT NULL,
                priority_confidence REAL NOT NULL,

                requires_review INTEGER NOT NULL,
                status TEXT NOT NULL,

                final_queue TEXT,
                final_priority TEXT,
                reviewer_notes TEXT,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TEXT
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_status
            ON tickets(status)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_requires_review
            ON tickets(requires_review)
            """
        )


def save_ticket(subject, body, prediction):
    subject = subject.strip()
    body = body.strip()

    if not subject and not body:
        raise ValueError(
            "A ticket must contain a subject or description."
        )

    predicted_queue = prediction["predicted_queue"]
    predicted_priority = prediction["predicted_priority"]
    requires_review = prediction["requires_review"]

    if requires_review:
        status = "needs_review"
    else:
        status = "auto_routed"

    alternative_queues = json.dumps(
        prediction["alternative_queues"]
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO tickets (
                subject,
                body,
                predicted_queue,
                queue_confidence,
                alternative_queues,
                predicted_priority,
                priority_confidence,
                requires_review,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject,
                body,
                predicted_queue["label"],
                predicted_queue["confidence"],
                alternative_queues,
                predicted_priority["label"],
                predicted_priority["confidence"],
                int(requires_review),
                status,
            ),
        )

        ticket_id = cursor.lastrowid

    return ticket_id


def convert_ticket_row(row):
    if row is None:
        return None

    ticket = dict(row)

    ticket["requires_review"] = bool(
        ticket["requires_review"]
    )

    ticket["alternative_queues"] = json.loads(
        ticket["alternative_queues"]
    )

    return ticket


def get_ticket(ticket_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM tickets
            WHERE id = ?
            """,
            (ticket_id,),
        ).fetchone()

    return convert_ticket_row(row)


def get_tickets(limit=100, review_only=False):
    if limit < 1:
        raise ValueError("The ticket limit must be at least 1.")

    query = """
        SELECT *
        FROM tickets
    """

    parameters = []

    if review_only:
        query += """
            WHERE requires_review = 1
            AND status = 'needs_review'
        """

    query += """
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """

    parameters.append(limit)

    with get_connection() as connection:
        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [
        convert_ticket_row(row)
        for row in rows
    ]


def review_ticket(
    ticket_id,
    final_queue,
    final_priority,
    reviewer_notes="",
):
    final_queue = final_queue.strip()
    final_priority = final_priority.strip()
    reviewer_notes = reviewer_notes.strip()

    if not final_queue:
        raise ValueError("A final queue must be selected.")

    if not final_priority:
        raise ValueError("A final priority must be selected.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE tickets
            SET final_queue = ?,
                final_priority = ?,
                reviewer_notes = ?,
                requires_review = 0,
                status = 'reviewed',
                reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                final_queue,
                final_priority,
                reviewer_notes,
                ticket_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Ticket {ticket_id} does not exist."
            )

    return get_ticket(ticket_id)


if __name__ == "__main__":
    from predict import predict_ticket

    initialise_database()

    example_subject = "Company VPN unavailable"
    example_body = (
        "Our entire remote team has lost access to the company VPN. "
        "Restarting devices and resetting passwords did not resolve "
        "the problem, and employees cannot access critical systems."
    )

    example_prediction = predict_ticket(
        example_subject,
        example_body,
    )

    saved_ticket_id = save_ticket(
        example_subject,
        example_body,
        example_prediction,
    )

    saved_ticket = get_ticket(saved_ticket_id)

    print(f"Database created at: {DATABASE_PATH}")
    print(f"Test ticket saved with ID: {saved_ticket_id}")
    print(f"Status: {saved_ticket['status']}")
    print(f"Queue: {saved_ticket['predicted_queue']}")
    print(f"Priority: {saved_ticket['predicted_priority']}")