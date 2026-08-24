from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.database import (
    get_ticket,
    get_tickets,
    initialise_database,
    review_ticket,
    save_ticket,
)
from src.predict import predict_ticket


class TicketSubmission(BaseModel):
    subject: str = ""
    body: str = ""


class TicketReview(BaseModel):
    final_queue: str
    final_priority: str
    reviewer_notes: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialise_database()
    yield


app = FastAPI(
    title="Support Ticket Triage API",
    description=(
        "Predicts the appropriate support queue and priority for "
        "customer-support tickets, while supporting human review."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "name": "Support Ticket Triage API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/tickets", status_code=201)
def create_ticket(ticket: TicketSubmission):
    try:
        prediction = predict_ticket(
            ticket.subject,
            ticket.body,
        )

        ticket_id = save_ticket(
            ticket.subject,
            ticket.body,
            prediction,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    saved_ticket = get_ticket(ticket_id)

    return saved_ticket


@app.get("/tickets")
def list_tickets(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    review_only: bool = False,
):
    tickets = get_tickets(
        limit=limit,
        review_only=review_only,
    )

    return {
        "count": len(tickets),
        "tickets": tickets,
    }


@app.get("/tickets/{ticket_id}")
def retrieve_ticket(ticket_id: int):
    ticket = get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket {ticket_id} was not found.",
        )

    return ticket


@app.put("/tickets/{ticket_id}/review")
def complete_ticket_review(
    ticket_id: int,
    review: TicketReview,
):
    existing_ticket = get_ticket(ticket_id)

    if existing_ticket is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket {ticket_id} was not found.",
        )

    try:
        reviewed_ticket = review_ticket(
            ticket_id=ticket_id,
            final_queue=review.final_queue,
            final_priority=review.final_priority,
            reviewer_notes=review.reviewer_notes,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return reviewed_ticket