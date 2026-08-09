"""Customer negotiation simulator: an evaluation layer over the agent team.

A customer agent plays the counterparty against the live quoting pipeline.
It is deliberately not a member of the team: it sees exactly what a real
customer sees, the guarded customer response text, and never any internal
state. A deterministic harness owns the round budget and the stop
conditions; the model owns only what a customer owns, which is whether to
accept, revise, or walk away, and the wording of the emails it sends.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from munder_difflin.models import RequestStatus, RunEvent, StrictModel
from munder_difflin.orchestrator import MunderDifflinSystem


class CustomerTurn(StrictModel):
    """One customer reply: a single email plus the decision it carries.

    On a revise, ``message`` is the complete email the customer sends back,
    answers and restated order together, and it becomes the next request the
    company processes; there is no separate cover note.
    """

    action: Literal["accept", "revise", "walk_away"]
    message: str = Field(
        min_length=1,
        description=(
            "The complete email the customer sends back: brief answers to the supplier's "
            "questions and, when revising, the restated order the supplier should process"
        ),
    )


class NegotiationRound(StrictModel):
    """One request-response exchange, with the customer's reaction if any."""

    round_number: int = Field(ge=1)
    request_text: str
    status: RequestStatus
    quoted_total: Decimal = Field(ge=0)
    response_text: str
    customer_action: str | None = None
    customer_message: str | None = None


class NegotiationResult(StrictModel):
    """Complete transcript and terminal outcome of one negotiation."""

    outcome: Literal["fulfilled", "partial", "accepted", "walked_away", "round_limit"]
    rounds: list[NegotiationRound]


customer_agent = Agent(
    output_type=CustomerTurn,
    retries=2,
    instructions=(
        "You are playing a customer of Munder Difflin, a paper company, negotiating an order "
        "over email. Stay in character: you are the person described in the context, with a "
        "real event coming up, and you want your supplies, not a perfect spec sheet. Read the "
        "supplier's latest reply and write the single email you would actually send back: "
        "briefly answer each question they asked, then, if you still want goods, restate the "
        "complete order as a short list the supplier can process, one line per item in the "
        "form 'N units of product name', for example '200 sheets of A4 paper' or '2,500 "
        "envelopes'. Keep it plain: no SKUs, no dimensions, no substitution clauses, no "
        "conditions; this supplier matches items by simple product names. If they said an "
        "item is not something they carry, drop it or switch to a similar simply-named paper "
        "product; if they asked about a pack size, answer with the total unit count instead. "
        "Keep the whole email under 120 words and write it the way a busy person actually "
        "writes to a supplier. Choose one action: revise when your email restates an order to "
        "process, accept when you are satisfied with what they concluded, walk_away only when "
        "they clearly cannot help."
    ),
)


def _customer_prompt(
    customer_context: str,
    event: str,
    request_date: date,
    rounds: list[NegotiationRound],
) -> str:
    """Assemble the customer's view: persona plus the customer-visible transcript.

    The transcript contains only what a real customer would hold: their own
    messages and the supplier's guarded replies. Internal state never
    appears here.
    """

    lines = [
        f"You are: {customer_context} (event: {event}). Today is {request_date.isoformat()}.",
        "This is your order conversation with the paper supplier so far:",
    ]
    for index, record in enumerate(rounds):
        lines.append("")
        lines.append(f"You sent:\n{record.request_text}")
        lines.append(f"\nThe supplier replied:\n{record.response_text}")
        echoed_as_next_request = (
            index + 1 < len(rounds) and record.customer_message == rounds[index + 1].request_text
        )
        if record.customer_message and not echoed_as_next_request:
            lines.append(f"\nYou answered:\n{record.customer_message}")
    lines.append(
        "\nWrite the single email you send back now, responding to the supplier's latest reply."
    )
    return "\n".join(lines)


def run_negotiation(
    system: MunderDifflinSystem,
    *,
    request: str,
    request_date: date,
    customer_context: str = "customer",
    event: str = "business event",
    max_rounds: int = 3,
    request_id_prefix: str = "negotiation",
    customer_model: Model | None = None,
    on_round_start: Callable[[int, str], None] | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    on_round: Callable[[NegotiationRound], None] | None = None,
) -> NegotiationResult:
    """Negotiate one request to a terminal outcome within a bounded round budget.

    Rounds run clarify-first: while any line is still clarifiable
    (ambiguous), the company holds all stock and ledger actions and asks the
    customer to answer; products it does not carry decline outright. A round
    that commits anything (fulfilled or partial) is terminal, so a revision
    can never re-order lines that were already sold. The other stop
    conditions are deterministic too: an accept or walk-away decision from
    the customer, or the round budget running out. The customer agent is
    consulted only between rounds, and only with the guarded response text a
    real customer would have received; its reply email becomes the next
    request verbatim.
    """

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    rounds: list[NegotiationRound] = []
    current_request = request
    outcome: Literal["fulfilled", "partial", "accepted", "walked_away", "round_limit"] = (
        "round_limit"
    )
    for round_number in range(1, max_rounds + 1):
        if on_round_start is not None:
            on_round_start(round_number, current_request)
        result = system.process_request(
            current_request,
            request_date=request_date,
            customer_context=customer_context,
            event=event,
            request_id=f"{request_id_prefix}-r{round_number}",
            on_event=on_event,
            clarify_first=True,
        )
        record = NegotiationRound(
            round_number=round_number,
            request_text=current_request,
            status=result.fulfillment.status,
            quoted_total=result.fulfillment.total,
            response_text=result.customer_response.render(),
        )
        if result.fulfillment.fulfilled_lines:
            outcome = (
                "fulfilled" if result.fulfillment.status is RequestStatus.FULFILLED else "partial"
            )
            rounds.append(record)
            if on_round is not None:
                on_round(record)
            break
        if round_number == max_rounds:
            outcome = "round_limit"
            rounds.append(record)
            if on_round is not None:
                on_round(record)
            break

        turn = customer_agent.run_sync(
            _customer_prompt(customer_context, event, request_date, [*rounds, record]),
            model=customer_model or system.model,
        ).output
        record = record.model_copy(
            update={
                "customer_action": turn.action,
                "customer_message": turn.message,
            }
        )
        rounds.append(record)
        if on_round is not None:
            on_round(record)
        if turn.action == "accept":
            outcome = "accepted"
            break
        if turn.action == "walk_away":
            outcome = "walked_away"
            break
        current_request = turn.message

    return NegotiationResult(outcome=outcome, rounds=rounds)
