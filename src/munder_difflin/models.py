"""Typed contracts shared by agents, tools, and evaluation code."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects accidental contract drift."""

    model_config = ConfigDict(extra="forbid")


class RequestStatus(StrEnum):
    """Externally visible request outcomes."""

    FULFILLED = "fulfilled"
    PARTIAL = "partial"
    REJECTED = "rejected"


class ResolutionStatus(StrEnum):
    """Catalog resolution outcomes."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class ParsedLineItem(StrictModel):
    """One requested product before inventory assessment."""

    raw_text: str
    requested_quantity: int = Field(gt=0)
    requested_unit: str
    requested_item: str
    deadline: date
    resolution_status: ResolutionStatus
    catalog_item: str | None = None
    normalized_quantity: int | None = Field(default=None, gt=0)
    resolution_reason: str | None = None


class ParsedRequest(StrictModel):
    """Structured interpretation of a customer request."""

    request_id: str
    request_date: date
    deadline: date
    customer_context: str
    event: str
    original_request: str
    line_items: list[ParsedLineItem]


class InventoryDecision(StrictModel):
    """Availability and sourcing decision for a resolved item."""

    catalog_item: str
    requested_quantity: int = Field(gt=0)
    stock_on_request_date: int = Field(ge=0)
    restock_quantity: int = Field(ge=0)
    supplier_delivery_date: date | None = None
    deliverable: bool
    delivery_date: date | None = None
    reason_code: str | None = None
    customer_reason: str | None = None


class QuoteLine(StrictModel):
    """Auditable line-level price calculation."""

    catalog_item: str
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    discount_rate: Decimal = Field(ge=0, le=1)
    subtotal: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)
    rationale: str


class QuoteResult(StrictModel):
    """Quote generated for all currently deliverable lines."""

    lines: list[QuoteLine]
    subtotal: Decimal = Field(ge=0)
    discount_total: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)
    historical_quotes_consulted: int = Field(ge=0)


class FulfilledLine(StrictModel):
    """A line item committed to the transaction ledger."""

    catalog_item: str
    quantity: int = Field(gt=0)
    delivery_date: date
    sale_transaction_id: int
    stock_order_transaction_id: int | None = None


class DeclinedLine(StrictModel):
    """A line item that could not safely be fulfilled."""

    requested_item: str
    requested_quantity: int = Field(gt=0)
    reason_code: str
    customer_reason: str


class FulfillmentResult(StrictModel):
    """Committed order state after all side effects complete."""

    status: RequestStatus
    fulfilled_lines: list[FulfilledLine]
    declined_lines: list[DeclinedLine]
    total: Decimal = Field(ge=0)
    cash_before: Decimal
    cash_after: Decimal

    @property
    def cash_delta(self) -> Decimal:
        """Return the request-attributable cash movement."""

        return self.cash_after - self.cash_before


class CustomerResponse(StrictModel):
    """Allowlisted customer-facing projection of internal results."""

    request_id: str
    status: RequestStatus
    summary: str
    supplied_items: list[str]
    declined_items: list[str]
    quoted_total: Decimal = Field(ge=0)
    delivery_message: str | None
    pricing_rationale: list[str]

    def render(self) -> str:
        """Render a transparent response without exposing internal state."""

        sections = [self.summary]
        if self.supplied_items:
            sections.append("Supplied: " + "; ".join(self.supplied_items) + ".")
        if self.declined_items:
            sections.append("Not supplied: " + "; ".join(self.declined_items) + ".")
        if self.supplied_items:
            sections.append(f"Quoted total: ${self.quoted_total:.2f}.")
        if self.pricing_rationale:
            sections.append("Pricing: " + " ".join(self.pricing_rationale))
        if self.delivery_message:
            sections.append(self.delivery_message)
        return " ".join(sections)


class RunEvent(StrictModel):
    """Structured event consumed by logs and the terminal experience."""

    trace_id: str
    sequence: int = Field(ge=1)
    agent: str
    action: str
    outcome: str
    detail: str


class ProcessResult(StrictModel):
    """Complete result returned by the orchestrator."""

    trace_id: str
    parsed_request: ParsedRequest
    inventory_decisions: list[InventoryDecision]
    quote: QuoteResult
    fulfillment: FulfillmentResult
    customer_response: CustomerResponse
    events: list[RunEvent]
