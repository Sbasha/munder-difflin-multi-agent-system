"""Four-agent Pydantic AI team over deterministic business tools.

The orchestrator agent interprets each request and delegates to the inventory,
quoting, and fulfillment agents through tools. Every money, stock, and date
decision is computed by deterministic Python inside the tools and recorded on
the shared `AgentDependencies` state; the language models decide which tools
to run and how outcomes are worded, never what the numbers are.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from munder_difflin.catalog import CATALOG, resolve_requested_line
from munder_difflin.config import Settings
from munder_difflin.db.helpers import (
    create_transaction,
    generate_financial_report,
    get_all_inventory,
    get_cash_balance,
    get_min_stock_level,
    get_stock_level,
    get_supplier_delivery_date,
    search_quote_history,
)
from munder_difflin.models import (
    AdvisoryRecommendation,
    AdvisoryReport,
    DeclinedLine,
    FulfilledLine,
    FulfillmentResult,
    InventoryDecision,
    ParsedLineItem,
    QuoteResult,
    RequestStatus,
    ResolutionStatus,
    RunEvent,
)
from munder_difflin.pricing import build_quote, compare_with_history


class ExtractedLine(BaseModel):
    """One product line the orchestrator extracted from the raw request text."""

    item_text: str = Field(
        description=(
            "The product name as the customer wrote it, without sizes, specifications, "
            "parentheticals, or conditions, e.g. 'A4 paper', 'kraft paper envelopes'"
        )
    )
    quantity: int = Field(gt=0, description="The total numeric amount requested for this line")
    unit: str = Field(description="The unit word used, e.g. sheets, reams, packs, cups")


@dataclass
class AgentDependencies:
    """Mutable per-request state shared by every agent tool.

    Tools write authoritative, deterministically computed facts here; the
    final customer response is assembled from this state, not from model text.
    """

    settings: Settings
    request_id: str
    trace_id: str
    request_date: date
    original_request: str
    clarify_first: bool = False
    deadline: date | None = None
    cash_before: Decimal | None = None
    line_items: list[ParsedLineItem] = field(default_factory=list)
    decisions: dict[str, InventoryDecision] = field(default_factory=dict)
    stock_order_ids: dict[str, int] = field(default_factory=dict)
    comparables_count: int = 0
    comparable_totals: list[Decimal] | None = None
    quote: QuoteResult | None = None
    fulfilled_lines: list[FulfilledLine] = field(default_factory=list)
    fulfillment: FulfillmentResult | None = None
    events: list[RunEvent] = field(default_factory=list)
    on_event: Callable[[RunEvent], None] | None = None

    @property
    def effective_deadline(self) -> date:
        """The customer deadline, failing closed to the request date."""

        return self.deadline or self.request_date

    def emit(self, agent: str, action: str, outcome: str, detail: str) -> None:
        """Append one structured run event and stream it to any live observer."""

        event = RunEvent(
            trace_id=self.trace_id,
            sequence=len(self.events) + 1,
            agent=agent,
            action=action,
            outcome=outcome,
            detail=detail,
        )
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)


orchestrator_agent = Agent(
    deps_type=AgentDependencies,
    output_type=str,
    retries=2,
    instructions=(
        "You are the orchestrator for Munder Difflin, a paper company. Handle one customer "
        "request end to end by calling your tools in this exact order:\n"
        "1. Extract every requested product line (item text, quantity, unit word) and the "
        "required delivery date from the request, then call resolve_catalog_items exactly once "
        "with all lines.\n"
        "2. Call consult_inventory exactly once.\n"
        "3. Call request_quote exactly once.\n"
        "4. If the quote total exceeds 500 dollars, call financial_health_report once as an "
        "internal check before committing.\n"
        "5. Call finalize_order exactly once.\n"
        "Then reply with one or two short sentences summarizing the outcome for the customer in "
        "a warm, professional tone, briefly explaining why any line could not be supplied. If "
        "the outcome status is needs_clarification, look at the resolution reason for each "
        "ambiguous line and ask for exactly what is missing: if the reason mentions catalog "
        "item names, ask the customer to confirm which named item they want (quote the names "
        "directly); if the reason mentions a packet or box size was not specified, ask the "
        "customer to provide the total unit count instead. Items that are merely being held "
        "will be processed automatically once the questions are answered - do not ask the "
        "customer to re-specify those. Do not ask for product specifications such as size, "
        "finish, weight, closure, or window style; resolution requires only a catalog product "
        "name, a total quantity, and a delivery date. Do not repeat specific prices, quantities, "
        "or dates (an itemized statement is attached automatically) and never mention internal "
        "finances, tooling, or errors."
    ),
)

inventory_agent = Agent(
    deps_type=AgentDependencies,
    output_type=str,
    retries=2,
    instructions=(
        "You are the inventory agent. First call inventory_snapshot once to see current stock. "
        "Then call assess_availability once for each line you were given, using the exact "
        "catalog item name. Whenever an assessment reports restock_required=true, immediately "
        "call place_restock_order for that item. Availability and purchasing policy are "
        "deterministic inside the tools and cannot be overridden. Finish with one short "
        "sentence summarizing availability."
    ),
)

quoting_agent = Agent(
    deps_type=AgentDependencies,
    output_type=str,
    retries=2,
    instructions=(
        "You are the quoting agent. First call retrieve_comparable_quotes once, using the "
        "catalog item names as search terms, to review historical context. Then call "
        "compute_quote once; it prices deterministically with the published bulk-discount "
        "tiers and never sells below cost. Finish with one short sentence on the pricing "
        "rationale."
    ),
)

fulfillment_agent = Agent(
    deps_type=AgentDependencies,
    output_type=str,
    retries=2,
    instructions=(
        "You are the fulfillment agent. Call commit_sale once for each deliverable line you "
        "were given, using the exact catalog item name. Never change prices and never restock. "
        "Finish with one short sentence stating how many lines were committed."
    ),
)


def build_live_model(settings: Settings) -> OpenAIChatModel:
    """Build the OpenAI-compatible model (Vocareum proxy by default)."""

    if not settings.llm_api_key:
        raise ValueError(
            "Set UDACITY_OPENAI_API_KEY, OPENAI_API_KEY, or LLM_API_KEY to run the agents"
        )
    return OpenAIChatModel(
        settings.llm_model,
        provider=OpenAIProvider(
            base_url=str(settings.llm_base_url),
            api_key=settings.llm_api_key,
        ),
    )


# --- Orchestrator tools -----------------------------------------------------


@orchestrator_agent.tool
def resolve_catalog_items(
    ctx: RunContext[AgentDependencies],
    lines: list[ExtractedLine],
    deadline: date | None = None,
) -> list[dict[str, Any]]:
    """Resolve extracted lines against the catalog deterministically.

    Records the request deadline and the resolved line items on shared state.
    Duplicate resolutions of the same catalog item are merged.
    """

    deps = ctx.deps
    deps.deadline = deadline
    resolved: dict[str, ParsedLineItem] = {}
    ordered: list[ParsedLineItem] = []
    for line in lines:
        parsed = resolve_requested_line(
            line.item_text, line.unit, line.quantity, deps.effective_deadline
        )
        if parsed.catalog_item and parsed.catalog_item in resolved:
            existing = resolved[parsed.catalog_item]
            merged = existing.model_copy(
                update={
                    "requested_quantity": existing.requested_quantity + parsed.requested_quantity,
                    "normalized_quantity": (existing.normalized_quantity or 0)
                    + (parsed.normalized_quantity or 0),
                    "raw_text": f"{existing.raw_text}; {parsed.raw_text}",
                }
            )
            resolved[parsed.catalog_item] = merged
            ordered[ordered.index(existing)] = merged
            continue
        if parsed.catalog_item:
            resolved[parsed.catalog_item] = parsed
        ordered.append(parsed)
    deps.line_items = ordered
    deps.emit(
        "Orchestrator",
        "resolve_catalog_items",
        "completed",
        f"{sum(1 for item in ordered if item.catalog_item)} of {len(ordered)} lines resolved",
    )
    return [
        {
            "requested": item.raw_text,
            "resolution_status": item.resolution_status.value,
            "catalog_item": item.catalog_item,
            "sellable_units": item.normalized_quantity,
            "reason": item.resolution_reason,
        }
        for item in ordered
    ]


@orchestrator_agent.tool
async def consult_inventory(ctx: RunContext[AgentDependencies]) -> list[dict[str, Any]]:
    """Delegate availability and restocking to the inventory agent.

    Returns the authoritative inventory decisions recorded by its tools.
    In clarify-first mode, an order with clarifiable (ambiguous) lines is
    held: no stock or ledger action is taken until the customer has
    answered. Unsupported products cannot be fixed by clarification, so they
    decline normally and never hold the order on their own.
    """

    deps = ctx.deps
    if deps.cash_before is None:
        deps.cash_before = Decimal(str(get_cash_balance(deps.effective_deadline.isoformat())))
    clarifiable = [
        item for item in deps.line_items if item.resolution_status is ResolutionStatus.AMBIGUOUS
    ]
    if deps.clarify_first and clarifiable:
        deps.emit(
            "Orchestrator",
            "clarification_hold",
            "held",
            f"{len(clarifiable)} line(s) can be clarified; no stock or ledger action taken",
        )
        return []
    resolved = [item for item in deps.line_items if item.catalog_item and item.normalized_quantity]
    if resolved:
        prompt = (
            f"Request date {deps.request_date.isoformat()}, deadline "
            f"{deps.effective_deadline.isoformat()}. Assess these lines:\n"
            + "\n".join(
                f"- {item.normalized_quantity:,} units of {item.catalog_item}" for item in resolved
            )
        )
        await inventory_agent.run(prompt, deps=deps, model=ctx.model, usage=ctx.usage)
    return [decision.model_dump(mode="json") for decision in deps.decisions.values()]


@orchestrator_agent.tool
async def request_quote(ctx: RunContext[AgentDependencies]) -> dict[str, Any]:
    """Delegate pricing to the quoting agent and return the authoritative quote."""

    deps = ctx.deps
    deliverable = [decision for decision in deps.decisions.values() if decision.deliverable]
    if deliverable:
        prompt = "Price these deliverable lines:\n" + "\n".join(
            f"- {decision.requested_quantity:,} units of {decision.catalog_item}"
            for decision in deliverable
        )
        await quoting_agent.run(prompt, deps=deps, model=ctx.model, usage=ctx.usage)
    if deps.quote is None:
        deps.quote = build_quote(
            [(decision.catalog_item, decision.requested_quantity) for decision in deliverable],
            markup_rate=deps.settings.markup_rate,
            historical_quotes_consulted=deps.comparables_count,
            comparable_totals=deps.comparable_totals,
        )
    return deps.quote.model_dump(mode="json")


@orchestrator_agent.tool
async def finalize_order(ctx: RunContext[AgentDependencies]) -> dict[str, Any]:
    """Delegate sale commitment to the fulfillment agent and assemble the outcome."""

    deps = ctx.deps
    if deps.quote is None:
        raise ModelRetry("Call request_quote before finalize_order")
    deliverable = [decision for decision in deps.decisions.values() if decision.deliverable]
    if deliverable:
        prompt = "Commit these deliverable lines:\n" + "\n".join(
            f"- {decision.requested_quantity:,} units of {decision.catalog_item}"
            for decision in deliverable
        )
        await fulfillment_agent.run(prompt, deps=deps, model=ctx.model, usage=ctx.usage)
    deps.fulfillment = assemble_fulfillment(deps)
    deps.emit(
        "Fulfillment",
        "commit_order",
        deps.fulfillment.status.value,
        (
            f"{len(deps.fulfillment.fulfilled_lines)} committed; "
            f"{len(deps.fulfillment.declined_lines)} declined"
        ),
    )
    return {
        "status": deps.fulfillment.status.value,
        "fulfilled_line_count": len(deps.fulfillment.fulfilled_lines),
        "declined_line_count": len(deps.fulfillment.declined_lines),
        "declined_reasons": [line.customer_reason for line in deps.fulfillment.declined_lines],
    }


@orchestrator_agent.tool
def financial_health_report(
    ctx: RunContext[AgentDependencies],
    as_of_date: date | None = None,
) -> dict[str, Any]:
    """Internal-only financial health check using generate_financial_report."""

    deps = ctx.deps
    report = generate_financial_report((as_of_date or deps.request_date).isoformat())
    deps.emit(
        "Orchestrator",
        "financial_health_report",
        "completed",
        f"Total assets ${report['total_assets']:,.2f} on {report['as_of_date']}",
    )
    return {
        "as_of_date": report["as_of_date"],
        "cash_balance": report["cash_balance"],
        "inventory_value": report["inventory_value"],
        "total_assets": report["total_assets"],
        "top_selling_products": report["top_selling_products"],
    }


# --- Inventory tools --------------------------------------------------------


def _resolved_line(deps: AgentDependencies, item_name: str) -> ParsedLineItem:
    for item in deps.line_items:
        if item.catalog_item == item_name and item.normalized_quantity:
            return item
    valid = sorted(item.catalog_item for item in deps.line_items if item.catalog_item)
    raise ModelRetry(f"'{item_name}' is not a resolved line; valid item names: {valid}")


@inventory_agent.tool
def inventory_snapshot(ctx: RunContext[AgentDependencies]) -> dict[str, int]:
    """Report all positive stock on the request date using get_all_inventory."""

    deps = ctx.deps
    snapshot = get_all_inventory(deps.request_date.isoformat())
    deps.emit(
        "Inventory",
        "inventory_snapshot",
        "completed",
        f"{len(snapshot)} items in stock on {deps.request_date.isoformat()}",
    )
    return snapshot


@inventory_agent.tool
def assess_availability(ctx: RunContext[AgentDependencies], item_name: str) -> dict[str, Any]:
    """Assess stock and supplier feasibility for one resolved line.

    Uses get_stock_level and get_supplier_delivery_date; the reorder policy
    (top up to min_stock_level times the configured multiplier) is
    deterministic and cannot be overridden.
    """

    deps = ctx.deps
    line = _resolved_line(deps, item_name)
    quantity = line.normalized_quantity or line.requested_quantity
    stock = int(get_stock_level(item_name, deps.request_date.isoformat())["current_stock"].iloc[0])
    if stock >= quantity:
        deps.decisions[item_name] = InventoryDecision(
            catalog_item=item_name,
            requested_quantity=quantity,
            stock_on_request_date=stock,
            restock_quantity=0,
            deliverable=True,
            delivery_date=deps.request_date,
        )
        deps.emit(
            "Inventory",
            "assess_availability",
            "deliverable",
            f"{item_name}: {quantity:,} units available from stock",
        )
        return {
            "item_name": item_name,
            "deliverable": True,
            "restock_required": False,
            "stock": stock,
        }

    minimum = get_min_stock_level(item_name)
    target = minimum * deps.settings.restock_target_multiplier
    restock_quantity = max(quantity - stock, target - stock)
    supplier_date = date.fromisoformat(
        get_supplier_delivery_date(deps.request_date.isoformat(), restock_quantity)
    )
    if supplier_date > deps.effective_deadline:
        deps.decisions[item_name] = InventoryDecision(
            catalog_item=item_name,
            requested_quantity=quantity,
            stock_on_request_date=stock,
            restock_quantity=restock_quantity,
            supplier_delivery_date=supplier_date,
            deliverable=False,
            reason_code="supplier_after_deadline",
            customer_reason=(
                f"Supplier replenishment would arrive {supplier_date.isoformat()}, "
                f"after the required {deps.effective_deadline.isoformat()} deadline"
            ),
        )
        deps.emit(
            "Inventory",
            "assess_availability",
            "declined",
            f"{item_name}: supplier arrives {supplier_date.isoformat()}, after the deadline",
        )
        return {
            "item_name": item_name,
            "deliverable": False,
            "restock_required": False,
            "reason": "supplier_after_deadline",
        }

    deps.decisions[item_name] = InventoryDecision(
        catalog_item=item_name,
        requested_quantity=quantity,
        stock_on_request_date=stock,
        restock_quantity=restock_quantity,
        supplier_delivery_date=supplier_date,
        deliverable=False,
        reason_code="restock_required",
        customer_reason="Required replenishment was identified but has not been arranged",
    )
    deps.emit(
        "Inventory",
        "assess_availability",
        "restock_required",
        f"{item_name}: short {quantity - stock:,}; restock {restock_quantity:,} units",
    )
    return {
        "item_name": item_name,
        "deliverable": False,
        "restock_required": True,
        "restock_quantity": restock_quantity,
        "supplier_delivery_date": supplier_date.isoformat(),
    }


@inventory_agent.tool
def place_restock_order(ctx: RunContext[AgentDependencies], item_name: str) -> dict[str, Any]:
    """Place a cash-guarded restock using get_cash_balance and create_transaction."""

    deps = ctx.deps
    decision = deps.decisions.get(item_name)
    if decision is None or not decision.restock_quantity or not decision.supplier_delivery_date:
        raise ModelRetry(
            "place_restock_order requires a prior assess_availability that reported "
            f"restock_required=true; nothing to restock for '{item_name}'"
        )
    if item_name in deps.stock_order_ids:
        return {
            "authorized": True,
            "transaction_id": deps.stock_order_ids[item_name],
            "note": "restock already placed",
        }

    supplier_date = decision.supplier_delivery_date
    cost = Decimal(decision.restock_quantity) * CATALOG[item_name].unit_price
    available_cash = Decimal(str(get_cash_balance(supplier_date.isoformat())))
    if available_cash - cost < Decimal(str(deps.settings.cash_reserve)):
        deps.decisions[item_name] = decision.model_copy(
            update={
                "deliverable": False,
                "reason_code": "restock_not_authorized",
                "customer_reason": (
                    "Required stock cannot be replenished within current purchasing limits"
                ),
            }
        )
        deps.emit(
            "Inventory",
            "place_restock_order",
            "declined",
            f"{item_name}: restock of {decision.restock_quantity:,} exceeds purchasing limits",
        )
        return {"authorized": False, "reason": "restock_not_authorized"}

    transaction_id = create_transaction(
        item_name,
        "stock_orders",
        decision.restock_quantity,
        float(cost),
        supplier_date.isoformat(),
    )
    deps.stock_order_ids[item_name] = transaction_id
    deps.decisions[item_name] = decision.model_copy(
        update={
            "deliverable": True,
            "delivery_date": supplier_date,
            "reason_code": None,
            "customer_reason": None,
        }
    )
    deps.emit(
        "Inventory",
        "place_restock_order",
        "authorized",
        f"{item_name}: {decision.restock_quantity:,} units arriving {supplier_date.isoformat()}",
    )
    return {
        "authorized": True,
        "transaction_id": transaction_id,
        "delivery_date": supplier_date.isoformat(),
    }


# --- Quoting tools ----------------------------------------------------------


@quoting_agent.tool
def retrieve_comparable_quotes(
    ctx: RunContext[AgentDependencies],
    search_terms: list[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve historical quote context using search_quote_history."""

    deps = ctx.deps
    rows = search_quote_history(search_terms, limit)
    deps.comparables_count = len(rows)
    totals: list[Decimal] = []
    for row in rows:
        value = row.get("total_amount")
        if value is None:
            continue
        try:
            totals.append(Decimal(str(value)))
        except InvalidOperation:
            continue
    deps.comparable_totals = totals
    deps.emit(
        "Quoting",
        "retrieve_comparable_quotes",
        "completed",
        f"{len(rows)} comparable historical quotes found",
    )
    return [
        {
            "total_amount": row.get("total_amount"),
            "quote_explanation": str(row.get("quote_explanation", ""))[:300],
            "job_type": row.get("job_type"),
            "order_size": row.get("order_size"),
            "event_type": row.get("event_type"),
        }
        for row in rows
    ]


@quoting_agent.tool
def compute_quote(ctx: RunContext[AgentDependencies]) -> dict[str, Any]:
    """Price all deliverable lines deterministically (markup plus volume tiers)."""

    deps = ctx.deps
    deliverable = [
        (decision.catalog_item, decision.requested_quantity)
        for decision in deps.decisions.values()
        if decision.deliverable
    ]
    deps.quote = build_quote(
        deliverable,
        markup_rate=deps.settings.markup_rate,
        historical_quotes_consulted=deps.comparables_count,
        comparable_totals=deps.comparable_totals,
    )
    deps.emit(
        "Quoting",
        "compute_quote",
        "completed",
        f"{len(deps.quote.lines)} priced lines totaling ${deps.quote.total:.2f}",
    )
    if deps.comparable_totals is not None:
        outcome, note = compare_with_history(deps.quote.total, deps.comparable_totals)
        deps.emit("Quoting", "comparables_check", outcome, note)
    return deps.quote.model_dump(mode="json")


# --- Fulfillment tools ------------------------------------------------------


@fulfillment_agent.tool
def commit_sale(ctx: RunContext[AgentDependencies], item_name: str) -> dict[str, Any]:
    """Revalidate stock with get_stock_level and record the sale with create_transaction."""

    deps = ctx.deps
    decision = deps.decisions.get(item_name)
    if decision is None or not decision.deliverable or decision.delivery_date is None:
        raise ModelRetry(f"'{item_name}' is not a deliverable line; do not commit it")
    for line in deps.fulfilled_lines:
        if line.catalog_item == item_name:
            return {
                "committed": True,
                "transaction_id": line.sale_transaction_id,
                "note": "sale already committed",
            }
    quote_line = None
    if deps.quote is not None:
        quote_line = next(
            (line for line in deps.quote.lines if line.catalog_item == item_name), None
        )
    if quote_line is None:
        raise ModelRetry(f"No priced quote line exists for '{item_name}'")

    stock = int(
        get_stock_level(item_name, decision.delivery_date.isoformat())["current_stock"].iloc[0]
    )
    if stock < decision.requested_quantity:
        deps.decisions[item_name] = decision.model_copy(
            update={
                "deliverable": False,
                "reason_code": "stock_changed_before_commit",
                "customer_reason": "Available stock changed before the sale could be committed",
            }
        )
        deps.emit(
            "Fulfillment",
            "commit_sale",
            "declined",
            f"{item_name}: stock changed before commit",
        )
        return {"committed": False, "reason": "stock_changed_before_commit"}

    transaction_id = create_transaction(
        item_name,
        "sales",
        decision.requested_quantity,
        float(quote_line.total),
        decision.delivery_date.isoformat(),
    )
    deps.fulfilled_lines.append(
        FulfilledLine(
            catalog_item=item_name,
            quantity=decision.requested_quantity,
            delivery_date=decision.delivery_date,
            sale_transaction_id=transaction_id,
            stock_order_transaction_id=deps.stock_order_ids.get(item_name),
        )
    )
    deps.emit(
        "Fulfillment",
        "commit_sale",
        "committed",
        f"{item_name}: {decision.requested_quantity:,} units, ${quote_line.total:.2f}",
    )
    return {"committed": True, "transaction_id": transaction_id}


# --- Deterministic outcome assembly ------------------------------------------


def assemble_fulfillment(deps: AgentDependencies) -> FulfillmentResult:
    """Assemble the authoritative order outcome from recorded tool state."""

    if deps.fulfillment is not None:
        return deps.fulfillment

    held_for_clarification = deps.clarify_first and any(
        item.resolution_status is ResolutionStatus.AMBIGUOUS for item in deps.line_items
    )
    declined: list[DeclinedLine] = []
    for item in deps.line_items:
        if not item.catalog_item or not item.normalized_quantity:
            declined.append(
                DeclinedLine(
                    requested_item=item.requested_item,
                    requested_quantity=item.requested_quantity,
                    reason_code=item.resolution_status.value,
                    customer_reason=item.resolution_reason
                    or "The requested product could not be resolved",
                )
            )
        elif item.catalog_item not in deps.decisions:
            if held_for_clarification:
                reason_code = "awaiting_clarification"
                customer_reason = "Held pending resolution of other items in this order"
            else:
                reason_code = "not_assessed"
                customer_reason = "Availability could not be confirmed for this line"
            declined.append(
                DeclinedLine(
                    requested_item=item.requested_item,
                    requested_quantity=item.requested_quantity,
                    reason_code=reason_code,
                    customer_reason=customer_reason,
                )
            )

    committed_items = {line.catalog_item for line in deps.fulfilled_lines}
    for item_name, decision in deps.decisions.items():
        if item_name in committed_items:
            continue
        if decision.deliverable:
            reason_code = "commit_incomplete"
            customer_reason = "The line was approved but could not be committed in time"
        else:
            reason_code = decision.reason_code or "inventory_unavailable"
            customer_reason = (
                decision.customer_reason
                or "The requested quantity is not available by the deadline"
            )
        declined.append(
            DeclinedLine(
                requested_item=item_name,
                requested_quantity=decision.requested_quantity,
                reason_code=reason_code,
                customer_reason=customer_reason,
            )
        )

    if held_for_clarification and not deps.fulfilled_lines:
        status = RequestStatus.NEEDS_CLARIFICATION
    else:
        status = (
            RequestStatus.FULFILLED
            if deps.fulfilled_lines and not declined
            else RequestStatus.PARTIAL
            if deps.fulfilled_lines
            else RequestStatus.REJECTED
        )
    total = Decimal("0")
    if deps.quote is not None:
        total = sum(
            (line.total for line in deps.quote.lines if line.catalog_item in committed_items),
            Decimal("0"),
        )
    cash_after = Decimal(str(get_cash_balance(deps.effective_deadline.isoformat())))
    cash_before = deps.cash_before if deps.cash_before is not None else cash_after
    return FulfillmentResult(
        status=status,
        fulfilled_lines=list(deps.fulfilled_lines),
        declined_lines=declined,
        total=total,
        cash_before=cash_before,
        cash_after=cash_after,
    )


# --- Business advisor agent ---------------------------------------------------


@dataclass
class AdvisorDependencies:
    """Read-only context for the business advisor agent."""

    settings: Settings
    as_of_date: str


advisor_agent = Agent(
    deps_type=AdvisorDependencies,
    output_type=AdvisoryReport,
    retries=2,
    instructions=(
        "You are the business advisor for Munder Difflin, a paper company. Your role is "
        "read-only: you analyze committed transaction data and produce actionable "
        "recommendations to improve operational efficiency and revenue. You never place "
        "orders or modify any records. Call your tools in this order:\n"
        "1. Call read_financial_report once to understand the current cash and inventory state.\n"
        "2. Call analyze_stock_gaps once to identify catalog items with demand but low or zero "
        "stock.\n"
        "3. Call review_demand_patterns once to find recurring demand the catalog does not "
        "currently cover.\n"
        "Synthesize your findings into 3 to 5 specific, prioritized recommendations. Each "
        "recommendation must name the specific catalog items or patterns it targets and "
        "quantify the expected impact where the data supports it. Avoid generic advice. "
        "Assign priority 'high' for issues that directly limit revenue or create cash risk, "
        "'medium' for operational improvements, and 'low' for longer-term opportunities."
    ),
)


@advisor_agent.tool
def read_financial_report(ctx: RunContext[AdvisorDependencies]) -> dict[str, Any]:
    """Retrieve cash balance, inventory valuation, and top-selling products."""

    report = generate_financial_report(ctx.deps.as_of_date)
    zero_stock = [
        item["item_name"]
        for item in report["inventory_summary"]
        if item["stock"] <= 0
    ]
    low_stock = [
        {"item": item["item_name"], "stock": item["stock"]}
        for item in report["inventory_summary"]
        if 0 < item["stock"] < 100
    ]
    return {
        "as_of_date": report["as_of_date"],
        "cash_balance": report["cash_balance"],
        "inventory_value": report["inventory_value"],
        "total_assets": report["total_assets"],
        "top_selling_products": report["top_selling_products"],
        "zero_stock_items": zero_stock,
        "low_stock_items": low_stock,
        "zero_stock_count": len(zero_stock),
    }


@advisor_agent.tool
def analyze_stock_gaps(ctx: RunContext[AdvisorDependencies]) -> dict[str, Any]:
    """Identify catalog items that received recent demand but have no or low stock.

    Compares the items in historic quote requests against current inventory
    to surface pre-stocking opportunities.
    """

    report = generate_financial_report(ctx.deps.as_of_date)
    stock_by_item = {item["item_name"]: item["stock"] for item in report["inventory_summary"]}
    top_sellers = {row["item_name"] for row in report["top_selling_products"]}

    gaps = []
    for item_name, stock in stock_by_item.items():
        if stock <= 0 and item_name in top_sellers:
            gaps.append({"item": item_name, "stock": stock, "gap_type": "sold_out_with_demand"})
        elif stock <= 0:
            gaps.append({"item": item_name, "stock": stock, "gap_type": "zero_stock"})

    # Items that sell well but have low buffer
    low_buffer = [
        {"item": row["item_name"], "stock": stock_by_item.get(row["item_name"], 0),
         "total_units_sold": row["total_units"]}
        for row in report["top_selling_products"]
        if stock_by_item.get(row["item_name"], 0) < row["total_units"]
    ]

    return {
        "stock_gaps": gaps[:10],
        "low_buffer_vs_demand": low_buffer,
        "total_zero_stock": sum(1 for g in gaps if g["stock"] <= 0),
    }


@advisor_agent.tool
def review_demand_patterns(
    ctx: RunContext[AdvisorDependencies],
    search_terms: list[str] | None = None,
) -> dict[str, Any]:
    """Review historical quote request patterns to find recurring demand themes.

    Searches quote history for the most common product categories and event
    types to identify seasonal patterns and catalog gaps.
    """

    terms = search_terms or ["paper", "envelopes", "cardstock", "poster", "recycled"]
    rows = search_quote_history(terms, limit=10)
    event_types: dict[str, int] = {}
    job_types: dict[str, int] = {}
    for row in rows:
        et = str(row.get("event_type") or "unknown")
        jt = str(row.get("job_type") or "unknown")
        event_types[et] = event_types.get(et, 0) + 1
        job_types[jt] = job_types.get(jt, 0) + 1

    top_events = sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:5]
    top_jobs = sorted(job_types.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "comparable_quotes_reviewed": len(rows),
        "top_event_types": [{"event": e, "count": c} for e, c in top_events],
        "top_customer_roles": [{"role": j, "count": c} for j, c in top_jobs],
        "sample_requests": [
            str(row.get("original_request", ""))[:120] for row in rows[:3]
        ],
    }


def run_advisory(
    settings: Settings,
    as_of_date: str,
    model: "OpenAIChatModel | None" = None,
) -> AdvisoryReport:
    """Run the business advisor agent against committed transaction data."""

    from munder_difflin.orchestrator import MunderDifflinSystem

    deps = AdvisorDependencies(settings=settings, as_of_date=as_of_date)
    live_model = model or MunderDifflinSystem(settings).model
    result = advisor_agent.run_sync(
        f"Analyze the business state as of {as_of_date} and produce your recommendations.",
        deps=deps,
        model=live_model,
    )
    return result.output
