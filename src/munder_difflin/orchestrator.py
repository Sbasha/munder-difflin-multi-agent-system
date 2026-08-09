"""Request lifecycle harness around the four-agent Pydantic AI team.

The harness owns everything that must never depend on a model: database
setup, run identity, the customer-safety guard, and assembling the final
response from the authoritative state recorded by the agents' tools. The
agents own interpretation, delegation, and wording.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from hashlib import sha256

from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from munder_difflin.agents.team import (
    AgentDependencies,
    assemble_fulfillment,
    build_live_model,
    orchestrator_agent,
)
from munder_difflin.config import Settings
from munder_difflin.db.helpers import configure_engine, get_cash_balance, init_database
from munder_difflin.models import (
    CustomerResponse,
    FulfillmentResult,
    ParsedRequest,
    ProcessResult,
    QuoteResult,
    RequestStatus,
)

FORBIDDEN_CUSTOMER_TERMS = (
    "cash balance",
    "profit margin",
    "internal error",
    "stack trace",
    "traceback",
    "api key",
)

_DEFAULT_SUMMARIES = {
    RequestStatus.FULFILLED: "Your request can be fulfilled in full.",
    RequestStatus.PARTIAL: (
        "Part of your request can be fulfilled; unavailable lines are explained below."
    ),
    RequestStatus.REJECTED: "We cannot safely fulfill this request by the required deadline.",
}

_MAX_SUMMARY_LENGTH = 600


class MunderDifflinSystem:
    """Deterministic harness that runs the orchestrator agent per request."""

    def __init__(self, settings: Settings | None = None, model: Model | None = None) -> None:
        self.settings = settings or Settings()
        configure_engine(self.settings.database_url)
        self._model = model

    @property
    def model(self) -> Model:
        """Return the injected model or build the configured live model."""

        if self._model is None:
            self._model = build_live_model(self.settings)
        return self._model

    def initialize(self, seed: int = 137) -> None:
        """Reset application state from the configured data directory."""

        init_database(seed=seed, data_dir=self.settings.data_dir)

    def process_request(
        self,
        request: str,
        request_date: date,
        customer_context: str = "customer",
        event: str = "business event",
        request_id: str | None = None,
    ) -> ProcessResult:
        """Run one customer request through the agent team and package the result."""

        stable_id = (
            request_id or sha256(f"{request_date.isoformat()}:{request}".encode()).hexdigest()[:16]
        )
        trace_id = sha256(f"trace:{stable_id}".encode()).hexdigest()[:16]
        deps = AgentDependencies(
            settings=self.settings,
            request_id=stable_id,
            trace_id=trace_id,
            request_date=request_date,
            original_request=request,
        )
        deps.emit("Orchestrator", "interpret_request", "started", "Reading the customer request")

        prompt = (
            f"Customer context: {customer_context} (event: {event}).\n"
            f"Request date: {request_date.isoformat()}.\n"
            f"Customer request:\n{request}"
        )
        run = orchestrator_agent.run_sync(
            prompt,
            deps=deps,
            model=self.model,
            usage_limits=UsageLimits(request_limit=30),
        )

        self._ensure_complete(deps)
        response = self._build_customer_response(deps, run.output)
        deps.emit(
            "Orchestrator",
            "compose_response",
            "completed",
            "Customer-safe response projection validated",
        )
        assert deps.quote is not None and deps.fulfillment is not None
        parsed = ParsedRequest(
            request_id=stable_id,
            request_date=request_date,
            deadline=deps.effective_deadline,
            customer_context=customer_context,
            event=event,
            original_request=request,
            line_items=deps.line_items,
        )
        return ProcessResult(
            trace_id=trace_id,
            parsed_request=parsed,
            inventory_decisions=list(deps.decisions.values()),
            quote=deps.quote,
            fulfillment=deps.fulfillment,
            customer_response=response,
            events=deps.events,
        )

    @staticmethod
    def _ensure_complete(deps: AgentDependencies) -> None:
        """Fail safe when a run ends without a committed outcome.

        The harness never performs business actions behind the agents' backs:
        an incomplete run becomes an honest rejection with no charges rather
        than a silent auto-commit.
        """

        if deps.quote is None:
            deps.quote = QuoteResult(
                lines=[],
                subtotal=Decimal("0"),
                discount_total=Decimal("0"),
                total=Decimal("0"),
                historical_quotes_consulted=deps.comparables_count,
            )
        if deps.fulfillment is None:
            cash = deps.cash_before
            if cash is None:
                cash = Decimal(str(get_cash_balance(deps.effective_deadline.isoformat())))
            if deps.line_items:
                deps.fulfillment = assemble_fulfillment(deps)
            else:
                deps.fulfillment = FulfillmentResult(
                    status=RequestStatus.REJECTED,
                    fulfilled_lines=[],
                    declined_lines=[],
                    total=Decimal("0"),
                    cash_before=cash,
                    cash_after=cash,
                )
            deps.emit(
                "Orchestrator",
                "finalize_backstop",
                deps.fulfillment.status.value,
                "Run ended without finalize_order; outcome assembled fail-safe",
            )

    def _build_customer_response(
        self,
        deps: AgentDependencies,
        candidate_summary: str,
    ) -> CustomerResponse:
        """Assemble the response from authoritative state plus the agent's wording."""

        fulfillment = deps.fulfillment
        quote = deps.quote
        assert fulfillment is not None and quote is not None
        supplied = [
            (
                f"{line.quantity:,} units of {line.catalog_item}, "
                f"delivery {line.delivery_date.isoformat()}"
            )
            for line in fulfillment.fulfilled_lines
        ]
        declined = [
            f"{line.requested_quantity:,} of {line.requested_item}: {line.customer_reason}"
            for line in fulfillment.declined_lines
        ]
        committed_items = {line.catalog_item for line in fulfillment.fulfilled_lines}
        rationale = [line.rationale for line in quote.lines if line.catalog_item in committed_items]
        delivery_message = None
        if fulfillment.fulfilled_lines:
            latest = max(line.delivery_date for line in fulfillment.fulfilled_lines)
            delivery_message = (
                f"All supplied lines are scheduled no later than {latest.isoformat()}, "
                f"within the requested {deps.effective_deadline.isoformat()} deadline."
            )

        summary = candidate_summary.strip()
        if not summary or len(summary) > _MAX_SUMMARY_LENGTH or _contains_forbidden_terms(summary):
            summary = _DEFAULT_SUMMARIES[fulfillment.status]

        response = CustomerResponse(
            request_id=deps.request_id,
            status=fulfillment.status,
            summary=summary,
            supplied_items=supplied,
            declined_items=declined,
            quoted_total=fulfillment.total,
            delivery_message=delivery_message,
            pricing_rationale=rationale,
        )
        if _contains_forbidden_terms(response.render()):
            response = response.model_copy(
                update={"summary": _DEFAULT_SUMMARIES[fulfillment.status]}
            )
        leaked = sorted(
            term for term in FORBIDDEN_CUSTOMER_TERMS if term in response.render().lower()
        )
        if leaked:
            raise ValueError(f"Customer response contains forbidden internal terms: {leaked}")
        return response


def _contains_forbidden_terms(rendered: str) -> bool:
    lowered = rendered.lower()
    return any(term in lowered for term in FORBIDDEN_CUSTOMER_TERMS)
