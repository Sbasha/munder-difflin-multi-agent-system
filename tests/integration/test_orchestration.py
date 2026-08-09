"""End-to-end lifecycle tests driving the real agent graph with scripted models."""

from contextlib import ExitStack
from datetime import date

from munder_difflin.agents import team
from munder_difflin.config import Settings
from munder_difflin.models import ProcessResult, RequestStatus
from munder_difflin.orchestrator import MunderDifflinSystem


def _run(
    settings: Settings,
    scripted_model,
    request: str,
    orchestrator_script,
    request_id: str,
    *,
    inventory_script=None,
    quoting_script=None,
    fulfillment_script=None,
    final_text: str = "Thanks for your order.",
    on_event=None,
    clarify_first: bool = False,
) -> ProcessResult:
    system = MunderDifflinSystem(settings)
    system.initialize()
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(model=scripted_model(orchestrator_script, final_text))
        )
        if inventory_script is not None:
            stack.enter_context(
                team.inventory_agent.override(model=scripted_model(inventory_script))
            )
        if quoting_script is not None:
            stack.enter_context(team.quoting_agent.override(model=scripted_model(quoting_script)))
        if fulfillment_script is not None:
            stack.enter_context(
                team.fulfillment_agent.override(model=scripted_model(fulfillment_script))
            )
        return system.process_request(
            request,
            request_date=date(2025, 4, 1),
            request_id=request_id,
            on_event=on_event,
            clarify_first=clarify_first,
        )


def test_partial_order_commits_stocked_line_and_explains_the_rest(
    settings: Settings, scripted_model
) -> None:
    result = _run(
        settings,
        scripted_model,
        "I need 200 sheets of A4 paper and 100 balloons delivered by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [
                        {"item_text": "A4 paper", "quantity": 200, "unit": "sheets"},
                        {"item_text": "balloons", "quantity": 100, "unit": "balloons"},
                    ],
                    "deadline": "2025-04-15",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-partial",
        inventory_script=[
            ("inventory_snapshot", {}),
            ("assess_availability", {"item_name": "A4 paper"}),
        ],
        quoting_script=[
            ("retrieve_comparable_quotes", {"search_terms": ["A4 paper"]}),
            ("compute_quote", {}),
        ],
        fulfillment_script=[("commit_sale", {"item_name": "A4 paper"})],
        final_text="Your paper is confirmed; unfortunately we do not stock balloons.",
    )

    assert result.fulfillment.status is RequestStatus.PARTIAL
    assert result.fulfillment.cash_delta != 0
    assert len(result.fulfillment.fulfilled_lines) == 1
    assert result.fulfillment.fulfilled_lines[0].sale_transaction_id > 0
    rendered = result.customer_response.render().lower()
    assert "balloons" in rendered
    assert "cash balance" not in rendered
    assert result.quote.total == result.fulfillment.total
    assert any(event.action == "commit_sale" for event in result.events)


def test_restock_path_orders_stock_then_commits_the_sale(
    settings: Settings, scripted_model
) -> None:
    result = _run(
        settings,
        scripted_model,
        "Please send 200 sheets of colored paper by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [{"item_text": "colored paper", "quantity": 200, "unit": "sheets"}],
                    "deadline": "2025-04-15",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-restock",
        inventory_script=[
            ("inventory_snapshot", {}),
            ("assess_availability", {"item_name": "Colored paper"}),
            ("place_restock_order", {"item_name": "Colored paper"}),
        ],
        quoting_script=[
            ("retrieve_comparable_quotes", {"search_terms": ["Colored paper"]}),
            ("compute_quote", {}),
        ],
        fulfillment_script=[("commit_sale", {"item_name": "Colored paper"})],
    )

    assert result.fulfillment.status is RequestStatus.FULFILLED
    line = result.fulfillment.fulfilled_lines[0]
    assert line.stock_order_transaction_id is not None
    assert line.sale_transaction_id > line.stock_order_transaction_id
    assert result.fulfillment.cash_delta != 0
    decision = result.inventory_decisions[0]
    assert decision.restock_quantity > 0
    assert decision.supplier_delivery_date is not None


def test_impossible_request_is_rejected_with_customer_reason(
    settings: Settings, scripted_model
) -> None:
    result = _run(
        settings,
        scripted_model,
        "I need 5,000 sheets of A3 paper delivered by April 2, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [{"item_text": "A3 paper", "quantity": 5000, "unit": "sheets"}],
                    "deadline": "2025-04-02",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-rejected",
        final_text="Unfortunately we cannot supply A3 paper.",
    )

    assert result.fulfillment.status is RequestStatus.REJECTED
    assert result.fulfillment.declined_lines
    assert result.fulfillment.declined_lines[0].reason_code == "unsupported"
    assert result.fulfillment.cash_delta == 0
    rendered = result.customer_response.render().lower()
    assert "not available" in rendered
    assert "cash balance" not in rendered


def test_incomplete_run_fails_safe_without_charges(settings: Settings, scripted_model) -> None:
    result = _run(
        settings,
        scripted_model,
        "I need 200 sheets of A4 paper by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [{"item_text": "A4 paper", "quantity": 200, "unit": "sheets"}],
                    "deadline": "2025-04-15",
                },
            ),
        ],
        request_id="integration-backstop",
    )

    assert result.fulfillment.status is RequestStatus.REJECTED
    assert result.fulfillment.cash_delta == 0
    assert result.fulfillment.declined_lines[0].reason_code == "not_assessed"
    assert any(event.action == "finalize_backstop" for event in result.events)


def test_clarify_first_holds_all_side_effects_for_ambiguous_lines(
    settings: Settings, scripted_model
) -> None:
    result = _run(
        settings,
        scripted_model,
        "I need 200 sheets of A4 paper and 5 packs of glossy paper by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [
                        {"item_text": "A4 paper", "quantity": 200, "unit": "sheets"},
                        {"item_text": "glossy paper", "quantity": 5, "unit": "packs"},
                    ],
                    "deadline": "2025-04-15",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-clarify-hold",
        clarify_first=True,
    )

    assert result.fulfillment.status is RequestStatus.NEEDS_CLARIFICATION
    assert result.fulfillment.cash_delta == 0
    assert not result.fulfillment.fulfilled_lines
    codes = {line.reason_code for line in result.fulfillment.declined_lines}
    assert codes == {"ambiguous", "awaiting_clarification"}
    actions = {event.action for event in result.events}
    assert "clarification_hold" in actions
    assert actions.isdisjoint({"place_restock_order", "commit_sale"})
    rendered = result.customer_response.render()
    assert "Please clarify:" in rendered
    # The awaiting_clarification item (A4 paper, just held) must NOT appear in the
    # customer-facing "Please clarify" list - it misleads the customer into thinking
    # their A4 paper order is being questioned. Only the genuinely ambiguous line
    # (packs of glossy paper) should prompt the customer to clarify.
    assert "pending resolution" not in rendered.lower()
    assert "glossy paper" in rendered.lower()


def test_clarify_first_still_executes_when_nothing_is_clarifiable(
    settings: Settings, scripted_model
) -> None:
    result = _run(
        settings,
        scripted_model,
        "I need 200 sheets of A4 paper and 100 balloons delivered by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [
                        {"item_text": "A4 paper", "quantity": 200, "unit": "sheets"},
                        {"item_text": "balloons", "quantity": 100, "unit": "balloons"},
                    ],
                    "deadline": "2025-04-15",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-clarify-executes",
        inventory_script=[
            ("inventory_snapshot", {}),
            ("assess_availability", {"item_name": "A4 paper"}),
        ],
        quoting_script=[
            ("retrieve_comparable_quotes", {"search_terms": ["A4 paper"]}),
            ("compute_quote", {}),
        ],
        fulfillment_script=[("commit_sale", {"item_name": "A4 paper"})],
        clarify_first=True,
    )

    assert result.fulfillment.status is RequestStatus.PARTIAL
    assert len(result.fulfillment.fulfilled_lines) == 1
    assert result.fulfillment.declined_lines[0].reason_code == "unsupported"


def test_events_stream_to_the_callback_as_the_run_progresses(
    settings: Settings, scripted_model
) -> None:
    streamed = []
    result = _run(
        settings,
        scripted_model,
        "Please send 200 sheets of colored paper by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [{"item_text": "colored paper", "quantity": 200, "unit": "sheets"}],
                    "deadline": "2025-04-15",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-stream",
        inventory_script=[
            ("inventory_snapshot", {}),
            ("assess_availability", {"item_name": "Colored paper"}),
            ("place_restock_order", {"item_name": "Colored paper"}),
        ],
        quoting_script=[
            ("retrieve_comparable_quotes", {"search_terms": ["Colored paper"]}),
            ("compute_quote", {}),
        ],
        fulfillment_script=[("commit_sale", {"item_name": "Colored paper"})],
        on_event=streamed.append,
    )

    assert streamed == result.events
    assert [event.sequence for event in streamed] == list(range(1, len(streamed) + 1))
    assert any(event.action == "comparables_check" for event in streamed)


def test_leaky_model_summary_is_replaced_by_safe_default(
    settings: Settings, scripted_model
) -> None:
    result = _run(
        settings,
        scripted_model,
        "I need 100 balloons by April 15, 2025.",
        orchestrator_script=[
            (
                "resolve_catalog_items",
                {
                    "lines": [{"item_text": "balloons", "quantity": 100, "unit": "balloons"}],
                    "deadline": "2025-04-15",
                },
            ),
            ("consult_inventory", {}),
            ("request_quote", {}),
            ("finalize_order", {}),
        ],
        request_id="integration-leak-guard",
        final_text="Do not worry, our cash balance can absorb this rejection.",
    )

    rendered = result.customer_response.render().lower()
    assert "cash balance" not in rendered
    assert result.customer_response.summary != (
        "Do not worry, our cash balance can absorb this rejection."
    )
