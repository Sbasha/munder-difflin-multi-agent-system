"""Negotiation simulator tests with scripted team and customer models."""

from contextlib import ExitStack
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from munder_difflin.agents import team
from munder_difflin.config import Settings
from munder_difflin.models import RequestStatus
from munder_difflin.negotiation import CustomerTurn, customer_agent, run_negotiation
from munder_difflin.orchestrator import MunderDifflinSystem

_BALLOONS_SCRIPT = [
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
]

_COLORED_PAPER_SCRIPT = [
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
]


def _customer_model(
    turns: list[dict[str, Any]],
    transcript: list[str] | None = None,
) -> FunctionModel:
    """Replay one structured customer turn per agent run, capturing prompts.

    ``transcript`` receives one joined string per model call, so tests can
    inspect exactly what the customer agent saw on each turn.
    """

    queue = list(turns)

    def run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if transcript is not None:
            seen = [
                content
                for message in messages
                for part in getattr(message, "parts", [])
                if isinstance(content := getattr(part, "content", ""), str)
            ]
            transcript.append("\n".join(seen))
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args=queue.pop(0))])

    return FunctionModel(run)


def _sequential_scripts_model(
    scripts: list[list[tuple[str, dict[str, Any]]]],
    final_text: str = "Thanks for your order.",
) -> FunctionModel:
    """Serve a different tool-call script for each successive agent run."""

    state = {"run_index": -1}

    def run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        completed = sum(
            1
            for message in messages
            for part in getattr(message, "parts", [])
            if getattr(part, "part_kind", "") == "tool-return"
        )
        if completed == 0:
            state["run_index"] += 1
        script = scripts[state["run_index"]]
        if completed < len(script):
            tool_name, args = script[completed]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(final_text)])

    return FunctionModel(run)


def _system(settings: Settings) -> MunderDifflinSystem:
    system = MunderDifflinSystem(settings)
    system.initialize()
    return system


def test_customer_revision_converts_a_rejection_into_a_sale(
    settings: Settings, scripted_model
) -> None:
    system = _system(settings)
    revision = "Please send 200 sheets of colored paper by April 15, 2025."
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(
                model=_sequential_scripts_model([_BALLOONS_SCRIPT, _COLORED_PAPER_SCRIPT])
            )
        )
        stack.enter_context(
            team.inventory_agent.override(
                model=scripted_model(
                    [
                        ("inventory_snapshot", {}),
                        ("assess_availability", {"item_name": "Colored paper"}),
                        ("place_restock_order", {"item_name": "Colored paper"}),
                    ]
                )
            )
        )
        stack.enter_context(
            team.quoting_agent.override(
                model=scripted_model(
                    [
                        ("retrieve_comparable_quotes", {"search_terms": ["Colored paper"]}),
                        ("compute_quote", {}),
                    ]
                )
            )
        )
        stack.enter_context(
            team.fulfillment_agent.override(
                model=scripted_model([("commit_sale", {"item_name": "Colored paper"})])
            )
        )
        stack.enter_context(
            customer_agent.override(
                model=_customer_model([{"action": "revise", "message": revision}])
            )
        )
        result = run_negotiation(
            system,
            request="I need 100 balloons by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=3,
        )

    assert result.outcome == "fulfilled"
    assert len(result.rounds) == 2
    assert result.rounds[0].status is RequestStatus.REJECTED
    assert result.rounds[0].customer_action == "revise"
    assert result.rounds[1].request_text == revision
    assert result.rounds[1].status is RequestStatus.FULFILLED
    assert result.rounds[1].customer_action is None
    assert result.rounds[1].quoted_total > 0


def test_customer_walks_away_seeing_only_customer_safe_text(settings: Settings) -> None:
    system = _system(settings)
    transcript: list[str] = []
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(model=_sequential_scripts_model([_BALLOONS_SCRIPT]))
        )
        stack.enter_context(
            customer_agent.override(
                model=_customer_model(
                    [{"action": "walk_away", "message": "We will source balloons elsewhere."}],
                    transcript=transcript,
                )
            )
        )
        result = run_negotiation(
            system,
            request="I need 100 balloons by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=3,
        )

    assert result.outcome == "walked_away"
    assert len(result.rounds) == 1
    assert result.rounds[0].status is RequestStatus.REJECTED
    assert result.rounds[0].customer_action == "walk_away"
    seen = "\n".join(transcript).lower()
    assert "balloons" in seen
    for term in ("cash balance", "profit margin", "traceback"):
        assert term not in seen


def test_round_budget_bounds_the_negotiation(settings: Settings) -> None:
    system = _system(settings)
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(
                model=_sequential_scripts_model([_BALLOONS_SCRIPT, _BALLOONS_SCRIPT])
            )
        )
        stack.enter_context(
            customer_agent.override(
                model=_customer_model(
                    [
                        {
                            "action": "revise",
                            "message": "Please retry: 100 balloons by April 20, 2025.",
                        }
                    ]
                )
            )
        )
        result = run_negotiation(
            system,
            request="I need 100 balloons by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=2,
        )

    assert result.outcome == "round_limit"
    assert len(result.rounds) == 2
    assert result.rounds[0].customer_action == "revise"
    assert result.rounds[1].customer_action is None


def test_partial_commitment_ends_the_negotiation(settings: Settings, scripted_model) -> None:
    system = _system(settings)
    partial_script = [
        (
            "resolve_catalog_items",
            {
                "lines": [
                    {"item_text": "A4 paper", "quantity": 200, "unit": "sheets"},
                    {"item_text": "standard copy paper", "quantity": 250_000, "unit": "sheets"},
                ],
                "deadline": "2025-04-02",
            },
        ),
        ("consult_inventory", {}),
        ("request_quote", {}),
        ("finalize_order", {}),
    ]
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(model=_sequential_scripts_model([partial_script]))
        )
        stack.enter_context(
            team.inventory_agent.override(
                model=scripted_model(
                    [
                        ("inventory_snapshot", {}),
                        ("assess_availability", {"item_name": "A4 paper"}),
                        ("assess_availability", {"item_name": "Standard copy paper"}),
                    ]
                )
            )
        )
        stack.enter_context(
            team.quoting_agent.override(
                model=scripted_model(
                    [
                        ("retrieve_comparable_quotes", {"search_terms": ["A4 paper"]}),
                        ("compute_quote", {}),
                    ]
                )
            )
        )
        stack.enter_context(
            team.fulfillment_agent.override(
                model=scripted_model([("commit_sale", {"item_name": "A4 paper"})])
            )
        )
        stack.enter_context(customer_agent.override(model=_customer_model([])))
        result = run_negotiation(
            system,
            request="200 sheets of A4 paper and 250,000 sheets of copy paper by April 2, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=3,
        )

    assert result.outcome == "partial"
    assert len(result.rounds) == 1
    assert result.rounds[0].status is RequestStatus.PARTIAL
    assert result.rounds[0].customer_action is None
    assert result.rounds[0].quoted_total > 0


def test_customer_sees_the_full_conversation_history(settings: Settings) -> None:
    system = _system(settings)
    transcript: list[str] = []
    first_answer = (
        "We really do need the balloons for the gala decorations. "
        "Please send 100 balloons by April 18, 2025."
    )
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(
                model=_sequential_scripts_model(
                    [_BALLOONS_SCRIPT, _BALLOONS_SCRIPT, _BALLOONS_SCRIPT]
                )
            )
        )
        stack.enter_context(
            customer_agent.override(
                model=_customer_model(
                    [
                        {"action": "revise", "message": first_answer},
                        {
                            "action": "revise",
                            "message": "One last try: 100 balloons by April 25, 2025.",
                        },
                    ],
                    transcript=transcript,
                )
            )
        )
        result = run_negotiation(
            system,
            request="I need 100 balloons by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=3,
        )

    assert result.outcome == "round_limit"
    assert len(result.rounds) == 3
    assert len(transcript) == 2
    assert first_answer in transcript[1]
    assert "I need 100 balloons by April 15, 2025." in transcript[1]


def test_customer_turn_requires_a_message() -> None:
    with pytest.raises(ValidationError):
        CustomerTurn(action="revise", message="")


def test_round_budget_must_be_positive(settings: Settings) -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        run_negotiation(
            MunderDifflinSystem(settings),
            request="I need 100 balloons by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=0,
        )


_KRAFT_ENVELOPES_HOLD_SCRIPT = [
    (
        "resolve_catalog_items",
        {
            "lines": [{"item_text": "kraft paper envelopes", "quantity": 500, "unit": "envelopes"}],
            "deadline": "2025-04-15",
        },
    ),
    ("consult_inventory", {}),
    ("request_quote", {}),
    ("finalize_order", {}),
]

_ENVELOPES_RESOLVED_SCRIPT = [
    (
        "resolve_catalog_items",
        {
            "lines": [{"item_text": "Envelopes", "quantity": 500, "unit": "envelopes"}],
            "deadline": "2025-04-15",
        },
    ),
    ("consult_inventory", {}),
    ("request_quote", {}),
    ("finalize_order", {}),
]


def test_catalog_suggestion_in_response_enables_convergence_in_two_rounds(
    settings: Settings, scripted_model
) -> None:
    """An ambiguous request holds with catalog suggestions; the customer picks one and closes."""

    system = _system(settings)
    revision = "Please send 500 Envelopes by April 15, 2025."
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(
                model=_sequential_scripts_model(
                    [_KRAFT_ENVELOPES_HOLD_SCRIPT, _ENVELOPES_RESOLVED_SCRIPT]
                )
            )
        )
        stack.enter_context(
            team.inventory_agent.override(
                model=scripted_model(
                    [
                        ("inventory_snapshot", {}),
                        ("assess_availability", {"item_name": "Envelopes"}),
                    ]
                )
            )
        )
        stack.enter_context(
            team.quoting_agent.override(
                model=scripted_model(
                    [
                        ("retrieve_comparable_quotes", {"search_terms": ["Envelopes"]}),
                        ("compute_quote", {}),
                    ]
                )
            )
        )
        stack.enter_context(
            team.fulfillment_agent.override(
                model=scripted_model([("commit_sale", {"item_name": "Envelopes"})])
            )
        )
        stack.enter_context(
            customer_agent.override(
                model=_customer_model([{"action": "revise", "message": revision}])
            )
        )
        result = run_negotiation(
            system,
            request="Please send 500 kraft paper envelopes by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=3,
        )

    assert result.outcome == "fulfilled"
    assert len(result.rounds) == 2
    assert result.rounds[0].status is RequestStatus.NEEDS_CLARIFICATION
    assert result.rounds[0].customer_action == "revise"
    assert result.rounds[1].request_text == revision
    assert result.rounds[1].status is RequestStatus.FULFILLED
    assert result.rounds[1].quoted_total > 0


def test_clarification_response_contains_catalog_suggestions(
    settings: Settings, scripted_model
) -> None:
    """The customer-visible response for an ambiguous request names items from our catalog."""

    system = _system(settings)
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(
                model=_sequential_scripts_model([_KRAFT_ENVELOPES_HOLD_SCRIPT])
            )
        )
        stack.enter_context(
            customer_agent.override(
                model=_customer_model(
                    [{"action": "walk_away", "message": "Nevermind, found another supplier."}]
                )
            )
        )
        result = run_negotiation(
            system,
            request="Please send 500 kraft paper envelopes by April 15, 2025.",
            request_date=date(2025, 4, 1),
            max_rounds=2,
        )

    assert result.rounds[0].status is RequestStatus.NEEDS_CLARIFICATION
    response_text = result.rounds[0].response_text.lower()
    assert "envelopes" in response_text or "kraft paper" in response_text
