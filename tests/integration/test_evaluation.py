"""Evaluation harness contract tests with an autopilot agent team."""

from contextlib import ExitStack
from pathlib import Path

from munder_difflin.agents import team
from munder_difflin.config import Settings
from munder_difflin.evaluation import run_evaluation


def test_evaluation_emits_auditable_artifacts(
    settings: Settings, tmp_path: Path, scripted_model
) -> None:
    output = tmp_path / "test_results.csv"
    orchestrator_script = [
        (
            "resolve_catalog_items",
            {"lines": [{"item_text": "A4 paper", "quantity": 100, "unit": "sheets"}]},
        ),
        ("consult_inventory", {}),
        ("request_quote", {}),
        ("finalize_order", {}),
    ]
    inventory_script = [
        ("inventory_snapshot", {}),
        ("assess_availability", {"item_name": "A4 paper"}),
    ]
    quoting_script = [
        ("retrieve_comparable_quotes", {"search_terms": ["A4 paper"]}),
        ("compute_quote", {}),
    ]
    fulfillment_script = [("commit_sale", {"item_name": "A4 paper"})]

    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(model=scripted_model(orchestrator_script))
        )
        stack.enter_context(team.inventory_agent.override(model=scripted_model(inventory_script)))
        stack.enter_context(team.quoting_agent.override(model=scripted_model(quoting_script)))
        stack.enter_context(
            team.fulfillment_agent.override(model=scripted_model(fulfillment_script))
        )
        results, metrics = run_evaluation(settings, output, limit=3)

    assert len(results) == 3
    assert metrics.total_requests == 3
    assert metrics.fully_fulfilled == 3
    assert {
        "status",
        "reason_codes",
        "cash_before",
        "cash_after",
        "cash_delta",
        "customer_response",
        "trace_id",
    }.issubset(results.columns)
    assert output.exists()
    assert output.with_name("evaluation-manifest.json").exists()
    assert output.with_name("run-events.jsonl").exists()
    assert not output.with_name("advisor-recommendations.json").exists()
