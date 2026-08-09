"""Evaluation harness contract tests with an autopilot agent team."""

import re
from contextlib import ExitStack
from pathlib import Path

from munder_difflin.agents import team
from munder_difflin.config import Settings
from munder_difflin.evaluation import run_evaluation, timestamped_output_path


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

    started: list[str] = []
    streamed: list[str] = []
    completed: list[str] = []
    with ExitStack() as stack:
        stack.enter_context(
            team.orchestrator_agent.override(model=scripted_model(orchestrator_script))
        )
        stack.enter_context(team.inventory_agent.override(model=scripted_model(inventory_script)))
        stack.enter_context(team.quoting_agent.override(model=scripted_model(quoting_script)))
        stack.enter_context(
            team.fulfillment_agent.override(model=scripted_model(fulfillment_script))
        )
        results, metrics = run_evaluation(
            settings,
            output,
            limit=3,
            on_request_start=lambda request_id, _date: started.append(request_id),
            on_event=lambda event: streamed.append(event.action),
            on_request_complete=lambda record: completed.append(str(record["request_id"])),
        )

    assert len(results) == 3
    assert metrics.total_requests == 3
    assert metrics.fully_fulfilled == 3
    assert started == completed == list(results["request_id"])
    assert "commit_sale" in streamed
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


def test_timestamped_output_path_gives_each_run_its_own_folder(tmp_path: Path) -> None:
    path = timestamped_output_path(tmp_path)

    assert path.name == "test_results.csv"
    assert path.parent.parent == tmp_path
    assert re.fullmatch(r"\d{8}-\d{6}", path.parent.name)
    assert timestamped_output_path().parts[0] == "runs"
