"""Munder Difflin multi-agent system - evaluation entry point.

Running this script evaluates the four-agent team against the complete
20-request sample dataset and writes ``test_results.csv`` with its run
manifest into a timestamped folder under ``runs/``; pass
``--output test_results.csv`` to reproduce the committed submission
artifact at the repository root. The implementation lives in ``src/munder_difflin``:
the agents and their tools in ``agents/team.py``, the per-request harness
in ``orchestrator.py``, and the seven required database helpers in
``db/helpers.py`` (re-exported below so they are importable from this
module as well).

Usage:
    python project_starter.py                            # evaluation into runs/<timestamp>/
    python project_starter.py --limit 3                  # quick smoke run
    python project_starter.py --watch                    # stream agent events while it runs
    python project_starter.py --output test_results.csv  # write the submission artifact

Model credentials are read from the environment or a ``.env`` file; see
``.env.example``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rich.console import Console

from munder_difflin.config import Settings
from munder_difflin.db.helpers import (
    create_transaction,
    generate_financial_report,
    get_all_inventory,
    get_cash_balance,
    get_stock_level,
    get_supplier_delivery_date,
    init_database,
    search_quote_history,
)
from munder_difflin.evaluation import check_rubric_gates, run_evaluation, timestamped_output_path
from munder_difflin.orchestrator import MunderDifflinSystem
from munder_difflin.ui import evaluation_reporters

__all__ = [
    "MunderDifflinSystem",
    "create_transaction",
    "generate_financial_report",
    "get_all_inventory",
    "get_cash_balance",
    "get_stock_level",
    "get_supplier_delivery_date",
    "init_database",
    "run_test_scenarios",
    "search_quote_history",
]


def run_test_scenarios(
    limit: int | None = None,
    watch: bool = False,
    output: Path | None = None,
) -> None:
    """Run the sample requests through the agent team and report the results."""

    settings = Settings(_env_file=_PROJECT_ROOT / ".env", data_dir=_PROJECT_ROOT / "data")
    if not settings.live_model_enabled:
        raise SystemExit(
            "No model credentials found. Set UDACITY_OPENAI_API_KEY, OPENAI_API_KEY, "
            "or LLM_API_KEY in your environment or .env file."
        )
    results_path = output or timestamped_output_path()
    on_start, on_event, on_complete = evaluation_reporters(Console(), watch=watch)
    _, metrics = run_evaluation(
        settings,
        results_path,
        limit=limit,
        on_request_start=on_start,
        on_event=on_event,
        on_request_complete=on_complete,
    )
    print(metrics)
    print(f"Results written to {results_path}")
    if limit is None:
        for message in check_rubric_gates(results_path):
            print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="evaluate only the first N requests")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="stream agent events while the evaluation runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="results CSV path (default: runs/<timestamp>/test_results.csv)",
    )
    arguments = parser.parse_args()
    run_test_scenarios(limit=arguments.limit, watch=arguments.watch, output=arguments.output)
