"""Cross-platform command-line interface for demos and evaluation."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from rich.console import Console

from munder_difflin.config import Settings
from munder_difflin.evaluation import check_rubric_gates, run_evaluation
from munder_difflin.orchestrator import MunderDifflinSystem
from munder_difflin.ui import present_result

_DEMO_REQUEST = (
    "I need 500 sheets of A4 paper, 200 sheets of colored paper, and "
    "100 balloons delivered by April 15, 2025."
)


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        database_url=args.database_url,
        data_dir=args.data_dir,
    )


def _default_data_dir() -> Path:
    """Prefer the repository's data directory when running from the repo root."""

    candidate = Path("data")
    if (candidate / "quote_requests_sample.csv").exists():
        return candidate
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI contract."""

    parser = argparse.ArgumentParser(
        prog="munder-difflin",
        description="Multi-agent inventory, quoting, and fulfillment system",
    )
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--database-url", default="sqlite:///munder_difflin.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run one customer request")
    demo.add_argument("--request", default=_DEMO_REQUEST)
    demo.add_argument("--request-date", type=date.fromisoformat, default=date(2025, 4, 1))
    demo.add_argument("--no-animation", action="store_true")
    demo.add_argument("--no-reset", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Run the required sample evaluation")
    evaluate.add_argument("--output", type=Path, default=Path("test_results.csv"))
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--seed", type=int, default=137)

    check = subparsers.add_parser("check", help="Check rubric acceptance gates")
    check.add_argument("results", type=Path, nargs="?", default=Path("test_results.csv"))
    check.add_argument("--expected-requests", type=int, default=20)
    return parser


def main() -> None:
    """Execute a CLI command."""

    args = build_parser().parse_args()
    console = Console()
    if args.command in {"demo", "evaluate"}:
        settings = _settings(args)
        if not settings.live_model_enabled:
            console.print(
                "[red]No model credentials found.[/red] Set LLM_API_KEY or "
                "UDACITY_OPENAI_API_KEY in your environment or .env file."
            )
            raise SystemExit(2)
    if args.command == "demo":
        system = MunderDifflinSystem(settings)
        if not args.no_reset:
            system.initialize()
        result = system.process_request(
            args.request,
            request_date=args.request_date,
            customer_context="demo customer",
            event="demo",
            request_id="demo-request",
        )
        present_result(result, animate=not args.no_animation)
        return
    if args.command == "evaluate":
        _, metrics = run_evaluation(
            settings,
            args.output,
            limit=args.limit,
            seed=args.seed,
        )
        console.print(metrics)
        return
    for message in check_rubric_gates(
        args.results,
        expected_requests=args.expected_requests,
    ):
        console.print(message)


if __name__ == "__main__":
    main()
