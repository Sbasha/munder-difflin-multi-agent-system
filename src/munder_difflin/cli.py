"""Cross-platform command-line interface for demos and evaluation."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

from munder_difflin.config import Settings
from munder_difflin.evaluation import check_rubric_gates, run_evaluation, timestamped_output_path
from munder_difflin.agents.team import run_advisory
from munder_difflin.negotiation import run_negotiation
from munder_difflin.orchestrator import MunderDifflinSystem
from munder_difflin.ui import (
    WorkflowStream,
    evaluation_reporters,
    negotiation_reporters,
    present_advisory_report,
    present_negotiation_outcome,
    present_outcome,
    present_result,
)

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
    evaluate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="results CSV path (default: runs/<timestamp>/test_results.csv)",
    )
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--seed", type=int, default=137)
    evaluate.add_argument(
        "--watch",
        action="store_true",
        help="stream every agent event while the evaluation runs",
    )

    negotiate = subparsers.add_parser(
        "negotiate",
        help="Simulate a negotiating customer against the agent team",
    )
    negotiate.add_argument(
        "--sample",
        type=int,
        default=9,
        help="1-based row from quote_requests_sample.csv to negotiate (default: 9)",
    )
    negotiate.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="maximum negotiation rounds, 1 to 5 (default: 3)",
    )
    negotiate.add_argument("--no-reset", action="store_true")

    advise = subparsers.add_parser(
        "advise",
        help="Run the business advisor agent and print operational recommendations",
    )
    advise.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Analysis date in YYYY-MM-DD format (default: today)",
    )
    advise.add_argument("--no-reset", action="store_true")

    check = subparsers.add_parser("check", help="Check rubric acceptance gates")
    check.add_argument("results", type=Path, nargs="?", default=Path("test_results.csv"))
    check.add_argument("--expected-requests", type=int, default=20)
    return parser


def main() -> None:
    """Execute a CLI command."""

    args = build_parser().parse_args()
    console = Console()
    if args.command in {"demo", "evaluate", "negotiate", "advise"}:
        settings = _settings(args)
        if not settings.live_model_enabled:
            console.print(
                "[red]No model credentials found.[/red] Set UDACITY_OPENAI_API_KEY, "
                "OPENAI_API_KEY, or LLM_API_KEY in your environment or .env file."
            )
            raise SystemExit(2)
    if args.command == "demo":
        system = MunderDifflinSystem(settings)
        if not args.no_reset:
            system.initialize()
        if args.no_animation:
            result = system.process_request(
                args.request,
                request_date=args.request_date,
                customer_context="demo customer",
                event="demo",
                request_id="demo-request",
            )
            present_result(result, console)
            return
        with WorkflowStream(args.request, "demo-request", console=console) as stream:
            result = system.process_request(
                args.request,
                request_date=args.request_date,
                customer_context="demo customer",
                event="demo",
                request_id="demo-request",
                on_event=stream.add,
            )
        present_outcome(result, console)
        return
    if args.command == "evaluate":
        output = args.output or timestamped_output_path()
        on_start, on_event, on_complete = evaluation_reporters(console, watch=args.watch)
        _, metrics = run_evaluation(
            settings,
            output,
            limit=args.limit,
            seed=args.seed,
            on_request_start=on_start,
            on_event=on_event,
            on_request_complete=on_complete,
        )
        console.print(metrics)
        console.print(f"Results written to {output}", highlight=False)
        return
    if args.command == "advise":
        system = MunderDifflinSystem(settings)
        if not args.no_reset:
            system.initialize()
        as_of = (args.as_of or date.today()).isoformat()
        console.print(f"[bold blue]Running business advisor as of {as_of}...[/bold blue]")
        report = run_advisory(settings, as_of, model=system.model)
        present_advisory_report(report, console)
        return
    if args.command == "negotiate":
        if not 1 <= args.rounds <= 5:
            raise SystemExit("--rounds must be between 1 and 5")
        samples = pd.read_csv(settings.data_dir / "quote_requests_sample.csv")
        if not 1 <= args.sample <= len(samples):
            raise SystemExit(f"--sample must be between 1 and {len(samples)}")
        row = samples.iloc[args.sample - 1]
        request_date = pd.to_datetime(str(row["request_date"]), format="%m/%d/%y").date()
        system = MunderDifflinSystem(settings)
        if not args.no_reset:
            system.initialize()
        on_round_start, on_event, on_round = negotiation_reporters(console)
        negotiation = run_negotiation(
            system,
            request=str(row["request"]),
            request_date=request_date,
            customer_context=str(row["job"]),
            event=str(row["event"]),
            max_rounds=args.rounds,
            request_id_prefix=f"negotiation-{args.sample:02d}",
            on_round_start=on_round_start,
            on_event=on_event,
            on_round=on_round,
        )
        present_negotiation_outcome(negotiation, console)
        return
    for message in check_rubric_gates(
        args.results,
        expected_requests=args.expected_requests,
    ):
        console.print(message)


if __name__ == "__main__":
    main()
