"""Rich terminal presentation driven only by structured run events."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from munder_difflin.models import AdvisoryReport, ProcessResult, RunEvent
from munder_difflin.negotiation import NegotiationResult, NegotiationRound

_STATUS_STYLES = {
    "fulfilled": "green",
    "partial": "yellow",
    "rejected": "red",
    "needs_clarification": "cyan",
}


def _status_label(status_value: str) -> str:
    return status_value.replace("_", " ")


def _event_table(events: list[RunEvent]) -> Table:
    table = Table(expand=True)
    table.add_column("Step", width=5, justify="right")
    table.add_column("Agent", width=14)
    table.add_column("Action", width=24)
    table.add_column("Outcome", width=12)
    table.add_column("Detail")
    for event in events:
        table.add_row(
            str(event.sequence),
            event.agent,
            event.action,
            event.outcome,
            event.detail,
        )
    return table


def _request_panel(request_text: str, request_id: str) -> Panel:
    return Panel(request_text, title=f"Request {request_id}", border_style="blue")


def _response_panel(result: ProcessResult) -> Panel:
    return Panel(
        result.customer_response.render(),
        title=f"Customer response - {result.fulfillment.status.value}",
        border_style="green" if result.fulfillment.fulfilled_lines else "yellow",
    )


class WorkflowStream:
    """Live terminal view fed by run events the moment the agents emit them.

    Use as a context manager and pass ``add`` as the ``on_event`` callback of
    ``MunderDifflinSystem.process_request``; the table grows while the real
    run is still in flight, so the pacing on screen is the actual processing.
    """

    def __init__(
        self,
        request_text: str,
        request_id: str,
        console: Console | None = None,
    ) -> None:
        self._header = _request_panel(request_text, request_id)
        self._events: list[RunEvent] = []
        self._live = Live(console=console or Console(), refresh_per_second=12)

    def __enter__(self) -> WorkflowStream:
        self._live.__enter__()
        self._live.update(Group(self._header, _event_table(self._events)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._live.__exit__(exc_type, exc, tb)

    def add(self, event: RunEvent) -> None:
        """Record one event; safe to call from tool worker threads."""

        self._events.append(event)
        self._live.update(Group(self._header, _event_table(self._events)))


def present_outcome(result: ProcessResult, console: Console | None = None) -> None:
    """Render the final customer-safe response panel."""

    (console or Console()).print(_response_panel(result))


def present_result(result: ProcessResult, console: Console | None = None) -> None:
    """Render a completed run statically: request, event table, response."""

    target = console or Console()
    target.print(
        _request_panel(result.parsed_request.original_request, result.parsed_request.request_id)
    )
    target.print(_event_table(result.events))
    target.print(_response_panel(result))


def evaluation_reporters(
    console: Console,
    *,
    watch: bool,
) -> tuple[
    Callable[[str, str], None],
    Callable[[RunEvent], None] | None,
    Callable[[dict[str, Any]], None],
]:
    """Build (on_request_start, on_event, on_request_complete) printers.

    The completion line always prints so an evaluation run shows progress;
    ``watch`` additionally streams every agent event as it happens.
    """

    def on_request_start(request_id: str, request_date: str) -> None:
        if watch:
            console.print(f"[bold blue]{request_id}[/bold blue] · {request_date}", highlight=False)

    def on_event(event: RunEvent) -> None:
        console.print(
            f"  [dim]{event.agent}[/dim] · {event.action} · [cyan]{event.outcome}[/cyan] "
            f"· {event.detail}",
            highlight=False,
        )

    def on_request_complete(record: dict[str, Any]) -> None:
        status = str(record["status"])
        style = _STATUS_STYLES.get(status, "white")
        console.print(
            f"[bold]{record['request_id']}[/bold] {record['request_date']} · "
            f"[{style}]{status}[/{style}] · "
            f"{record['fulfilled_line_count']} supplied / {record['declined_line_count']} declined "
            f"· ${record['quoted_total']:.2f}",
            highlight=False,
        )

    return on_request_start, (on_event if watch else None), on_request_complete


def negotiation_reporters(
    console: Console,
) -> tuple[
    Callable[[int, str], None],
    Callable[[RunEvent], None],
    Callable[[NegotiationRound], None],
]:
    """Build (on_round_start, on_event, on_round) printers for a negotiation."""

    def on_round_start(round_number: int, request_text: str) -> None:
        if round_number == 1:
            console.print(Panel(request_text, title="Customer request", border_style="blue"))
            return
        # Rounds after the first open with the customer email already on
        # screen; a rule avoids reprinting the same text as a second panel.
        console.rule(f"[bold blue]Round {round_number}[/bold blue]")

    def on_event(event: RunEvent) -> None:
        console.print(
            f"  [dim]{event.agent}[/dim] · {event.action} · [cyan]{event.outcome}[/cyan] "
            f"· {event.detail}",
            highlight=False,
        )

    def on_round(round_record: NegotiationRound) -> None:
        style = _STATUS_STYLES.get(round_record.status.value, "white")
        console.print(
            Panel(
                round_record.response_text,
                title=f"Company response - {_status_label(round_record.status.value)}",
                border_style=style,
            )
        )
        if round_record.customer_action:
            # A revision is not repeated here; it opens the next round's panel.
            console.print(
                Panel(
                    round_record.customer_message or "",
                    title=f"Customer - {round_record.customer_action.replace('_', ' ')}",
                    border_style="magenta",
                )
            )

    return on_round_start, on_event, on_round


def present_advisory_report(report: AdvisoryReport, console: Console | None = None) -> None:
    """Render the business advisor's recommendations to the terminal."""

    target = console or Console()
    _PRIORITY_STYLES = {"high": "red", "medium": "yellow", "low": "cyan"}
    target.print(
        Panel(
            f"As of {report.as_of_date} | Cash: ${report.cash_balance:,.2f} | "
            f"Total assets: ${report.total_assets:,.2f}\n\n{report.summary}",
            title="Business Advisor Report",
            border_style="blue",
        )
    )
    for i, rec in enumerate(report.recommendations, 1):
        priority_style = _PRIORITY_STYLES.get(rec.priority.lower(), "white")
        target.print(
            Panel(
                f"[bold]Finding:[/bold] {rec.finding}\n\n"
                f"[bold]Recommendation:[/bold] {rec.recommendation}",
                title=f"[{priority_style}]{i}. {rec.category} [{rec.priority.upper()}][/{priority_style}]",
                border_style=priority_style,
            )
        )


def present_negotiation_outcome(result: NegotiationResult, console: Console | None = None) -> None:
    """Render the terminal outcome of a negotiation transcript."""

    target = console or Console()
    final_round = result.rounds[-1]
    style = _STATUS_STYLES.get(final_round.status.value, "white")
    target.print(
        Panel(
            f"Outcome: {result.outcome.replace('_', ' ')} after {len(result.rounds)} "
            f"round{'s' if len(result.rounds) != 1 else ''}; "
            f"final order status {_status_label(final_round.status.value)}, "
            f"committed total ${final_round.quoted_total:.2f}.",
            title="Negotiation result",
            border_style=style,
        )
    )
