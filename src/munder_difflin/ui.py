"""Rich terminal presentation driven only by structured run events."""

from __future__ import annotations

import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from munder_difflin.models import ProcessResult, RunEvent


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


def present_result(
    result: ProcessResult,
    *,
    animate: bool = True,
    delay_seconds: float = 0.18,
) -> None:
    """Render the agent workflow and final customer-safe response."""

    console = Console()
    visible: list[RunEvent] = []
    with Live(console=console, refresh_per_second=12, transient=animate) as live:
        for event in result.events:
            visible.append(event)
            live.update(
                Group(
                    Panel(
                        result.parsed_request.original_request,
                        title=f"Request {result.parsed_request.request_id}",
                        border_style="blue",
                    ),
                    _event_table(visible),
                )
            )
            if animate:
                time.sleep(delay_seconds)
    console.print(
        Panel(
            result.customer_response.render(),
            title=f"Customer response - {result.fulfillment.status.value}",
            border_style="green" if result.fulfillment.fulfilled_lines else "yellow",
        )
    )
