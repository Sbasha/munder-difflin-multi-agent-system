"""Terminal presentation smoke tests over an in-memory console."""

from io import StringIO

from rich.console import Console

from munder_difflin.models import RunEvent
from munder_difflin.ui import WorkflowStream, evaluation_reporters


def _event(sequence: int) -> RunEvent:
    return RunEvent(
        trace_id="trace",
        sequence=sequence,
        agent="Inventory",
        action="assess_availability",
        outcome="deliverable",
        detail="A4 paper: 200 units available from stock",
    )


def _console(buffer: StringIO) -> Console:
    return Console(file=buffer, width=120, force_terminal=False)


def test_workflow_stream_renders_streamed_events() -> None:
    buffer = StringIO()
    with WorkflowStream("200 sheets of A4 paper", "ui-test", console=_console(buffer)) as stream:
        stream.add(_event(1))
        stream.add(_event(2))
    output = buffer.getvalue()

    assert "Request ui-test" in output
    assert "assess_availability" in output


def test_evaluation_reporters_print_progress_and_optionally_events() -> None:
    buffer = StringIO()
    on_start, on_event, on_complete = evaluation_reporters(_console(buffer), watch=True)
    assert on_event is not None
    on_start("sample-01", "2025-04-01")
    on_event(_event(1))
    on_complete(
        {
            "request_id": "sample-01",
            "request_date": "2025-04-01",
            "status": "partial",
            "fulfilled_line_count": 1,
            "declined_line_count": 2,
            "quoted_total": 123.45,
        }
    )
    output = buffer.getvalue()

    assert "sample-01" in output
    assert "assess_availability" in output
    assert "1 supplied / 2 declined" in output

    _, quiet_on_event, _ = evaluation_reporters(_console(StringIO()), watch=False)
    assert quiet_on_event is None
