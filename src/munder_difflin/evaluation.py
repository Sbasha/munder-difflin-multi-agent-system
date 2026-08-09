"""Reproducible evaluation harness and executable rubric gates."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from munder_difflin.config import Settings
from munder_difflin.db.helpers import generate_financial_report
from munder_difflin.models import RequestStatus, RunEvent
from munder_difflin.orchestrator import FORBIDDEN_CUSTOMER_TERMS, MunderDifflinSystem


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Auditable aggregate outcomes for one complete evaluation run."""

    total_requests: int
    fully_fulfilled: int
    partially_fulfilled: int
    rejected: int
    requests_with_cash_change: int
    customer_data_leaks: int
    negative_inventory_items: int


_FAILURE_RESPONSE = (
    "We were unable to process this request automatically. No charges were made; "
    "please contact us and we will follow up directly."
)


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def timestamped_output_path(base_dir: Path | None = None) -> Path:
    """Return a fresh results location under the gitignored runs directory.

    Every run gets its own folder, so experiments never overwrite the
    committed submission artifacts; promoting a run is an explicit
    ``--output test_results.csv``.
    """

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (base_dir or Path("runs")) / stamp / "test_results.csv"


def run_evaluation(
    settings: Settings,
    output_path: Path,
    *,
    limit: int | None = None,
    seed: int = 137,
    on_request_start: Callable[[str, str], None] | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    on_request_complete: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, EvaluationMetrics]:
    """Run sample requests in date order against a fresh seeded database.

    The optional callbacks report live progress: request start, every agent
    event as it is emitted, and the recorded outcome of each request. The
    evaluation itself stays headless; presentation belongs to the caller.
    """

    sample_path = settings.data_dir / "quote_requests_sample.csv"
    requests = pd.read_csv(sample_path)
    requests["request_date"] = pd.to_datetime(
        requests["request_date"],
        format="%m/%d/%y",
        errors="raise",
    )
    requests = requests.sort_values("request_date", kind="stable")
    if limit is not None:
        requests = requests.iloc[:limit]
    requests = requests.assign(
        request_label=[f"sample-{int(index) + 1:02d}" for index in requests.index.tolist()]
    )

    system = MunderDifflinSystem(settings)
    system.initialize(seed=seed)
    records: list[dict[str, Any]] = []
    event_path = output_path.with_name("run-events.jsonl")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("w", encoding="utf-8") as event_file:
        for _, row in requests.iterrows():
            request_date = row["request_date"].date()
            request_id = str(row["request_label"])
            if on_request_start is not None:
                on_request_start(request_id, request_date.isoformat())
            result = None
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    result = system.process_request(
                        str(row["request"]),
                        request_date=request_date,
                        customer_context=str(row["job"]),
                        event=str(row["event"]),
                        request_id=request_id,
                        on_event=on_event,
                    )
                    break
                except Exception as error:
                    last_error = error
            if result is None:
                records.append(_failure_record(request_id, request_date, row, error=last_error))
                if on_request_complete is not None:
                    on_request_complete(records[-1])
                continue
            for event in result.events:
                event_file.write(event.model_dump_json() + "\n")
            records.append(
                {
                    "request_id": request_id,
                    "request_date": request_date.isoformat(),
                    "job": row["job"],
                    "event": row["event"],
                    "status": result.fulfillment.status.value,
                    "fulfilled_line_count": len(result.fulfillment.fulfilled_lines),
                    "declined_line_count": len(result.fulfillment.declined_lines),
                    "fulfilled_items": json.dumps(
                        [
                            line.model_dump(mode="json")
                            for line in result.fulfillment.fulfilled_lines
                        ],
                        sort_keys=True,
                    ),
                    "declined_items": json.dumps(
                        [
                            line.model_dump(mode="json")
                            for line in result.fulfillment.declined_lines
                        ],
                        sort_keys=True,
                    ),
                    "reason_codes": json.dumps(
                        [line.reason_code for line in result.fulfillment.declined_lines]
                    ),
                    "quoted_total": float(result.fulfillment.total),
                    "cash_before": float(result.fulfillment.cash_before),
                    "cash_after": float(result.fulfillment.cash_after),
                    "cash_delta": float(result.fulfillment.cash_delta),
                    "customer_response": result.customer_response.render(),
                    "trace_id": result.trace_id,
                }
            )
            if on_request_complete is not None:
                on_request_complete(records[-1])

    results = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    final_date = (
        max(requests["request_date"]).date()
        if not requests.empty
        else date.fromisoformat("2025-01-01")
    )
    financial_report = generate_financial_report(final_date.isoformat())
    negative_inventory = sum(
        1 for item in financial_report["inventory_summary"] if item["stock"] < 0
    )
    leaks = sum(
        any(term in response.lower() for term in FORBIDDEN_CUSTOMER_TERMS)
        for response in results.get("customer_response", pd.Series(dtype=str))
    )
    metrics = EvaluationMetrics(
        total_requests=len(results),
        fully_fulfilled=int((results["status"] == RequestStatus.FULFILLED.value).sum()),
        partially_fulfilled=int((results["status"] == RequestStatus.PARTIAL.value).sum()),
        rejected=int((results["status"] == RequestStatus.REJECTED.value).sum()),
        requests_with_cash_change=int((results["cash_delta"].abs() > 0.001).sum()),
        customer_data_leaks=int(leaks),
        negative_inventory_items=negative_inventory,
    )
    manifest = {
        "dataset": str(sample_path.name),
        "dataset_sha256": _hash_file(sample_path),
        "seed": seed,
        "model": settings.llm_model,
        "request_count": len(results),
        "metrics": asdict(metrics),
    }
    output_path.with_name("evaluation-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return results, metrics


def _failure_record(
    request_id: str,
    request_date: date,
    row: pd.Series,
    error: Exception | None,
) -> dict[str, Any]:
    """Record a safe rejection when a request could not be processed at all."""

    detail = type(error).__name__ if error is not None else "unknown"
    return {
        "request_id": request_id,
        "request_date": request_date.isoformat(),
        "job": row["job"],
        "event": row["event"],
        "status": RequestStatus.REJECTED.value,
        "fulfilled_line_count": 0,
        "declined_line_count": 0,
        "fulfilled_items": "[]",
        "declined_items": "[]",
        "reason_codes": json.dumps([f"processing_error:{detail}"]),
        "quoted_total": 0.0,
        "cash_before": 0.0,
        "cash_after": 0.0,
        "cash_delta": 0.0,
        "customer_response": _FAILURE_RESPONSE,
        "trace_id": "",
    }


def check_rubric_gates(results_path: Path, expected_requests: int = 20) -> list[str]:
    """Return passing gate messages or raise with all failures."""

    results = pd.read_csv(results_path)
    failures: list[str] = []
    passes: list[str] = []

    def check(condition: bool, message: str) -> None:
        (passes if condition else failures).append(message)

    check(len(results) == expected_requests, f"processed exactly {expected_requests} requests")
    check(
        int((results["status"] == RequestStatus.FULFILLED.value).sum()) >= 3,
        "at least three requests were fully fulfilled",
    )
    check(
        int((results["cash_delta"].abs() > 0.001).sum()) >= 3,
        "at least three requests changed the cash balance",
    )
    not_fulfilled = results[results["status"] != RequestStatus.FULFILLED.value]
    check(not not_fulfilled.empty, "at least one request was not fully fulfilled")
    check(
        bool(
            not_fulfilled["reason_codes"]
            .fillna("[]")
            .map(lambda value: value not in {"[]", ""})
            .all()
        ),
        "every non-fulfilled request includes reason codes",
    )
    forbidden_pattern = "|".join(re.escape(term) for term in FORBIDDEN_CUSTOMER_TERMS)
    check(
        not results["customer_response"]
        .str.contains(forbidden_pattern, case=False, regex=True)
        .any(),
        "customer responses contain no forbidden internal terms",
    )
    if failures:
        formatted = "\n".join(f"[FAIL] {failure}" for failure in failures)
        raise AssertionError(f"Rubric gates failed:\n{formatted}")
    return [f"[PASS] {message}" for message in passes]
