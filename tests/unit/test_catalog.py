"""Deterministic catalog resolution tests."""

from datetime import date

from munder_difflin.catalog import CATALOG, resolve_requested_line
from munder_difflin.models import ResolutionStatus

_DEADLINE = date(2025, 4, 15)


def test_alias_resolution_maps_to_exact_catalog_item() -> None:
    line = resolve_requested_line("printer paper", "sheets", 100, _DEADLINE)

    assert line.resolution_status is ResolutionStatus.RESOLVED
    assert line.catalog_item == "Standard copy paper"
    assert line.normalized_quantity == 100


def test_reams_normalize_to_five_hundred_sheets_each() -> None:
    line = resolve_requested_line("A4 paper", "reams", 10, _DEADLINE)

    assert line.catalog_item == "A4 paper"
    assert line.normalized_quantity == 5_000


def test_unknown_products_fail_closed() -> None:
    line = resolve_requested_line("balloons", "balloons", 100, _DEADLINE)

    assert line.resolution_status is ResolutionStatus.UNSUPPORTED
    assert line.catalog_item is None
    assert line.resolution_reason is not None


def test_unsized_packs_fail_closed_as_ambiguous() -> None:
    line = resolve_requested_line("sticky notes", "packs", 5, _DEADLINE)

    assert line.resolution_status is ResolutionStatus.AMBIGUOUS
    assert line.catalog_item is None
    assert "pack" in (line.resolution_reason or "")


def test_every_alias_targets_a_real_catalog_item() -> None:
    from munder_difflin.catalog import _ALIASES

    missing = {canonical for canonical in _ALIASES.values() if canonical not in CATALOG}

    assert not missing
