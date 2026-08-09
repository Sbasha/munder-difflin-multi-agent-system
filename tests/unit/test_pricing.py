"""Deterministic pricing and bulk-discount tier tests."""

from decimal import Decimal

from munder_difflin.catalog import CATALOG
from munder_difflin.pricing import build_quote, discount_rate_for, price_line


def test_discount_tiers_and_boundaries() -> None:
    assert discount_rate_for(499) == Decimal("0")
    assert discount_rate_for(500) == Decimal("0.05")
    assert discount_rate_for(999) == Decimal("0.05")
    assert discount_rate_for(1_000) == Decimal("0.10")
    assert discount_rate_for(4_999) == Decimal("0.10")
    assert discount_rate_for(5_000) == Decimal("0.15")


def test_top_tier_line_is_discounted_and_explained() -> None:
    line = price_line("A4 paper", 5_000, markup_rate=1.3)

    assert line.discount_rate == Decimal("0.15")
    assert line.unit_price >= CATALOG["A4 paper"].unit_price
    assert "15%" in line.rationale


def test_price_never_falls_below_catalog_cost() -> None:
    line = price_line("A4 paper", 5_000, markup_rate=1.05)

    assert line.unit_price == CATALOG["A4 paper"].unit_price


def test_multi_line_quote_reconciles() -> None:
    quote = build_quote(
        [("A4 paper", 5_000), ("Cardstock", 200)],
        markup_rate=1.3,
        historical_quotes_consulted=2,
    )

    assert quote.total == sum((line.total for line in quote.lines), Decimal("0"))
    assert quote.discount_total == quote.subtotal - quote.total
    assert quote.historical_quotes_consulted == 2
