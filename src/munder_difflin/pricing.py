"""Deterministic pricing rules with auditable cost protection."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from munder_difflin.catalog import CATALOG
from munder_difflin.models import QuoteLine, QuoteResult

_CENT = Decimal("0.01")


def money(value: Decimal | float | str) -> Decimal:
    """Round currency consistently to cents."""

    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


def discount_rate_for(quantity: int) -> Decimal:
    """Return the published volume tier for a normalized unit quantity."""

    if quantity >= 5_000:
        return Decimal("0.15")
    if quantity >= 1_000:
        return Decimal("0.10")
    if quantity >= 500:
        return Decimal("0.05")
    return Decimal("0")


def compare_with_history(total: Decimal, comparable_totals: list[Decimal]) -> tuple[str, str]:
    """Compare a quote total with retrieved historical comparables.

    Returns a deterministic (outcome, customer_note) pair. The band is wide,
    half the lowest to double the highest comparable total, because historical
    orders differ in composition: the check flags scale anomalies for the
    audit trail and never adjusts a price.
    """

    count = len(comparable_totals)
    if count == 0:
        return (
            "no_comparables",
            "No comparable historical quotes were found for this order profile.",
        )
    lower = min(comparable_totals) / 2
    upper = max(comparable_totals) * 2
    noun = "quote" if count == 1 else "quotes"
    if lower <= total <= upper:
        return (
            "in_band",
            f"The total is consistent with the {count} comparable historical {noun} reviewed.",
        )
    return (
        "out_of_band",
        f"The total falls outside the range of the {count} comparable historical {noun} "
        "reviewed; the published price list and volume tiers still determine the price.",
    )


def price_line(catalog_item: str, quantity: int, markup_rate: float) -> QuoteLine:
    """Price one line without allowing the sale price below catalog cost."""

    cost = CATALOG[catalog_item].unit_price
    marked_up_unit = money(cost * Decimal(str(markup_rate)))
    discount_rate = discount_rate_for(quantity)
    discounted_unit = money(marked_up_unit * (Decimal("1") - discount_rate))
    safe_unit = max(cost, discounted_unit)
    subtotal = money(marked_up_unit * quantity)
    total = money(safe_unit * quantity)
    if discount_rate:
        rationale = (
            f"A {discount_rate:.0%} volume discount was applied for {quantity:,} units; "
            "the final amount is shown in the line total."
        )
    else:
        rationale = "Standard transparent pricing applies because this line is below a volume tier."
    return QuoteLine(
        catalog_item=catalog_item,
        quantity=quantity,
        unit_price=safe_unit,
        discount_rate=discount_rate,
        subtotal=subtotal,
        total=total,
        rationale=rationale,
    )


def build_quote(
    deliverable_lines: list[tuple[str, int]],
    markup_rate: float,
    historical_quotes_consulted: int,
    comparable_totals: list[Decimal] | None = None,
) -> QuoteResult:
    """Build a reconciled quote from deterministic line calculations.

    ``comparable_totals`` is None when no historical search ran; a list
    (possibly empty) records the search result and yields a comparables note.
    """

    lines = [price_line(item, quantity, markup_rate) for item, quantity in deliverable_lines]
    subtotal = money(sum((line.subtotal for line in lines), Decimal("0")))
    total = money(sum((line.total for line in lines), Decimal("0")))
    note = None if comparable_totals is None else compare_with_history(total, comparable_totals)[1]
    return QuoteResult(
        lines=lines,
        subtotal=subtotal,
        discount_total=money(subtotal - total),
        total=total,
        historical_quotes_consulted=historical_quotes_consulted,
        comparables_note=note,
    )
