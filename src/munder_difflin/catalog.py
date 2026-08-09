"""Deterministic catalog resolution for lines extracted by the orchestrator agent.

The language model extracts what the customer asked for; this module decides,
without guessing, whether each extracted line maps to exactly one catalog item
and how many sellable units it represents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher

from munder_difflin.models import ParsedLineItem, ResolutionStatus


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """Canonical product sold by the company."""

    name: str
    category: str
    unit_price: Decimal


_SUPPLY_ROWS = [
    ("A4 paper", "paper", "0.05"),
    ("Letter-sized paper", "paper", "0.06"),
    ("Cardstock", "paper", "0.15"),
    ("Colored paper", "paper", "0.10"),
    ("Glossy paper", "paper", "0.20"),
    ("Matte paper", "paper", "0.18"),
    ("Recycled paper", "paper", "0.08"),
    ("Eco-friendly paper", "paper", "0.12"),
    ("Poster paper", "paper", "0.25"),
    ("Banner paper", "paper", "0.30"),
    ("Kraft paper", "paper", "0.10"),
    ("Construction paper", "paper", "0.07"),
    ("Wrapping paper", "paper", "0.15"),
    ("Glitter paper", "paper", "0.22"),
    ("Decorative paper", "paper", "0.18"),
    ("Letterhead paper", "paper", "0.12"),
    ("Legal-size paper", "paper", "0.08"),
    ("Crepe paper", "paper", "0.05"),
    ("Photo paper", "paper", "0.25"),
    ("Uncoated paper", "paper", "0.06"),
    ("Butcher paper", "paper", "0.10"),
    ("Heavyweight paper", "paper", "0.20"),
    ("Standard copy paper", "paper", "0.04"),
    ("Bright-colored paper", "paper", "0.12"),
    ("Patterned paper", "paper", "0.15"),
    ("Paper plates", "product", "0.10"),
    ("Paper cups", "product", "0.08"),
    ("Paper napkins", "product", "0.02"),
    ("Disposable cups", "product", "0.10"),
    ("Table covers", "product", "1.50"),
    ("Envelopes", "product", "0.05"),
    ("Sticky notes", "product", "0.03"),
    ("Notepads", "product", "2.00"),
    ("Invitation cards", "product", "0.50"),
    ("Flyers", "product", "0.15"),
    ("Party streamers", "product", "0.05"),
    ("Decorative adhesive tape (washi tape)", "product", "0.20"),
    ("Paper party bags", "product", "0.25"),
    ("Name tags with lanyards", "product", "0.75"),
    ("Presentation folders", "product", "0.50"),
    ("Large poster paper (24x36 inches)", "large_format", "1.00"),
    ("Rolls of banner paper (36-inch width)", "large_format", "2.50"),
    ("100 lb cover stock", "specialty", "0.50"),
    ("80 lb text paper", "specialty", "0.40"),
    ("250 gsm cardstock", "specialty", "0.30"),
    ("220 gsm poster paper", "specialty", "0.35"),
]

CATALOG = {
    name: CatalogItem(name=name, category=category, unit_price=Decimal(price))
    for name, category, price in _SUPPLY_ROWS
}

_ALIASES = {
    "a4 paper": "A4 paper",
    "a4 white paper": "A4 paper",
    "a4 printer paper": "A4 paper",
    "a4 printing paper": "A4 paper",
    "a4 size printer paper": "A4 paper",
    "a4 white printer paper": "A4 paper",
    "standard printer paper": "Standard copy paper",
    "standard printing paper": "Standard copy paper",
    "standard copy paper": "Standard copy paper",
    "white printer paper": "Standard copy paper",
    "printing paper": "Standard copy paper",
    "printer paper": "Standard copy paper",
    "glossy paper": "Glossy paper",
    "high quality glossy paper": "Glossy paper",
    "a4 glossy paper": "Glossy paper",
    "glossy a4 paper": "Glossy paper",
    "matte paper": "Matte paper",
    "a4 matte paper": "Matte paper",
    "cardstock": "Cardstock",
    "sturdy cardstock": "Cardstock",
    "white cardstock": "Cardstock",
    "high quality white cardstock": "Cardstock",
    "high quality cardstock": "Cardstock",
    "colorful cardstock": "Cardstock",
    "cardstock in various colors": "Cardstock",
    "cardstock in assorted colors": "Cardstock",
    "heavy cardstock": "250 gsm cardstock",
    "heavyweight cardstock": "250 gsm cardstock",
    "colored paper": "Colored paper",
    "colorful paper": "Colored paper",
    "colored paper assorted colors": "Colored paper",
    "colorful construction paper": "Construction paper",
    "construction paper": "Construction paper",
    "poster paper": "Poster paper",
    "colorful poster paper": "Poster paper",
    "poster paper in various colors": "Poster paper",
    "poster board": "Poster paper",
    "poster boards": "Poster paper",
    "large poster paper": "Large poster paper (24x36 inches)",
    "recycled paper": "Recycled paper",
    "a4 recycled paper": "Recycled paper",
    "paper napkins": "Paper napkins",
    "table napkins": "Paper napkins",
    "napkins": "Paper napkins",
    "paper cups": "Paper cups",
    "cups": "Paper cups",
    "paper plates": "Paper plates",
    "plates": "Paper plates",
    "streamers": "Party streamers",
    "party streamers": "Party streamers",
    "decorative washi tape": "Decorative adhesive tape (washi tape)",
    "decorative adhesive tape": "Decorative adhesive tape (washi tape)",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "flyers": "Flyers",
    "posters": "Poster paper",
    "envelopes": "Envelopes",
    "sticky notes": "Sticky notes",
    "notepads": "Notepads",
    "invitation cards": "Invitation cards",
    "invitations": "Invitation cards",
    "banner paper": "Banner paper",
    "kraft paper": "Kraft paper",
    "wrapping paper": "Wrapping paper",
    "glitter paper": "Glitter paper",
    "decorative paper": "Decorative paper",
    "letterhead paper": "Letterhead paper",
    "legal size paper": "Legal-size paper",
    "crepe paper": "Crepe paper",
    "photo paper": "Photo paper",
    "uncoated paper": "Uncoated paper",
    "butcher paper": "Butcher paper",
    "heavyweight paper": "Heavyweight paper",
    "letter sized paper": "Letter-sized paper",
    "letter size paper": "Letter-sized paper",
    "eco friendly paper": "Eco-friendly paper",
    "table covers": "Table covers",
    "disposable cups": "Disposable cups",
    "paper party bags": "Paper party bags",
    "party bags": "Paper party bags",
    "name tags with lanyards": "Name tags with lanyards",
    "name tags": "Name tags with lanyards",
    "presentation folders": "Presentation folders",
    "100 lb cover stock": "100 lb cover stock",
    "80 lb text paper": "80 lb text paper",
    "250 gsm cardstock": "250 gsm cardstock",
    "220 gsm poster paper": "220 gsm poster paper",
}

_UNSUPPORTED_MARKERS = {
    "a3",
    "a5",
    "balloon",
    "ticket",
    "cardboard",
    "recycled cardstock",
}


def normalize_text(value: str) -> str:
    """Normalize customer product text for deterministic matching."""

    value = value.lower().replace("-", " ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(
        r"\b(?:white|assorted colors?|various colors?|high quality|sturdy)\b", " ", value
    )
    value = re.sub(r"\b(?:for|in time for)\s+(?:our|the|an|upcoming).*", "", value)
    value = re.sub(r"\b(?:delivered|delivery|needed|please)\b.*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,:-")


def _resolve_item(item_text: str, unit: str) -> tuple[ResolutionStatus, str | None, str | None]:
    self_describing_units = {
        "flyer": "flyers",
        "flyers": "flyers",
        "poster": "posters",
        "posters": "posters",
        "ticket": "tickets",
        "tickets": "tickets",
        "balloon": "balloons",
        "balloons": "balloons",
        "napkin": "napkins",
        "napkins": "napkins",
        "table napkin": "table napkins",
        "table napkins": "table napkins",
        "paper cup": "paper cups",
        "paper cups": "paper cups",
        "cup": "cups",
        "cups": "cups",
        "paper plate": "paper plates",
        "paper plates": "paper plates",
        "plate": "plates",
        "plates": "plates",
        "poster board": "poster board",
        "poster boards": "poster boards",
    }
    normalized_unit = normalize_text(unit)
    normalized_item = normalize_text(item_text)
    candidate = normalized_item or self_describing_units.get(normalized_unit, normalized_unit)

    if any(marker in candidate for marker in _UNSUPPORTED_MARKERS):
        return (
            ResolutionStatus.UNSUPPORTED,
            None,
            f"'{candidate}' is not available in the current catalog",
        )

    if "24x36" in item_text.replace(" ", "").lower() or "24 x 36" in item_text.lower():
        return ResolutionStatus.RESOLVED, "Large poster paper (24x36 inches)", None

    if candidate in _ALIASES:
        return ResolutionStatus.RESOLVED, _ALIASES[candidate], None

    scored = sorted(
        (
            (SequenceMatcher(None, candidate, normalize_text(alias)).ratio(), canonical)
            for alias, canonical in _ALIASES.items()
        ),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.86:
        return ResolutionStatus.RESOLVED, scored[0][1], None
    return (
        ResolutionStatus.AMBIGUOUS,
        None,
        f"'{candidate}' could not be matched to one catalog item without guessing",
    )


def _normalize_quantity(
    quantity: int,
    unit: str,
    catalog_item: str | None,
) -> tuple[int | None, str | None]:
    normalized_unit = normalize_text(unit)
    if normalized_unit in {"ream", "reams"}:
        if catalog_item and CATALOG[catalog_item].category in {"paper", "specialty"}:
            return quantity * 500, None
        return None, "Reams are only supported for sheet-based paper products"
    if normalized_unit in {"pack", "packs", "packet", "packets", "box", "boxes"}:
        return None, f"The quantity contained in each {normalized_unit} was not specified"
    return quantity, None


def resolve_requested_line(
    item_text: str,
    unit: str,
    quantity: int,
    deadline: date,
) -> ParsedLineItem:
    """Resolve one extracted request line against the catalog without guessing.

    Unknown or ambiguous products fail closed so no downstream agent can sell
    something the company does not stock.
    """

    status, catalog_item, reason = _resolve_item(item_text, unit)
    normalized_quantity, unit_reason = _normalize_quantity(quantity, unit, catalog_item)
    if unit_reason:
        status = ResolutionStatus.AMBIGUOUS
        catalog_item = None
        reason = unit_reason
    return ParsedLineItem(
        raw_text=f"{quantity} {unit} {item_text}".strip(),
        requested_quantity=quantity,
        requested_unit=unit,
        requested_item=item_text or unit,
        deadline=deadline,
        resolution_status=status,
        catalog_item=catalog_item,
        normalized_quantity=normalized_quantity,
        resolution_reason=reason,
    )
