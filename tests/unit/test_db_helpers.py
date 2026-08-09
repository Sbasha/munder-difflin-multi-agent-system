"""Round-trip tests for the seven starter-faithful database helpers."""

import pytest

from munder_difflin.config import Settings
from munder_difflin.db.helpers import (
    configure_engine,
    create_transaction,
    generate_financial_report,
    get_all_inventory,
    get_cash_balance,
    get_stock_level,
    get_supplier_delivery_date,
    init_database,
    search_quote_history,
)


def test_all_seven_starter_helpers_round_trip(settings: Settings) -> None:
    configure_engine(settings.database_url)
    init_database(seed=137, data_dir=settings.data_dir)

    cash_before = get_cash_balance("2025-04-01")
    stock_before = int(get_stock_level("A4 paper", "2025-04-01")["current_stock"].iloc[0])

    transaction_id = create_transaction("A4 paper", "stock_orders", 25, 1.25, "2025-04-01")

    assert transaction_id > 0
    assert get_cash_balance("2025-04-01") == pytest.approx(cash_before - 1.25)
    stock_after = int(get_stock_level("A4 paper", "2025-04-01")["current_stock"].iloc[0])
    assert stock_after == stock_before + 25
    assert get_all_inventory("2025-04-01")["A4 paper"] == stock_after
    assert get_supplier_delivery_date("2025-04-01", 101) == "2025-04-05"
    report = generate_financial_report("2025-04-01")
    assert report["total_assets"] > 0
    assert search_quote_history(["paper"], limit=3)


def test_create_transaction_rejects_invalid_input(settings: Settings) -> None:
    configure_engine(settings.database_url)
    init_database(seed=137, data_dir=settings.data_dir)

    with pytest.raises(ValueError):
        create_transaction("A4 paper", "refund", 5, 1.0, "2025-04-01")
    with pytest.raises(ValueError):
        create_transaction("A4 paper", "sales", 0, 1.0, "2025-04-01")
