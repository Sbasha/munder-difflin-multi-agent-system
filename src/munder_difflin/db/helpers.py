"""SQLite helpers matching the provided starter functions, plus engine management.

The seven business helpers keep the starter code's signatures and semantics
(`create_transaction`, `get_all_inventory`, `get_stock_level`,
`get_supplier_delivery_date`, `get_cash_balance`, `generate_financial_report`,
`search_quote_history`) so agent tools demonstrably build on the provided
functions. Engine configuration is injectable so tests run on isolated
temporary databases.
"""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine, text

from munder_difflin.catalog import CATALOG

_db_engine: Engine = create_engine("sqlite:///munder_difflin.db")


def configure_engine(database_url: str) -> Engine:
    """Set the process engine, allowing tests to inject an isolated database."""

    global _db_engine
    _db_engine = create_engine(database_url)
    return _db_engine


def get_engine() -> Engine:
    """Return the configured SQLAlchemy engine."""

    return _db_engine


def _iso_day(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return datetime.fromisoformat(value.split("T")[0]).date().isoformat()


def init_database(
    db_engine: Engine | None = None,
    seed: int = 137,
    data_dir: Path | str | None = None,
) -> Engine:
    """Create a reproducible ledger, catalog, and quote history.

    Mirrors the starter's initialization: quote history tables from the two
    CSV files, a seeded random 40% of the catalog stocked, and a $50,000
    opening cash balance recorded as the first sales transaction.
    """

    global _db_engine
    if db_engine is not None:
        _db_engine = db_engine
    engine = _db_engine
    source_dir = Path(data_dir) if data_dir is not None else Path.cwd()
    quote_requests_path = source_dir / "quote_requests.csv"
    quotes_path = source_dir / "quotes.csv"
    if not quote_requests_path.exists() or not quotes_path.exists():
        raise FileNotFoundError(
            "quote_requests.csv and quotes.csv must exist in the configured data directory"
        )

    quote_requests = pd.read_csv(quote_requests_path)
    quote_requests["id"] = range(1, len(quote_requests) + 1)

    quotes = pd.read_csv(quotes_path)
    quotes["request_id"] = range(1, len(quotes) + 1)
    quotes["order_date"] = "2025-01-01"
    metadata = quotes.get("request_metadata", pd.Series([{}] * len(quotes))).apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value
    )
    for key in ("job_type", "order_size", "event_type"):
        quotes[key] = metadata.apply(
            lambda value, field=key: value.get(field, "") if isinstance(value, dict) else ""
        )
    quotes = quotes[
        [
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type",
        ]
    ]

    rng = np.random.default_rng(seed)
    catalog_rows = [
        {
            "item_name": item.name,
            "category": item.category,
            "unit_price": float(item.unit_price),
            "min_stock_level": int(rng.integers(50, 151)),
        }
        for item in CATALOG.values()
    ]
    stocked_names = set(
        rng.choice(
            list(CATALOG),
            size=int(len(CATALOG) * 0.4),
            replace=False,
        ).tolist()
    )

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS transactions"))
        connection.execute(
            text(
                """
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT,
                    transaction_type TEXT NOT NULL
                        CHECK (transaction_type IN ('stock_orders', 'sales')),
                    units INTEGER,
                    price REAL NOT NULL,
                    transaction_date TEXT NOT NULL
                )
                """
            )
        )

    quote_requests.to_sql("quote_requests", engine, if_exists="replace", index=False)
    quotes.to_sql("quotes", engine, if_exists="replace", index=False)
    pd.DataFrame(catalog_rows).to_sql("inventory", engine, if_exists="replace", index=False)

    initial_transactions: list[dict[str, Any]] = [
        {
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50_000.0,
            "transaction_date": "2025-01-01",
        }
    ]
    for item in catalog_rows:
        if item["item_name"] not in stocked_names:
            continue
        stock = int(rng.integers(200, 801))
        initial_transactions.append(
            {
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": stock,
                "price": stock * item["unit_price"],
                "transaction_date": "2025-01-01",
            }
        )
    pd.DataFrame(initial_transactions).to_sql(
        "transactions",
        engine,
        if_exists="append",
        index=False,
    )
    return engine


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: str | datetime,
) -> int:
    """Record one stock order or sale and return its id (starter signature)."""

    if transaction_type not in {"stock_orders", "sales"}:
        raise ValueError("transaction_type must be 'stock_orders' or 'sales'")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    with _db_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO transactions
                    (item_name, transaction_type, units, price, transaction_date)
                VALUES
                    (:item_name, :transaction_type, :units, :price, :transaction_date)
                """
            ),
            {
                "item_name": item_name,
                "transaction_type": transaction_type,
                "units": quantity,
                "price": float(price),
                "transaction_date": _iso_day(date),
            },
        )
        return int(connection.execute(text("SELECT last_insert_rowid()")).scalar_one())


def get_all_inventory(as_of_date: str) -> dict[str, int]:
    """Return positive inventory quantities as of an ISO date."""

    query = text(
        """
        SELECT item_name,
               SUM(CASE
                   WHEN transaction_type = 'stock_orders' THEN units
                   WHEN transaction_type = 'sales' THEN -units
                   ELSE 0
               END) AS stock
        FROM transactions
        WHERE item_name IS NOT NULL
          AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
        """
    )
    result = pd.read_sql(query, _db_engine, params={"as_of_date": _iso_day(as_of_date)})
    return {
        str(name): int(stock)
        for name, stock in zip(result["item_name"], result["stock"], strict=True)
    }


def get_stock_level(item_name: str, as_of_date: str | datetime) -> pd.DataFrame:
    """Return the net stock level for one exact catalog item."""

    query = text(
        """
        SELECT :item_name AS item_name,
               COALESCE(SUM(CASE
                   WHEN transaction_type = 'stock_orders' THEN units
                   WHEN transaction_type = 'sales' THEN -units
                   ELSE 0
               END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
          AND transaction_date <= :as_of_date
        """
    )
    return pd.read_sql(
        query,
        _db_engine,
        params={"item_name": item_name, "as_of_date": _iso_day(as_of_date)},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """Estimate supplier delivery using the starter quantity bands (0/1/4/7 days)."""

    if quantity <= 0:
        raise ValueError("quantity must be positive")
    input_date = datetime.fromisoformat(input_date_str.split("T")[0]).date()
    days = 0 if quantity <= 10 else 1 if quantity <= 100 else 4 if quantity <= 1_000 else 7
    return (input_date + timedelta(days=days)).isoformat()


def get_cash_balance(as_of_date: str | datetime) -> float:
    """Return sales less stock-order spend through the supplied date."""

    query = text(
        """
        SELECT COALESCE(SUM(CASE
            WHEN transaction_type = 'sales' THEN price
            WHEN transaction_type = 'stock_orders' THEN -price
            ELSE 0
        END), 0)
        FROM transactions
        WHERE transaction_date <= :as_of_date
        """
    )
    with _db_engine.connect() as connection:
        return float(connection.execute(query, {"as_of_date": _iso_day(as_of_date)}).scalar_one())


def generate_financial_report(as_of_date: str | datetime) -> dict[str, Any]:
    """Generate cash, inventory valuation, assets, and top-selling products."""

    as_of = _iso_day(as_of_date)
    cash = get_cash_balance(as_of)
    inventory = pd.read_sql("SELECT * FROM inventory", _db_engine)
    inventory_summary: list[dict[str, Any]] = []
    inventory_value = 0.0
    for row in inventory.to_dict(orient="records"):
        item_name = str(row["item_name"])
        unit_price = float(row["unit_price"])
        stock = int(get_stock_level(item_name, as_of)["current_stock"].iloc[0])
        value = stock * unit_price
        inventory_value += value
        inventory_summary.append(
            {
                "item_name": item_name,
                "stock": stock,
                "unit_price": unit_price,
                "value": value,
            }
        )
    top_sales = pd.read_sql(
        text(
            """
            SELECT item_name, SUM(units) AS total_units, SUM(price) AS total_revenue
            FROM transactions
            WHERE transaction_type = 'sales'
              AND transaction_date <= :as_of_date
              AND item_name IS NOT NULL
            GROUP BY item_name
            ORDER BY total_revenue DESC
            LIMIT 5
            """
        ),
        _db_engine,
        params={"as_of_date": as_of},
    )
    return {
        "as_of_date": as_of,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_sales.to_dict(orient="records"),
    }


def search_quote_history(search_terms: list[str], limit: int = 5) -> list[dict[str, Any]]:
    """Find comparable historical requests and quote explanations.

    Terms are combined with OR (any-term match) rather than the starter's AND
    so multi-item searches still surface useful comparables.
    """

    safe_terms = [term.strip().lower() for term in search_terms if term.strip()][:5]
    safe_limit = max(1, min(int(limit), 10))
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": safe_limit}
    for index, term in enumerate(safe_terms):
        key = f"term_{index}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{key} OR LOWER(q.quote_explanation) LIKE :{key})"
        )
        params[key] = f"%{term}%"
    where_clause = " OR ".join(conditions) if conditions else "1=1"
    query = text(
        f"""
        SELECT qr.response AS original_request,
               q.total_amount,
               q.quote_explanation,
               q.job_type,
               q.order_size,
               q.event_type,
               q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT :limit
        """
    )
    with _db_engine.connect() as connection:
        return [dict(row._mapping) for row in connection.execute(query, params)]


def get_min_stock_level(item_name: str) -> int:
    """Return the configured reorder threshold for one SKU."""

    with _db_engine.connect() as connection:
        result = connection.execute(
            text("SELECT min_stock_level FROM inventory WHERE item_name = :item_name"),
            {"item_name": item_name},
        ).scalar_one()
    return int(result)
