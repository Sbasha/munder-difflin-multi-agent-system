"""Agent roster, least-privilege tool registration, and helper traceability."""

import inspect
import re

from pydantic_ai import Agent

from munder_difflin.agents import team


def test_exactly_five_framework_agents_are_defined() -> None:
    agents = [
        team.orchestrator_agent,
        team.inventory_agent,
        team.quoting_agent,
        team.fulfillment_agent,
        team.advisor_agent,
    ]

    assert len(agents) == 5
    assert all(isinstance(agent, Agent) for agent in agents)


def test_tools_are_registered_to_least_privilege_agents() -> None:
    assert set(team.orchestrator_agent._function_toolset.tools) == {
        "resolve_catalog_items",
        "consult_inventory",
        "request_quote",
        "finalize_order",
        "financial_health_report",
    }
    assert set(team.inventory_agent._function_toolset.tools) == {
        "inventory_snapshot",
        "assess_availability",
        "place_restock_order",
    }
    assert set(team.quoting_agent._function_toolset.tools) == {
        "retrieve_comparable_quotes",
        "compute_quote",
    }
    assert set(team.fulfillment_agent._function_toolset.tools) == {"commit_sale"}
    assert set(team.advisor_agent._function_toolset.tools) == {
        "read_financial_report",
        "analyze_stock_gaps",
        "review_demand_patterns",
    }


REQUIRED_HELPERS = frozenset(
    {
        "create_transaction",
        "get_all_inventory",
        "get_stock_level",
        "get_supplier_delivery_date",
        "get_cash_balance",
        "generate_financial_report",
        "search_quote_history",
    }
)

HELPERS_BY_TOOL = {
    team.resolve_catalog_items: frozenset(),
    team.consult_inventory: frozenset({"get_cash_balance"}),
    team.request_quote: frozenset(),
    team.finalize_order: frozenset(),
    team.financial_health_report: frozenset({"generate_financial_report"}),
    team.inventory_snapshot: frozenset({"get_all_inventory"}),
    team.assess_availability: frozenset({"get_stock_level", "get_supplier_delivery_date"}),
    team.place_restock_order: frozenset({"get_cash_balance", "create_transaction"}),
    team.retrieve_comparable_quotes: frozenset({"search_quote_history"}),
    team.compute_quote: frozenset(),
    team.commit_sale: frozenset({"get_stock_level", "create_transaction"}),
    team.read_financial_report: frozenset({"generate_financial_report"}),
    team.analyze_stock_gaps: frozenset({"generate_financial_report"}),
    team.review_demand_patterns: frozenset({"search_quote_history"}),
}


def test_every_tool_uses_exactly_its_documented_helpers() -> None:
    for tool, expected in HELPERS_BY_TOOL.items():
        source = inspect.getsource(tool)
        used = frozenset(name for name in REQUIRED_HELPERS if re.search(rf"\b{name}\b", source))
        assert used == expected, (
            f"{tool.__name__} helper usage drifted from the documented mapping: "
            f"expected {sorted(expected)}, found {sorted(used)}"
        )


def test_every_required_helper_is_called_inside_a_framework_tool() -> None:
    covered = frozenset().union(*HELPERS_BY_TOOL.values())

    assert covered == REQUIRED_HELPERS
