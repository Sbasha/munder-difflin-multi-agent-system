"""Agent roster, least-privilege tool registration, and helper traceability."""

import inspect

from pydantic_ai import Agent

from munder_difflin.agents import team


def test_exactly_four_framework_agents_are_defined() -> None:
    agents = [
        team.orchestrator_agent,
        team.inventory_agent,
        team.quoting_agent,
        team.fulfillment_agent,
    ]

    assert len(agents) == 4
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


def test_every_required_helper_is_called_inside_a_framework_tool() -> None:
    tool_source = "\n".join(
        inspect.getsource(function)
        for function in (
            team.resolve_catalog_items,
            team.consult_inventory,
            team.request_quote,
            team.finalize_order,
            team.financial_health_report,
            team.inventory_snapshot,
            team.assess_availability,
            team.place_restock_order,
            team.retrieve_comparable_quotes,
            team.compute_quote,
            team.commit_sale,
        )
    )

    for helper_name in (
        "create_transaction",
        "get_all_inventory",
        "get_stock_level",
        "get_supplier_delivery_date",
        "get_cash_balance",
        "generate_financial_report",
        "search_quote_history",
    ):
        assert helper_name in tool_source
