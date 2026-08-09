# Architecture

## System purpose

Munder Difflin turns unstructured customer requests into explainable inventory, quote, and
fulfillment outcomes. It is deliberately more constrained than a general autonomous agent:
language models interpret requests, decide delegation, and word the customer response, while
deterministic software owns catalog validity, pricing, purchasing limits, dates, and every
transaction side effect.

This split is the central design principle: use probabilistic reasoning where ambiguity is real
(free-text interpretation, communication), and use normal software where the business needs an
invariant (money, stock, dates).

![Four-agent workflow](diagrams/agent-workflow.png)

## Why four agents

The boundaries follow business authority, not prompt-writing convenience:

1. **Orchestrator Agent** owns request interpretation, delegation, and the customer response. It
   extracts product lines and the deadline from the raw request, resolves them against the
   catalog through a deterministic tool, and delegates each business decision. It cannot write to
   the ledger.
2. **Inventory Agent** owns stock visibility, supplier feasibility, and cash-guarded restocking.
   It cannot set prices or record sales.
3. **Quoting Agent** owns comparable retrieval and quote construction. Prices come from a
   deterministic pricing tool; the agent contributes the rationale, never the arithmetic.
4. **Fulfillment Agent** owns commit-time revalidation and sale recording. It cannot restock or
   change prices.

A single agent could technically call all eleven tools, but that would concentrate authority,
make failures harder to localize, and create a broad prompt-injection blast radius. Four
boundaries make ownership, tool permissions, tests, and audit evidence explicit, and each
required responsibility (orchestration, inventory, quoting, sales finalization) has exactly
one owner. A fifth business-advisor agent was considered and deferred to the
[roadmap](roadmap.md): no requirement or correctness need justified it.

## Agents, tools, and starter helpers

Each tool is deterministic Python; the table shows which provided starter helper functions each
tool builds on.

| Agent | Tool | Starter helper(s) |
| --- | --- | --- |
| Orchestrator | `resolve_catalog_items` | none (catalog aliases, unit normalization, fuzzy threshold) |
| Orchestrator | `consult_inventory`, `request_quote`, `finalize_order` | delegation to the worker agents; `get_cash_balance` baselines the cash position |
| Orchestrator | `financial_health_report` | `generate_financial_report` |
| Inventory | `inventory_snapshot` | `get_all_inventory` |
| Inventory | `assess_availability` | `get_stock_level`, `get_supplier_delivery_date` |
| Inventory | `place_restock_order` | `get_cash_balance`, `create_transaction` (stock_orders) |
| Quoting | `retrieve_comparable_quotes` | `search_quote_history` |
| Quoting | `compute_quote` | deterministic markup and volume tiers |
| Fulfillment | `commit_sale` | `get_stock_level`, `create_transaction` (sales) |

All seven provided helpers (`create_transaction`, `get_all_inventory`, `get_stock_level`,
`get_supplier_delivery_date`, `get_cash_balance`, `generate_financial_report`,
`search_quote_history`) are therefore used inside tool definitions, and a structural test
(`tests/unit/test_agent_tools.py`) fails if that mapping drifts.

## Shared state is the source of truth

Every tool writes its deterministically computed result to a shared, typed `AgentDependencies`
object (resolved line items, inventory decisions, the quote, committed sales, trace events). The
final customer response is assembled from that recorded state, not from model text; the model's
final output contributes only the summary sentence, and only after it passes the safety guard.
A model that misquotes a number in its summary cannot change what was committed, priced, or
charged.

If a run ends without the orchestrator calling `finalize_order`, the harness fails safe: the
request becomes an honest rejection with no charges. The harness never performs business actions
behind the agents' backs.

## Deterministic policies

- **Catalog resolution** uses an alias table, unit normalization (a ream is 500 sheets of a
  sheet-based product), and a 0.86 similarity threshold. Unknown or ambiguous products fail
  closed with a customer-readable reason; the system never guesses what to sell.
- **Pricing** is cost times a configured markup with published volume tiers (5% at 500 units,
  10% at 1,000, 15% at 5,000), computed in `Decimal`, never below catalog cost.
- **Inventory and restocking** compare net stock as of the request date; shortages trigger a
  reorder sized to top the item up to `min_stock_level` times a configured multiplier, gated by
  supplier delivery timing against the customer deadline and by a cash-reserve purchasing limit.
- **Fulfillment** revalidates stock at commit time and records each sale with the starter's
  transaction helper; a line whose stock changed is declined, not silently shorted.

## Customer safety boundary

Customer-facing text is produced only by the `CustomerResponse` projection: an itemized statement
of supplied and declined lines with reasons, the quoted total, pricing rationale, and delivery
commitments. A guard rejects any output containing internal terms (cash balance, profit margin,
error internals); a model-written summary that trips the guard is replaced by a safe default.
The evaluation harness independently re-checks every response for the same terms.

## Failure semantics

Declines always carry a named reason code and a customer-readable explanation:

- `unsupported` / `ambiguous`: the product could not be resolved against the catalog;
- `supplier_after_deadline`: replenishment cannot arrive before the customer deadline;
- `restock_not_authorized`: the reorder would breach the cash-reserve purchasing limit;
- `restock_required` / `not_assessed` / `commit_incomplete`: an agent stopped before completing
  the line, and the system failed safe rather than committing it;
- `stock_changed_before_commit`: the commit-time revalidation caught a stock change.

## Evaluation boundary

`python project_starter.py` runs the full 20-request `quote_requests_sample.csv` in date order
against a freshly seeded database (seed 137), writes one `test_results.csv` row per request, and
records the dataset SHA-256, seed, model, and aggregate metrics in a manifest. Each request is
retried once on failure and otherwise recorded as a safe rejection, so one bad model turn cannot
sink the run. `munder-difflin check` asserts the acceptance gates in code: exactly 20 rows, at
least three fully fulfilled, at least three cash-moving, at least one non-fulfilled with reasons,
and zero forbidden-term leaks.

## Testing strategy

The suite runs entirely offline. Deterministic logic (catalog resolution, pricing tiers, the
seven helpers) is tested directly. The agent graph itself is exercised with scripted
`FunctionModel`s that drive the real agents and tools through the full lifecycle: partial
fulfillment, restock-then-sell, rejection with reasons, the fail-safe backstop, and the
leak guard. Live model behavior is validated by the evaluation run, whose results ship with the
repository.
