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

![Agent workflow](diagrams/agent-workflow.png)

## Why five agents

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
5. **Business Advisor Agent** owns post-run operational analysis. It is strictly read-only: it
   reads the transaction ledger and quote history to surface demand gaps, supplier lead-time
   patterns, and discount-tier effects on realized margin, and cannot place orders or modify any
   records.

A single agent could technically call all fourteen tools, but that would concentrate authority,
make failures harder to localize, and create a broad prompt-injection blast radius. Five
boundaries make ownership, tool permissions, tests, and audit evidence explicit, and each
responsibility (orchestration, inventory, quoting, fulfillment, operational analysis) has exactly
one owner. The negotiating customer agent described below is a counterparty for evaluation, not a
sixth team member.

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
| Quoting | `compute_quote` | deterministic markup, volume tiers, and comparables check |
| Fulfillment | `commit_sale` | `get_stock_level`, `create_transaction` (sales) |
| Advisor | `read_financial_report` | `generate_financial_report` |
| Advisor | `analyze_stock_gaps` | `generate_financial_report` |
| Advisor | `review_demand_patterns` | `search_quote_history` |

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
  sheet-based product), and a 0.86 similarity threshold. Unknown products fail closed as
  `unsupported`. When a name clears the unsupported check but scores below the 0.86 threshold,
  the resolver returns `ambiguous` and includes the nearest matching catalog item names in the
  reason string (e.g., "nearest items we carry: Envelopes, Kraft paper"), so the customer
  response is immediately actionable. The system never silently substitutes a product;
  disambiguation is always left to the customer.
- **Pricing** is cost times a configured markup with published volume tiers (5% at 500 units,
  10% at 1,000, 15% at 5,000), computed in `Decimal`, never below catalog cost. Each quote also
  records a deterministic comparables check: the total is compared against the range of the
  retrieved historical quotes (half the lowest to double the highest) and anomalies are noted
  in the quote, never silently repriced.
- **Inventory and restocking** compare net stock as of the request date; shortages trigger a
  reorder sized to the larger of the shortfall and a top-up to `min_stock_level` times a
  configured multiplier, gated by supplier delivery timing against the customer deadline and by
  a cash-reserve purchasing limit.
- **Fulfillment** revalidates stock at commit time and records each sale with the starter's
  transaction helper; a line whose stock changed is declined, not silently shorted.

## Customer safety boundary

Customer-facing text is produced only by the `CustomerResponse` projection: an itemized statement
of supplied and declined lines with reasons, the quoted total, pricing rationale, and delivery
commitments. A guard rejects any output containing internal terms (cash balance, profit margin,
error internals); a model-written summary that trips the guard is replaced by a safe default.
The evaluation harness independently re-checks every response for the same terms.

## Negotiation evaluation layer

`munder-difflin negotiate` runs a bounded negotiation between the team and a customer agent
seeded with a sample request's context. The customer agent is a counterparty, not a team
member: it receives only the guarded customer response text, exactly what a real customer
would see, and decides whether to accept, revise, or walk away; its reply is a single email
that both answers the company's questions and restates the order, and that email becomes the
next request verbatim. The deterministic harness owns the round budget and every stop condition
(a committing round, accept, walk-away, or the budget running out), so two language models can
never loop unboundedly. Negotiation rounds are clarify-first: while any line is still
clarifiable, an unspecified pack size or an unmatched product name, the company holds every
stock and ledger action and asks the customer to answer (`needs_clarification`, with resolved
lines noted as held); products the catalog does not carry decline outright because no answer
can fix them. When a line is ambiguous, the clarification names the nearest catalog items
(e.g., "nearest items we carry: Envelopes, Kraft paper") so the customer can reply with a
resolvable product name rather than a specification the catalog cannot use. A request executes once nothing clarifiable remains, and a
round that commits anything ends the negotiation, so a revision can never re-order lines that
were already sold. The Phase 1 state machine in the [roadmap](roadmap.md) grows this hold into
a full quote-accept lifecycle for batch traffic.

## Failure semantics

Declines always carry a named reason code and a customer-readable explanation:

- `unsupported`: the product is not in the catalog and no answer can fix it;
- `ambiguous`: the product name could not be matched to one catalog item; the reason string
  includes the nearest catalog alternatives so the customer can restate with a resolvable name;
- `supplier_after_deadline`: replenishment cannot arrive before the customer deadline;
- `restock_not_authorized`: the reorder would breach the cash-reserve purchasing limit;
- `restock_required` / `not_assessed` / `commit_incomplete`: an agent stopped before completing
  the line, and the system failed safe rather than committing it;
- `awaiting_clarification` (negotiation mode): resolved lines held, with no stock or ledger
  action, while another line in the order can still be clarified;
- `stock_changed_before_commit`: the commit-time revalidation caught a stock change.

## Evaluation boundary

`project_starter.py` runs the full 20-request `quote_requests_sample.csv` in date order
against a freshly seeded database (seed 137), writes one `test_results.csv` row per request, and
records the dataset SHA-256, seed, model, and aggregate metrics in a manifest. Each run writes
to its own timestamped folder under `runs/`; the committed results at the repository root are a
deliberately promoted run, so experiments never overwrite the submission. Each request is
retried once on failure and otherwise recorded as a safe rejection, so one bad model turn cannot
sink the run. `munder-difflin check` asserts the acceptance gates in code: exactly 20 rows, at
least three fully fulfilled, at least three cash-moving, at least one non-fulfilled with reasons,
and zero forbidden-term leaks.

## Testing strategy

The suite runs entirely offline. Deterministic logic (catalog resolution, pricing tiers, the
seven helpers) is tested directly. The agent graph itself is exercised with scripted
`FunctionModel`s that drive the real agents and tools through the full lifecycle: partial
fulfillment, restock-then-sell, rejection with reasons, the fail-safe backstop, the leak guard,
and the negotiation loop's stop conditions. Live model behavior is validated by the evaluation
run, whose results ship with the repository.
