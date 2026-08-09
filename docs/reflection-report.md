# Reflection Report

This report explains the multi-agent system I built for the Munder Difflin (Beaver's Choice)
paper company project, discusses the evaluation results in `test_results.csv`, and proposes
improvements. The workflow diagram referenced throughout is
`docs/diagrams/agent-workflow.png` (source: `docs/diagrams/agent-workflow.mmd`).

## The agent workflow and how I arrived at it

The system is four Pydantic AI agents, within the project's five-agent cap:

1. **Orchestrator Agent** (diagram step 1) owns the request lifecycle. It reads the free-text
   request, extracts product lines and the required delivery date, and calls its
   `resolve_catalog_items` tool, which matches lines against the catalog deterministically
   (alias table, unit normalization such as one ream equals 500 sheets, a 0.86 fuzzy-match
   threshold, and fail-closed handling of unknown or ambiguous products). It then delegates,
   in order, to the three workers through tools (`consult_inventory`, `request_quote`,
   `finalize_order`), can run an internal `financial_health_report` check built on
   `generate_financial_report` before committing large orders, and finally words the customer
   response.
2. **Inventory Agent** (step 2) owns stock and supply. `inventory_snapshot` reports dated stock
   with `get_all_inventory`; `assess_availability` combines `get_stock_level` with
   `get_supplier_delivery_date` and the reorder policy (top up to the item's minimum stock level
   times a configured multiplier); `place_restock_order` places cash-guarded replenishment with
   `get_cash_balance` and `create_transaction`, refusing any order that would breach the cash
   reserve.
3. **Quoting Agent** (step 3) owns pricing. `retrieve_comparable_quotes` consults history with
   `search_quote_history`; `compute_quote` prices deterministically with a cost markup and
   published volume tiers (5% at 500 units, 10% at 1,000, 15% at 5,000), never below catalog
   cost.
4. **Fulfillment Agent** (step 4) owns commitment. `commit_sale` revalidates stock with
   `get_stock_level` at the delivery date and records the sale with `create_transaction`.

All seven provided helper functions are used inside these tool definitions, and a structural
test fails the build if that mapping ever drifts.

Two decisions shaped this architecture. First, the split between probabilistic and deterministic
work: language models are good at reading messy customer prose and writing warm, clear
responses, and bad at being trusted with arithmetic and side effects. So every tool computes its
result in plain Python, records it on shared typed state, and the final customer response is
assembled from that recorded state; the model's text contributes only the summary sentence,
behind a guard that blocks internal terms (cash balances, margins, error internals). A model
that misquotes a number cannot change what was committed or charged. Second, the agent count: I
initially considered a fifth business-advisor agent, and cut it. Each of the four remaining
agents owns exactly one business authority (orchestration, inventory, quoting, fulfillment)
with least-privilege tools, which keeps responsibilities non-overlapping and failures easy to
localize; the advisor added an agent without adding a requirement it alone could satisfy, so it
moved to the roadmap.

The diagram's outcome fan (FULL, PARTIAL, REJECTED) is a real code path, not decoration: partial
fulfillment commits the deliverable lines and explains every declined line with a named reason
code, and a run that stops early fails safe into an honest rejection with no charges.

## Evaluation results and strengths

The evaluation harness runs the complete 20-request `quote_requests_sample.csv` in date order
against a freshly seeded database (seed 137) and writes one row per request to
`test_results.csv`. The committed run used gpt-5-mini through the Vocareum proxy:

| Metric | Value |
| --- | --- |
| Requests processed | 20 of 20 |
| Fully fulfilled | 3 |
| Partially fulfilled | 14 |
| Rejected with reasons | 3 |
| Requests that changed the cash balance | 16 |
| Customer information leaks | 0 |
| Negative-inventory occurrences | 0 |

This satisfies every acceptance gate, and the gates themselves are executable
(`munder-difflin check test_results.csv`), so the claims above are reproducible rather than
transcribed.

Specific strengths visible in the results:

- **Impossible constraints are recognized, with reasons.** The three rejections and every
  declined line carry a named reason code and a customer-readable explanation: nine lines
  declined because supplier replenishment would land after the customer's deadline
  (`supplier_after_deadline`), eleven because the product is not in the catalog
  (`unsupported`, for example balloons and A3 paper), and four because the request was too
  ambiguous to price honestly (`ambiguous`, for example packs of unspecified size).
- **Commit-time revalidation earned its place.** Four lines were approved on assessment but
  declined at commitment (`stock_changed_before_commit`) because earlier same-day orders
  consumed the stock. Without the revalidation these would have been oversold into negative
  inventory; the run ended with zero negative-inventory items.
- **The restocking policy works end to end.** Sixteen requests moved cash, a mix of sales
  revenue and cash-guarded stock orders placed by the inventory agent, and the company ended
  the run with more cash ($49,075.43) than it started with ($48,132.64) while never breaching
  the purchasing reserve.
- **The customer boundary held under a live model.** Across 20 model-worded responses, the
  independent leak check found zero occurrences of internal terms, and no request needed the
  fail-safe backstop or the per-request retry.

The clearest area for improvement in the numbers is the partial-fulfillment rate: fourteen of
twenty requests were only partly supplied. Most of that is the dataset asking for products the
company genuinely does not stock, which the system correctly refuses to guess about, but the
four `ambiguous` declines represent real demand lost to interpretation strictness rather than to
inventory.

## Suggestions for improvement

1. **A clarification loop for ambiguous lines.** Today an ambiguous line (an unmatchable product
   name, or "5 packs" with no pack size) fails closed and is declined. A better system would let
   the orchestrator ask the customer one targeted follow-up question and re-resolve the line,
   converting some of the `ambiguous` declines into sales. The same mechanism generalizes into a
   customer negotiation agent that plays the counterparty for evaluation.
2. **A business advisor agent over the transaction ledger.** The deferred fifth agent would
   read `generate_financial_report` output across runs and recommend operational changes:
   which zero-stock items actually receive demand and deserve pre-stocking, whether the
   supplier-lead-time rejections cluster on specific products, and how discount tiers affect
   realized margin. Its recommendations should remain read-only until evaluation and governance
   controls exist.
3. **Demand-aware restocking.** The reorder policy is a static top-up multiplier. Replacing it
   with a forecast built from the quote-history table (seasonality by event type is visible in
   the data) would cut both the `supplier_after_deadline` rejections, by pre-stocking ahead of
   demand, and the `stock_changed_before_commit` contention between same-day orders.
