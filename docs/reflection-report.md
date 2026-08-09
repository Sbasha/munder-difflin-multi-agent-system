# Reflection Report

This report explains the multi-agent system I built for the Munder Difflin (Beaver's Choice)
paper company project, discusses the evaluation results in `test_results.csv`, and proposes
improvements. The workflow diagram referenced throughout is
`docs/diagrams/agent-workflow.png` (source: `docs/diagrams/agent-workflow.mmd`).

## The agent workflow and how I arrived at it

The system is five Pydantic AI agents, at the project's five-agent cap:

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
   `get_supplier_delivery_date` and the reorder policy (order the shortfall or top up to the
   item's minimum stock level times a configured multiplier, whichever is larger);
   `place_restock_order` places cash-guarded replenishment with
   `get_cash_balance` and `create_transaction`, refusing any order that would breach the cash
   reserve.
3. **Quoting Agent** (step 3) owns pricing. `retrieve_comparable_quotes` consults history with
   `search_quote_history`; `compute_quote` prices deterministically with a cost markup and
   published volume tiers (5% at 500 units, 10% at 1,000, 15% at 5,000), never below catalog
   cost, and records a deterministic check of the total against the retrieved comparables.
4. **Fulfillment Agent** (step 4) owns commitment. `commit_sale` revalidates stock with
   `get_stock_level` at the delivery date and records the sale with `create_transaction`.
5. **Business Advisor Agent** (post-run, read-only) owns operational analysis. It runs after
   the evaluation via `munder-difflin advise` and never takes any transactional action.
   `read_financial_report` retrieves cash, inventory valuation, and top-selling products via
   `generate_financial_report`; `analyze_stock_gaps` cross-references demand with current stock
   to identify pre-stocking opportunities; `review_demand_patterns` calls `search_quote_history`
   to surface recurring customer demand the catalog does not currently capture.

All seven provided helper functions are used inside these tool definitions, and a structural
test fails the build if that mapping ever drifts.

Two decisions shaped this architecture. First, the split between probabilistic and deterministic
work: language models are good at reading messy customer prose and writing warm, clear
responses, and bad at being trusted with arithmetic and side effects. So every tool computes its
result in plain Python, records it on shared typed state, and the final customer response is
assembled from that recorded state; the model's text contributes only the summary sentence,
behind a guard that blocks internal terms (cash balances, margins, error internals). A model
that misquotes a number cannot change what was committed or charged. Second, the agent
separation: each agent owns exactly one business authority (orchestration, inventory, quoting,
fulfillment, advisory) with least-privilege tools, which keeps responsibilities non-overlapping
and failures easy to localize. The business advisor is deliberately read-only - it reads
`generate_financial_report` output to identify demand gaps and pre-stocking opportunities across
the run, but never places orders or modifies any records. Keeping it read-only is the governance
boundary that makes it safe to run automatically after each evaluation.

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
  declined at commitment (`stock_changed_before_commit`) because orders committed earlier in
  the run consumed the stock: one line lost to a same-day order, and three to earlier requests
  whose sales were recorded at future delivery dates and so became visible only when the commit
  revalidated stock on the delivery date. Without the revalidation these would have been
  oversold into negative inventory; the run ended with zero negative-inventory items.
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

### Production safety gaps

**Retry without idempotency is the most urgent issue.** The evaluation harness retries each
request once on exception. `AgentDependencies` is fresh on each retry, which resets the
in-memory deduplication guards (`fulfilled_lines`, `stock_order_ids`). A request that partially
commits on attempt 1 - restock lands, then an LLM timeout kills the run before the sale commits
- will re-run all tool calls on attempt 2 and write a second restock and a second sale. The
test in `tests/integration/test_financial_safety.py` documents this current behavior explicitly.
An idempotency key table closes this gap: the `request_id` written with a UNIQUE constraint
before any tool runs, and checked on retry to identify and skip already-committed work.

**The TOCTOU in `commit_sale` and `place_restock_order` is dormant, not closed.** Both tools
read a database value (stock level or cash balance) and then write a transaction in two separate
calls with no lock held between them. In the current single-process sequential evaluation model
no two requests interleave, so this never fires. It fires the moment any concurrent processing
is added - multiple workers, an async request queue, or two requests for the same item in the
same batch. `commit_sale` already has a revalidation guard that declines the sale rather than
overselling when it detects stock has moved (verified in `test_financial_safety.py`), but the
guard detects the symptom rather than closing the race. The structural fix is database-level: holding the write lock across the read-check-write
sequence - `BEGIN IMMEDIATE` in SQLite or `SELECT ... FOR UPDATE` in PostgreSQL.

**A committed restock with a failed sale leaves an orphaned ledger entry.** If
`place_restock_order` writes its transaction and the run then fails before `commit_sale` runs,
the restock is permanent: cash is debited, inventory is credited, and no sale is ever recorded.
The fail-safe backstop rejects the request but does not issue a compensating transaction. A
two-phase commit design - recording the restock as PENDING and confirming it only when the sale
commits - would close this gap without requiring a full saga framework.

**The forbidden-term denylist is not a sufficient customer safety boundary on its own.** The
denylist catches exact substrings and misses Unicode homoglyphs, pluralization, hyphenation,
and semantic paraphrasing (a model can reveal cash position without using any listed term).
The structural protection - `CustomerResponse` assembled from tool-recorded state rather than
model text - is the real boundary. The model's summary sentence is the only prose field that can
carry a leak, and it should be treated as untrusted text replaced by a safe template on any
ambiguity, not scanned for known strings. Unicode normalization (NFC) is applied before the
denylist check as a baseline hardening measure; structural output constraints are the correct
long-term approach.

**The `customer_context` field is a prompt injection surface.** It is interpolated directly into
the orchestrator's system prompt. A customer who controls this field can inject instructions
that manipulate which tools the orchestrator calls and in what order. The `financial_health_report`
tool, registered on the orchestrator, returns `cash_balance` and `total_assets` and is reachable
by injection. The correct boundary separates LLM extraction - a capability-reduced step with no access to
financial tools - from orchestration, so that only the structured result (product lines,
quantity, deadline) ever reaches the orchestration step.

### Capability extensions

**Clarification into the batch flow.** The negotiation mode already holds every stock and ledger
action while a line can still be clarified, asks the customer to answer, and names the nearest
catalog items so the customer can reply with a resolvable product name rather than a
specification. The remaining step is bringing that loop into the batch flow through an async
follow-up email channel or a human-in-the-loop escalation path - the four `ambiguous` declines
in the evaluation run represent real demand the system currently fails closed on.

**Demand-aware restocking.** The reorder policy is a static top-up multiplier. Replacing it
with a forecast built from the quote-history table - where seasonality by event type is visible
in the data - would reduce both `supplier_after_deadline` rejections (by pre-stocking ahead of
known demand peaks) and `stock_changed_before_commit` contention between overlapping orders.

**The business advisor agent** (`munder-difflin advise`) provides read-only analysis across the
committed transaction ledger: which catalog items have demand but zero stock, which supplier
lead times create the most rejections, and how the discount tiers affect realized margin across
the run. Its recommendations identify pre-stocking opportunities and operational patterns that
the per-request agents cannot see. It is strictly read-only and cannot place orders or modify
any records.
