# Production Roadmap

This roadmap starts from the risks created when a classroom simulation begins accepting real
orders. It prioritizes correctness and recoverability before adding more autonomous behavior.

## Current baseline

The repository already provides:

- four Pydantic AI agents (orchestrator, inventory, quoting, fulfillment) over deterministic
  tools;
- typed contracts for every agent handoff and tool result;
- deterministic catalog, pricing, inventory, and safety policies;
- structured trace events;
- executable evaluation gates; and
- a customer-safe response projection with a leak guard.

These controls make the simulation inspectable, but they do not make SQLite, file-based input, or
automatic quote acceptance suitable for real commerce.

## Deferred by design

The following were considered during development and deliberately left out of the current
system because no correctness need justified their complexity. They remain the first candidates
for the phases below:

- **Business advisor agent.** A fifth, read-only agent that analyzes the transaction ledger and
  recommends operational changes. It belongs in Phase 7 once evaluation and governance exist.
- **Request-level idempotency and replay cache.** Safe replays of duplicate requests matter for a
  real API surface (Phase 1); the batch evaluation resets its database per run and never
  resubmits, so the cache earned nothing in the classroom system.
- **Row-level transaction dedupe and atomic multi-line commits.** The current system commits
  line by line with a pre-commit stock revalidation, which is honest for a single-threaded
  simulation. Real concurrency requires the Phase 1 state machine plus database constraints.
- **Customer negotiation simulator.** An external agent that plays the customer and negotiates
  with the team. Useful as a Phase 5 evaluation layer, not as part of the company system.

## Phase 1: Separate quote, order, and fulfillment state

Introduce an explicit state machine:

`REQUESTED -> QUOTED -> CUSTOMER_ACCEPTED -> RESERVED -> FULFILLED -> INVOICED`

Cancellation and expiry paths become first-class states rather than ad hoc exceptions. A quote
must never create a sale until the customer accepts it. Inventory reservations receive expiry
times and are released automatically.

Exit criteria:

- every state transition is idempotent and auditable;
- duplicate webhooks cannot create duplicate orders;
- expired quotes release reservations;
- compensation tests prove failed multi-line orders leave no partial state.

## Phase 2: Durable service and integration boundaries

Expose commands through a FastAPI service and move state to Postgres with schema migrations.
Separate synchronous customer APIs from asynchronous supplier and fulfillment work using a durable
queue. Use a transactional outbox so database commits and emitted events cannot diverge.

Integrate through narrow adapters:

- ERP for customer, order, and invoice records;
- warehouse management for available-to-promise inventory;
- supplier APIs for price and delivery commitments;
- identity provider for workforce access; and
- email or CRM channels for inbound and outbound communication.

Exit criteria:

- horizontal workers process the same request safely;
- an integration outage cannot lose an accepted order;
- retry and dead-letter behavior is tested;
- reconciliation detects drift between the ledger, ERP, and warehouse systems.

## Phase 3: Reliability and human authority

Add timeouts, bounded retries with jitter, circuit breakers, and provider fallbacks around model
and supplier calls. Put human approval in front of high-value quotes, policy exceptions, new
catalog mappings, and recommendations that would change pricing or purchasing.

Define service objectives separately for:

- request acknowledgement;
- quote completion;
- order confirmation;
- supplier commitment; and
- customer notification.

Runbooks must name the owner, evidence, rollback, and customer communication path for each failure
class.

Exit criteria:

- chaos tests cover unavailable models, suppliers, queues, and databases;
- operators can replay or compensate a failed request from its trace;
- model failure degrades to deterministic handling or human review;
- no automated recommendation changes a policy without accountable approval.

## Phase 4: Observability and cost control

Map `RunEvent` to OpenTelemetry spans with one trace across API, agent, tool, database, and
integration calls. Record model, prompt version, tool schema version, latency, tokens, cost,
retries, and outcome without storing unnecessary customer text.

Dashboards should connect technical signals to business outcomes:

- quote latency and abandonment;
- full, partial, and rejected fulfillment rates;
- rejection reasons and recoverable demand;
- reservation expiry and inventory contention;
- gross-margin bands without exposing them to customers;
- model and tool error rates;
- cost per completed quote.

Exit criteria:

- every customer outcome is explainable from one trace;
- alerts are tied to service objectives rather than raw log volume;
- cost regressions fail release gates;
- telemetry retention follows privacy and residency policy.

## Phase 5: Evaluation and controlled release

Build three evaluation layers:

1. deterministic unit and contract tests on every change;
2. a versioned golden set covering parsing, tools, policies, and customer explanations;
3. production monitoring calibrated against human review samples.

Use LLM-as-judge only for subjective communication qualities, and calibrate it against human
ratings before trusting aggregates. Keep financial arithmetic, tool authorization, inventory
invariants, and leakage checks programmatic.

Prompt, model, tool, and policy changes receive separate versions. Release with shadow traffic,
then canary cohorts, then progressive rollout. Rollback must restore both model configuration and
policy version.

Exit criteria:

- regressions are attributable to a specific versioned component;
- canary rollback is rehearsed;
- judge agreement with humans is measured and reported;
- drift alerts distinguish input changes from model behavior changes.

## Phase 6: Security, privacy, and governance

Apply least-privilege service identities to every tool and data store. Classify inbound text as
untrusted, scan attachments outside the agent runtime, and prevent retrieved content from granting
new authority. Encrypt data in transit and at rest, rotate secrets, and keep credentials out of
prompts and traces.

Add:

- customer and employee authentication;
- role and attribute-based authorization;
- tenant isolation if the service becomes multi-company;
- PII minimization and purpose-limited retention;
- regional deployment and data-residency controls;
- signed audit evidence for high-risk decisions;
- dependency, container, and secret scanning;
- incident response and notification procedures.

Exit criteria:

- threat modeling covers prompt injection, tool misuse, data exfiltration, and supply chain risk;
- authorization is enforced by services, never only by agent instructions;
- privacy deletion propagates through operational and analytical stores;
- external security testing closes all high-severity findings.

## Phase 7: Business optimization without unsafe autonomy

A business advisor agent (deferred by design, above) can graduate from descriptive
recommendations to constrained experiments only after the earlier controls exist. Candidate capabilities include demand forecasting, supplier scorecards,
quote-expiry optimization, and inventory policy simulation.

Recommendations should include evidence, expected impact, uncertainty, affected stakeholders, and
a rollback condition. Humans remain accountable for pricing, credit, procurement, and customer
policy.

Exit criteria:

- experiments use predeclared success and guardrail metrics;
- recommendations are compared with a non-agent baseline;
- realized outcomes feed back into evaluation;
- the organization can explain who approved every material policy change.
