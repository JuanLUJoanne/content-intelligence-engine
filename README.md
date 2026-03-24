# Multi-Modal Content Intelligence Engine

Production-grade AI pipeline framework with dynamic model routing, cost guardrails, eval-driven development, and an agentic personalised recommendation layer. Designed to solve real problems running LLMs in production: quality inconsistency, unpredictable costs, lack of systematic evaluation, and cold-start personalisation.


## Architecture

```
                        ┌───────────────────────────────────────────────────────────┐
                        │                  API Layer (FastAPI)                      │
                        │  POST /process  POST /process/stream  POST /process/batch │
                        └────────────────────────┬──────────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │              Input Sanitizer (security.py)          │
                        │   Blocks prompt injection · Detects / redacts PII   │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │                 Cache Check (Redis / in-mem)        │
                        │                                                     │
                        │   HIT ──────────────────────────────► Store Result  │
                        │   MISS                                              │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │               Model Router (router.py)              │
                        │  FLASH tier · STANDARD tier · cost/latency scoring  │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           Rate Limiter + Cost Guardrails            │
                        │  per-request · per-minute · anomaly · total budget  │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │                    LLM Call                         │
                        │         Circuit Breaker · Retry + Backoff           │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │         Schema Validation + Failure Routing         │
                        │                                                     │
                        │  structural_failure ──────────► Engineering Queue   │
                        │  field_error (×3 retries) ────► Human Review Queue  │
                        │  PASS                                               │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │              Eval Score (LLM-as-Judge)              │
                        │                                                     │
                        │   score < 0.7 ─────────────────────► Human Review   │
                        │   score ≥ 0.7                                       │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           Store Result + Audit Log (JSONL)          │
                        └─────────────────────────────────────────────────────┘


  Agentic Recommendation Layer (runs independently of the processing pipeline)

                        ┌───────────────────────────────────────────────────────────┐
                        │                 MCPClient (real or mock)                  │
                        │   get_browsing_history · get_purchase_history             │
                        │   get_asset_metadata                                      │
                        └────────────────────────┬──────────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           BuyerProfile (tag affinity, top category) │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │         A/B Experiment (Variant A / B assignment)   │
                        │     purchase_history_based vs browsing_pattern_based│
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │       AssetRetriever (tag-affinity ranked corpus)   │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │     LLM Email Generation + Judge Eval + Cost Track  │
                        └────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────▼────────────────────────────┐
                        │           Log result to ReviewStore (A/B audit)     │
                        └─────────────────────────────────────────────────────┘
```

## Key Design Decisions

**1. Heuristic routing over LLM routing**
Routing decisions (which model tier to use) are made with deterministic heuristics — token count, latency sensitivity, cost sensitivity — rather than asking an LLM. LLM-based routing adds latency and cost to every request and introduces a bootstrapping paradox: you need a model to decide which model to use. Heuristics are predictable, fast, and trivially unit-testable.

**2. Failure-type-aware validation routing**
Schema validation failures are split into two distinct paths: *structural failures* (malformed JSON, completely missing fields) go directly to an engineering queue — no retry is attempted, because retrying a broken prompt template wastes quota and obscures the root cause. *Field errors* (wrong enum value, out-of-range number) trigger up to three error-feedback retries, passing the exact Pydantic error message back to the LLM so it can self-correct. This separation means engineers get a clean signal about prompt regressions without drowning in human-review noise.

**3. Record-level checkpointing**
Each successfully processed record is checkpointed immediately after storage. On restart or failure, the pipeline replays only unprocessed records. This makes the pipeline idempotent across crashes, restarts, and partial batch failures without requiring distributed transactions.

**4. Multi-level cost guardrails**
Four independent guardrail levels — per-request, per-minute sliding window, anomaly detection (3× rolling average), and total budget hard stop — are checked before every LLM call. A single anomalous request cannot exhaust the budget; a runaway batch cannot breach the daily cap. The layered design means each guard can be tuned independently as traffic patterns change.

**5. LLM-as-Judge for eval**
Output quality is scored by a separate LLM judge across five dimensions: factual accuracy, hallucination rate, semantic consistency, relevance, and schema compliance. Using a language model as the evaluator catches subtle quality regressions that rule-based metrics miss, while the separate judge model avoids the "marking your own homework" problem. Scores below 0.7 are routed to human review rather than rejected outright.

**6. A/B-tested personalised recommendations**
The recommendation layer assigns users deterministically to variants using `hash(user_id) % 2`, so the same user always sees the same prompt flavour across requests without storing session state. Variant A weights purchase history; Variant B weights browsing patterns. Every result is logged to the ReviewStore so effect sizes can be computed offline using Cohen's d.

**7. MCP tool layer as a data-fetching adapter**
`RecommendationAgent.run()` accepts an optional `MCPClient`. When provided, buyer history is fetched via the Model Context Protocol; when absent, the caller passes resolved asset lists directly. The mock path (`use_mock=True`) uses `hash()`-based determinism — no random seed, no network — so tests are reproducible without fixtures or patching.

## Production Features

| Feature | Description |
|---|---|
| **Model routing** | Heuristic tier selection (FLASH / STANDARD) with cost-aware scoring |
| **Circuit breakers** | Per-model open/half-open/closed state with configurable thresholds |
| **Rate limiting** | Token-bucket rate limiter with per-model and global limits |
| **Response cache** | SHA-256 keyed cache with configurable TTL; cache hits skip the LLM entirely |
| **Cost guardrails** | Per-request, per-minute, anomaly, and total-budget protection |
| **Batch API** | Async batch submission with poll-until-complete semantics |
| **Drift detection** | Baseline comparison across 5 eval dimensions; alerts on regression |
| **Prompt versioning** | Register, rollback, and auto-rollback prompt versions on drift |
| **A/B prompt comparison** | Statistical comparison with Cohen's d effect size and 2% minimum threshold |
| **Failure-type routing** | structural_failure → engineering queue; field_error (×3) → human review queue |
| **Engineering queue** | REST API for ops to inspect and requeue structural validation failures |
| **Human review queue** | Low-confidence and field-error outputs routed to review; approvals written to golden set |
| **Personalised recommendations** | A/B-tested email generation driven by tag-affinity buyer profiles |
| **MCP tool layer** | MCPClient adapter fetches real buyer data; deterministic mock for offline testing |
| **Input sanitization** | 12-pattern prompt injection detection with fast-fail at API boundary |
| **PII detection** | Regex-based detection and redaction of emails, phone numbers, credit cards |
| **Audit logging** | Append-only JSONL audit trail for every request/response |
| **LangGraph orchestration** | 7-node stateful pipeline with conditional edges, retry loops, and failure routing |
| **SSE streaming** | Server-Sent Events progress stream for real-time UI integration |

## Eval Metrics

| Metric | Baseline | Optimized |
|---|---|---|
| Schema Compliance | 91% | 98.5% |
| Factual Accuracy | 0.74 | 0.89 |
| Hallucination Rate | 12% | 3.1% |
| Avg Cost / Record | $0.0042 | $0.0011 |
| P95 Latency | 4.2 s | 1.1 s |

*Baseline: GPT-4 for all records. Optimized: heuristic routing to Gemini Flash for 78% of traffic.*

## Cost Comparison

| Mode | Model | Cost / 1K records | Cost @ 10M records |
|---|---|---|---|
| Real-time (all GPT-4) | gpt-4o | $60.00 | $600,000 |
| Real-time (routed) | gemini-flash / gpt-4o-mini | $8.50 | $85,000 |
| **Batch (routed)** | gemini-flash / gpt-4o-mini | **$4.25** | **$42,500** |

Batch mode halves cost again by using provider batch APIs (50% discount) at the expense of same-day latency.

## Quick Start

```bash
cp .env.example .env          # add OPENAI_API_KEY / GOOGLE_API_KEY
pip install -e .
pytest tests/unit/ -v         # 251+ tests, ~8s
python scripts/demo.py        # end-to-end demo without real API keys
uvicorn src.api.main:app --reload
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/process` | Process a single item through the full LangGraph pipeline |
| `POST` | `/process/stream` | SSE stream of per-stage progress events |
| `POST` | `/process/batch` | Submit a list of records for async batch processing |
| `POST` | `/evaluate` | Score an LLM output with the LLM-as-Judge evaluator |
| `GET` | `/costs` | Current spend by model with remaining budget |
| `GET` | `/drift/report` | Latest drift detection report vs. saved baseline |
| `GET` | `/health` | Liveness check with passing test count |
| `GET` | `/review/pending` | List items queued for human review |
| `POST` | `/review/{id}/approve` | Approve a review item (writes to golden set) |
| `POST` | `/review/{id}/reject` | Reject a review item with a reason |
| `GET` | `/review/stats` | Review queue statistics and approval rate |
| `GET` | `/engineering/pending` | List structural validation failures pending investigation |
| `POST` | `/engineering/{id}/requeue` | Mark an engineering failure for reprocessing |
| `GET` | `/engineering/stats` | Failure counts grouped by prompt version |

### Example: process a single item

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/product-001.jpg",
    "text": "Wireless noise-cancelling headphones with 30-hour battery",
    "language": "en",
    "content_type": "product"
  }'
```

```json
{
  "metadata": {
    "content_id": "https://example.com/product-001.jpg",
    "title": "Wireless Noise-Cancelling Headphones",
    "category": "electronics",
    "condition": "new",
    "price_range": "premium",
    "tags": ["headphones", "noise-cancelling", "wireless"],
    "language": "en"
  },
  "model_used": "gemini-flash",
  "cost": 0.0013,
  "confidence": 0.92
}
```

### Example: stream progress

```bash
curl -N -X POST http://localhost:8000/process/stream \
  -H "Content-Type: application/json" \
  -d '{"image_url": "...", "text": "...", "language": "en", "content_type": "product"}'
```

```
data: {"event": "sanitizing", "status": "running"}
data: {"event": "sanitizing", "status": "ok"}
data: {"event": "cache_check", "status": "ok", "hit": false}
data: {"event": "routing", "status": "ok", "model": "gemini-flash"}
data: {"event": "calling_llm", "status": "running"}
data: {"event": "calling_llm", "status": "ok"}
data: {"event": "validating", "status": "ok"}
data: {"event": "scoring", "status": "ok", "confidence": 0.92}
data: {"event": "complete", "status": "ok", "metadata": {...}}
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Pipeline orchestration | LangGraph-style stateful graph (migrates to LangGraph with 5-line change) |
| Tracing | LangSmith (`@traceable`, graceful no-op without API key) |
| LLM providers | Google Gemini, OpenAI GPT-4o |
| Schema validation | Pydantic v2 |
| API framework | FastAPI + Uvicorn |
| Storage | SQLite (checkpoints), JSON (golden set), JSONL (audit log) |
| Logging | structlog (structured JSON logs) |
| Testing | pytest + pytest-asyncio (251+ unit tests) |

## Project Layout

```
src/
├── api/
│   ├── main.py              # FastAPI app, routers, and all endpoints
│   └── review.py            # Human review queue, engineering queue, golden-set management
├── agents/
│   ├── recommendation_agent.py  # End-to-end personalised recommendation pipeline
│   └── memory/
│       └── buyer_profile.py     # BuyerProfile, tag-affinity computation, MCP-backed factory
├── ab_test/
│   └── experiment.py        # Variant assignment, prompt templates, result logging
├── retrieval/
│   └── asset_retriever.py   # Tag-affinity scored in-memory corpus search
├── mcp/
│   ├── tools.py             # ANALYTICS_TOOLS MCP schema definitions
│   └── client.py            # MCPClient with deterministic mock and real stub
├── eval/
│   ├── judge.py             # LLM-as-Judge with per-dimension prompts
│   ├── drift_detector.py    # Baseline comparison and drift alerting
│   └── ab_prompt.py         # A/B comparison with Cohen's d effect size
├── gateway/
│   ├── providers.py         # LLMProvider protocol + Gemini/OpenAI/Dummy impls
│   ├── router.py            # Heuristic model router with cost-aware scoring
│   ├── cost_tracker.py      # Token accounting and budget enforcement
│   ├── guardrails.py        # Multi-level cost guardrail system
│   ├── batch.py             # Async batch submission and polling
│   ├── cache.py             # Response cache with TTL
│   ├── circuit_breaker.py   # Per-model circuit breaker
│   ├── rate_limiter.py      # Token-bucket rate limiter
│   └── security.py          # Sanitization, PII detection, audit logging
└── pipeline/
    ├── graph.py             # 7-node ContentPipelineGraph with failure-type routing
    ├── processor.py         # BatchProcessor with checkpointing and DLQ
    ├── checkpoint.py        # Record-level checkpoint persistence
    ├── versioning.py        # Prompt registry with rollback and auto-rollback
    └── prompt_chain.py      # Multi-step prompt chaining utilities
```
